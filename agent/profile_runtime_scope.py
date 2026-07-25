"""Context-local, serialized profile runtime scope for shared surfaces.

A shared dashboard can serve several local Hermes profiles from one process.
This module binds one selected profile's home and native secret mapping together
without copying resolved provider credentials into ``os.environ``.

The lock deliberately covers the full scoped interval.  Most Hermes paths are
contextvar-safe, but a few third-party and legacy helpers still cache/import
process-global state.  Serializing the interval makes those paths deterministic
while the secret mapping itself remains context-local.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from threading import RLock
from typing import Iterator

from agent.secret_scope import (
    current_secret_scope,
    is_multiplex_active,
    reset_secret_scope,
    set_secret_scope,
)
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


_PROFILE_RUNTIME_SCOPE_LOCK = RLock()


def _initial_scope(home: Path) -> dict[str, str]:
    """Read profile bootstrap env files without mutating ``os.environ``."""
    from agent.secret_scope import load_env_file

    scope = load_env_file(home / ".env")
    # ``.op.env`` supplies only bootstrap authentication for native 1Password.
    # It must not override an explicit profile .env entry.
    for key, value in load_env_file(home / ".op.env").items():
        scope.setdefault(key, value)
    return scope


def _refresh_native_secret_sources(home: Path, scope: dict[str, str]) -> None:
    """Resolve enabled native sources directly into the local scope mapping.

    ``apply_all(..., environ=scope)`` preserves the established source
    precedence rules while avoiding its default process-global environment.
    Errors intentionally remain fail-closed at the consumer: an unavailable
    selected-profile credential is absent from the scope and cannot borrow an
    ambient or another profile's value.
    """
    from hermes_cli.env_loader import _load_secrets_config
    from agent.secret_sources.registry import apply_all

    cfg = _load_secrets_config(home)
    if cfg:
        apply_all(cfg, home, environ=scope)


def profile_secret(name: str, default: str = "") -> str:
    """Read a selected-profile value without ambient-environment fallback.

    Profile-sensitive discovery paths that historically used ``os.environ``
    call this helper. Outside a selected runtime scope the caller receives
    ``default``; no dashboard-process credential is ever considered.
    """
    scope = current_secret_scope()
    if scope is None:
        return default if is_multiplex_active() else str(os.environ.get(name) or default)
    return str(scope.get(name) or default)


@contextmanager
def profile_runtime_scope(profile_home: Path | str) -> Iterator[None]:
    """Bind one profile's home and secrets for a complete sensitive interval.

    Supports nesting through ``RLock`` and ContextVar tokens.  The local mapping
    is cleared after reset so resolved values are not retained by this helper.
    """
    home = Path(profile_home).resolve()
    if not home.is_dir():
        raise ValueError("selected profile home is unavailable")

    with _PROFILE_RUNTIME_SCOPE_LOCK:
        home_token = set_hermes_home_override(home)
        scope = _initial_scope(home)
        secret_token = set_secret_scope(scope)
        try:
            _refresh_native_secret_sources(home, scope)
            yield
        finally:
            reset_secret_scope(secret_token)
            reset_hermes_home_override(home_token)
            scope.clear()
