"""Behavioral coverage for shared Desktop selected-profile isolation."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from agent import secret_scope
from agent.profile_runtime_scope import profile_runtime_scope
from hermes_constants import get_hermes_home, get_hermes_home_override
from hermes_cli.inventory import build_model_options_payload, load_picker_context
from hermes_cli.runtime_provider import resolve_runtime_provider
from tui_gateway import server


@pytest.fixture(autouse=True)
def _clean_scope():
    secret_scope.set_multiplex_active(True)
    yield
    secret_scope.set_multiplex_active(False)


def _profile(root: Path, name: str, marker: str | None) -> Path:
    home = root / "profiles" / name
    home.mkdir(parents=True)
    (home / "state.db").touch()
    (home / "config.yaml").write_text(
        "model:\n  provider: openrouter\n  default: openai/gpt-5.6-sol\n",
        encoding="utf-8",
    )
    if marker is not None:
        (home / ".env").write_text(f"OPENROUTER_API_KEY={marker}\n", encoding="utf-8")
    return home


def _resolve(home: Path) -> dict:
    with profile_runtime_scope(home):
        return resolve_runtime_provider(
            requested="openrouter", target_model="openai/gpt-5.6-sol"
        )


def test_selected_profile_provider_resolution_and_construction_do_not_cross_leak(tmp_path, monkeypatch):
    blogger = _profile(tmp_path, "blogger", "synthetic-blogger")
    vcf = _profile(tmp_path, "vcf-expert", "synthetic-vcf")
    monkeypatch.setenv("OPENROUTER_API_KEY", "ambient-forbidden")

    blogger_runtime = _resolve(blogger)
    vcf_runtime = _resolve(vcf)
    assert blogger_runtime["api_key"] == "synthetic-blogger"
    assert vcf_runtime["api_key"] == "synthetic-vcf"

    # Exercise TUI agent construction after real provider resolution.  The
    # AIAgent constructor is replaced only at its network-free boundary.
    captured = []
    with (
        patch("tui_gateway.server._load_cfg", return_value={"model": {"provider": "openrouter", "default": "openai/gpt-5.6-sol"}, "agent": {}}),
        patch("tui_gateway.server._load_enabled_toolsets", return_value=[]),
        patch("run_agent.AIAgent", side_effect=lambda **kw: captured.append(kw) or object()),
    ):
        with profile_runtime_scope(blogger):
            server._make_agent("a", "a", model_override={"model": "openai/gpt-5.6-sol", "provider": "openrouter"})
        with profile_runtime_scope(vcf):
            server._make_agent("b", "b", model_override={"model": "openai/gpt-5.6-sol", "provider": "openrouter"})
    assert [item["api_key"] for item in captured] == ["synthetic-blogger", "synthetic-vcf"]


def test_sequential_nested_and_failure_restoration(tmp_path, monkeypatch):
    process_home = tmp_path / "neutral"
    process_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(process_home))
    a = _profile(tmp_path, "a", "marker-a")
    b = _profile(tmp_path, "b", "marker-b")

    with profile_runtime_scope(a):
        assert get_hermes_home() == a
        assert secret_scope.get_secret("OPENROUTER_API_KEY") == "marker-a"
        with profile_runtime_scope(b):
            assert get_hermes_home() == b
            assert secret_scope.get_secret("OPENROUTER_API_KEY") == "marker-b"
        assert get_hermes_home() == a
        assert secret_scope.get_secret("OPENROUTER_API_KEY") == "marker-a"

    assert get_hermes_home_override() is None
    assert secret_scope.current_secret_scope() is None
    with pytest.raises(RuntimeError):
        with profile_runtime_scope(a):
            raise RuntimeError("synthetic provider failure")
    with profile_runtime_scope(b):
        assert secret_scope.get_secret("OPENROUTER_API_KEY") == "marker-b"


def test_overlapping_scope_operations_are_serialized_and_isolated(tmp_path):
    a = _profile(tmp_path, "a", "marker-a")
    b = _profile(tmp_path, "b", "marker-b")
    entered_a = threading.Event()
    release_a = threading.Event()
    entered_b = threading.Event()
    seen: dict[str, str] = {}

    def worker_a():
        with profile_runtime_scope(a):
            seen["a"] = secret_scope.get_secret("OPENROUTER_API_KEY") or ""
            entered_a.set()
            assert release_a.wait(3)

    def worker_b():
        assert entered_a.wait(3)
        with profile_runtime_scope(b):
            seen["b"] = secret_scope.get_secret("OPENROUTER_API_KEY") or ""
            entered_b.set()

    ta = threading.Thread(target=worker_a)
    tb = threading.Thread(target=worker_b)
    ta.start(); tb.start()
    assert entered_a.wait(3)
    assert not entered_b.wait(0.1)
    release_a.set()
    ta.join(3); tb.join(3)
    assert seen == {"a": "marker-a", "b": "marker-b"}


def test_provider_visibility_and_no_secret_persistence(tmp_path, monkeypatch):
    with_key = _profile(tmp_path, "with-key", "synthetic-only-marker")
    without_key = _profile(tmp_path, "without-key", None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "ambient-forbidden")

    with profile_runtime_scope(with_key):
        providers = build_model_options_payload(load_picker_context(), explicit_only=True)["providers"]
        assert any(p["slug"] == "openrouter" and p.get("authenticated") for p in providers)
    with profile_runtime_scope(without_key):
        providers = build_model_options_payload(load_picker_context(), explicit_only=True)["providers"]
        assert not any(p["slug"] == "openrouter" and p.get("authenticated") for p in providers)

    # Scope creation/resolution must not write raw values into durable state.
    for path in tmp_path.rglob("*"):
        if path.is_file() and path.name not in {".env"}:
            assert "synthetic-only-marker" not in path.read_text(errors="ignore")
    assert "synthetic-only-marker" not in json.dumps(dict(os.environ))


def test_selected_profile_model_resolution_ignores_ambient_launch_seeds(tmp_path, monkeypatch):
    profile = _profile(tmp_path, "selected", None)
    monkeypatch.setenv("HERMES_MODEL", "ambient-model-forbidden")
    monkeypatch.setenv("HERMES_TUI_PROVIDER", "ambient-provider-forbidden")
    monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "ambient-provider-forbidden")

    with profile_runtime_scope(profile):
        assert server._resolve_model() == "openai/gpt-5.6-sol"
        assert server._resolve_startup_runtime() == ("openai/gpt-5.6-sol", None)


def test_model_picker_never_uses_ambient_profile_credentials(tmp_path, monkeypatch):
    with_key = _profile(tmp_path, "with-key", "synthetic-picker-marker")
    without_key = _profile(tmp_path, "without-key", None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "ambient-forbidden")

    with profile_runtime_scope(with_key):
        rows = build_model_options_payload(load_picker_context(), explicit_only=True)["providers"]
        assert any(row["slug"] == "openrouter" and row.get("authenticated") for row in rows)
    with profile_runtime_scope(without_key):
        rows = build_model_options_payload(load_picker_context(), explicit_only=True)["providers"]
        assert not any(row["slug"] == "openrouter" and row.get("authenticated") for row in rows)


def test_dispatch_serializes_selected_profile_metadata_calls(tmp_path, monkeypatch):
    a = _profile(tmp_path, "a", "marker-a")
    b = _profile(tmp_path, "b", "marker-b")
    monkeypatch.setattr(server, "_profile_home", lambda profile: {"a": a, "b": b}.get(profile))
    monkeypatch.setattr(server, "_hermes_home", tmp_path / "neutral")
    seen: list[tuple[Path, str]] = []

    @server.method("test.profile-scope")
    def handler(_rid, _params):
        seen.append((get_hermes_home(), secret_scope.get_secret("OPENROUTER_API_KEY") or ""))
        return {"result": "ok"}

    try:
        assert server.dispatch({"id": 1, "method": "test.profile-scope", "params": {"profile": "a"}}) == {"result": "ok"}
        assert server.dispatch({"id": 2, "method": "test.profile-scope", "params": {"profile": "b"}}) == {"result": "ok"}
    finally:
        server._methods.pop("test.profile-scope", None)
    assert seen == [(a, "marker-a"), (b, "marker-b")]
