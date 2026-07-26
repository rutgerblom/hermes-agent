"""Closure regressions for shared Desktop profile isolation and async events."""

from __future__ import annotations

import queue
import threading
from contextlib import contextmanager

from tui_gateway import server


class _CapturedSessionDB:
    created: list[dict] = []

    def __init__(self, *, db_path):
        self.db_path = db_path

    def create_session(self, key, source=None, model=None, model_config=None, parent_session_id=None, cwd=None, profile_name=None):
        self.created.append(
            {
                "key": key,
                "model": model,
                "model_config": model_config,
                "profile_name": profile_name,
            }
        )

    def close(self):
        pass


def test_profile_session_initial_row_uses_profile_config_not_ambient_model(monkeypatch, tmp_path):
    """First DB write for a selected profile must not inherit dashboard launch state."""
    _CapturedSessionDB.created = []
    entered: list[str] = []

    @contextmanager
    def profile_scope(home):
        entered.append(str(home))
        yield

    monkeypatch.setattr("hermes_state.SessionDB", _CapturedSessionDB)
    monkeypatch.setattr(server, "profile_runtime_scope", profile_scope)
    monkeypatch.setattr(server, "_config_model_target", lambda: ("profile/default", "openai-codex"))
    monkeypatch.setattr(server, "_resolve_model", lambda: "ambient/dashboard-model")

    server._ensure_session_db_row(
        {
            "session_key": "profile-session",
            "profile_home": str(tmp_path / "selected-profile"),
            "profile_name": "selected-profile",
        }
    )

    assert entered == [str(tmp_path / "selected-profile")]
    assert _CapturedSessionDB.created == [
        {
            "key": "profile-session",
            "model": "profile/default",
            "model_config": None,
            "profile_name": "selected-profile",
        }
    ]


def test_notification_poller_emits_one_message_start_for_one_async_completion(monkeypatch):
    """The completion runner owns its single message.start frame, not the poller."""
    from tools import async_delegation
    from tools.process_registry import process_registry

    stop = threading.Event()
    completion_queue = queue.Queue()
    completion_queue.put(
        {
            "type": "async_delegation",
            "delegation_id": "deleg_once",
            "session_key": "owner-key",
            "origin_ui_session_id": "desktop-tab",
            "results": [{"status": "completed"}],
        }
    )
    session = {
        "history_lock": threading.Lock(),
        "running": False,
        "_finalized": False,
        "session_key": "owner-key",
    }
    emitted: list[str] = []
    deliveries: list[str] = []

    monkeypatch.setattr(process_registry, "completion_queue", completion_queue)
    monkeypatch.setattr(server, "_notification_event_belongs_elsewhere", lambda *_args: False)
    monkeypatch.setattr(server, "_notification_event_requires_owner", lambda _evt: True)
    monkeypatch.setattr(server, "_session_owns_notification_event", lambda *_args: True)
    monkeypatch.setattr("tools.process_registry.format_process_notification", lambda _evt: "completion")
    monkeypatch.setattr(async_delegation, "claim_event_delivery", lambda _evt, _consumer: "claim")
    monkeypatch.setattr(async_delegation, "complete_event_delivery", lambda _evt, claim: deliveries.append(claim))
    monkeypatch.setattr(async_delegation, "release_event_delivery", lambda *_args: None)
    monkeypatch.setattr(server, "_emit", lambda event, *_args, **_kwargs: emitted.append(event))

    def run_prompt(_rid, _sid, active_session, _text, **_kwargs):
        server._emit("message.start", "desktop-tab")
        active_session["running"] = False
        stop.set()

    monkeypatch.setattr(server, "_run_prompt_submit", run_prompt)

    server._notification_poller_loop(stop, "desktop-tab", session)

    assert emitted.count("message.start") == 1
    assert deliveries == ["claim"]
