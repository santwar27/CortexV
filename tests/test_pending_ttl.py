"""
Staged-action TTL (M1-A9, review Fable 5).

The staged S3 action survives restarts by design — but it used to
survive FOREVER: "staged last night, forgotten, 'sim' today to
something else" executed a stale email. The drop-on-other-input guard
only helps when there IS an intermediate turn. Now:

  - every staging path stamps `staged_at` (single chokepoint:
    `_persist_pending_action`, which all staging sites call);
  - an approval to an expired slot discards it and SAYS so (a silently
    eaten "sim" reads as a missed command);
  - an expired file is discarded at load (no user to warn → audit);
  - missing/unparseable `staged_at` counts as expired — unknown age
    fails safe.
"""
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.core.orchestrator import CortexOrchestrator


def _iso(minutes_ago: float) -> str:
    return (datetime.now(timezone.utc)
            - timedelta(minutes=minutes_ago)).isoformat()


def _orch(tmp_path, monkeypatch, pending=None):
    monkeypatch.setattr(
        CortexOrchestrator, "_PENDING_ACTION_FILE",
        str(tmp_path / "pending_action.json"),
    )
    o = CortexOrchestrator.__new__(CortexOrchestrator)
    o._pending_action = pending
    o._pending_query = None
    o._reply_lang = "pt"
    o._shadow_classify = lambda text: None
    o._execute_pending_action = MagicMock(
        return_value=("Enviado.", "AGENT_EXECUTED"))
    o._cancel_pending_action = MagicMock(
        return_value=("Cancelado.", "AGENT_CANCELLED"))
    return o


class TestStagedAtStamp:
    def test_persist_stamps_staged_at(self, tmp_path, monkeypatch):
        o = _orch(tmp_path, monkeypatch,
                  {"type": "send_email", "args": {}})
        o._persist_pending_action()
        assert "staged_at" in o._pending_action
        on_disk = json.loads((tmp_path / "pending_action.json").read_text())
        assert on_disk["staged_at"] == o._pending_action["staged_at"]

    def test_repersist_does_not_refresh_age(self, tmp_path, monkeypatch):
        o = _orch(tmp_path, monkeypatch,
                  {"type": "send_email", "args": {},
                   "staged_at": _iso(10)})
        first = o._pending_action["staged_at"]
        o._persist_pending_action()
        assert o._pending_action["staged_at"] == first


class TestApprovalTTL:
    def test_fresh_sim_executes(self, tmp_path, monkeypatch):
        o = _orch(tmp_path, monkeypatch,
                  {"type": "send_email", "args": {}, "staged_at": _iso(1)})
        reply, mode = o._process_core("sim")
        assert mode == "AGENT_EXECUTED"
        o._execute_pending_action.assert_called_once()

    def test_expired_sim_discards_and_says_so(self, tmp_path, monkeypatch):
        o = _orch(tmp_path, monkeypatch,
                  {"type": "send_email", "args": {}, "staged_at": _iso(16)})
        reply, mode = o._process_core("sim")
        assert mode == "STALE_ACTION_DISCARDED"
        assert "descartei" in reply and "15" in reply
        assert o._pending_action is None
        o._execute_pending_action.assert_not_called()

    def test_expired_nao_also_acked_not_cancelled(self, tmp_path, monkeypatch):
        o = _orch(tmp_path, monkeypatch,
                  {"type": "send_email", "args": {}, "staged_at": _iso(16)})
        reply, mode = o._process_core("não")
        assert mode == "STALE_ACTION_DISCARDED"
        o._cancel_pending_action.assert_not_called()

    def test_missing_staged_at_fails_safe(self, tmp_path, monkeypatch):
        # Pre-M1-A9 persisted action (or hand-edited JSON): unknown age
        # must never execute.
        o = _orch(tmp_path, monkeypatch,
                  {"type": "send_email", "args": {}})
        reply, mode = o._process_core("sim")
        assert mode == "STALE_ACTION_DISCARDED"
        o._execute_pending_action.assert_not_called()

    def test_expired_with_unrelated_input_routes_normally(
        self, tmp_path, monkeypatch
    ):
        o = _orch(tmp_path, monkeypatch,
                  {"type": "send_email", "args": {}, "staged_at": _iso(16)})
        reply, mode = o._process_core("que horas são")
        # Slot discarded silently; the new query flows (reflex answers).
        assert mode == "REFLEX"
        assert o._pending_action is None

    def test_en_notice(self, tmp_path, monkeypatch):
        o = _orch(tmp_path, monkeypatch,
                  {"type": "send_email", "args": {}, "staged_at": _iso(16)})
        reply, mode = o._process_core("yes", language="en")
        assert mode == "STALE_ACTION_DISCARDED"
        assert "discarded it for safety" in reply


class TestLoadTTL:
    def _write(self, tmp_path, action):
        (tmp_path / "pending_action.json").write_text(json.dumps(action))

    def test_fresh_file_loads(self, tmp_path, monkeypatch):
        o = _orch(tmp_path, monkeypatch)
        self._write(tmp_path, {"type": "send_email", "args": {},
                               "staged_at": _iso(2)})
        loaded = o._load_pending_action()
        assert loaded is not None and loaded["type"] == "send_email"

    def test_stale_file_discarded_and_removed(self, tmp_path, monkeypatch):
        o = _orch(tmp_path, monkeypatch)
        self._write(tmp_path, {"type": "send_email", "args": {},
                               "staged_at": _iso(120)})
        assert o._load_pending_action() is None
        assert not (tmp_path / "pending_action.json").exists()

    def test_pre_ttl_file_without_stamp_discarded(self, tmp_path, monkeypatch):
        o = _orch(tmp_path, monkeypatch)
        self._write(tmp_path, {"type": "send_email", "args": {}})
        assert o._load_pending_action() is None

    def test_ttl_env_tunable(self, tmp_path, monkeypatch):
        o = _orch(tmp_path, monkeypatch)
        monkeypatch.setattr(type(o), "_PENDING_TTL_MIN", 60.0)
        self._write(tmp_path, {"type": "send_email", "args": {},
                               "staged_at": _iso(30)})
        assert o._load_pending_action() is not None
