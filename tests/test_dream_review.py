"""
Tests for the dream-review dispatch (Fase 2, DORMANT wiring).

DREAM_REVIEW_RE classifies show/keep/drop. The handler is gated by
CORTEX_DREAM_LOOP: OFF → None (fall through); ON → reads the DreamStore.
"""
import pytest

import src.core.dream_loop as dl
from src.core.dispatch.dream import DREAM_REVIEW_RE


def _action(m):
    return ("show" if m.group("show") else "keep" if m.group("keep") else "drop")


# --- regex -----------------------------------------------------------------

@pytest.mark.parametrize("text,action", [
    ("me conta o último sonho", "show"),
    ("mostra o sonho", "show"),
    ("ler sonho", "show"),
    ("show me the dream", "show"),
    ("guarda esse sonho", "keep"),
    ("aprova o sonho", "keep"),
    ("keep the dream", "keep"),
    ("descarta o sonho", "drop"),
    ("rejeita esse sonho", "drop"),
    ("discard dream", "drop"),
])
def test_dream_re_actions(text, action):
    m = DREAM_REVIEW_RE.match(text)
    assert m and _action(m) == action


@pytest.mark.parametrize("text", ["oi cortex", "sonho com viajar", "que sonho bom"])
def test_dream_re_ignores(text):
    assert not DREAM_REVIEW_RE.match(text)


# --- handler (gated) -------------------------------------------------------

def _orch():
    from src.core.orchestrator import CortexOrchestrator
    return CortexOrchestrator.__new__(CortexOrchestrator)


def test_handler_dormant_returns_none(monkeypatch):
    monkeypatch.delenv("CORTEX_DREAM_LOOP", raising=False)  # default off
    assert _orch()._handle_dream_review("show") is None


def _enabled_store(tmp_path, monkeypatch, seed=None):
    monkeypatch.setenv("CORTEX_DREAM_LOOP", "1")
    store = dl.DreamStore(path=str(tmp_path / "dreams.jsonl"))
    if seed:
        store.save(seed, [1], ["frag"])
    monkeypatch.setattr(dl, "DreamStore", lambda: store)
    return store


def test_handler_show_returns_text(tmp_path, monkeypatch):
    _enabled_store(tmp_path, monkeypatch, seed="um sonho sobre viajar")
    reply, mode = _orch()._handle_dream_review("show")
    assert mode == "DREAM_SHOW" and "viajar" in reply


def test_handler_show_no_pending(tmp_path, monkeypatch):
    _enabled_store(tmp_path, monkeypatch)  # empty store
    reply, mode = _orch()._handle_dream_review("show")
    assert mode == "DREAM_NONE"


def test_handler_keep_approves(tmp_path, monkeypatch):
    store = _enabled_store(tmp_path, monkeypatch, seed="sonho X")
    reply, mode = _orch()._handle_dream_review("keep")
    assert mode == "DREAM_KEPT"
    assert store.pending() == []         # no longer pending
    assert len(store.list_by_status("kept")) == 1


def test_handler_drop_rejects(tmp_path, monkeypatch):
    store = _enabled_store(tmp_path, monkeypatch, seed="sonho Y")
    reply, mode = _orch()._handle_dream_review("drop")
    assert mode == "DREAM_DROPPED"
    assert store.pending() == []
    assert len(store.list_by_status("discarded")) == 1
