"""
Tests for the deep-research dispatch (Fase 2, DORMANT wiring).

DEEP_RESEARCH_RE must fire only on an explicit deep/a-fundo qualifier (plain
'pesquise X' stays with normal search). The handler is gated by
CORTEX_DEEP_STUDIER: OFF → None (fall through); ON → routes to DeepStudier.
"""
import pytest

from src.core.dispatch.deep_research import DEEP_RESEARCH_RE


# --- regex -----------------------------------------------------------------

@pytest.mark.parametrize("text,topic", [
    ("pesquise a fundo sobre transformers", "transformers"),
    ("pesquisa profunda sobre o conflito", "o conflito"),
    ("investigue profundamente a história X", "história X"),
    ("estuda a fundo o modelo Gemma", "modelo Gemma"),
    ("deep research on quantization", "quantization"),
    ("deep research about LLMs", "LLMs"),
])
def test_deep_re_matches(text, topic):
    m = DEEP_RESEARCH_RE.match(text)
    assert m and m.group("topic") == topic


@pytest.mark.parametrize("text", [
    "pesquise o clima de hoje",   # plain search → normal path
    "pesquisa sobre gatos",       # no deep qualifier
    "oi cortex",
])
def test_deep_re_ignores(text):
    assert not DEEP_RESEARCH_RE.match(text)


# --- handler (gated) -------------------------------------------------------

def _orch():
    from src.core.orchestrator import CortexOrchestrator
    return CortexOrchestrator.__new__(CortexOrchestrator)


def test_handler_dormant_returns_none(monkeypatch):
    monkeypatch.delenv("CORTEX_DEEP_STUDIER", raising=False)  # default off
    assert _orch()._handle_deep_research("transformers") is None


def test_handler_enabled_enqueues_never_runs_inline(monkeypatch, tmp_path):
    """M8-A2: the handler ENQUEUES and returns — the deep loop (minutes of
    searches + syntheses) must never run in the request thread. If anything
    even constructs a DeepStudier here, that's the regression."""
    monkeypatch.setenv("CORTEX_DEEP_STUDIER", "1")
    monkeypatch.setenv("CORTEX_DEEP_QUEUE", str(tmp_path / "deep_queue.jsonl"))
    import src.core.deep_studier as ds

    class _NeverConstructed:
        def __init__(self, *a, **k):
            raise AssertionError("DeepStudier must NOT run inline (M8-A2)")

    monkeypatch.setattr(ds, "DeepStudier", _NeverConstructed)
    reply, mode = _orch()._handle_deep_research("transformers")
    assert mode == "DEEP_RESEARCH_QUEUED"
    assert "transformers" in reply and "fila" in reply
    # And the topic actually landed in the deep queue.
    pending = ds.deep_queue().list_pending()
    assert len(pending) == 1
    assert pending[0]["topic"] == "transformers"


def test_handler_enabled_empty_topic_none(monkeypatch):
    monkeypatch.setenv("CORTEX_DEEP_STUDIER", "1")
    assert _orch()._handle_deep_research("  ") is None


def test_handler_duplicate_topic_rejected_softly(monkeypatch, tmp_path):
    monkeypatch.setenv("CORTEX_DEEP_STUDIER", "1")
    monkeypatch.setenv("CORTEX_DEEP_QUEUE", str(tmp_path / "deep_queue.jsonl"))
    orch = _orch()
    assert orch._handle_deep_research("transformers")[1] == "DEEP_RESEARCH_QUEUED"
    reply, mode = orch._handle_deep_research("transformers")
    assert mode == "DEEP_RESEARCH_REJECTED"
    assert "[SYSTEM ERROR]" not in reply


def test_handler_enabled_failure_is_soft(monkeypatch):
    monkeypatch.setenv("CORTEX_DEEP_STUDIER", "1")
    import src.core.deep_studier as ds

    def _boom():
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(ds, "deep_queue", _boom)
    reply, mode = _orch()._handle_deep_research("x")
    assert mode == "DEEP_RESEARCH_ERROR"
