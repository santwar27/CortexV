"""Per-turn token usage crosses the stream's worker thread (s88, Codex lote 13)."""
import threading
from src.core import turn_metrics
from tests.test_dispatch_parity import _build_real_orch


def test_usage_survives_the_stream_worker(tmp_path, monkeypatch):
    """The UI opens the token bucket on its own thread, and the deferred
    path of process_stream runs the engine on another: add() found no bucket
    there and the turn reported no usage (s88, Codex lote 13)."""
    o = _build_real_orch(tmp_path)

    def process_core(*_a, **_k):
        turn_metrics.add(1200, 40)      # what the engine does per LLM call
        return "resposta", "OK"
    monkeypatch.setattr(o, "_process_core", process_core)
    text = "meu hobby é escalada. qual arquivo eu leio sobre isso no meu computador?"
    assert not o._is_pure_chat(text)
    out = {}

    def ui_thread():
        turn_metrics.reset()
        list(o.process_stream(text))
        out["usage"] = turn_metrics.collect()
    t = threading.Thread(target=ui_thread)
    t.start(); t.join()
    assert out["usage"] == {"prompt_eval_count": 1200, "eval_count": 40, "calls": 1,
                           "models": [], "online": []}
