"""The semantic router ACTING, not just observing (item 1c, s83).

Calibrated offline on 614 real sentences from the audit log
(scripts/router_replay.py): at a 0.74 floor it routes 14 of the 340 that
reach it, all genuine rescues, the first false positive sitting at 0.719.

Three properties are pinned here, in the order that matters if one of them
breaks: it stays OFF unless the flag says otherwise; it only acts on intents
it can act on WITHOUT slots; and it runs at both entry points, because a
cross-cutting layer wired into one of them is the recurring bug of this
codebase.
"""
import pytest

from src.core.orchestrator import CortexOrchestrator, _CITY_TAIL_RE


class _Decision:
    def __init__(self, mode, accepted=True, score=0.80, margin=0.10,
                 example="exemplo"):
        self.mode, self.accepted = mode, accepted
        self.score, self.margin, self.example = score, margin, example
        self.reason = None if accepted else "below_score"


@pytest.fixture
def orch(monkeypatch):
    o = CortexOrchestrator.__new__(CortexOrchestrator)
    o._reply_lang = "pt"
    monkeypatch.setenv("CORTEX_SEMANTIC_ROUTER", "1")
    return o


def _router(orch, monkeypatch, decision):
    monkeypatch.setattr(orch, "_semantic_router",
                        lambda: type("R", (), {
                            "explain": staticmethod(lambda _t: decision)})(),
                        raising=False)


class TestItStaysOffUntilTold:

    def test_off_by_default(self, orch, monkeypatch):
        monkeypatch.delenv("CORTEX_SEMANTIC_ROUTER", raising=False)
        _router(orch, monkeypatch, _Decision("JOBS_TOP"))
        assert orch._semantic_route_live("tem vaga?") is None

    def test_a_decline_keeps_the_conversation(self, orch, monkeypatch):
        _router(orch, monkeypatch, _Decision("JOBS_TOP", accepted=False))
        assert orch._semantic_route_live("tem vaga?") is None

    def test_empty_text_is_not_routed(self, orch, monkeypatch):
        _router(orch, monkeypatch, _Decision("JOBS_TOP"))
        assert orch._semantic_route_live("   ") is None


class TestOnlySlotFreeIntentsAct:
    """The router reports a mode and never slots: "manda matue tocando" is
    confidently SPOTIFY_PLAY and says nothing about WHICH track."""

    def test_an_intent_that_needs_an_argument_falls_through(self, orch,
                                                            monkeypatch):
        _router(orch, monkeypatch, _Decision("SPOTIFY_PLAY"))
        assert orch._semantic_route_live("manda matue tocando") is None

    def test_note_create_falls_through_too(self, orch, monkeypatch):
        _router(orch, monkeypatch, _Decision("NOTE_CREATE"))
        assert orch._semantic_route_live("me conte uma história") is None

    def test_jobs_top_is_acted_on(self, orch, monkeypatch):
        _router(orch, monkeypatch, _Decision("JOBS_TOP"))
        monkeypatch.setattr(orch, "_handle_jobs_top",
                            lambda raw_text="": ("suas vagas", "JOBS_TOP"),
                            raising=False)
        assert orch._semantic_route_live("tem vaga?") == ("suas vagas", "JOBS_TOP")

    def test_a_handler_that_declines_keeps_the_conversation(self, orch,
                                                            monkeypatch):
        _router(orch, monkeypatch, _Decision("JOBS_TOP"))
        monkeypatch.setattr(orch, "_handle_jobs_top",
                            lambda raw_text="": None, raising=False)
        assert orch._semantic_route_live("tem vaga?") is None

    def test_a_handler_that_raises_never_breaks_the_turn(self, orch,
                                                         monkeypatch):
        def boom(raw_text=""):
            raise RuntimeError("handler down")
        _router(orch, monkeypatch, _Decision("JOBS_TOP"))
        monkeypatch.setattr(orch, "_handle_jobs_top", boom, raising=False)
        assert orch._semantic_route_live("tem vaga?") is None

    def test_a_broken_router_never_breaks_the_turn(self, orch, monkeypatch):
        def boom():
            raise RuntimeError("embeddings down")
        monkeypatch.setattr(orch, "_semantic_router", boom, raising=False)
        assert orch._semantic_route_live("tem vaga?") is None


class TestWeatherNeedsACity:
    """WEATHER_RE wants the weather noun first, so "How's the weather like in
    London?" misses it — which is why the router catches the intent. The city
    is only read off the tail once the intent is settled."""

    @pytest.mark.parametrize("text, city", [
        ("How's the weather like in London?", "London"),
        ("What is the weather like in London today?", "London"),
        ("How the weather looks like in London", "London"),
        ("como está o tempo em Nova York hoje", "Nova York"),
    ])
    def test_the_city_is_read_off_the_tail(self, text, city):
        assert _CITY_TAIL_RE.search(text).group("city") == city

    @pytest.mark.parametrize("text", [
        "para hoje e qual o clima de amanhã.",
        "tem vaga nova?",
        "o que eu tenho para hoje",
    ])
    def test_no_city_means_no_guess(self, text):
        assert _CITY_TAIL_RE.search(text) is None

    def test_weather_without_a_city_keeps_the_conversation(self, orch,
                                                           monkeypatch):
        _router(orch, monkeypatch, _Decision("WEATHER"))
        called = []
        monkeypatch.setattr(orch, "_handle_weather",
                            lambda city, raw_text="": called.append(city),
                            raising=False)
        assert orch._semantic_route_live("para hoje e qual o clima de amanhã.") is None
        assert called == []          # the handler was never even asked

    def test_weather_with_a_city_is_acted_on(self, orch, monkeypatch):
        _router(orch, monkeypatch, _Decision("WEATHER"))
        monkeypatch.setattr(
            orch, "_handle_weather",
            lambda city, raw_text="": (f"tempo em {city}", "WEATHER"),
            raising=False)
        assert orch._semantic_route_live("How's the weather like in London?") == (
            "tempo em London", "WEATHER")


class TestBothEntryPoints:

    def test_the_live_router_is_wired_in_process_stream_and_process_core(self):
        """Pinned textually, the way this repo pins the other cross-cutting
        layers: the regexes can't prove a call site exists."""
        from pathlib import Path
        source = (Path(__file__).resolve().parents[1]
                  / "src" / "core" / "orchestrator.py").read_text()
        assert source.count("self._semantic_route_live(route_text)") == 2
