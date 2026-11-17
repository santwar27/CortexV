"""
VoiceTelemetry — phase emission, latency math, fail-soft.

No audit file, no clock wall-time: a fake logger records calls and a
fake monotonic clock makes latencies deterministic.
"""
import pytest

from src.voice.telemetry import STAGE, VoiceTelemetry


class FakeAudit:
    """Records .log() calls; hands out predictable trace ids."""
    def __init__(self):
        self.events = []
        self._n = 0

    def new_trace(self):
        self._n += 1
        return f"trace{self._n}"

    def log(self, trace_id, stage, event, payload=None, latency_ms=None):
        self.events.append({
            "trace_id": trace_id, "stage": stage, "event": event,
            "payload": payload, "latency_ms": latency_ms,
        })


class FakeClock:
    """Monotonic clock the test advances by hand (seconds)."""
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, secs):
        self.t += secs


@pytest.fixture
def setup():
    audit = FakeAudit()
    clock = FakeClock()
    telem = VoiceTelemetry(audit, clock=clock)
    return telem, audit, clock


class TestHappyPath:
    def test_full_session_phases_and_order(self, setup):
        telem, audit, clock = setup
        trace = telem.wake(score=0.42)
        clock.advance(1.5)            # user speaks 1.5s
        telem.listened(1.5)
        clock.advance(2.0)            # STT 2s
        telem.transcribed("que horas são", lang="pt", lang_prob=0.97)
        clock.advance(0.8)            # model to first audio 0.8s
        telem.thinking_done()
        clock.advance(1.2)            # speaking 1.2s
        telem.replied(status_mode="REFLEX")
        telem.session_done()

        events = [e["event"] for e in audit.events]
        assert events == [
            "wake", "listened", "transcribed", "thinking", "replied", "done",
        ]
        # all under one trace, all stage=voice
        assert {e["trace_id"] for e in audit.events} == {trace}
        assert all(e["stage"] == STAGE for e in audit.events)

    def test_latencies_are_inter_phase(self, setup):
        telem, audit, clock = setup
        telem.wake()
        clock.advance(2.0)
        telem.listened(2.0)
        clock.advance(0.5)
        telem.transcribed("oi")
        by = {e["event"]: e for e in audit.events}
        # wake carries no latency (session start)
        assert by["wake"]["latency_ms"] is None
        # listened = 2.0s since wake; transcribed = 0.5s since listened
        assert by["listened"]["latency_ms"] == pytest.approx(2000.0)
        assert by["transcribed"]["latency_ms"] == pytest.approx(500.0)

    def test_done_is_total_since_wake(self, setup):
        telem, audit, clock = setup
        telem.wake()
        clock.advance(3.0)
        telem.listened(3.0)
        clock.advance(1.0)
        telem.session_done()
        done = [e for e in audit.events if e["event"] == "done"][0]
        assert done["latency_ms"] == pytest.approx(4000.0)

    def test_payloads_captured(self, setup):
        telem, audit, _ = setup
        telem.wake(score=0.5)
        telem.transcribed("toca matuê", lang="pt", lang_prob=0.9)
        telem.replied(status_mode="SPOTIFY_PLAY")
        by = {e["event"]: e for e in audit.events}
        assert by["wake"]["payload"] == {"score": 0.5}
        assert by["transcribed"]["payload"]["text"] == "toca matuê"
        assert by["transcribed"]["payload"]["lang"] == "pt"
        assert by["replied"]["payload"] == {"status_mode": "SPOTIFY_PLAY"}

    def test_text_truncated(self, setup):
        telem, audit, _ = setup
        telem.wake()
        telem.transcribed("x" * 500)
        t = [e for e in audit.events if e["event"] == "transcribed"][0]
        assert len(t["payload"]["text"]) == 120

    def test_dictated_secret_redacted_before_cap(self, setup):
        # M5-A9: a spoken "minha senha é X" must be masked before it
        # reaches the (fake) audit — and before the 120-char cap, so the
        # mask itself can't be clipped in half.
        telem, audit, _ = setup
        telem.wake()
        telem.transcribed("minha senha do banco é 1234 e o resto da frase " + "y" * 100)
        t = [e for e in audit.events if e["event"] == "transcribed"][0]
        assert "1234" not in t["payload"]["text"]
        assert "[REDACTED]" in t["payload"]["text"]


class TestTranscribedExtra:
    """Mid-session transcriptions (barge-in / approval follow-up) — s76:
    the smoke's "yes"→"Ah!" follow-up was invisible in the audit because
    only the main-loop capture logged `transcribed`."""

    def test_payload_and_event_name(self, setup):
        telem, audit, _ = setup
        telem.wake()
        telem.transcribed_extra("followup", "Ah!", "en", 0.531)
        ev = [e for e in audit.events if e["event"] == "transcribed_extra"]
        assert len(ev) == 1
        assert ev[0]["payload"] == {
            "context": "followup", "text": "Ah!",
            "lang": "en", "lang_prob": 0.53,
        }

    def test_does_not_advance_phase_clock(self, setup):
        # Off-clock like speaker_shadow: the /ui latency timeline must not
        # absorb the follow-up wait.
        telem, audit, clock = setup
        telem.wake()
        clock.advance(1.0)
        telem.transcribed_extra("followup", "yes")
        clock.advance(1.0)
        telem.replied("AUTO_SEARCH")
        replied = [e for e in audit.events if e["event"] == "replied"][0]
        assert replied["latency_ms"] == 2000.0

    def test_does_not_collide_with_command_transcript(self, setup):
        # The /ui row keys text on `transcribed` (last-write-wins per event
        # name) — the follow-up must land under a DIFFERENT event name.
        telem, audit, _ = setup
        telem.wake()
        telem.transcribed("How's the weather like in London?", "en", 0.88)
        telem.transcribed_extra("followup", "Ah!", "en", 0.5)
        t = [e for e in audit.events if e["event"] == "transcribed"]
        assert len(t) == 1
        assert "London" in t[0]["payload"]["text"]

    def test_secret_redacted_before_cap(self, setup):
        telem, audit, _ = setup
        telem.wake()
        telem.transcribed_extra(
            "followup", "minha senha do banco é 1234 " + "y" * 120)
        ev = [e for e in audit.events if e["event"] == "transcribed_extra"][0]
        assert "1234" not in ev["payload"]["text"]
        assert "[REDACTED]" in ev["payload"]["text"]

    def test_noop_without_wake(self, setup):
        telem, audit, _ = setup
        telem.transcribed_extra("barge_in", "x")
        assert audit.events == []


class TestReuseAndOutOfOrder:
    def test_second_wake_starts_fresh_trace(self, setup):
        telem, audit, _ = setup
        t1 = telem.wake()
        telem.session_done()
        t2 = telem.wake()
        assert t1 != t2

    def test_phase_before_wake_is_noop(self, setup):
        telem, audit, _ = setup
        telem.listened(1.0)
        telem.transcribed("orphan")
        telem.session_done()
        assert audit.events == []  # nothing logged without a session

    def test_session_done_without_wake_is_noop(self, setup):
        telem, audit, _ = setup
        telem.session_done()
        assert audit.events == []


class TestFailSoft:
    def test_logger_raising_does_not_propagate(self):
        class BoomAudit:
            def new_trace(self):
                return "t"
            def log(self, *a, **k):
                raise RuntimeError("disk full")
        telem = VoiceTelemetry(BoomAudit())
        # None of these may raise.
        telem.wake(0.5)
        telem.listened(1.0)
        telem.transcribed("x")
        telem.thinking_done()
        telem.replied("OK")
        telem.session_done()

    def test_new_trace_raising_degrades_to_no_session(self):
        class BoomTrace:
            def new_trace(self):
                raise RuntimeError("nope")
            def log(self, *a, **k):
                raise AssertionError("should never log without a trace")
        telem = VoiceTelemetry(BoomTrace())
        assert telem.wake(0.5) is None
        telem.listened(1.0)  # no-op, must not call log
        telem.session_done()


# --- per-turn language trail (s82) ------------------------------------------
# "Falei em inglês e ele respondeu em português" was UNDIAGNOSABLE: the audit
# recorded only the STT language, and that was correct (en @ 0.84). The break
# is downstream — model, server announcement, or TTS voice picker — and the
# log could not tell them apart.

def _telem():
    import src.voice.telemetry as t
    class FakeAudit:
        def __init__(self): self.events = []
        def new_trace(self): return "trace-1"
        def log(self, trace, stage, event, payload=None, latency_ms=None):
            self.events.append((event, payload))
    a = FakeAudit()
    return t.VoiceTelemetry(a), a


def test_replied_carries_the_three_languages():
    telem, audit = _telem()
    telem.wake(score=0.5)
    telem.languages(stt="en", server="en", spoke="pt")
    telem.replied(status_mode="OFFLINE")
    payload = [p for e, p in audit.events if e == "replied"][-1]
    assert payload["stt_lang"] == "en"
    assert payload["server_lang"] == "en"
    assert payload["spoke_lang"] == "pt"   # → the TTS voice picker is at fault


def test_languages_are_not_carried_between_turns():
    telem, audit = _telem()
    telem.wake(score=0.5)
    telem.languages(stt="en", server="en", spoke="en")
    telem.replied(status_mode="OK")
    telem.wake(score=0.5)                  # a new turn starts clean
    telem.replied(status_mode="OK")
    last = [p for e, p in audit.events if e == "replied"][-1]
    assert "stt_lang" not in (last or {})


def test_partial_languages_are_fine():
    telem, audit = _telem()
    telem.wake(score=0.5)
    telem.languages(stt="pt")              # server never announced
    telem.replied(status_mode="OK")
    payload = [p for e, p in audit.events if e == "replied"][-1]
    assert payload["stt_lang"] == "pt" and "server_lang" not in payload


def test_replied_without_any_language_still_works():
    telem, audit = _telem()
    telem.wake(score=0.5)
    telem.replied(status_mode="OK")
    payload = [p for e, p in audit.events if e == "replied"][-1]
    assert payload == {"status_mode": "OK"}
