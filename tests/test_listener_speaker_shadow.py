"""Onda 12 Fase A: speaker-ID shadow no listener — o contrato de observação.

O que está pinado aqui:
  - o evento speaker_shadow NÃO mexe no clock de fases da telemetria
    (o timeline wake→listened→… do /ui mede latência real; uma
    observação fora de ordem não pode distorcê-lo);
  - o hook é fail-soft em todas as direções (sem verifier, sem sessão,
    verifier explodindo) e converte o buffer int16 pro float32 [-1,1]
    que o SpeakerVerifier espera;
  - shadow é shadow: speaker_shadow_check devolve None e NENHUM call
    site do listener decide nada com base nele (pin textual, mesma
    técnica do test_model_profile).

Sem hardware de áudio, sem ONNX: verifier e audit são fakes.
"""

import re
from pathlib import Path

import numpy as np

import src.voice.listener as l
from src.voice.telemetry import VoiceTelemetry

REPO = Path(__file__).resolve().parents[1]


class FakeAudit:
    def __init__(self):
        self.events = []

    def new_trace(self):
        return "trace-1"

    def log(self, trace, stage, event, payload=None, latency_ms=None):
        self.events.append(
            {"event": event, "payload": payload, "latency_ms": latency_ms}
        )


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class FakeVerifier:
    def __init__(self, score=0.8):
        self._score = score
        self.received = None

    def score(self, samples, sample_rate=16000):
        self.received = np.asarray(samples)
        return self._score


PCM = np.full(16000, 16384, dtype=np.int16)  # 1s a meia escala


# ── telemetria: evento fora do clock de fases ────────────────────────

def test_speaker_shadow_event_payload():
    audit = FakeAudit()
    telem = VoiceTelemetry(audit)
    telem.wake(0.5)
    telem.speaker_shadow(context="command", score=0.812, duration_s=3.21,
                         score_ms=41.7)
    ev = [e for e in audit.events if e["event"] == "speaker_shadow"]
    assert len(ev) == 1
    assert ev[0]["payload"] == {
        "context": "command", "duration_s": 3.21,
        "score": 0.812, "score_ms": 41.7,
    }


def test_speaker_shadow_logs_none_score_as_unavailable():
    audit = FakeAudit()
    telem = VoiceTelemetry(audit)
    telem.wake()
    telem.speaker_shadow(context="followup", score=None, duration_s=0.4)
    ev = [e for e in audit.events if e["event"] == "speaker_shadow"][0]
    assert ev["payload"]["score"] is None  # ausência visível, não pulada


def test_speaker_shadow_does_not_advance_phase_clock():
    audit = FakeAudit()
    clock = FakeClock()
    telem = VoiceTelemetry(audit, clock)
    telem.wake()
    clock.t = 1.0
    telem.speaker_shadow(context="command", score=0.9, duration_s=2.0)
    clock.t = 2.0
    telem.listened(2.0)
    listened = [e for e in audit.events if e["event"] == "listened"][0]
    # 2000ms desde o wake — se o shadow tivesse tocado o _t_last, seria 1000.
    assert listened["latency_ms"] == 2000.0


def test_speaker_shadow_without_session_is_noop():
    audit = FakeAudit()
    telem = VoiceTelemetry(audit)
    telem.speaker_shadow(context="command", score=0.9, duration_s=2.0)
    assert audit.events == []


# ── hook do listener: fail-soft + conversão ──────────────────────────

def _telem_with_session():
    audit = FakeAudit()
    telem = VoiceTelemetry(audit)
    telem.wake()
    return audit, telem


def test_check_scores_and_logs():
    audit, telem = _telem_with_session()
    verifier = FakeVerifier(score=0.731)
    l.speaker_shadow_check(PCM, telem, verifier, "command")
    ev = [e for e in audit.events if e["event"] == "speaker_shadow"][0]
    assert ev["payload"]["score"] == 0.731
    assert ev["payload"]["context"] == "command"
    assert ev["payload"]["duration_s"] == 1.0
    assert ev["payload"]["score_ms"] is not None


def test_check_converts_int16_to_unit_float32():
    _, telem = _telem_with_session()
    verifier = FakeVerifier()
    l.speaker_shadow_check(PCM, telem, verifier, "command")
    assert verifier.received.dtype == np.float32
    assert abs(float(verifier.received[0]) - 0.5) < 1e-3  # 16384/32768

def test_check_without_verifier_or_telem_is_noop():
    audit, telem = _telem_with_session()
    l.speaker_shadow_check(PCM, telem, None, "command")
    l.speaker_shadow_check(PCM, None, FakeVerifier(), "command")
    assert [e for e in audit.events if e["event"] == "speaker_shadow"] == []


def test_check_swallows_verifier_exception():
    audit, telem = _telem_with_session()

    class Exploding:
        def score(self, samples, sample_rate=16000):
            raise RuntimeError("onnx died")

    l.speaker_shadow_check(PCM, telem, Exploding(), "command")  # não levanta
    assert [e for e in audit.events if e["event"] == "speaker_shadow"] == []


# ── shadow é shadow: nenhum call site decide nada com o resultado ────

def test_shadow_call_sites_exist_and_do_not_branch():
    source = (REPO / "src/voice/listener.py").read_text()
    calls = re.findall(r"speaker_shadow_check\(pcm, telem, verifier, \"(\w+)\"\)",
                       source)
    assert sorted(calls) == ["barge_in", "command", "followup"]
    # Nenhum site captura retorno nem condiciona fluxo no shadow:
    assert "= speaker_shadow_check" not in source
    assert "if speaker_shadow_check" not in source
