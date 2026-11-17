"""
Voice-pipeline telemetry.

Answers "is the voice assistant actually working, and where does the time
go?" by emitting one structured event per phase of a single spoken
interaction:

    wake → listened → transcribed → thinking → replied → done

Each phase carries the latency since the previous one, so a session reads
as a timeline: woke at T, listened Xs, STT took Yms, the model took Zms
to first audio, spoke for Wms, total Vms.

Storage reuses the existing `AuditLogger` (`~/.cortex/audit.jsonl`) with
`stage="voice"` — no new sink, and the `/ui` dashboard + jq queries that
already read that file get voice events for free. The listener runs as a
separate process from uvicorn, but both append to the same JSONL (POSIX
append of small lines is atomic), so a `VoiceTelemetry` built in the
listener lands in the same log the dashboard reads.

Fail-soft like audit/cost: a telemetry error must NEVER crash the voice
loop. Every emit is wrapped; methods called out of order (before `wake`)
are no-ops rather than exceptions.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

from src.core.redaction import redact_text

STAGE = "voice"

# Canonical phase order, exported so the aggregator / UI render a stable
# timeline even when a session is cut short (e.g. interrupted reply).
PHASES = ("wake", "listened", "transcribed", "thinking", "replied", "done")


def _default_audit():
    """Lazy import of the module singleton so importing telemetry doesn't
    drag in the whole core package (and tests can inject a fake)."""
    from src.core.audit import audit
    return audit


class VoiceTelemetry:
    """One instance per listener process; one *session* per spoken turn.

    Call `wake()` to open a session (returns the trace id), then the phase
    methods in order. Latency between phases is computed automatically
    from a monotonic clock. `session_done()` closes it and records the
    total. Re-calling `wake()` starts a fresh session."""

    def __init__(
        self,
        audit_logger: Any = None,
        clock: Callable[[], float] = time.perf_counter,
    ):
        # `audit_logger is None` → resolve the real singleton lazily.
        # Pass an explicit fake in tests (anything with a `.log()` and
        # `.new_trace()`); pass it positionally to avoid the lazy import.
        self._audit = audit_logger if audit_logger is not None else _default_audit()
        self._clock = clock
        self._trace: Optional[str] = None
        self._t_start = 0.0
        self._t_last = 0.0
        # Languages seen this turn (see `languages`/`replied`).
        self._langs: dict = {}

    # ------------------------------------------------------------------
    # session lifecycle
    # ------------------------------------------------------------------

    def wake(self, score: Optional[float] = None) -> Optional[str]:
        """Open a new voice session (wake word fired). Returns the trace
        id, or None if even starting failed (fail-soft)."""
        try:
            self._trace = self._audit.new_trace()
        except Exception:
            self._trace = None
            return None
        self._t_start = self._t_last = self._clock()
        self._langs = {}     # languages are per-turn, never carried over
        payload = {"score": round(score, 3)} if score is not None else None
        self._emit("wake", payload)
        return self._trace

    def listened(self, duration_s: float) -> None:
        """User finished speaking; `duration_s` is how long they spoke."""
        self._phase("listened", {"duration_s": round(duration_s, 2)})

    def transcribed(
        self, text: str, lang: Optional[str] = None, lang_prob: Optional[float] = None
    ) -> None:
        """STT done. Latency on this event = transcription time."""
        # Redact BEFORE capping (M5-A9): everything the user dictates lands in
        # the audit JSONL via this field — a spoken "minha senha é X" must
        # be masked here, and capping after keeps the mask intact. The
        # audit sink redacts again (defense in depth), but this border is
        # the one the spec names.
        payload: Dict[str, Any] = {"text": redact_text(text or "")[:120]}
        if lang:
            payload["lang"] = lang
        if lang_prob is not None:
            payload["lang_prob"] = round(lang_prob, 2)
        self._phase("transcribed", payload)

    def thinking_done(self) -> None:
        """First audio is about to play. Latency on this event = time the
        model/dispatch took from end-of-STT to first spoken word."""
        self._phase("thinking", None)

    def replied(self, status_mode: Optional[str] = None,
                stt_lang: Optional[str] = None,
                server_lang: Optional[str] = None,
                spoke_lang: Optional[str] = None) -> None:
        """Reply finished speaking. Latency = time spent speaking.

        The three language fields exist because "falei em inglês e ele
        respondeu em português" (s82) is currently UNDIAGNOSABLE from the
        audit: only the STT language was recorded, and it was CORRECT
        (en @ 0.84). The break is somewhere downstream — the model ignoring
        the directive, the server announcing the wrong language, or the TTS
        voice picker — and the log could not tell them apart. Recording all
        three makes the next occurrence a one-line diagnosis:
          stt=en server=en spoke=pt  → the TTS voice picker
          stt=en server=pt spoke=pt  → the model ignored the directive
          stt=pt ...                 → the STT (and then it is upstream)"""
        payload = {}
        if status_mode:
            payload["status_mode"] = status_mode
        merged = {
            "stt_lang": stt_lang or self._langs.get("stt_lang"),
            "server_lang": server_lang or self._langs.get("server_lang"),
            "spoke_lang": spoke_lang or self._langs.get("spoke_lang"),
        }
        for key, value in merged.items():
            if value:
                payload[key] = value
        self._phase("replied", payload or None)

    def languages(self, stt: Optional[str] = None, server: Optional[str] = None,
                  spoke: Optional[str] = None) -> None:
        """Record the languages seen during the turn, for `replied` to emit.

        Stored on the telemetry object rather than passed through
        `stream_and_speak`'s return: the values become known deep inside the
        streaming loop while `replied()` is called by the caller, and widening
        that return type would ripple through the SSE contract tests for a
        logging field. Best-effort by design — a missing value just doesn't
        appear in the payload."""
        for key, value in (("stt_lang", stt), ("server_lang", server),
                           ("spoke_lang", spoke)):
            if value:
                self._langs[key] = value

    def speaker_shadow(
        self,
        context: str,
        score: Optional[float],
        duration_s: float,
        score_ms: Optional[float] = None,
    ) -> None:
        """Onda 12 Fase A: speaker-ID shadow observation — the cosine score
        of this utterance against the enrolled voiceprint. OBSERVATION
        ONLY; nothing in the pipeline branches on it (enforcement is Fase
        B, gated on this very data). `score=None` means the check couldn't
        run (no voiceprint / model / too-short audio) — logged rather than
        skipped so absence is visible when reading a week of shadow.

        Emitted OUTSIDE the phase clock (`_emit` direct, no `_t_last`
        update): the wake→…→done latency timeline must stay untouched,
        and the /ui aggregator ignores event names it doesn't know."""
        if self._trace is None:
            return
        payload: Dict[str, Any] = {
            "context": context,
            "duration_s": round(duration_s, 2),
            "score": round(score, 3) if score is not None else None,
        }
        if score_ms is not None:
            payload["score_ms"] = round(score_ms, 1)
        self._emit("speaker_shadow", payload)

    def transcribed_extra(
        self,
        context: str,
        text: str,
        lang: Optional[str] = None,
        lang_prob: Optional[float] = None,
    ) -> None:
        """A transcription that happens MID-session — the barge-in command
        or the approval follow-up (s76: the smoke's "yes"→"Ah!" follow-up
        was invisible in the audit because only the main-loop capture
        logged `transcribed`). Same redaction border as `transcribed`.

        Emitted OUTSIDE the phase clock, like `speaker_shadow`: the /ui
        row keys its text/lang on the session's single `transcribed`
        event (last-write-wins), so re-emitting that name would replace
        the command transcript on the panel. `context` mirrors
        speaker_shadow's ("barge_in" / "followup")."""
        if self._trace is None:
            return
        payload: Dict[str, Any] = {
            "context": context,
            "text": redact_text(text or "")[:120],
        }
        if lang:
            payload["lang"] = lang
        if lang_prob is not None:
            payload["lang_prob"] = round(lang_prob, 2)
        self._emit("transcribed_extra", payload)

    def session_done(self) -> None:
        """Close the session; records total elapsed since `wake`."""
        if self._trace is None:
            return
        total = (self._clock() - self._t_start) * 1000.0
        self._emit("done", None, latency_ms=total)
        self._trace = None

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _phase(self, event: str, payload: Optional[Dict[str, Any]]) -> None:
        # No-op if no session is open (out-of-order / failed wake).
        if self._trace is None:
            return
        now = self._clock()
        latency_ms = (now - self._t_last) * 1000.0
        self._t_last = now
        self._emit(event, payload, latency_ms=latency_ms)

    def _emit(
        self,
        event: str,
        payload: Optional[Dict[str, Any]],
        latency_ms: Optional[float] = None,
    ) -> None:
        if self._trace is None:
            return
        try:
            self._audit.log(
                self._trace, STAGE, event, payload=payload, latency_ms=latency_ms
            )
        except Exception:
            # Telemetry must never crash the voice loop.
            pass
