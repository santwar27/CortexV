"""
System/device-control skill handlers for `CortexOrchestrator`.

Extracted from `src/core/orchestrator.py` (s59) as part of the per-family
handler-mixin refactor. Groups the handlers that act on the machine / its
output:

  - silence — stop the voice listener's current speech immediately (drops a
    flag file + pkills `say`); acts directly, no staging.
  - mac_quit_regex / mac_sleep_regex — S3 actions (can lose work / sleep the
    Mac), so they STAGE a `_pending_action` and wait for the user's "sim".

Mixed in so `self._handle_*(...)` resolves through the MRO; the staging
handlers reach `_pending_action` / `_persist_pending_action` /
`_human_confirmation_prompt` via the MRO.
"""
from __future__ import annotations

from typing import Tuple

# The flag the listener polls (src/voice/listener.py, SILENCE_FLAG).
SILENCE_FLAG = "/tmp/cortex_silence"

# The Piper playback, as pkill -f sees it: the listener plays per-process
# temp WAVs (`listener._mk_tts_wav`: cortex_tts_*.wav under $TMPDIR). The
# old pattern, "afplay /tmp/cortex_tts.wav", named a file that no longer
# exists, so the mute never stopped the sentence being spoken (s88, Codex
# lote 12). Anchored on afplay so the chime and any other afplay survive;
# scripts/cortex-shush.sh uses the same pattern.
CORTEX_AFPLAY_PATTERN = r"^afplay .*/cortex_tts_[^/]*\.wav$"


class SystemControlMixin:
    """Speech-silence + Mac quit/sleep (staged) handlers."""

    def _handle_silence(self) -> Tuple[str, str]:
        """
        Stop the voice listener's current speech immediately.

        Drops a flag file at /tmp/cortex_silence that the listener checks
        between SSE chunks — when seen, it kills its current playback
        subprocess, drains the rest of the HTTP stream, and returns to
        wake-word listening. Also kills any in-flight playback directly in
        case the listener is between chunk reads: the TTS engine is either
        Piper (plays the listener's cortex_tts_*.wav via `afplay`) or the
        `say` fallback, so both are targeted. The afplay match is pinned to
        the Cortex WAVs so other afplay processes are untouched.
        """
        import subprocess
        from src.core.flag_file import raise_flag
        try:
            raise_flag(SILENCE_FLAG)   # never through a link (s88, lote 12)
        except Exception:
            pass
        # Bounded, like every command the controller runs (s88, lote 12).
        for argv in (["pkill", "-x", "say"], ["pkill", "-f", CORTEX_AFPLAY_PATTERN]):
            try:
                subprocess.run(argv, check=False, timeout=5)
            except Exception:
                pass
        return (self._t("Calado.", "Quiet."), "SPEECH_INTERRUPTED")

    def _handle_mac_quit_regex(self, app_name: str) -> Tuple[str, str]:
        """Stage an app-quit (S3) and return the confirmation prompt.

        Same staging shape as the calendar/reminder regex handlers: write
        the action into _pending_action, persist it, and hand back the
        deterministic human prompt. The user's next 'sim'/'não' turn is
        resolved by the generic pending-action flow, which dispatches to
        the mac_quit_app branch in StagingMixin._execute_pending_action.
        """
        self._pending_action = {
            "type": "mac_quit_app",
            "args": {"app_name": app_name},
        }
        self._persist_pending_action()
        return (
            self._human_confirmation_prompt(self._pending_action),
            "AGENT_AWAITING_CONFIRMATION",
        )

    def _handle_mac_sleep_regex(self, phrased_as_shutdown: bool = False
                                ) -> Tuple[str, str]:
        """Stage a Mac sleep (S3) and return the confirmation prompt.
        Same staging shape as mac_quit_app so the sim/não flow dispatches
        through the existing pending-action plumbing.

        `phrased_as_shutdown` is set when the user said "desliga o mac".
        Cortex sleeps instead of shutting down, and the flag makes the
        confirmation say that before the user answers 'sim' — the action
        itself is identical (the executor ignores the args)."""
        self._pending_action = {
            "type": "mac_sleep",
            "args": {"phrased_as_shutdown": True} if phrased_as_shutdown else {},
        }
        self._persist_pending_action()
        return (
            self._human_confirmation_prompt(self._pending_action),
            "AGENT_AWAITING_CONFIRMATION",
        )
