"""
The mute button must kill the REAL TTS playback (M5-A1, review Fable 5).

Since 2026-06-01 the voice listener speaks via Piper → `afplay
/tmp/cortex_tts.wav` (`say` is only the fallback engine), but both mute
call-sites — `_handle_silence` ("cala a boca" typed/spoken) and
`scripts/cortex-shush.sh` (hotkey) — historically killed only `say`,
leaving the current Piper sentence playing to the end. These tests pin
the contract: muting kills BOTH engines, and the afplay kill is pinned
to the Cortex WAV path so unrelated afplay processes survive.
"""
import os
import stat
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from src.core.i18n import I18nMixin
from src.core.system_control_mixin import SystemControlMixin


SHUSH_SCRIPT = (
    Path(__file__).resolve().parent.parent / "scripts" / "cortex-shush.sh"
)

from src.core.system_control_mixin import CORTEX_AFPLAY_PATTERN


class _Host(SystemControlMixin, I18nMixin):
    pass


@pytest.fixture
def _clean_silence_flag():
    flag = "/tmp/cortex_silence"
    try:
        os.remove(flag)
    except OSError:
        pass
    yield
    try:
        os.remove(flag)
    except OSError:
        pass


class TestHandleSilenceKillsPlayback:
    def _killed(self):
        """Run _handle_silence with subprocess patched; return the list of
        argv lists it tried to run."""
        host = _Host()
        with patch("subprocess.run") as run:
            reply, mode = host._handle_silence()
        assert mode == "SPEECH_INTERRUPTED"
        return [call.args[0] for call in run.call_args_list]

    def test_kills_say_fallback(self, _clean_silence_flag):
        assert ["pkill", "-x", "say"] in self._killed()

    def test_kills_piper_afplay(self, _clean_silence_flag):
        # The real engine since 01/06: Piper playing the Cortex WAV.
        assert ["pkill", "-f", CORTEX_AFPLAY_PATTERN] in self._killed()

    def test_afplay_kill_is_pinned_to_cortex_wav(self, _clean_silence_flag):
        # Pattern-match kill must target ONLY the Cortex playback, never a
        # bare "afplay" that would take down the user's own audio.
        import re
        for argv in self._killed():
            if argv[:2] == ["pkill", "-f"]:
                assert re.search(argv[2], "afplay /var/folders/kc/x/T/cortex_tts_ab12cd.wav")
                assert not re.search(argv[2], "afplay /System/Library/Sounds/Blow.aiff")
                assert not re.search(argv[2], "afplay /Users/me/Music/song.wav")


class TestThePatternMatchesWhatIsPlaying:
    """The pattern named /tmp/cortex_tts.wav, a file the listener stopped
    using when its WAVs became per-process temp files: the live listener
    plays $TMPDIR/cortex_tts_XXXX.wav, so the mute never stopped the
    sentence being spoken (s88, Codex lote 12)."""

    def test_the_listeners_own_wavs(self):
        import re
        from src.voice import listener
        for wav in listener._TTS_WAV_PATHS:
            assert re.search(CORTEX_AFPLAY_PATTERN, f"afplay {wav}")

    def test_the_script_uses_the_same_pattern(self):
        assert f"'{CORTEX_AFPLAY_PATTERN}'" in SHUSH_SCRIPT.read_text()


class TestShushScriptKillsPlayback:
    """Black-box: run the script with pkill/killall shimmed onto PATH and
    inspect what it tried to kill (same sandbox style as backup.sh tests)."""

    def _run_shush(self, tmp_path, pkill_rc=0, killall_rc=1):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        log = tmp_path / "kills.log"
        for name, rc in (("pkill", pkill_rc), ("killall", killall_rc)):
            shim = bin_dir / name
            shim.write_text(
                f'#!/bin/sh\necho "{name} $@" >> "{log}"\nexit {rc}\n'
            )
            shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
        env = dict(os.environ)
        env["PATH"] = f"{bin_dir}:{env['PATH']}"
        env["CORTEX_SILENCE_FLAG"] = str(tmp_path / "cortex_silence")
        result = subprocess.run(
            ["/bin/sh", str(SHUSH_SCRIPT)],
            env=env, capture_output=True, text=True, timeout=10,
        )
        kills = log.read_text().splitlines() if log.exists() else []
        return result, kills, tmp_path / "cortex_silence"

    def test_kills_piper_afplay_and_say(self, tmp_path):
        result, kills, flag = self._run_shush(tmp_path)
        assert f"pkill -f {CORTEX_AFPLAY_PATTERN}" in kills
        assert "killall say" in kills
        assert flag.exists(), "silence flag must be raised for the stream"
        assert "stopped" in result.stdout

    def test_reports_stopped_when_only_afplay_was_playing(self, tmp_path):
        # Piper engine active, no `say` running: the script must still
        # report a real stop (pre-fix it only looked at killall's rc).
        result, _, _ = self._run_shush(tmp_path, pkill_rc=0, killall_rc=1)
        assert "stopped" in result.stdout

    def test_nothing_playing_still_raises_flag(self, tmp_path):
        result, _, flag = self._run_shush(tmp_path, pkill_rc=1, killall_rc=1)
        assert flag.exists()
        assert "abort flag set" in result.stdout
