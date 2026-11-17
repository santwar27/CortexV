"""A flag in /tmp is never raised through a symlink (s88, Codex lote 12).

/tmp is writable by every account on the Mac. `open(path, "w")` and the
shell's `: > path` followed a link planted in a free flag name and emptied
the file it pointed to — reproduced on disposable files with
_handle_screen_eyes_off and the three scripts; _handle_silence used the
same open(). Each test here points a flag name at a "personal" file and
checks the file survives.
"""
import os
import stat
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from src.core import system_control_mixin
from src.core.flag_file import own_flag, raise_flag
from src.core.i18n import I18nMixin
from src.core.screen_context_mixin import ScreenContextMixin
from src.core.system_control_mixin import SystemControlMixin

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
PERSONAL = "conteúdo pessoal\n"


@pytest.fixture
def planted(tmp_path):
    """A personal file and a link to it sitting in a free flag name."""
    victim = tmp_path / "zshrc"
    victim.write_text(PERSONAL)
    link = tmp_path / "flag"
    link.symlink_to(victim)
    return victim, link


class TestRaiseFlag:

    def test_creates_the_flag(self, tmp_path):
        flag = tmp_path / "flag"
        raise_flag(str(flag))
        raise_flag(str(flag))                     # idempotent
        assert flag.is_file() and not flag.is_symlink()

    def test_refuses_a_link_and_leaves_its_target(self, planted):
        victim, link = planted
        with pytest.raises(OSError):
            raise_flag(str(link))
        assert victim.read_text() == PERSONAL


class _Host(SystemControlMixin, ScreenContextMixin, I18nMixin):
    pass


class TestTheHandlers:

    def test_silence_through_a_link(self, planted, monkeypatch):
        victim, link = planted
        monkeypatch.setattr(system_control_mixin, "SILENCE_FLAG", str(link))
        with patch("subprocess.run"):
            _reply, mode = _Host()._handle_silence()
        assert mode == "SPEECH_INTERRUPTED"       # the kill still happens
        assert victim.read_text() == PERSONAL

    def test_screen_off_through_a_link(self, planted, monkeypatch):
        victim, link = planted
        monkeypatch.setenv("CORTEX_SCREEN_OFF_FLAG", str(link))
        with patch("src.core.screen_context_mixin.audit"):
            _reply, mode = _Host()._handle_screen_eyes_off()
        assert mode == "SCREEN_EYES_ERROR"        # honest: it did not turn off
        assert victim.read_text() == PERSONAL


def _run(script, env_var, flag, tmp_path, *args):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name in ("pkill", "killall"):           # cortex-shush kills playback
        shim = bin_dir / name
        shim.write_text("#!/bin/sh\nexit 1\n")
        shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env[env_var] = str(flag)
    env["CORTEX_RECORD_DIR"] = str(tmp_path / "rec")
    return subprocess.run(["/bin/sh", str(SCRIPTS / script), *args], env=env,
                          capture_output=True, text=True, timeout=10)


_SCRIPTS = [
    ("cortex-shush.sh", "CORTEX_SILENCE_FLAG", ()),
    ("cortex-record.sh", "CORTEX_RECORD_FLAG", ("on",)),
    ("cortex-eyes.sh", "CORTEX_SCREEN_OFF_FLAG", ("off",)),
]


class TestTheScripts:

    @pytest.mark.parametrize("script,env_var,args", _SCRIPTS)
    def test_a_link_in_the_flag_name_is_not_written(self, planted, tmp_path,
                                                    script, env_var, args):
        victim, link = planted
        _run(script, env_var, link, tmp_path, *args)
        assert victim.read_text() == PERSONAL

    @pytest.mark.parametrize("script,env_var,args", _SCRIPTS)
    def test_a_free_name_still_raises_the_flag(self, tmp_path,
                                               script, env_var, args):
        flag = tmp_path / "flag"
        _run(script, env_var, flag, tmp_path, *args)
        assert flag.is_file()


class TestOwnFlag:

    def test_an_own_file(self, tmp_path):
        flag = tmp_path / "flag"
        flag.touch()
        assert own_flag(str(flag))

    def test_absent(self, tmp_path):
        assert not own_flag(str(tmp_path / "flag"))

    def test_a_link(self, planted):
        _victim, link = planted
        assert not own_flag(str(link))

    def test_another_accounts_file(self, tmp_path, monkeypatch):
        flag = tmp_path / "flag"
        flag.touch()
        real = os.getuid()
        monkeypatch.setattr(os, "getuid", lambda: real + 1)
        assert not own_flag(str(flag))
