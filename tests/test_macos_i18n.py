"""
macOS controller + system/scheduling mixin reply localisation (s60, Claude-A).

The dispatch wrappers pass `lang=o._reply_lang` into the controller, which
replies in the spoken language (PT default). Tests mock subprocess so no real
volume/lock/sleep happens. PT stays the default → no regression.
"""
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest

from src.tools.macos_controller import MacOSController
from src.core.system_control_mixin import SystemControlMixin
from src.core.scheduling_mixin import SchedulingMixin
from src.core.i18n import I18nMixin


@pytest.fixture
def controller():
    return MacOSController()


def _ok(stdout=""):
    return MagicMock(returncode=0, stdout=stdout, stderr="")


def _fail(stderr="boom", returncode=1):
    return MagicMock(returncode=returncode, stdout="", stderr=stderr)


class TestSleepConfirmationWording:
    """"Desliga o mac" sleeps instead of shutting down, so the prompt has
    to say that BEFORE the user answers 'sim' (s83)."""

    def test_shutdown_phrasing_says_it_will_sleep(self):
        from src.core.tool_risk import build_preview
        human = build_preview("mac_sleep", {"phrased_as_shutdown": True},
                              lang="pt")["human"]
        assert human.startswith("Eu não desligo o Mac")
        assert "dormir" in human

    def test_plain_sleep_keeps_the_short_prompt(self):
        from src.core.tool_risk import build_preview
        human = build_preview("mac_sleep", {}, lang="pt")["human"]
        assert human.startswith("Colocar o Mac pra dormir?")

    def test_shutdown_phrasing_en(self):
        from src.core.tool_risk import build_preview
        human = build_preview("mac_sleep", {"phrased_as_shutdown": True},
                              lang="en")["human"]
        assert human.startswith("I don't shut the Mac down")


class TestControllerRepliesEN:
    @pytest.mark.parametrize("actual", (48, 50, 52))
    def test_set_volume_en(self, controller, actual):
        with patch("src.tools.macos_controller._osascript",
                   side_effect=[(True, "70"), (True, ""), (True, str(actual))]) as osa:
            assert controller.set_system_volume(50, lang="en") == f"Mac volume at {actual}%."
        assert [c.args[0] for c in osa.call_args_list] == [
            "output volume of (get volume settings)", "set volume output volume 50",
            "output volume of (get volume settings)",
        ]

    @pytest.mark.parametrize("actual", (48, 50, 52))
    def test_set_volume_pt_default(self, controller, actual):
        with patch("src.tools.macos_controller._osascript",
                   side_effect=[(True, "70"), (True, ""), (True, str(actual))]) as osa:
            assert controller.set_system_volume(50) == f"Volume do Mac em {actual}%."
        assert [c.args[0] for c in osa.call_args_list] == [
            "output volume of (get volume settings)", "set volume output volume 50",
            "output volume of (get volume settings)",
        ]

    def test_mute_en(self, controller):
        with patch("src.tools.macos_controller.subprocess.run", return_value=_ok()):
            assert controller.mute(lang="en") == "System muted."

    def test_dark_on_en(self, controller):
        with patch("src.tools.macos_controller.subprocess.run", return_value=_ok()):
            assert controller.enable_dark_mode(lang="en") == "Dark mode on."

    def test_lock_en(self, controller):
        with patch("src.tools.macos_controller.subprocess.run", return_value=_ok()):
            assert controller.lock_screen(lang="en") == "Screen locked."

    def test_sleep_en(self, controller):
        with patch("src.tools.macos_controller.subprocess.run", return_value=_ok()):
            assert controller.sleep_mac(lang="en") == "Mac going to sleep."

    def test_open_app_en(self, controller):
        with patch("src.tools.macos_controller.subprocess.run", return_value=_ok()):
            assert controller.open_app("Notion", lang="en") == "Opening Notion."

    def test_open_app_no_name_en(self, controller):
        assert controller.open_app("", lang="en") == "I need the app name to open it."

    def test_quit_app_en(self, controller):
        with patch("src.tools.macos_controller.subprocess.run", return_value=_ok()):
            assert controller.quit_app("Notion", lang="en") == "Closing Notion."

    # `tell application X to quit` exits 0 for a name that doesn't exist,
    # so without the LaunchServices check the reply claimed a quit that
    # never happened ("Fechando tela.", 2026-09-19).
    _NO_SUCH_APP = 'execution error: Can’t get application "tela". (-1728)'

    def test_quit_unknown_app_is_honest_pt(self, controller):
        with patch("src.tools.macos_controller.subprocess.run",
                   return_value=_fail(stderr=self._NO_SUCH_APP)) as run:
            assert controller.quit_app("tela") == "não achei nenhum app chamado 'tela'."
        assert run.call_count == 1   # never attempted the quit

    def test_quit_unknown_app_is_honest_en(self, controller):
        with patch("src.tools.macos_controller.subprocess.run",
                   return_value=_fail(stderr=self._NO_SUCH_APP)):
            assert controller.quit_app("tela", lang="en") == "I couldn't find an app called 'tela'."

    def test_quit_check_other_failure_still_tries_quit(self, controller):
        # Only -1728 means "no such app"; anything else must not block a
        # real quit on a flaky lookup.
        with patch("src.tools.macos_controller.subprocess.run",
                   side_effect=[_fail(stderr="boom"), _ok()]):
            assert controller.quit_app("Notion") == "Fechando Notion."

    def test_list_apps_empty_en(self, controller):
        with patch("src.tools.macos_controller.subprocess.run", return_value=_ok(stdout="")):
            assert controller.list_apps(lang="en") == "No foreground apps right now."

    @pytest.mark.parametrize("outcomes", (
        [(False, "error -1743")],
        [(True, "70"), (False, "error -1743"), (True, "70")],
        [(True, "70"), (True, ""), (False, "read failed")],
    ), ids=("initial_read", "write", "readback"))
    def test_volume_failure_en(self, controller, outcomes):
        with patch("src.tools.macos_controller._osascript", side_effect=outcomes) as osa:
            reply = controller.set_system_volume(50, lang="en")
        assert reply == "I couldn't change the Mac volume — nothing changed."
        assert osa.call_count == len(outcomes)

    @pytest.mark.parametrize("outcomes", (
        [(False, "error -1743")],
        [(True, "70"), (False, "error -1743"), (True, "70")],
        [(True, "70"), (True, ""), (False, "read failed")],
    ), ids=("initial_read", "write", "readback"))
    def test_volume_failure_pt_default(self, controller, outcomes):
        with patch("src.tools.macos_controller._osascript", side_effect=outcomes) as osa:
            reply = controller.set_system_volume(50)
        assert reply == "Não consegui mudar o volume do Mac — nada mudou."
        assert osa.call_count == len(outcomes)


# --- mixin composites: just the handler + the i18n surface ---

class _SilenceHost(SystemControlMixin, I18nMixin):
    pass


class _SchedHost(SchedulingMixin, I18nMixin):
    # Minimal staging surface the calendar-clarify path never reaches; the
    # vague-body branch returns BEFORE staging, so we only need the regexes.
    import re as _re
    _REMINDER_BODY_RE = _re.compile(r"^(?P<title>.+?)\s+(?P<time_expr>amanhã.*)$")
    _VAGUE_CALENDAR_RE = _re.compile(r"\b(esses|essas|aqueles)\b")

    def _parse_reminder_due(self, expr, now):
        return None


@pytest.fixture
def _clean_silence_flag():
    """`_handle_silence` drops a real /tmp/cortex_silence flag the voice
    listener watches. Remove it before AND after so the flag never leaks
    into other tests (it broke test_listener_barge once)."""
    import os
    flag = "/tmp/cortex_silence"
    for _ in (0, 1):
        try:
            os.remove(flag)
        except OSError:
            pass
    yield
    try:
        os.remove(flag)
    except OSError:
        pass


class TestMixinRepliesEN:
    def test_silence_en(self, _clean_silence_flag):
        host = _SilenceHost()
        host._reply_lang = "en"
        with patch("subprocess.run"):
            reply, mode = host._handle_silence()
        assert reply == "Quiet." and mode == "SPEECH_INTERRUPTED"

    def test_silence_pt_default(self, _clean_silence_flag):
        host = _SilenceHost()
        with patch("subprocess.run"):
            reply, _ = host._handle_silence()
        assert reply == "Calado."

    def test_calendar_clarify_en(self):
        host = _SchedHost()
        host._reply_lang = "en"
        reply, mode = host._handle_calendar_regex("esses jogos", datetime.now(timezone.utc))
        assert mode == "CALENDAR_CLARIFY"
        assert "I need each event" in reply

    def test_calendar_clarify_pt_default(self):
        host = _SchedHost()
        reply, mode = host._handle_calendar_regex("esses jogos", datetime.now(timezone.utc))
        assert mode == "CALENDAR_CLARIFY"
        assert "cada evento" in reply
