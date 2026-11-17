"""Tests for src/tools/macos_controller.py.

Strategy: inject fake DisplayServices functions for brightness and mock
`subprocess.run` for shell controls, without changing the developer's
screen, volume or lock state. Every public method is
covered for both the happy path and at least one error path (TCC
permission, timeout, app not found).

Each test gets a fresh `MacOSController()` — no shared fixture state.
"""
import ctypes
import os
import subprocess
from unittest.mock import patch, MagicMock

import pytest

from src.tools.macos_controller import MacOSController, _clamp


@pytest.fixture
def controller():
    return MacOSController()


def _ok(stdout=""):
    """Build a successful CompletedProcess stand-in."""
    return MagicMock(returncode=0, stdout=stdout, stderr="")


def _fail(stderr="boom", returncode=1):
    return MagicMock(returncode=returncode, stdout="", stderr=stderr)


# =====================================================================
# Helpers
# =====================================================================

class TestClamp:
    @pytest.mark.parametrize("value, lo, hi, expected", [
        (50, 0, 100, 50),
        (-10, 0, 100, 0),
        (200, 0, 100, 100),
        (0, 0, 100, 0),
        (100, 0, 100, 100),
        (5, 1, 50, 5),
        (60, 1, 50, 50),
    ])
    def test_clamp_bounds(self, value, lo, hi, expected):
        assert _clamp(value, lo, hi) == expected


# =====================================================================
# System volume
# =====================================================================

class TestSystemVolume:

    def test_set_volume_happy(self, controller):
        with patch("src.tools.macos_controller.subprocess.run",
                   side_effect=[_ok("70"), _ok(), _ok("50")]) as run:
            reply = controller.set_system_volume(50)
        assert "50%" in reply
        args = run.call_args_list[1].args[0]
        assert args[0] == "osascript"
        assert "set volume output volume 50" in args[2]

    def test_set_volume_clamps_high(self, controller):
        with patch("src.tools.macos_controller.subprocess.run",
                   side_effect=[_ok("70"), _ok(), _ok("100")]) as run:
            controller.set_system_volume(500)
        assert "set volume output volume 100" in run.call_args_list[1].args[0][2]

    def test_set_volume_clamps_negative(self, controller):
        with patch("src.tools.macos_controller.subprocess.run",
                   side_effect=[_ok("70"), _ok(), _ok("0")]) as run:
            controller.set_system_volume(-30)
        assert "set volume output volume 0" in run.call_args_list[1].args[0][2]

    def test_set_volume_tcc_failure_does_not_claim_success(self, controller):
        with patch("src.tools.macos_controller.subprocess.run",
                   return_value=_fail(stderr="execution error: System Events got an error: -1743")):
            reply = controller.set_system_volume(50)
        assert reply == "Não consegui mudar o volume do Mac — nada mudou."

    def test_volume_up_reads_then_sets(self, controller):
        # Read current, set, then verify the actual output volume.
        runs = [_ok(stdout="40"), _ok(), _ok("50")]
        with patch("src.tools.macos_controller.subprocess.run", side_effect=runs) as run:
            reply = controller.volume_up(step=10)
        assert "50%" in reply
        # Second call should set to 50
        assert "set volume output volume 50" in run.call_args_list[1].args[0][2]

    def test_volume_up_clamps_at_100(self, controller):
        runs = [_ok(stdout="95"), _ok(), _ok("100")]
        with patch("src.tools.macos_controller.subprocess.run", side_effect=runs) as run:
            controller.volume_up(step=20)
        assert "set volume output volume 100" in run.call_args_list[1].args[0][2]

    def test_volume_down_reads_then_sets(self, controller):
        runs = [_ok(stdout="60"), _ok(), _ok("45")]
        with patch("src.tools.macos_controller.subprocess.run", side_effect=runs) as run:
            reply = controller.volume_down(step=15)
        assert "45%" in reply
        assert "set volume output volume 45" in run.call_args_list[1].args[0][2]

    def test_volume_up_read_failure_returns_friendly(self, controller):
        with patch("src.tools.macos_controller.subprocess.run", return_value=_fail()):
            reply = controller.volume_up()
        assert reply == "Não consegui mudar o volume do Mac — nada mudou."

    def test_volume_up_unparseable_read_returns_friendly(self, controller):
        with patch("src.tools.macos_controller.subprocess.run", return_value=_ok(stdout="???")):
            reply = controller.volume_up()
        assert reply == "Não consegui mudar o volume do Mac — nada mudou."

    def test_mute(self, controller):
        with patch("src.tools.macos_controller.subprocess.run", return_value=_ok()) as run:
            reply = controller.mute()
        assert "silenciado" in reply.lower()
        assert "set volume with output muted" in run.call_args.args[0][2]

    def test_unmute(self, controller):
        with patch("src.tools.macos_controller.subprocess.run", return_value=_ok()) as run:
            reply = controller.unmute()
        assert "desmutado" in reply.lower()
        assert "set volume without output muted" in run.call_args.args[0][2]


# =====================================================================
# Brightness
# =====================================================================

class TestBrightness:

    @pytest.mark.parametrize("lang", ["pt", "en"])
    @pytest.mark.parametrize("method, args, before, after, target", [
        ("set_brightness", (30,), 0.7, 0.31, 30),
        ("set_brightness", (30,), 0.7, 0.28, 30),
        ("set_brightness", (30,), 0.7, 0.32, 30),
        ("set_brightness", (500,), 0.7, 1.0, 100),
        ("set_brightness", (-20,), 0.7, 0.0, 0),
        ("brightness_down", (), 0.5, 0.41, 40),
        ("brightness_up", (), 0.5, 0.59, 60),
        ("brightness_down", (), 0.05, 0.0, 0),
        ("brightness_up", (), 0.95, 1.0, 100),
    ])
    def test_verified_reply(self, brightness, lang, method, args, before, after, target):
        controller, services, _ = brightness([before, after])
        reply = getattr(controller, method)(*args, lang=lang)
        actual = round(after * 100)
        assert reply == (f"Brilho em {actual}%." if lang == "pt" else f"Brightness at {actual}%.")
        display, value = services.DisplayServicesSetBrightness.call_args.args
        assert display == 77
        assert isinstance(value, ctypes.c_float)
        assert value.value == pytest.approx(target / 100)
        assert services.DisplayServicesGetBrightness.call_count == 2

    @pytest.mark.parametrize("lang", ["pt", "en"])
    @pytest.mark.parametrize("method, current, pt, en", [
        ("brightness_down", 0.0, "O brilho já está no mínimo (0%).",
         "Brightness is already at the minimum (0%)."),
        ("brightness_up", 1.0, "O brilho já está no máximo (100%).",
         "Brightness is already at the maximum (100%)."),
    ])
    def test_at_limit(self, brightness, lang, method, current, pt, en):
        controller, services, _ = brightness([current])
        assert getattr(controller, method)(lang=lang) == (pt if lang == "pt" else en)
        services.DisplayServicesSetBrightness.assert_not_called()

    @pytest.mark.parametrize("lang", ["pt", "en"])
    @pytest.mark.parametrize("reads, set_result, actual", [
        ([0.7, 0.7], 0, None),               # accepted write, unchanged screen
        ([0.7, 0.45], 0, 45),               # mismatched read-back
        ([0.7, 0.33], 0, 33),               # outside the two-point tolerance
        ([0.7, (1, 0.0)], 0, None),         # read-back rc != 0
        ([0.7, OSError("read failed")], 0, None),
        ([0.7, float("nan")], 0, None),
        ([0.7, 0.7], 1, None),              # failed write, no change
        ([0.7, 0.45], 1, 45),               # failed write, actual change
        ([0.7, 0.3], 1, 30),                # rc != 0 still cannot claim success
        ([0.7, 0.45], OSError("write failed"), 45),
    ])
    def test_honest_failure(self, brightness, lang, reads, set_result, actual):
        controller, services, _ = brightness(reads, set_result)
        reply = controller.set_brightness(30, lang=lang)
        if actual is None:
            assert reply == _brightness_failure(lang)
        else:
            assert reply == (f"Não consegui deixar em 30% — o brilho está em {actual}%."
                             if lang == "pt" else f"I couldn't set it to 30% — brightness is at {actual}%.")
        assert services.DisplayServicesGetBrightness.call_count == 2

    @pytest.mark.parametrize("lang", ["pt", "en"])
    @pytest.mark.parametrize("bad_read", [(1, 0.0), OSError("read failed"), float("nan"),
                                          float("inf"), -0.1, 1.1])
    @pytest.mark.parametrize("method, args", [("set_brightness", (30,)), ("brightness_down", ()),
                                              ("brightness_up", ())])
    def test_initial_read_failure_never_writes(self, brightness, lang, bad_read, method, args):
        controller, services, _ = brightness([bad_read])
        assert getattr(controller, method)(*args, lang=lang) == _brightness_failure(lang)
        services.DisplayServicesSetBrightness.assert_not_called()

    @pytest.mark.parametrize("lang", ["pt", "en"])
    @pytest.mark.parametrize("method, args", [("set_brightness", (30,)), ("brightness_down", ()),
                                              ("brightness_up", ())])
    def test_missing_framework(self, monkeypatch, lang, method, args):
        from src.tools.macos_controller import _DisplayServicesBrightness

        loader = MagicMock(side_effect=OSError("framework missing"))
        monkeypatch.setattr(ctypes, "CDLL", loader)
        controller = MacOSController()
        controller._brightness = _DisplayServicesBrightness()
        loader.assert_not_called()  # construction/import must not load frameworks
        assert getattr(controller, method)(*args, lang=lang) == _brightness_failure(lang)

    def test_lazy_framework_abi(self, brightness, monkeypatch):
        from src.tools.macos_controller import _DisplayServicesBrightness

        _, services, graphics = brightness([0.456])
        loader = MagicMock(side_effect=[services, graphics])
        monkeypatch.setattr(ctypes, "CDLL", loader)
        monkeypatch.setattr("ctypes.util.find_library", lambda name: "CoreGraphics-test")
        backend = _DisplayServicesBrightness()
        loader.assert_not_called()
        assert backend.read() == 46
        assert [c.args[0] for c in loader.call_args_list] == [
            "/System/Library/PrivateFrameworks/DisplayServices.framework/DisplayServices", "CoreGraphics-test",
        ]
        assert graphics.CGMainDisplayID.restype is ctypes.c_uint32
        assert graphics.CGMainDisplayID.argtypes == []
        assert services.DisplayServicesGetBrightness.argtypes == [ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)]
        assert services.DisplayServicesSetBrightness.argtypes == [ctypes.c_uint32, ctypes.c_float]
        assert services.DisplayServicesGetBrightness.restype is ctypes.c_int
        assert services.DisplayServicesSetBrightness.restype is ctypes.c_int

    def test_missing_symbol_or_main_display_fails_soft(self, brightness):
        controller, services, graphics = brightness([0.5])
        del services.DisplayServicesGetBrightness
        assert controller.set_brightness(30) == _brightness_failure("pt")
        controller, services, graphics = brightness([0.5])
        graphics.CGMainDisplayID.side_effect = OSError("no main display")
        assert controller.brightness_down() == _brightness_failure("pt")
        services.DisplayServicesSetBrightness.assert_not_called()


def _brightness_failure(lang):
    return ("Não consegui mudar o brilho desta tela — nada mudou." if lang == "pt"
            else "I couldn't change this screen's brightness — nothing changed.")


@pytest.fixture
def brightness(monkeypatch):
    """Inject fake native functions; no normal brightness test may touch the OS."""
    from src.tools.macos_controller import _DisplayServicesBrightness

    monkeypatch.setattr(ctypes, "CDLL", MagicMock(side_effect=AssertionError("real framework forbidden")))
    monkeypatch.setattr("src.tools.macos_controller.subprocess.run",
                        MagicMock(side_effect=AssertionError("brightness must not shell out")))

    def build(reads, set_result=0):
        remaining = iter(reads)

        def get(display, out):
            assert display == 77
            result = next(remaining)
            if isinstance(result, Exception):
                raise result
            rc, value = result if isinstance(result, tuple) else (0, result)
            ctypes.cast(out, ctypes.POINTER(ctypes.c_float))[0] = value
            return rc

        services = MagicMock()
        services.DisplayServicesGetBrightness.side_effect = get
        if isinstance(set_result, Exception):
            services.DisplayServicesSetBrightness.side_effect = set_result
        else:
            services.DisplayServicesSetBrightness.return_value = set_result
        graphics = MagicMock()
        graphics.CGMainDisplayID.return_value = 77
        controller = MacOSController()
        controller._brightness = _DisplayServicesBrightness(services=services, core_graphics=graphics)
        return controller, services, graphics

    return build


@pytest.mark.skipif(os.getenv("CORTEX_REAL_MAC_TESTS") != "1", reason="requires CORTEX_REAL_MAC_TESTS=1")
def test_real_displayservices_brightness_restores_original():
    from src.tools.macos_controller import _DisplayServicesBrightness

    backend = _DisplayServicesBrightness()
    original = backend.read()
    assert original is not None, "main display brightness must be readable"
    try:
        assert backend.set(40)
        current = backend.read()
        assert current is not None and abs(current - 40) <= 2
    finally:
        assert backend.set(original), "failed to restore original brightness"
        restored = backend.read()
        assert restored is not None and abs(restored - original) <= 2


# =====================================================================
# Dark mode
# =====================================================================

class TestDarkMode:

    def test_enable_dark_mode(self, controller):
        with patch("src.tools.macos_controller.subprocess.run", return_value=_ok()) as run:
            reply = controller.enable_dark_mode()
        assert "ativado" in reply.lower()
        assert "set dark mode to true" in run.call_args.args[0][2]

    def test_disable_dark_mode(self, controller):
        with patch("src.tools.macos_controller.subprocess.run", return_value=_ok()) as run:
            reply = controller.disable_dark_mode()
        assert "claro" in reply.lower()
        assert "set dark mode to false" in run.call_args.args[0][2]

    def test_toggle_dark_mode_from_dark_to_light(self, controller):
        runs = [_ok(stdout="true"), _ok()]
        with patch("src.tools.macos_controller.subprocess.run", side_effect=runs) as run:
            reply = controller.toggle_dark_mode()
        assert "claro" in reply.lower()
        # Toggle from true should write false
        assert "set dark mode to false" in run.call_args_list[1].args[0][2]

    def test_toggle_dark_mode_from_light_to_dark(self, controller):
        runs = [_ok(stdout="false"), _ok()]
        with patch("src.tools.macos_controller.subprocess.run", side_effect=runs) as run:
            reply = controller.toggle_dark_mode()
        assert "escuro" in reply.lower()
        assert "set dark mode to true" in run.call_args_list[1].args[0][2]


# =====================================================================
# Lock screen
# =====================================================================

class TestLockScreen:

    def test_lock_uses_ctrl_cmd_q_keystroke(self, controller):
        # The old CGSession binary 404'd in recent macOS; lock now sends the
        # documented Ctrl+Cmd+Q Lock Screen shortcut via System Events.
        with patch("src.tools.macos_controller.subprocess.run", return_value=_ok()) as run:
            reply = controller.lock_screen()
        assert "travada" in reply.lower()
        cmd = run.call_args.args[0]
        assert cmd[0] == "osascript"
        assert 'keystroke "q"' in cmd[2]
        assert "control down" in cmd[2] and "command down" in cmd[2]

    def test_lock_falls_back_to_pmset_when_keystroke_fails(self, controller):
        # No Accessibility permission → keystroke fails → sleep the display.
        with patch("src.tools.macos_controller.subprocess.run",
                   side_effect=[_fail(stderr="-1743"), _ok()]) as run:
            reply = controller.lock_screen()
        assert "travada" in reply.lower()
        second = run.call_args_list[1].args[0]
        # M4-A6: absolute path so the fallback actually fires under launchd.
        assert second == ["/usr/bin/pmset", "displaysleepnow"]

    def test_lock_both_paths_fail_returns_friendly(self, controller):
        with patch("src.tools.macos_controller.subprocess.run",
                   side_effect=[_fail(), _fail()]):
            reply = controller.lock_screen()
        assert "não consegui travar" in reply.lower()


# =====================================================================
# Screenshot
# =====================================================================

class TestScreenshot:

    def test_screenshot_writes_to_data_screenshots(self, tmp_path, monkeypatch, controller):
        monkeypatch.chdir(tmp_path)  # so data/screenshots/ resolves under tmp
        with patch("src.tools.macos_controller.subprocess.run", return_value=_ok()) as run:
            reply = controller.screenshot()
        # File path appears in reply
        assert "data/screenshots" in reply
        # screencapture invoked silently
        cmd = run.call_args.args[0]
        assert cmd[0] == "/usr/sbin/screencapture"
        assert "-x" in cmd
        # The destination directory was created
        assert (tmp_path / "data" / "screenshots").is_dir()

    def test_screenshot_with_custom_filename(self, tmp_path, monkeypatch, controller):
        monkeypatch.chdir(tmp_path)
        with patch("src.tools.macos_controller.subprocess.run", return_value=_ok()) as run:
            reply = controller.screenshot(filename="my_capture")
        assert "my_capture.png" in reply
        # Must auto-append .png

    def test_screenshot_strips_directory_traversal(self, tmp_path, monkeypatch, controller):
        monkeypatch.chdir(tmp_path)
        with patch("src.tools.macos_controller.subprocess.run", return_value=_ok()) as run:
            reply = controller.screenshot(filename="../../etc/passwd.png")
        # basename() should have stripped the path components
        assert "passwd.png" in reply
        assert "/etc/" not in reply

    @pytest.mark.parametrize("bad", ["/", "..", "...", "   "])
    def test_screenshot_degenerate_filename_falls_back_to_timestamp(
        self, tmp_path, monkeypatch, controller, bad
    ):
        # M4-A14f: basename("/") == "" → the old code produced a hidden
        # ".png"; degenerate names now fall back to the timestamp name.
        monkeypatch.chdir(tmp_path)
        with patch("src.tools.macos_controller.subprocess.run", return_value=_ok()) as run:
            reply = controller.screenshot(filename=bad)
        target = run.call_args.args[0][-1]
        import os as _os
        base = _os.path.basename(target)
        assert base.startswith("cortex_screenshot_")
        assert base != ".png"

    def test_screenshot_failure_returns_friendly(self, tmp_path, monkeypatch, controller):
        monkeypatch.chdir(tmp_path)
        with patch("src.tools.macos_controller.subprocess.run", return_value=_fail()):
            reply = controller.screenshot()
        assert "não consegui tirar" in reply.lower()

    # ---- locked-screen guard (s75 polish) ----
    # screencapture writes a black png + reports success when the screen is
    # locked; the guard reads IOConsoleLocked from ioreg and refuses early.

    _IOREG_LOCKED = _ok(stdout='  | "IOConsoleLocked" = Yes\n')
    _IOREG_UNLOCKED = _ok(stdout='  | "IOConsoleLocked" = No\n')

    def test_screenshot_refuses_when_locked(self, tmp_path, monkeypatch, controller):
        monkeypatch.chdir(tmp_path)
        with patch("src.tools.macos_controller.subprocess.run",
                   return_value=self._IOREG_LOCKED) as run:
            reply = controller.screenshot()
        assert "trancada" in reply.lower()
        # screencapture must NEVER run — only the ioreg probe did.
        for call in run.call_args_list:
            assert call.args[0][0] != "/usr/sbin/screencapture", (
                "must not attempt a capture while locked")
        # and no file path is promised
        assert "data/screenshots" not in reply

    def test_screenshot_proceeds_when_unlocked(self, tmp_path, monkeypatch, controller):
        monkeypatch.chdir(tmp_path)
        # ioreg says unlocked, then screencapture succeeds.
        runs = [self._IOREG_UNLOCKED, _ok()]
        with patch("src.tools.macos_controller.subprocess.run",
                   side_effect=runs) as run:
            reply = controller.screenshot()
        assert "data/screenshots" in reply
        assert run.call_args.args[0][0] == "/usr/sbin/screencapture"

    def test_screenshot_fails_open_when_ioreg_errors(self, tmp_path, monkeypatch, controller):
        # The probe itself failing must NOT block a legitimate capture —
        # denying a real screenshot on a detection glitch is worse than the
        # occasional black png the guard targets.
        monkeypatch.chdir(tmp_path)
        runs = [_fail(stderr="ioreg: broken"), _ok()]
        with patch("src.tools.macos_controller.subprocess.run",
                   side_effect=runs) as run:
            reply = controller.screenshot()
        assert "data/screenshots" in reply
        assert run.call_args.args[0][0] == "/usr/sbin/screencapture"

    def test_screen_is_locked_parses_yes_and_no(self, controller):
        with patch("src.tools.macos_controller.subprocess.run",
                   return_value=self._IOREG_LOCKED):
            assert controller._screen_is_locked() is True
        with patch("src.tools.macos_controller.subprocess.run",
                   return_value=self._IOREG_UNLOCKED):
            assert controller._screen_is_locked() is False


# =====================================================================
# App open
# =====================================================================

class TestOpenApp:

    def test_open_app_happy(self, controller):
        with patch("src.tools.macos_controller.subprocess.run", return_value=_ok()) as run:
            reply = controller.open_app("Notion")
        assert "Abrindo Notion" in reply
        cmd = run.call_args.args[0]
        assert cmd == ["open", "-a", "Notion"]

    def test_open_app_strips_whitespace(self, controller):
        with patch("src.tools.macos_controller.subprocess.run", return_value=_ok()) as run:
            controller.open_app("   Calendar   ")
        assert run.call_args.args[0] == ["open", "-a", "Calendar"]

    def test_open_app_empty_returns_friendly(self, controller):
        # No subprocess call expected when name is blank.
        with patch("src.tools.macos_controller.subprocess.run") as run:
            reply = controller.open_app("")
        assert "preciso do nome" in reply.lower()
        run.assert_not_called()

    def test_open_app_failure_surfaces_app_name(self, controller):
        with patch("src.tools.macos_controller.subprocess.run",
                   return_value=_fail(stderr="Unable to find application named 'Foo'")):
            reply = controller.open_app("Foo")
        assert "Foo" in reply
        assert "não consegui" in reply.lower()


class TestQuitApp:

    def test_quit_app_happy(self, controller):
        with patch("src.tools.macos_controller.subprocess.run", return_value=_ok()) as run:
            reply = controller.quit_app("Notion")
        assert "Fechando Notion" in reply
        cmd = run.call_args.args[0]
        assert cmd == ["osascript", "-e", 'tell application "Notion" to quit']

    def test_quit_app_strips_whitespace(self, controller):
        with patch("src.tools.macos_controller.subprocess.run", return_value=_ok()) as run:
            controller.quit_app("   Spotify   ")
        assert run.call_args.args[0] == [
            "osascript", "-e", 'tell application "Spotify" to quit'
        ]

    def test_quit_app_empty_returns_friendly(self, controller):
        with patch("src.tools.macos_controller.subprocess.run") as run:
            reply = controller.quit_app("")
        assert "preciso do nome" in reply.lower()
        run.assert_not_called()

    def test_quit_app_not_running_is_friendly_not_error(self, controller):
        with patch("src.tools.macos_controller.subprocess.run",
                   return_value=_fail(stderr="Application isn't running. (-600)")):
            reply = controller.quit_app("Mail")
        assert "já estava fechado" in reply
        assert "não consegui" not in reply.lower()

    def test_quit_app_failure_surfaces_app_name(self, controller):
        with patch("src.tools.macos_controller.subprocess.run",
                   return_value=_fail(stderr="boom")):
            reply = controller.quit_app("Foo")
        assert "Foo" in reply
        assert "não consegui" in reply.lower()

    def test_quit_app_escapes_quotes_in_name(self, controller):
        with patch("src.tools.macos_controller.subprocess.run", return_value=_ok()) as run:
            controller.quit_app('Find My "Mac"')
        cmd = run.call_args.args[0]
        assert cmd == [
            "osascript", "-e", r'tell application "Find My \"Mac\"" to quit'
        ]


class TestCloseWindow:
    """close_window = red button / Cmd+W (app keeps running), NOT a quit."""

    def test_close_window_activates_then_cmd_w(self, controller):
        # 1st osascript = the running check (M4-A14a), 2nd = activate+Cmd+W.
        with patch("src.tools.macos_controller.subprocess.run",
                   side_effect=[_ok("true"), _ok("closed")]) as run:
            reply = controller.close_window("Spotify")
        assert "janela do spotify fechada" in reply.lower()
        assert "continua aberto" in reply.lower()
        script = run.call_args.args[0][2]  # osascript -e <script>
        assert 'tell application "Spotify" to activate' in script
        assert 'keystroke "w" using command down' in script

    def test_cmd_w_waits_for_the_app_to_be_frontmost(self, controller):
        # s88 (Codex lote 12): a fixed `delay 0.1` then a global Cmd+W closed
        # the window of whatever app the user switched to meanwhile.
        with patch("src.tools.macos_controller.subprocess.run",
                   side_effect=[_ok("true"), _ok("closed")]) as run:
            controller.close_window("Spotify")
        script = run.call_args.args[0][2]
        assert "delay 0.1" not in script
        assert script.index("frontmost of p") < script.index('keystroke "w"')
        assert 'if not (frontmost of p) then return "notfront"' in script

    def test_another_app_in_front_closes_nothing(self, controller):
        with patch("src.tools.macos_controller.subprocess.run",
                   side_effect=[_ok("true"), _ok("notfront")]):
            reply = controller.close_window("Spotify")
        assert "não fechei" in reply.lower()
        assert "fechada" not in reply.lower()

    @pytest.mark.parametrize("seen,expected", [
        ("sheet", "aviso antes de fechar"),
        ("open", "continua aberta"),
        ("nowindow", "não tem janela"),
        ("", "continua aberta"),          # nothing confirmed, nothing claimed
    ])
    def test_closed_is_said_only_when_the_window_went_away(self, controller,
                                                           seen, expected):
        # s88 (Codex lote 12): TextEdit with unsaved changes answers Cmd+W
        # with a save sheet, and Cortex said the window was closed.
        with patch("src.tools.macos_controller.subprocess.run",
                   side_effect=[_ok("true"), _ok(seen)]):
            reply = controller.close_window("TextEdit")
        assert expected in reply.lower()
        assert "fechada" not in reply.lower()

    def test_close_window_not_running_does_not_launch(self, controller):
        # M4-A14a: `activate` LAUNCHES a closed app, and the Cmd+W lands on
        # whatever is frontmost. A not-running app gets an answer, not an
        # action — and only the check runs (no second osascript).
        with patch("src.tools.macos_controller.subprocess.run",
                   side_effect=[_ok("false")]) as run:
            reply = controller.close_window("Notion")
        assert "não está aberto" in reply.lower()
        assert run.call_count == 1
        script = run.call_args.args[0][2]
        assert "is running" in script
        assert "keystroke" not in script

    def test_close_window_empty_returns_friendly(self, controller):
        with patch("src.tools.macos_controller.subprocess.run") as run:
            reply = controller.close_window("")
        assert "preciso do nome" in reply.lower()
        run.assert_not_called()

    def test_close_window_failure_surfaces_app_name(self, controller):
        with patch("src.tools.macos_controller.subprocess.run",
                   return_value=_fail(stderr="boom")):
            reply = controller.close_window("Foo")
        assert "Foo" in reply
        assert "não consegui" in reply.lower()

    def test_close_window_escapes_quotes(self, controller):
        with patch("src.tools.macos_controller.subprocess.run",
                   side_effect=[_ok("true"), _ok()]) as run:
            controller.close_window('Find My "Mac"')
        script = run.call_args.args[0][2]
        assert r'tell application "Find My \"Mac\"" to activate' in script


class TestSleepMac:

    def test_sleep_happy(self, controller):
        with patch("src.tools.macos_controller.subprocess.run",
                   return_value=_ok()) as run:
            reply = controller.sleep_mac()
        assert "Mac" in reply and "dormir" in reply.lower()
        # Must use the absolute pmset path so launchd's restricted PATH
        # doesn't cause a silent ENOENT under uvicorn (same fix the
        # screencapture path needed).
        cmd = run.call_args.args[0]
        assert cmd == ["/usr/bin/pmset", "sleepnow"]

    def test_sleep_failure_surfaces_error(self, controller):
        with patch("src.tools.macos_controller.subprocess.run",
                   return_value=_fail(stderr="not authorised")):
            reply = controller.sleep_mac()
        assert "não consegui" in reply.lower()
        # The stderr text should make it into the user-facing string so
        # an operator can diagnose without grepping logs.
        assert "not authorised" in reply

    def test_sleep_takes_no_positional_args(self, controller):
        # Pure parameter check: the signature must accept zero positional
        # args (the orchestrator's call shape — staging passes {}). The
        # optional `lang` kwarg (s60 localisation) defaults to "pt".
        import inspect
        sig = inspect.signature(controller.sleep_mac)
        params = list(sig.parameters)
        assert params == ["lang"]
        assert sig.parameters["lang"].default == "pt"


class TestListApps:

    def test_list_happy_sorted(self, controller):
        with patch("src.tools.macos_controller.subprocess.run",
                   return_value=_ok(stdout="Safari, Notion, Finder")):
            reply = controller.list_apps()
        assert reply == "Apps abertos: Finder, Notion, Safari."  # sorted, case-insensitive

    def test_list_queries_system_events(self, controller):
        with patch("src.tools.macos_controller.subprocess.run",
                   return_value=_ok(stdout="Finder")) as run:
            controller.list_apps()
        cmd = run.call_args.args[0]
        assert cmd[0] == "osascript"
        assert "background only is false" in cmd[2]

    def test_list_empty(self, controller):
        with patch("src.tools.macos_controller.subprocess.run", return_value=_ok(stdout="")):
            reply = controller.list_apps()
        assert "Nenhum app" in reply

    def test_list_tcc_failure_surfaces_hint(self, controller):
        with patch("src.tools.macos_controller.subprocess.run",
                   return_value=_fail(stderr="execution error: -1743")):
            reply = controller.list_apps()
        assert "não consegui listar" in reply.lower()
        assert "accessibility" in reply.lower()


class TestSwitchToApp:

    def test_switch_happy(self, controller):
        with patch("src.tools.macos_controller.subprocess.run", return_value=_ok()) as run:
            reply = controller.switch_to_app("Notion")
        assert "Foco em Notion" in reply
        cmd = run.call_args.args[0]
        assert cmd == ["osascript", "-e", 'tell application "Notion" to activate']

    def test_switch_strips_whitespace(self, controller):
        with patch("src.tools.macos_controller.subprocess.run", return_value=_ok()) as run:
            controller.switch_to_app("   Spotify   ")
        assert run.call_args.args[0] == [
            "osascript", "-e", 'tell application "Spotify" to activate'
        ]

    def test_switch_empty_returns_friendly(self, controller):
        with patch("src.tools.macos_controller.subprocess.run") as run:
            reply = controller.switch_to_app("")
        assert "preciso do nome" in reply.lower()
        run.assert_not_called()

    def test_switch_failure_surfaces_app_name(self, controller):
        with patch("src.tools.macos_controller.subprocess.run",
                   return_value=_fail(stderr="boom")):
            reply = controller.switch_to_app("Foo")
        assert "Foo" in reply
        assert "não consegui focar" in reply.lower()

    def test_switch_escapes_quotes(self, controller):
        with patch("src.tools.macos_controller.subprocess.run", return_value=_ok()) as run:
            controller.switch_to_app('Find My "Mac"')
        assert run.call_args.args[0] == [
            "osascript", "-e", r'tell application "Find My \"Mac\"" to activate'
        ]


# =====================================================================
# Subprocess error paths (covered via _run indirectly)
# =====================================================================

class TestSubprocessErrorPaths:

    def test_timeout_surfaces_friendly(self, controller):
        def raiser(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=args[0], timeout=5)
        with patch("src.tools.macos_controller.subprocess.run", side_effect=raiser):
            reply = controller.mute()
        assert "timeout" in reply.lower()

    def test_missing_binary_surfaces_friendly(self, controller):
        def raiser(*args, **kwargs):
            raise FileNotFoundError("no such binary")
        with patch("src.tools.macos_controller.subprocess.run", side_effect=raiser):
            reply = controller.mute()
        assert "não encontrado" in reply.lower()
