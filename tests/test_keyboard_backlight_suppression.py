"""Tasks 118b–118d: hardware-free tests, also runnable with stdlib unittest."""
import sys
import unittest
from itertools import product
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from src.tools import macos_controller as module


class KeyboardSuppressionTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.client.copyKeyboardBacklightIDs.return_value = [77]
        self.client.isBacklightDimmedOnKeyboard_.return_value = False
        self.client.isBacklightSuppressedOnKeyboard_.return_value = True
        self.client.brightnessForKeyboard_.return_value = 0.0
        self.client.setBrightness_forKeyboard_.return_value = True
        self.objc = SimpleNamespace(lookUpClass=Mock(), registerMetaDataForSelector=Mock(), _C_NSBOOL=b"Z")
        self.objc.lookUpClass.return_value.alloc.return_value.init.return_value = self.client
        self.bundle = Mock()
        self.bundle.bundleWithPath_.return_value.load.return_value = True
        for patcher in (
            patch.dict(sys.modules, {"objc": self.objc, "Foundation": SimpleNamespace(NSBundle=self.bundle)}),
            patch.object(module.subprocess, "run", side_effect=AssertionError("no real subprocess")),
            patch.object(module, "_osascript", side_effect=AssertionError("no osascript")),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def tearDown(self):
        self.assertFalse(any(c[0].startswith(("suspendIdleDimming", "setIdleDimTime",
                                             "enableAutoBrightness"))
                             for c in self.client.method_calls))

    def test_dimmed_fake_client(self):
        for value, expected in [(True, True), (False, False), (1, True), (0, False),
                                (None, None), ("bad", None), (2, None)]:
            with self.subTest(value=value):
                self.client.reset_mock()
                self.client.copyKeyboardBacklightIDs.return_value = [2**63 + 7]
                self.client.isBacklightDimmedOnKeyboard_.return_value = value
                helper = module._KeyboardBacklightBrightness(self.client)
                self.assertIs(helper.dimmed(), expected)
                self.assertEqual(self.client.method_calls, [call.copyKeyboardBacklightIDs(),
                    call.isBacklightDimmedOnKeyboard_(2**63 + 7)])

    def test_dimmed_selector_abi_is_lazy_and_registered_once(self):
        helper = module._KeyboardBacklightBrightness()
        self.assertEqual(helper.read(), 0)
        selector_call = call(b"KeyboardBrightnessClient", b"isBacklightDimmedOnKeyboard:",
            {"retval": {"type": b"Z"}, "arguments": {2: {"type": b"Q"}}})
        self.assertNotIn(selector_call, self.objc.registerMetaDataForSelector.call_args_list)
        self.assertIs(helper.dimmed(), False)
        self.assertIs(helper.dimmed(), False)
        self.assertEqual(self.objc.registerMetaDataForSelector.call_args_list.count(selector_call), 1)

    def test_dimmed_failures_return_none(self):
        failures = [
            patch.object(self.bundle.bundleWithPath_.return_value.load, "return_value", False),
            patch.object(self.objc.registerMetaDataForSelector, "side_effect", RuntimeError("ABI")),
            patch.object(self.client.copyKeyboardBacklightIDs, "side_effect", RuntimeError("IDs")),
            patch.object(self.client.copyKeyboardBacklightIDs, "return_value", []),
            patch.object(self.client.isBacklightDimmedOnKeyboard_, "side_effect", AttributeError("selector")),
            patch.dict(sys.modules, {"objc": None}),
        ]
        for index, failure in enumerate(failures):
            with self.subTest(failure=index), failure:
                self.assertIsNone(module._KeyboardBacklightBrightness().dimmed())

    def test_dimmed_missed_readback_takes_precedence_over_suppression(self):
        methods = [
            ("keyboard_brightness_set", (50,), 0.0, 50),
            ("keyboard_brightness_set", (0,), 0.5, 0),
            ("keyboard_brightness_up", (), 0.4, 50),
            ("keyboard_brightness_down", (), 0.4, 30),
        ]
        for lang, lid, ok, suppression, missing_readback, (method, args, before, target) in product(
                ("pt", "en"), (False, None), (True, False), (True, False, None), (False, True), methods):
            with self.subTest(lang=lang, lid=lid, ok=ok, suppression=suppression,
                              missing_readback=missing_readback, method=method):
                self.client.reset_mock()
                self.client.isBacklightDimmedOnKeyboard_.return_value = True
                self.client.isBacklightSuppressedOnKeyboard_.return_value = suppression
                self.client.brightnessForKeyboard_.side_effect = [before, None if missing_readback else before]
                self.client.setBrightness_forKeyboard_.return_value = ok
                controller = module.MacOSController()
                controller._keyboard_brightness = module._KeyboardBacklightBrightness(self.client)
                with patch.object(controller, "_lid_closed", return_value=lid):
                    self.assertEqual(getattr(controller, method)(*args, lang=lang),
                        "O Mac está sem uso há alguns minutos, então o macOS apagou a luz do teclado. "
                        "Ela volta quando você mexer no teclado ou no trackpad." if lang == "pt" else
                        "The Mac has been idle for a few minutes, so macOS turned the keyboard light off. "
                        "It comes back when you touch the keyboard or trackpad.")
                self.assertEqual(self.client.method_calls, [call.copyKeyboardBacklightIDs(),
                    call.brightnessForKeyboard_(77), call.setBrightness_forKeyboard_(target / 100, 77),
                    call.brightnessForKeyboard_(77), call.isBacklightDimmedOnKeyboard_(77)])

    def test_failed_dimmed_query_or_metadata_keeps_read_write_and_suppression(self):
        def metadata(class_name, selector, metadata):
            if selector == b"isBacklightDimmedOnKeyboard:":
                raise RuntimeError("dimmed ABI unavailable")

        for failure, suppression in product(("selector", "metadata"), (True, False, None)):
            with self.subTest(failure=failure, suppression=suppression):
                self.client.reset_mock()
                self.client.brightnessForKeyboard_.side_effect = [0.4, 0.4]
                self.client.isBacklightSuppressedOnKeyboard_.return_value = suppression
                self.client.isBacklightDimmedOnKeyboard_.side_effect = (
                    AttributeError("selector unavailable") if failure == "selector" else None)
                self.objc.registerMetaDataForSelector.side_effect = metadata if failure == "metadata" else None
                controller = module.MacOSController()
                with patch.object(controller, "_lid_closed", return_value=False):
                    self.assertEqual(controller.keyboard_brightness_set(50, lang="en"),
                        "macOS is keeping the keyboard light off right now. Nothing changed." if suppression else
                        "I couldn't set it to 50% — keyboard light is at 40%.")
                self.client.setBrightness_forKeyboard_.assert_called_once_with(0.5, 77)
                self.client.isBacklightSuppressedOnKeyboard_.assert_called_once_with(77)

    def test_suppressed_fake_client(self):
        for value, expected in [(True, True), (False, False), (1, True), (0, False),
                                (None, None), ("bad", None), (2, None)]:
            with self.subTest(value=value):
                self.client.reset_mock()
                self.client.copyKeyboardBacklightIDs.return_value = [2**63 + 7]
                self.client.isBacklightSuppressedOnKeyboard_.return_value = value
                helper = module._KeyboardBacklightBrightness(self.client)
                self.assertIs(helper.suppressed(), expected)
                self.assertEqual(self.client.method_calls, [call.copyKeyboardBacklightIDs(),
                    call.isBacklightSuppressedOnKeyboard_(2**63 + 7)])

    def test_suppression_selector_abi(self):
        self.assertIs(module._KeyboardBacklightBrightness().suppressed(), True)
        self.assertIn(call(b"KeyboardBrightnessClient", b"isBacklightSuppressedOnKeyboard:",
            {"retval": {"type": b"Z"}, "arguments": {2: {"type": b"Q"}}}),
            self.objc.registerMetaDataForSelector.call_args_list)

    def test_suppression_failures_return_none(self):
        failures = [
            patch.object(self.bundle.bundleWithPath_.return_value.load, "return_value", False),
            patch.object(self.objc.registerMetaDataForSelector, "side_effect", RuntimeError("ABI")),
            patch.object(self.client.copyKeyboardBacklightIDs, "side_effect", RuntimeError("IDs")),
            patch.object(self.client.copyKeyboardBacklightIDs, "return_value", []),
            patch.object(self.client.isBacklightSuppressedOnKeyboard_, "side_effect", AttributeError("selector")),
            patch.dict(sys.modules, {"objc": None}),
        ]
        for index, failure in enumerate(failures):
            with self.subTest(failure=index), failure:
                self.assertIsNone(module._KeyboardBacklightBrightness().suppressed())

    def test_lid_reader_parses_ioreg(self):
        for output, expected in [
            ('  | "AppleClamshellState" = Yes\n', True),
            ('  | "AppleClamshellState" = No\n', False),
            ('"AppleClamshellState" = Maybe', None),
            ('"OtherState" = Yes', None), ("", None),
        ]:
            with self.subTest(output=output), patch.object(module.subprocess, "run") as runner:
                runner.return_value = module.subprocess.CompletedProcess([], 0, output, "")
                self.assertIs(module.MacOSController()._lid_closed(), expected)
                runner.assert_called_once_with(
                    ["/usr/sbin/ioreg", "-r", "-k", "AppleClamshellState", "-d", "1"],
                    capture_output=True, text=True, timeout=module._DEFAULT_TIMEOUT_S)

    def test_lid_reader_failures_return_none(self):
        for failure in [None, module.subprocess.TimeoutExpired("ioreg", 5),
                        FileNotFoundError("ioreg"), RuntimeError("ioreg")]:
            with self.subTest(failure=failure), patch.object(module.subprocess, "run") as runner:
                runner.return_value = module.subprocess.CompletedProcess(
                    [], 1, '"AppleClamshellState" = Yes', "failed")
                runner.side_effect = failure
                self.assertIsNone(module.MacOSController()._lid_closed())

    def test_suppressed_open_or_unknown_lid_write_sticks(self):
        self.client.isBacklightDimmedOnKeyboard_.return_value = True
        methods = [
            ("keyboard_brightness_set", (50,), 0.0, 50),
            ("keyboard_brightness_set", (100,), 0.0, 100),
            ("keyboard_brightness_set", (0,), 0.5, 0),
            ("keyboard_brightness_set", (-20,), 0.5, 0),
            ("keyboard_brightness_up", (), 0.4, 50),
            ("keyboard_brightness_down", (), 0.4, 30),
        ]
        for lang, lid, (method, args, before, target) in product(
                ("pt", "en"), (False, None), methods):
            with self.subTest(lang=lang, lid=lid, method=method, args=args):
                self.client.reset_mock()
                self.client.brightnessForKeyboard_.side_effect = [before, target / 100]
                controller = module.MacOSController()
                controller._keyboard_brightness = module._KeyboardBacklightBrightness(self.client)
                with patch.object(controller, "_lid_closed", return_value=lid):
                    self.assertEqual(getattr(controller, method)(*args, lang=lang),
                        f"Luz do teclado em {target}%." if lang == "pt" else f"Keyboard light at {target}%.")
                self.client.setBrightness_forKeyboard_.assert_called_once_with(target / 100, 77)
                self.assertEqual(self.client.brightnessForKeyboard_.call_count, 2)
                self.client.isBacklightDimmedOnKeyboard_.assert_not_called()
                self.client.isBacklightSuppressedOnKeyboard_.assert_not_called()

    def test_suppressed_write_does_not_stick_uses_neutral_explanation_after_dimmed(self):
        methods = [
            ("keyboard_brightness_set", (50,), 0.0, 50),
            ("keyboard_brightness_set", (0,), 0.5, 0),
            ("keyboard_brightness_up", (), 0.4, 50),
            ("keyboard_brightness_down", (), 0.4, 30),
        ]
        for lang, lid, ok, dimmed, (method, args, before, target) in product(
                ("pt", "en"), (False, None), (True, False), (False, None), methods):
            with self.subTest(lang=lang, lid=lid, ok=ok, dimmed=dimmed, method=method):
                self.client.reset_mock()
                self.client.isBacklightDimmedOnKeyboard_.return_value = dimmed
                self.client.brightnessForKeyboard_.side_effect = [before, before]
                self.client.setBrightness_forKeyboard_.return_value = ok
                controller = module.MacOSController()
                controller._keyboard_brightness = module._KeyboardBacklightBrightness(self.client)
                with patch.object(controller, "_lid_closed", return_value=lid):
                    expected = ("O macOS está deixando a luz do teclado apagada agora. Nada mudou."
                                if lang == "pt" else
                                "macOS is keeping the keyboard light off right now. Nothing changed.")
                    self.assertEqual(getattr(controller, method)(*args, lang=lang), expected)
                self.client.setBrightness_forKeyboard_.assert_called_once_with(target / 100, 77)
                self.assertEqual(self.client.brightnessForKeyboard_.call_count, 2)
                self.assertEqual(self.client.method_calls, [call.copyKeyboardBacklightIDs(),
                    call.brightnessForKeyboard_(77), call.setBrightness_forKeyboard_(target / 100, 77),
                    call.brightnessForKeyboard_(77), call.isBacklightDimmedOnKeyboard_(77),
                    call.isBacklightSuppressedOnKeyboard_(77)])

    def test_closed_lid_never_writes_regardless_of_suppression(self):
        methods = [("keyboard_brightness_set", (50,)), ("keyboard_brightness_set", (100,)),
                   ("keyboard_brightness_up", ()), ("keyboard_brightness_down", ())]
        for lang, suppression, (method, args) in product(
                ("pt", "en"), (True, False, None), methods):
            with self.subTest(lang=lang, suppression=suppression, method=method):
                self.client.reset_mock()
                self.client.isBacklightSuppressedOnKeyboard_.return_value = suppression
                controller = module.MacOSController()
                controller._keyboard_brightness = module._KeyboardBacklightBrightness(self.client)
                with patch.object(controller, "_lid_closed", return_value=True):
                    expected = ("A tampa do Mac está fechada — com ela fechada o macOS deixa a luz do teclado apagada. Nada mudou."
                                if lang == "pt" else
                                "The Mac's lid is closed — with it closed macOS keeps the keyboard light off. Nothing changed.")
                    self.assertEqual(getattr(controller, method)(*args, lang=lang), expected)
                self.client.setBrightness_forKeyboard_.assert_not_called()
                self.client.brightnessForKeyboard_.assert_not_called()
                self.client.isBacklightDimmedOnKeyboard_.assert_not_called()
                self.client.isBacklightSuppressedOnKeyboard_.assert_not_called()

    def test_closed_lid_off_is_already_off(self):
        for lang, suppression, level in product(("pt", "en"), (True, False, None), (0, -20)):
            with self.subTest(lang=lang, suppression=suppression, level=level):
                self.client.reset_mock()
                self.client.isBacklightSuppressedOnKeyboard_.return_value = suppression
                controller = module.MacOSController()
                controller._keyboard_brightness = module._KeyboardBacklightBrightness(self.client)
                with patch.object(controller, "_lid_closed", return_value=True):
                    self.assertEqual(controller.keyboard_brightness_set(level, lang=lang),
                        "A luz do teclado já está apagada." if lang == "pt" else "The keyboard light is already off.")
                self.client.setBrightness_forKeyboard_.assert_not_called()
                self.client.brightnessForKeyboard_.assert_not_called()

    def test_unsuppressed_or_unknown_preserves_118(self):
        methods = [
            ("keyboard_brightness_set", (50,), [0.0, 0.0], 50),
            ("keyboard_brightness_set", (0,), [0.5, 0.0], 0),
            ("keyboard_brightness_up", (), [0.4, 0.5], 50),
            ("keyboard_brightness_down", (), [0.4, 0.3], 30),
        ]
        for suppression, lid, lang, (method, args, readings, target) in product(
                (False, None), (False, None), ("pt", "en"), methods):
            with self.subTest(suppression=suppression, lang=lang, method=method, target=target):
                self.client.reset_mock()
                self.client.isBacklightSuppressedOnKeyboard_.return_value = suppression
                self.client.brightnessForKeyboard_.side_effect = readings
                controller = module.MacOSController()
                controller._keyboard_brightness = module._KeyboardBacklightBrightness(self.client)
                with patch.object(controller, "_lid_closed", return_value=lid) as lid_reader:
                    if readings[-1] == target / 100:
                        expected = f"Luz do teclado em {target}%." if lang == "pt" else f"Keyboard light at {target}%."
                    else:
                        expected = ("Não consegui deixar em 50% — a luz do teclado está em 0%." if lang == "pt"
                                    else "I couldn't set it to 50% — keyboard light is at 0%.")
                    self.assertEqual(getattr(controller, method)(*args, lang=lang), expected)
                    self.client.setBrightness_forKeyboard_.assert_called_once_with(target / 100, 77)
                    lid_reader.assert_called_once_with()

    def test_failed_suppression_query_or_metadata_preserves_118(self):
        def metadata(class_name, selector, metadata):
            if selector == b"isBacklightSuppressedOnKeyboard:":
                raise RuntimeError("suppression ABI unavailable")

        for failure, actual in product(("selector", "metadata"), (0.5, 0.4)):
            with self.subTest(failure=failure, actual=actual):
                self.client.reset_mock()
                self.client.brightnessForKeyboard_.side_effect = [0.4, actual]
                self.client.isBacklightSuppressedOnKeyboard_.side_effect = (
                    AttributeError("selector unavailable") if failure == "selector" else None)
                self.objc.registerMetaDataForSelector.side_effect = metadata if failure == "metadata" else None
                controller = module.MacOSController()
                with patch.object(controller, "_lid_closed", return_value=None) as lid_reader:
                    expected = ("Keyboard light at 50%." if actual == 0.5 else
                                "I couldn't set it to 50% — keyboard light is at 40%.")
                    self.assertEqual(controller.keyboard_brightness_set(50, lang="en"), expected)
                    self.client.setBrightness_forKeyboard_.assert_called_once_with(0.5, 77)
                    lid_reader.assert_called_once_with()

    def test_suppressed_but_readback_reached_target_preserves_118_failed_write(self):
        self.client.isBacklightDimmedOnKeyboard_.return_value = True
        for lang in ("pt", "en"):
            with self.subTest(lang=lang):
                self.client.reset_mock()
                self.client.brightnessForKeyboard_.side_effect = [0.4, 0.5]
                self.client.setBrightness_forKeyboard_.return_value = False
                controller = module.MacOSController()
                controller._keyboard_brightness = module._KeyboardBacklightBrightness(self.client)
                with patch.object(controller, "_lid_closed", return_value=False):
                    expected = ("Não consegui deixar em 50% — a luz do teclado está em 50%." if lang == "pt" else
                                "I couldn't set it to 50% — keyboard light is at 50%.")
                    self.assertEqual(controller.keyboard_brightness_set(50, lang=lang), expected)
                self.client.setBrightness_forKeyboard_.assert_called_once_with(0.5, 77)
                self.assertEqual(self.client.brightnessForKeyboard_.call_count, 2)
                self.client.isBacklightDimmedOnKeyboard_.assert_not_called()
                self.client.isBacklightSuppressedOnKeyboard_.assert_not_called()
