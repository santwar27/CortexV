"""Task 118: fake CoreBrightness only; never load the real private framework."""
import sys
from itertools import product
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call

import pytest

from src.core.dispatch import macos
from src.core.dispatch.heuristics import REAL_TOOL_HINT_RE, TOOL_HINT_RE
from src.core.dispatch.normalisation import strip_courtesy
from src.core.dispatch.registry import run, simulate
from src.core.hallucination import detect_hallucinated_mac_action
from src.core.i18n import _detect_lang
from src.tools import macos_controller as module
from tests.test_engine_mac_hallucination import MAC_FALLBACK, rig


@pytest.fixture(autouse=True)
def native(monkeypatch):
    client = Mock()
    client.copyKeyboardBacklightIDs.return_value = [77]
    client.brightnessForKeyboard_.return_value = 0.456
    client.setBrightness_forKeyboard_.return_value = True
    client.isBacklightSuppressedOnKeyboard_.return_value = None
    client.isBacklightDimmedOnKeyboard_.return_value = None
    objc = SimpleNamespace(lookUpClass=Mock(), registerMetaDataForSelector=Mock(), _C_NSBOOL=b"Z")
    objc.lookUpClass.return_value.alloc.return_value.init.return_value = client
    bundle = Mock()
    bundle.bundleWithPath_.return_value.load.return_value = True
    monkeypatch.setitem(sys.modules, "objc", objc)
    monkeypatch.setitem(sys.modules, "Foundation", SimpleNamespace(NSBundle=bundle))
    monkeypatch.setattr(module.subprocess, "run", Mock(side_effect=AssertionError("no subprocess")))
    monkeypatch.setattr(module, "_osascript", Mock(side_effect=AssertionError("no osascript")))
    return client, objc, bundle


def backend(client=None):
    return module._KeyboardBacklightBrightness(client=client)


def test_lazy_load_and_abi(native):
    client, objc, bundle = native
    helper = backend()
    bundle.bundleWithPath_.assert_not_called()
    assert helper.read() == 46
    assert helper.set(30) is True
    bundle.bundleWithPath_.assert_called_once_with(
        "/System/Library/PrivateFrameworks/CoreBrightness.framework")
    objc.lookUpClass.assert_called_once_with("KeyboardBrightnessClient")
    assert objc.registerMetaDataForSelector.call_args_list == [
        call(b"KeyboardBrightnessClient", b"brightnessForKeyboard:",
             {"retval": {"type": b"f"}, "arguments": {2: {"type": b"Q"}}}),
        call(b"KeyboardBrightnessClient", b"setBrightness:forKeyboard:",
             {"retval": {"type": b"Z"}, "arguments": {2: {"type": b"f"}, 3: {"type": b"Q"}}}),
    ]
    client.brightnessForKeyboard_.assert_called_once_with(77)
    client.setBrightness_forKeyboard_.assert_called_once_with(0.3, 77)
    client.isAutoBrightnessEnabledForKeyboard_.assert_not_called()
    assert {c[0] for c in client.method_calls} == {
        "copyKeyboardBacklightIDs", "brightnessForKeyboard_", "setBrightness_forKeyboard_"}


@pytest.mark.parametrize("value,expected", [(0, 0), (1, 100), (0.456, 46),
    (float("nan"), None), (float("inf"), None), (-0.1, None), (1.1, None), (None, None), ("bad", None)])
def test_fake_read(native, value, expected):
    client, _, bundle = native
    client.brightnessForKeyboard_.return_value = value
    assert backend(client).read() == expected
    bundle.bundleWithPath_.assert_not_called()


@pytest.mark.parametrize("level,expected", [(-20, 0), (30, 0.3), (500, 1)])
@pytest.mark.parametrize("ok", [True, False])
def test_fake_set(native, level, expected, ok):
    client = native[0]
    client.setBrightness_forKeyboard_.return_value = ok
    assert backend(client).set(level) is ok
    client.setBrightness_forKeyboard_.assert_called_once_with(expected, 77)


@pytest.mark.parametrize("failure", ["framework", "bundle", "nil_bundle", "lookup", "metadata", "ids", "no_ids", "none_ids", "invalid_id", "read", "set"])
def test_native_failures_never_raise(native, failure):
    client, objc, bundle = native
    if failure == "framework":
        bundle.bundleWithPath_.return_value.load.return_value = False
    elif failure == "bundle":
        bundle.bundleWithPath_.side_effect = RuntimeError("missing")
    elif failure == "nil_bundle":
        bundle.bundleWithPath_.return_value = None
    elif failure == "lookup":
        objc.lookUpClass.side_effect = RuntimeError("missing class")
    elif failure == "metadata":
        objc.registerMetaDataForSelector.side_effect = RuntimeError("bad ABI")
    elif failure == "ids":
        client.copyKeyboardBacklightIDs.side_effect = RuntimeError("ids")
    elif failure in ("no_ids", "none_ids"):
        client.copyKeyboardBacklightIDs.return_value = [] if failure == "no_ids" else None
    elif failure == "invalid_id":
        client.copyKeyboardBacklightIDs.return_value = ["bad"]
    elif failure == "read":
        client.brightnessForKeyboard_.side_effect = RuntimeError("read")
    else:
        client.setBrightness_forKeyboard_.side_effect = RuntimeError("set")
    helper = backend()
    if failure != "set":
        assert helper.read() is None
    if failure != "read":
        assert helper.set(30) is False


def test_missing_pyobjc_is_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "objc", None)
    assert backend().read() is None
    assert backend().set(30) is False


def test_unsigned_keyboard_id_is_preserved(native):
    client = native[0]
    client.copyKeyboardBacklightIDs.return_value = [2**63 + 7]
    helper = backend(client)
    assert helper.read() == 46
    assert helper.set(30) is True
    client.brightnessForKeyboard_.assert_called_once_with(2**63 + 7)
    client.setBrightness_forKeyboard_.assert_called_once_with(0.3, 2**63 + 7)


@pytest.mark.parametrize("method,args", [("keyboard_brightness_set", (30,)),
    ("keyboard_brightness_up", ()), ("keyboard_brightness_down", ())])
@pytest.mark.parametrize("failure", ["no_ids", "read", "set"])
def test_controller_with_native_failures(native, method, args, failure):
    client = native[0]
    if failure == "no_ids":
        client.copyKeyboardBacklightIDs.return_value = []
    elif failure == "read":
        client.brightnessForKeyboard_.side_effect = RuntimeError("read")
    else:
        client.setBrightness_forKeyboard_.side_effect = RuntimeError("write")
    controller = module.MacOSController()
    controller._keyboard_brightness = backend(client)
    assert getattr(controller, method)(*args) == "Não consegui mudar a luz do teclado — nada mudou."
    if failure != "set":
        client.setBrightness_forKeyboard_.assert_not_called()


def test_controller_reads_back_from_fake_native_client(native):
    client = native[0]
    client.brightnessForKeyboard_.side_effect = [0.7, 0.4]
    controller = module.MacOSController()
    controller._keyboard_brightness = backend(client)
    assert controller.keyboard_brightness_set(30) == (
        "Não consegui deixar em 30% — a luz do teclado está em 40%.")
    assert client.method_calls == [call.copyKeyboardBacklightIDs(), call.brightnessForKeyboard_(77),
                                 call.setBrightness_forKeyboard_(0.3, 77), call.brightnessForKeyboard_(77),
                                 call.isBacklightDimmedOnKeyboard_(77),
                                 call.isBacklightSuppressedOnKeyboard_(77)]


@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("method,args,before,after,target", [
    ("keyboard_brightness_set", (30,), 70, 30, 30),
    ("keyboard_brightness_set", (500,), 70, 100, 100),
    ("keyboard_brightness_set", (-20,), 70, 0, 0),
    ("keyboard_brightness_up", (), 40, 50, 50),
    ("keyboard_brightness_down", (), 40, 30, 30),
    ("keyboard_brightness_up", (), 95, 100, 100),
    ("keyboard_brightness_down", (), 5, 0, 0),
])
def test_verified_controller_reply(lang, method, args, before, after, target):
    controller = module.MacOSController()
    fake = Mock(read=Mock(side_effect=[before, after]), set=Mock(return_value=True))
    controller._keyboard_brightness = fake
    assert getattr(controller, method)(*args, lang=lang) == (
        f"Luz do teclado em {after}%." if lang == "pt" else f"Keyboard light at {after}%.")
    assert fake.method_calls == [call.read(), call.set(target), call.read()]


@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("method,args,current", [
    ("keyboard_brightness_down", (), 0), ("keyboard_brightness_up", (), 100),
    ("keyboard_brightness_set", (0,), 0), ("keyboard_brightness_set", (100,), 100),
])
def test_already_at_limit(lang, method, args, current):
    controller = module.MacOSController()
    controller._keyboard_brightness = Mock(read=Mock(return_value=current))
    expected = ("A luz do teclado já está em 0%." if current == 0 else
                "A luz do teclado já está no máximo (100%).") if lang == "pt" else (
                "Keyboard light is already at 0%." if current == 0 else
                "Keyboard light is already at the maximum (100%).")
    assert getattr(controller, method)(*args, lang=lang) == expected
    controller._keyboard_brightness.set.assert_not_called()


@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("before,after,ok", [
    (70, 28, True), (70, 32, True), (70, 40, True), (70, 30, False),
    (70, 70, False), (70, 70, True), (70, None, True), (None, None, True),
])
def test_honest_readback(lang, before, after, ok):
    controller = module.MacOSController()
    fake = Mock(read=Mock(side_effect=[before, after]), set=Mock(return_value=ok))
    controller._keyboard_brightness = fake
    reply = controller.keyboard_brightness_set(30, lang=lang)
    if after is not None and (ok or after != before):
        assert reply == (f"Não consegui deixar em 30% — a luz do teclado está em {after}%." if lang == "pt"
                         else f"I couldn't set it to 30% — keyboard light is at {after}%.")
    else:
        assert reply == ("Não consegui mudar a luz do teclado — nada mudou." if lang == "pt"
                         else "I couldn't change the keyboard light — nothing changed.")
    if before is None:
        fake.set.assert_not_called()
    else:
        expected_calls = [call.read(), call.set(30), call.read()]
        if after != 30:
            expected_calls.append(call.dimmed())
            expected_calls.append(call.suppressed())
        assert fake.method_calls == expected_calls


DOWN = "abaixa abaixe abaixar baixa baixe baixar diminui diminua diminuir reduz reduza reduzir desce desça desca descer escurece escureça escureca".split()
UP = "aumenta aumente aumentar sobe suba subir clareia clareie".split()
NOUNS = ("brilho do teclado", "luz do teclado", "brilho do teclado do mac", "luz do teclado do mac")
CASES = [
    (f"{verb} {article}{noun}{tail}", method, None)
    for verbs, method in ((DOWN, "keyboard_brightness_down"), (UP, "keyboard_brightness_up"))
    for verb, noun, tail in product(verbs, NOUNS, ("", " um pouco"))
    for article in ("", "o " if noun.startswith("brilho") else "a ")
] + [
    (f"{verb} {article}{noun} para 30%", "keyboard_brightness_set", 30)
    for verb, noun in product(DOWN + UP, NOUNS)
    for article in ("", "o " if noun.startswith("brilho") else "a ")
] + [
    (f"{verb} {article}keyboard {noun}", method, None)
    for verbs, method in ((("lower", "decrease", "dim", "turn down"), "keyboard_brightness_down"),
                          (("raise", "increase", "turn up"), "keyboard_brightness_up"))
    for verb, article, noun in product(verbs, ("", "the "), ("light", "brightness"))
] + [
    (f"{verb}{noun} {prep}{level}{percent}", "keyboard_brightness_set", level)
    for noun, (level, percent), prep in product((*NOUNS, "keyboard brightness", "keyboard light"),
        ((0, ""), (30, "%"), (100, "")), ("", "para ", "em ", "no ", "to ", "at "))
    for verb in ("", "põe ", "poe ", "coloca ", "deixa ", "ajusta ", "set ")
] + [
    (f"{noun} no {word}", "keyboard_brightness_set", level)
    for noun, (word, level) in product(NOUNS, (("mínimo", 0), ("minimo", 0), ("máximo", 100), ("maximo", 100)))
] + [
    (f"{verb} a luz do teclado", "keyboard_brightness_set", level)
    for verbs, level in (("apaga apague apagar desliga desligue desligar".split(), 0),
                         ("acende acenda acender liga ligue ligar".split(), 50)) for verb in verbs
] + [(text, method, target) for text, method, target in (
    ("Brilho do teclado do mac 0", "keyboard_brightness_set", 0),
    ("keyboard brightness up", "keyboard_brightness_up", None),
    ("keyboard brightness down", "keyboard_brightness_down", None),
    ("keyboard light off", "keyboard_brightness_set", 0),
    ("keyboard light on", "keyboard_brightness_set", 50),
    ("turn off the keyboard light", "keyboard_brightness_set", 0),
    ("turn on the keyboard light", "keyboard_brightness_set", 50),
    ("lower the keyboard brightness", "keyboard_brightness_down", None),
    ("raise the keyboard light", "keyboard_brightness_up", None),
)]


@pytest.mark.parametrize("lang", ["pt", "en"])
def test_all_keyboard_routes_and_screen_separation(lang):
    for text, method, level in CASES:
        for suffix in ("", " por favor", " please"):
            orch = SimpleNamespace(_reply_lang=lang, macos=MagicMock(), spotify=MagicMock())
            getattr(orch.macos, method).return_value = "verified reply"
            clean = strip_courtesy(text + suffix)
            assert run(orch, clean) == ("verified reply", "MAC_KEYBOARD_BRIGHTNESS"), text
            args = () if level is None else (level,)
            assert orch.macos.method_calls == [getattr(call, method)(*args, lang=lang)], text
            assert not orch.spotify.mock_calls, text
            assert not any(rx.match(clean) for rx in (macos.MAC_BRIGHTNESS_SET_RE,
                macos.MAC_BRIGHTNESS_UP_RE, macos.MAC_BRIGHTNESS_DOWN_RE)), text
            assert simulate(clean).mode == "MAC_KEYBOARD_BRIGHTNESS", text


@pytest.mark.parametrize("text", ["brilho do teclado -1", "keyboard brightness 30.5",
    "luz do teclado 30 amanhã", "turn off the keyboard light and play music",
    "por que abaixa o brilho do teclado", "qual a luz do teclado ideal?"])
def test_non_commands_never_touch_screen(text):
    orch = SimpleNamespace(_reply_lang="pt", macos=MagicMock(), spotify=MagicMock())
    run(orch, text)
    assert not orch.macos.mock_calls


@pytest.mark.parametrize("text", ["Muda a luz do teclado um pouquinho", "Could you change the keyboard light?"])
def test_unmatched_light_request_reaches_tool_guard(text):
    assert TOOL_HINT_RE.search(text)
    assert not REAL_TOOL_HINT_RE.search(text)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("text,method,args,lang", [
    ("Brilho do teclado do mac 0", "keyboard_brightness_set", (0,), "pt"),
    ("luz do teclado 30", "keyboard_brightness_set", (30,), "pt"),
    ("abaixa o brilho do teclado", "keyboard_brightness_down", (), "pt"),
    ("acende a luz do teclado", "keyboard_brightness_set", (50,), "pt"),
    ("apaga a luz do teclado", "keyboard_brightness_set", (0,), "pt"),
    ("keyboard brightness up", "keyboard_brightness_up", (), "en"),
    ("turn off the keyboard light", "keyboard_brightness_set", (0,), "en"),
])
def test_entry_points_and_voice_veto(stream, text, method, args, lang, monkeypatch):
    from src.core.orchestrator import CortexOrchestrator
    orch = CortexOrchestrator.__new__(CortexOrchestrator)
    orch.engine_analytical = MagicMock()
    orch.engine_creative = MagicMock()
    orch.macos = MagicMock()
    orch.spotify = MagicMock()
    getattr(orch.macos, method).return_value = "verified reply"
    orch._pending_action = orch._pending_query = None
    orch._shadow_classify = MagicMock()
    orch._maybe_announce_pending = lambda reply, mode: reply
    monkeypatch.setattr(orch, "_handle_user_paths", Mock(side_effect=AssertionError("registry missed")))
    assert _detect_lang(text) == lang
    assert orch._is_pure_chat(text) is False
    assert any(getattr(orch, name).match(text) for name in orch._PURE_CHAT_BLOCKING_RES)
    if stream:
        assert list(orch.process_stream(text)) == [("verified reply", True, "MAC_KEYBOARD_BRIGHTNESS", lang)]
    else:
        assert orch.process(text) == ("verified reply", "MAC_KEYBOARD_BRIGHTNESS")
    assert orch.macos.method_calls == [getattr(call, method)(*args, lang=lang)]
    for engine in (orch.engine_analytical, orch.engine_creative):
        for operation in ("chat", "chat_stream", "chat_with_tools", "generate"):
            getattr(engine, operation).assert_not_called()
    assert not orch.spotify.mock_calls


@pytest.mark.parametrize("lang,claim", [
    ("pt", "Diminuí a luz do teclado para 0%."), ("pt", "Apaguei a luz do teclado."),
    ("pt", "Acendi a luz do teclado."), ("pt", "Desliguei a luz do teclado."),
    ("pt", "Luz do teclado ajustada para 50%."),
    ("en", "I lowered the keyboard brightness."), ("en", "I turned off the keyboard light."),
    ("en", "Keyboard light adjusted to 50%."),
])
def test_no_tool_keyboard_claim_is_blocked(rig, lang, claim):
    engine, model, executor, audit = rig
    model.return_value = {"message": {"content": claim, "tool_calls": []}}
    text = "Muda a luz do teclado" if lang == "pt" else "Change the keyboard light"
    assert detect_hallucinated_mac_action(claim)
    assert engine.chat_with_tools(text, [], executor, lang) == (MAC_FALLBACK[lang], None)
    executor.assert_not_called()
    assert audit.call_args.kwargs["payload"]["kind"] == "mac"
    engine.memory.store_interaction.assert_called_once_with(text, MAC_FALLBACK[lang])


@pytest.mark.parametrize("text", ["Apaguei a luz da sala.", "I turned off the kitchen light.",
    "A luz do teclado depende do ambiente.", "Use the keyboard brightness keys."])
def test_guard_keeps_unrelated_light_prose(text):
    assert not detect_hallucinated_mac_action(text)
