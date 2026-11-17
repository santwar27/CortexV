"""146: invented failures never become attempted actions or silent retries."""
import json
from datetime import datetime, timedelta, timezone
from itertools import islice
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

from src.core.engine import OllamaEngine
from src.core.dispatch import registry
from tests.test_taught_facts import orch  # noqa: F401


TRUTH = {
    "pt": "Não entendi esse comando, então nada foi feito.",
    "en": "I didn't understand that command, so nothing was done.",
}
PLAY = "Não consegui acionar o Spotify pra tocar música. Tenta dizer 'toca matuê no spotify'."
PAUSE = "Não consegui pausar a música. Tenta dizer 'pausa' no Spotify."
MECHANISM_PAUSE = ('Não consegui acionar a ferramenta de controle de dispositivo para pausar a música. '
                   'Tenta dizer "pausa no Spotify" …')
MECHANISM_PAUSE_EN = ('I couldn\'t trigger the tool for device control to pause the music. '
                      'Try saying "pause on Spotify" …')
V9_PLAY = TRUTH["pt"] + " Tenta dizer 'toca veigh no spotify'."
V9_PLAY_EN = TRUTH["en"] + " Try saying 'play veigh on spotify'."


def expected(lang, suggestion=None):
    if suggestion is None:
        return TRUTH[lang]
    return TRUTH[lang] + (
        f" Quer que eu faça: '{suggestion}'? Diga sim." if lang == "pt" else
        f" Do you want me to do: '{suggestion}'? Say yes.")


def test_invented_failure_input_stays_unrouted():
    from src.core.dispatch.spotify import SPOTIFY_PLAY_RE
    from src.core.hallucination import routed_command_match

    text = "manda um veigh aí"
    reason = "Invented-failure fixtures must reach the faked LLM, not a fixed route."
    assert SPOTIFY_PLAY_RE.match(text) is None, reason
    assert routed_command_match(text) is None, reason


@pytest.fixture
def live(orch, monkeypatch):
    memory = orch.engine_analytical.memory
    monkeypatch.setattr(memory, "retrieve_relevant_context", Mock(return_value=""))
    monkeypatch.setattr(memory, "store_interaction", Mock())
    # Preserve the class's refusal classifier used after chat/chat_stream.
    from src.core.memory import CortexMemory
    monkeypatch.setattr("src.core.engine.CortexMemory", Mock(
        return_value=memory, _is_refusal=CortexMemory._is_refusal))
    engine = OllamaEngine()
    orch.engine_analytical = orch.engine_creative = engine
    model = Mock()

    def response(*args, **kwargs):
        text = model.reply
        if kwargs.get("stream"):
            return iter({"message": {"content": char}} for char in text)
        return {"message": {"content": text}}

    model.reply = PLAY
    model.side_effect = response
    monkeypatch.setattr("src.core.engine.ollama.chat", model)
    orch.spotify.play_anything.return_value = "Playing."
    orch.spotify.pause.return_value = "Paused."
    return orch, engine, model


def turn(o, text="solta um matuê aí", lang="pt", stream=False,
         channel="ui", conversation="A"):
    o.set_turn_origin(channel, conversation)
    try:
        if stream:
            frames = list(o.process_stream(text, language=lang))
            return "".join(f[0] for f in frames), frames[-1][2]
        return o.process(text, language=lang)
    finally:
        o.set_turn_origin(None)


CASES = [
    ("pt", PLAY, "toca matuê no spotify", "SPOTIFY_PLAY"),
    ("pt", PAUSE, "pausa", "SPOTIFY_PAUSE"),
    ("pt", "Não consegui ajustar a iluminação da tela. Tente usar o controle de brilho do seu Mac.", None, None),
    ("pt", "Não consegui acender o teclado. Tenta usar o controle de iluminação do teclado.", None, None),
    ("en", "I couldn't activate Spotify to play music. Try saying 'play matue on spotify'.", "play matue on spotify", "SPOTIFY_PLAY"),
    ("en", "I couldn't pause the music. Try saying 'pause' on Spotify.", "pause", "SPOTIFY_PAUSE"),
    ("en", "I couldn't adjust the screen lighting. Try using your Mac's brightness control.", None, None),
    ("en", "I couldn't turn on the keyboard. Try using the keyboard lighting control.", None, None),
    ("pt", "Não consegui ajustar o som. Tenta dizer 'manda email pro João'.", None, None),
    ("pt", "Não consegui abrir o app. Tenta dizer 'abre o Safari'.", "abre o Safari", "MAC_OPEN_APP"),
    ("en", 'I could not mute the volume. Try saying “Cortex, mute please.”', "Cortex, mute please.", "MAC_MUTE"),
    ("pt", "Não consegui acionar a função de ajustar a tela.", None, None),
    ("pt", "Não consegui acionar a função de aumentar a iluminância da tela.", None, None),
    ("pt", "Não consegui acionar a função de silenciar o computador.", None, None),
    ("pt", "Não consegui acionar a função de iluminar o teclado.", None, None),
    ("pt", "Não consegui acionar a função de tocar um som do Veigh.", None, None),
    ("pt", "Não consegui acionar a função de abaixar a tela.", None, None),
    ("pt", "Não consegui acionar a função de pausar a música.", None, None),
    ("en", "I couldn't trigger the function to adjust the screen.", None, None),
    ("en", "I couldn't use the function to pause the music.", None, None),
    ("pt", "Não consegui acionar o comando de ajuste de brilho. Tenta dizer 'ajusta brilho' ou 'muda brilho'.", None, None),
    ("pt", "Não consegui acionar o comando de ajuste de brilho da tela. Tenta dizer 'diminui brilho da tela' ou 'abaixa a luz da tela'.", "diminui brilho da tela", "MAC_BRIGHTNESS"),
    ("pt", "Não consegui acionar o comando de ajuste de iluminação do teclado. Tenta dizer 'ilumina teclado' …", None, None),
    ("pt", "Não consegui acionar a ferramenta de controle de tela.", None, None),
    ("pt", "Não consegui acionar o controle de iluminação do teclado. Tenta dizer 'ilumina teclado' no terminal.", None, None),
    ("pt", "Não consegui acionar a função de ajuste do brilho.", None, None),
    ("pt", "Não consegui acionar a ferramenta de controle de dispositivo para silenciar o computador. Tenta fechar a aplicação ou pressionar o botão de energia.", None, None),
    ("pt", "Não consegui acionar a ferramenta de controle de dispositivo para iluminar o teclado. Tenta pressionar …", None, None),
    ("pt", "Não consegui acionar a ferramenta de controle de dispositivo para abaixar a tela. Tenta pressionar o botão de ajuste de brilho …", None, None),
    ("pt", MECHANISM_PAUSE, "pausa no Spotify", "SPOTIFY_PAUSE"),
    ("en", "I couldn't trigger the tool for device control to mute the computer. Try closing the application or pressing the power button.", None, None),
    ("en", "I couldn't trigger the tool for device control to light up the keyboard. Try pressing …", None, None),
    ("en", "I couldn't trigger the tool for device control to lower the screen. Try pressing the brightness adjustment button …", None, None),
    ("en", MECHANISM_PAUSE_EN, "pause on Spotify", "SPOTIFY_PAUSE"),
]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang,reply,suggestion,mode", CASES)
def test_replacement_before_emission_and_storage(live, stream, lang, reply, suggestion, mode):
    o, engine, model = live
    model.reply = reply
    user_input = "quero o safari aberto" if mode == "MAC_OPEN_APP" else "solta um matuê aí"
    actual, _ = turn(o, user_input, lang=lang, stream=stream)
    assert actual == expected(lang, suggestion)
    assert engine.history[-1]["content"] == actual
    engine.memory.store_interaction.assert_called_once_with(user_input, actual)
    if suggestion:
        action = json.loads(Path(o._PENDING_ACTION_FILE).read_text())
        assert action["type"] == "routed_command"
        assert action["args"] == {"text": suggestion, "mode": mode, "lang": lang}
        assert action["origin"] == {"channel": "ui", "conversation": "A"}
    else:
        assert o._pending_action is None
    o.spotify.play_anything.assert_not_called()
    o.spotify.pause.assert_not_called()
    assert o.macos.mock_calls == []


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang,reply", [
    ("pt", "Eu não consegui acionar a ferramenta de controle de dispositivo."),
    ("pt", "Não consegui ativar a função de controle de dispositivo."),
    ("pt", "Não consegui usar a opção de controle de dispositivo."),
    ("pt", "Não consegui acionar o comando solicitado para este dispositivo."),
    ("pt", "Não consegui ativar o controle solicitado para este dispositivo."),
    ("pt", "Não consegui usar o recurso solicitado para este dispositivo."),
    ("en", "I couldn't trigger the tool requested for this device."),
    ("en", "I couldn't activate the function requested for this device."),
    ("en", "I couldn't use the command requested for this device."),
    ("en", "I could not use the control requested for this device."),
    ("pt", "Não consegui usar a ferramenta; mesmo com outra tentativa, não obtive acesso ao computador."),
    ("en", "I couldn’t use the tool; even after another attempt, I had no access to the computer."),
])
def test_mechanism_without_control_target_passes(live, stream, lang, reply):
    o, engine, model = live
    model.reply = reply
    assert turn(o, lang=lang, stream=stream)[0] == reply
    assert engine.history[-1]["content"] == reply
    assert o._pending_action is None
    o.spotify.play_anything.assert_not_called()
    o.spotify.pause.assert_not_called()
    assert o.macos.mock_calls == []


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("separator", [".", "!", "?", "\n"])
@pytest.mark.parametrize("lang,opening,following", [
    ("pt", "Não consegui usar a ferramenta de controle de sono", "Tenta usar o computador."),
    ("en", "I couldn't use the tool for sleep control", "Try using the computer."),
    ("pt", "Não consegui usar a ferramenta de controle de sono", "Tenta usar algo para silenciar o computador."),
    ("en", "I couldn't use the tool for sleep control", "Try using something to mute the computer."),
])
def test_mechanism_device_cannot_cross_first_sentence(live, stream, separator,
                                                    lang, opening, following):
    o, engine, model = live
    model.reply = f"{opening}{separator} {following}"
    assert turn(o, lang=lang, stream=stream)[0] == model.reply
    assert o._pending_action is None


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang,user_input,reply", [
    ("pt", "não consegui dormir ontem, alguma dica?", "Não consegui acionar a ferramenta de controle de sono para te ajudar com isso. Tenta relaxar com uma técnica de respiração profunda …"),
    ("en", "I couldn't sleep last night, any tips?", "I couldn't trigger the tool for sleep control to help you with that. Try relaxing with a deep breathing technique …"),
])
def test_sleep_control_failure_without_device_passes(live, stream, lang, user_input, reply):
    o, engine, model = live
    model.reply = reply
    assert turn(o, user_input, lang=lang, stream=stream)[0] == reply
    assert engine.history[-1]["content"] == reply
    engine.memory.store_interaction.assert_called_once_with(user_input, reply)
    assert o._pending_action is None


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang,reply", [
    ("pt", "Não consegui acionar a ferramenta de controle de dispositivo para te ajudar a dormir. Tenta relaxar."),
    ("pt", "Não consegui acionar a função de controle de sono que desliga a tela. Tenta x."),
    ("pt", "Não consegui acionar o comando para te ajudar a dormir, evite o computador à noite."),
    ("en", "I couldn't trigger the tool for device control to help you sleep. Try relaxing."),
    ("en", "I couldn't trigger the function for sleep control that turns off the screen. Try x."),
    ("en", "I couldn't trigger the command to help you sleep, avoid the computer at night."),
])
def test_sleep_failure_with_incidental_device_passes(live, stream, lang, reply):
    from src.core.hallucination import guard_invented_command_failure

    o, engine, model = live
    user_input = ("não consegui dormir ontem, alguma dica?" if lang == "pt" else
                  "I couldn't sleep last night, any tips?")
    assert guard_invented_command_failure(reply, user_input, lang) == reply
    model.reply = reply
    assert turn(o, user_input, lang=lang, stream=stream)[0] == reply
    assert engine.history[-1]["content"] == reply
    engine.memory.store_interaction.assert_called_once_with(user_input, reply)
    assert o._pending_action is None
    o.spotify.play_anything.assert_not_called()
    o.spotify.pause.assert_not_called()
    assert o.macos.mock_calls == []


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang,connector,action", [
    (lang, connector, action) for lang, action in [
        ("pt", "silenciar o computador"),
        ("pt", "mutar o computador"),
        ("pt", "iluminar o teclado"),
        ("pt", "abaixar a tela"),
        ("pt", "clarear a tela"),
        ("pt", "escurecer a tela"),
        ("en", "mute the computer"),
        ("en", "light up the keyboard"),
        ("en", "illuminate the keyboard"),
        ("en", "lower the screen"),
        ("en", "brighten the screen"),
        ("en", "dim the screen"),
        ("en", "darken the screen"),
    ] for connector in (("para", "pra") if lang == "pt" else ("to",))
])
def test_mechanism_purpose_control_target(live, stream, connector, lang, action):
    o, engine, model = live
    opening = ("Não consegui usar a ferramenta de controle de dispositivo" if lang == "pt" else
               "I couldn't use the tool for device control")
    model.reply = f"{opening} {connector} {action}."
    assert turn(o, lang=lang, stream=stream)[0] == TRUTH[lang]
    assert engine.history[-1]["content"] == TRUTH[lang]
    assert o._pending_action is None
    assert o.macos.mock_calls == []


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang,reply", [
    ("pt", "Não consegui usar a ferramenta para silenciar conselhos sobre o computador."),
    ("pt", "Não consegui usar a ferramenta para silenciar o dispositivo."),
    ("pt", "Não consegui usar a ferramenta para te ajudar a silenciar o computador."),
    ("pt", "Não consegui usar a ferramenta de sono que pode silenciar o computador."),
    ("pt", "Não consegui usar a ferramenta para silenciar o computadore."),
    ("en", "I couldn't use the tool to mute advice about the computer."),
    ("en", "I couldn't use the tool to mute the device."),
    ("en", "I couldn't use the tool to help you mute the computer."),
    ("en", "I couldn't use the sleep tool that can mute the computer."),
    ("en", "I couldn't use the tool to mute the computerized toy."),
])
def test_mechanism_purpose_requires_adjacent_control_verb_and_object(live, stream, lang, reply):
    o, engine, model = live
    model.reply = reply
    assert turn(o, lang=lang, stream=stream)[0] == reply
    assert o._pending_action is None


@pytest.mark.parametrize("lang,reply", [
    ("pt", "Não consegui ativar o comando para clarear a tela."),
    ("pt", "Não consegui usar o controle pra escurecer a tela."),
    ("pt", "Não consegui usar a opção de iluminar o teclado."),
    ("pt", "Não consegui acionar o recurso do volume."),
    ("pt", "Não consegui ativar a ferramenta da iluminancia."),
    ("pt", "Não consegui acionar a função de tela."),
    ("pt", "Não consegui acionar abaixar a tela."),
    ("pt", "Não consegui iluminar o teclado."),
    ("en", "I couldn't activate the command to increase the brightness."),
    ("en", "I couldn't use the control for the volume."),
    ("en", "I couldn't trigger the option to mute the computer."),
    ("en", "I couldn't activate the feature to play music."),
    ("en", "I couldn't use the tool to turn on the keyboard."),
    ("en", "I couldn't trigger the function for the screen."),
    ("en", "I couldn't use the volume."),
    ("en", "I couldn't trigger play music."),
])
def test_mechanism_control_failures(live, lang, reply):
    o, engine, model = live
    model.reply = reply
    assert turn(o, lang=lang)[0] == TRUTH[lang]
    assert o._pending_action is None
    o.spotify.play_anything.assert_not_called()
    o.spotify.pause.assert_not_called()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang,phrase", [
    ("pt", "acionar o comando de ajuste de brilho"),
    ("pt", "usar a função de controle do volume"),
    ("pt", "ativar o recurso para alteração da tela"),
    ("pt", "acionar a ferramenta pra alteracao de brilho"),
    ("pt", "usar a opção de mudança do som"),
    ("pt", "acionar o controle de mudanca da iluminação do teclado"),
    ("pt", "acionar o comando de regulagem de um volume"),
    ("pt", "acionar ajuste de brilho"),
    ("pt", "ajustar controle do volume"),
    ("en", "trigger the command for adjustment of the brightness"),
    ("en", "use the tool for control of your volume"),
    ("en", "activate the function for change of the screen"),
    ("en", "trigger adjustment for brightness"),
    ("en", "use control for volume"),
    ("en", "adjust change for the screen"),
])
def test_action_noun_control_failures(live, stream, lang, phrase):
    o, engine, model = live
    opening = "Não consegui" if lang == "pt" else "I couldn't"
    model.reply = f"{opening} {phrase}."
    assert turn(o, lang=lang, stream=stream)[0] == TRUTH[lang]
    assert o._pending_action is None


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("suggestions", [
    "'ajusta brilho' ou 'diminui brilho da tela' ou 'pausa'",
    "\"trava a tela\" ou “diminui brilho da tela” ou ‘pausa’",
    "«algo que não existe» ou ‘diminui brilho da tela’ ou \"pausa\"",
])
def test_first_allowed_quote_is_offered(live, monkeypatch, stream, suggestions):
    o, engine, model = live
    log = Mock()
    monkeypatch.setattr("src.core.engine.audit.log", log)
    model.reply = f"Não consegui ajustar o brilho. Tenta dizer {suggestions}."
    assert turn(o, stream=stream)[0] == expected("pt", "diminui brilho da tela")
    assert o._pending_action["args"] == {
        "text": "diminui brilho da tela", "mode": "MAC_BRIGHTNESS", "lang": "pt",
    }
    swaps = [c for c in log.call_args_list
             if len(c.args) > 2 and c.args[2] == "invented_command_failure"]
    assert len(swaps) == 1
    assert swaps[0].kwargs["payload"]["suggestion"] == "diminui brilho da tela"
    assert swaps[0].kwargs["payload"]["mode"] == "MAC_BRIGHTNESS"
    assert o.macos.mock_calls == []


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang,user_input,suggestion,allowed", [
    ("pt", "silencia o computador", "toca matue no spotify", False),
    ("pt", "manda um veigh aí", "toca matue no spotify", False),
    ("pt", "quero o teclado iluminado", "abra a luz no teclado", False),
    ("pt", "solta um matuê aí", "toca matue no spotify", True),
    ("en", "mute the computer", "play matue on spotify", False),
    ("en", "give me some veigh", "play matue on spotify", False),
    ("en", "I want the keyboard lit", "open the keyboard light", False),
    ("en", "give me some matuê", "play matue on spotify", True),
    ("pt", "manda um veigh aí", "toca veigh no spotify", True),
    ("pt", "quero o safari aberto", "abre o safari", True),
    ("en", "I want safari open", "open safari", True),
])
def test_free_argument_must_come_from_input(live, monkeypatch, stream, lang,
                                          user_input, suggestion, allowed):
    o, engine, model = live
    log = Mock()
    monkeypatch.setattr("src.core.engine.audit.log", log)
    model.reply = (f"Não consegui acionar o Spotify. Tenta dizer '{suggestion}'."
                   if lang == "pt" else
                   f"I couldn't activate Spotify. Try saying '{suggestion}'.")
    actual, _ = turn(o, user_input, lang=lang, stream=stream)
    assert actual == expected(lang, suggestion if allowed else None)
    assert engine.history[-1]["content"] == actual
    engine.memory.store_interaction.assert_called_once_with(user_input, actual)
    o.spotify.play_anything.assert_not_called()
    assert o.macos.mock_calls == []
    if allowed:
        assert o._pending_action["args"]["text"] == suggestion
        if "matue" in suggestion:
            assert turn(o, "sim" if lang == "pt" else "yes", lang=lang,
                        stream=stream) == ("Playing.", "SPOTIFY_PLAY")
            o.spotify.play_anything.assert_called_once_with("matue", device_hint=None, lang=lang)
            turn(o, "sim" if lang == "pt" else "yes", lang=lang, stream=stream)
            assert o.spotify.play_anything.call_count == 1
    else:
        assert o._pending_action is None
        assert not Path(o._PENDING_ACTION_FILE).exists()
        swaps = [c for c in log.call_args_list
                 if len(c.args) > 2 and c.args[2] == "invented_command_failure"]
        assert len(swaps) == 1
        assert swaps[0].kwargs["payload"]["suggestion_rejections"] == [{
            "suggestion": suggestion,
            "mode": "MAC_OPEN_APP" if suggestion.startswith(("abra", "open")) else "SPOTIFY_PLAY",
            "reason": "suggestion_argument_not_in_input",
        }]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang,reply,suggestion", [
    ("pt", V9_PLAY, "toca veigh no spotify"),
    ("en", V9_PLAY_EN, "play veigh on spotify"),
    ("pt", "\nNÃO ENTENDI ESSE COMANDO; ENTÃO NADA FOI FEITO! Tenta dizer “toca veigh no spotify”.",
     "toca veigh no spotify"),
    ("en", "\nI DIDN’T UNDERSTAND THAT COMMAND — SO NOTHING WAS DONE! Try saying “play veigh on spotify”.",
     "play veigh on spotify"),
    ("pt", "Não entendi esse comando então nada foi feito\nTenta dizer 'toca veigh no spotify'.",
     "toca veigh no spotify"),
    ("en", "I didnt understand that command so nothing was done\nTry saying 'play veigh on spotify'.",
     "play veigh on spotify"),
])
@pytest.mark.parametrize("argument_in_input", [False, True])
def test_v9_offer_before_emission_and_storage(live, monkeypatch, stream, lang,
                                            reply, suggestion, argument_in_input):
    o, engine, model = live
    log = Mock()
    monkeypatch.setattr("src.core.engine.audit.log", log)
    user_input = (("manda um veigh aí" if argument_in_input else "silencia o computador")
                  if lang == "pt" else
                  ("give me some veigh" if argument_in_input else "mute the computer"))
    model.reply = reply
    actual, _ = turn(o, user_input, lang=lang, stream=stream)
    # The engine sanitiser already strips outer whitespace before the guard.
    assert actual == (expected(lang, suggestion) if argument_in_input else reply.strip())
    assert engine.history[-1]["content"] == actual
    engine.memory.store_interaction.assert_called_once_with(user_input, actual)
    swaps = [c for c in log.call_args_list
             if len(c.args) > 2 and c.args[2] == "invented_command_failure"]
    assert len(swaps) == 1
    payload = swaps[0].kwargs["payload"]
    assert payload["suggestion"] == suggestion
    assert payload["mode"] == "SPOTIFY_PLAY"
    if argument_in_input:
        assert o._pending_action["args"] == {
            "text": suggestion, "mode": "SPOTIFY_PLAY", "lang": lang,
        }
        assert json.loads(Path(o._PENDING_ACTION_FILE).read_text())["args"] == o._pending_action["args"]
    else:
        assert o._pending_action is None
        assert payload["suggestion_rejections"] == [{
            "suggestion": suggestion, "mode": "SPOTIFY_PLAY",
            "reason": "suggestion_argument_not_in_input",
        }]
    o.spotify.play_anything.assert_not_called()
    o.spotify.pause.assert_not_called()
    assert o.macos.mock_calls == []


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang,advice", [
    ("pt", "Tenta dizer 'trava a tela' ou 'algo que não existe'."),
    ("en", "Try saying 'lock the screen' or 'something unknown'."),
    ("pt", "Tenta usar os botões do computador."),
    ("en", "Try using your computer's buttons."),
])
def test_v9_keeps_advice_without_eligible_suggestion(live, stream, lang, advice):
    o, engine, model = live
    model.reply = TRUTH[lang] + " " + advice
    assert turn(o, lang=lang, stream=stream)[0] == model.reply
    assert engine.history[-1]["content"] == model.reply
    assert o._pending_action is None
    assert o.macos.mock_calls == []


@pytest.mark.parametrize("lang,reply", [("pt", V9_PLAY), ("en", V9_PLAY_EN)])
@pytest.mark.parametrize("stager_result", [None, False, True])
def test_v9_uses_existing_stager_contract(lang, reply, stager_result):
    from src.core.hallucination import command_failure_scope, guard_invented_command_failure

    stager = Mock(return_value=stager_result) if stager_result is not None else None
    suggestion = "toca veigh no spotify" if lang == "pt" else "play veigh on spotify"
    with command_failure_scope(stager):
        assert guard_invented_command_failure(reply, "some veigh", lang) == (
            expected(lang, suggestion) if stager_result else reply)
    if stager:
        stager.assert_called_once_with(suggestion, "SPOTIFY_PLAY", lang)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang,user_input,content", [
    ("pt", "me dá uma receita rápida de omelete",
     "Se você quer uma receita de omelete, posso sugerir uma opção rápida: bata dois ovos e frite."),
    ("en", "give me a quick omelette recipe",
     "If you want an omelette recipe, I can suggest a quick option: beat two eggs and fry them."),
])
def test_v9_omelette_reply_passes_unchanged(live, stream, lang, user_input, content):
    o, engine, model = live
    model.reply = TRUTH[lang] + " " + content
    assert turn(o, user_input, lang=lang, stream=stream)[0] == model.reply
    assert engine.history[-1]["content"] == model.reply
    engine.memory.store_interaction.assert_called_once_with(user_input, model.reply)
    assert o._pending_action is None
    o.spotify.play_anything.assert_not_called()
    assert o.macos.mock_calls == []


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("failure_opening", [False, True], ids=["honest-fallback", "invented-failure"])
@pytest.mark.parametrize("lang,user_input,suggestion,word", [
    ("pt", "manda um veigh aí", "toca Veigh", "sim"),
    ("en", "give me some veigh", "play Veigh", "yes"),
])
def test_imitated_offer_stages_and_approval_plays_once(live, stream, failure_opening,
                                                     lang, user_input, suggestion, word):
    o, engine, model = live
    opening = (("Não consegui acionar o Spotify." if lang == "pt" else
                "I couldn't activate Spotify.") if failure_opening else TRUTH[lang])
    model.reply = opening + expected(lang, suggestion)[len(TRUTH[lang]):]
    assert turn(o, user_input, lang=lang, stream=stream)[0] == expected(lang, suggestion)
    assert engine.history[-1]["content"] == expected(lang, suggestion)
    engine.memory.store_interaction.assert_called_once_with(user_input, expected(lang, suggestion))
    assert o._pending_action["args"] == {
        "text": suggestion, "mode": "SPOTIFY_PLAY", "lang": lang,
    }
    o.spotify.play_anything.assert_not_called()
    assert turn(o, word, lang=lang, stream=stream) == ("Playing.", "SPOTIFY_PLAY")
    o.spotify.play_anything.assert_called_once_with("Veigh", device_hint=None, lang=lang)
    assert o._pending_action is None
    assert model.call_count == 1
    model.reply = "OK."
    turn(o, word, lang=lang, stream=stream)
    assert o.spotify.play_anything.call_count == 1
    assert o.macos.mock_calls == []


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang,user_input,suggestion,rest", [
    ("pt", "silencia o computador", "toca Veigh", "Posso explicar os comandos disponíveis."),
    ("en", "mute the computer", "play Veigh", "I can explain the available commands."),
    ("pt", "silencia o computador", "trava a tela", "Posso explicar os comandos disponíveis."),
    ("en", "mute the computer", "lock the screen", "I can explain the available commands."),
])
@pytest.mark.parametrize("placement", ["before", "after"])
def test_v9_ineligible_fake_offer_removes_only_offer(live, stream, lang, user_input,
                                                    suggestion, rest, placement):
    o, engine, model = live
    offer = expected(lang, suggestion)[len(TRUTH[lang]) + 1:]
    model.reply = TRUTH[lang] + " " + (rest + " " + offer if placement == "before" else
                                      offer + " " + rest)
    actual = TRUTH[lang] + " " + rest
    assert turn(o, user_input, lang=lang, stream=stream)[0] == actual
    assert engine.history[-1]["content"] == actual
    engine.memory.store_interaction.assert_called_once_with(user_input, actual)
    assert o._pending_action is None
    assert not Path(o._PENDING_ACTION_FILE).exists()
    o.spotify.play_anything.assert_not_called()
    assert o.macos.mock_calls == []


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang,user_input,suggestion", [
    ("pt", "silencia o computador", "toca Veigh"),
    ("en", "mute the computer", "play Veigh"),
])
def test_v9_bare_ineligible_imitated_offer_is_removed(live, stream, lang, user_input, suggestion):
    o, engine, model = live
    model.reply = expected(lang, suggestion)
    assert turn(o, user_input, lang=lang, stream=stream)[0] == TRUTH[lang]
    assert engine.history[-1]["content"] == TRUTH[lang]
    assert o._pending_action is None
    o.spotify.play_anything.assert_not_called()
    assert o.macos.mock_calls == []


@pytest.mark.parametrize("lang,content", [
    ("pt", "Se você quer uma receita de omelete, posso sugerir uma opção rápida."),
    ("en", "If you want an omelette recipe, I can suggest a quick option."),
])
def test_v9_guard_preserves_reply_formatting_without_offer(lang, content):
    from src.core.hallucination import guard_invented_command_failure

    reply = "\n  " + TRUTH[lang] + "\n\n" + content + "  \n"
    assert guard_invented_command_failure(reply, "omelette", lang) == reply


@pytest.mark.parametrize("lang,offer", [
    ("pt", "Quer que eu faça: 'trava a tela. Agora?'? Diga sim."),
    ("en", "Do you want me to do: 'lock the screen. Now?'? Say yes."),
])
def test_v9_fake_offer_sentence_keeps_quoted_punctuation_together(lang, offer):
    from src.core.hallucination import guard_invented_command_failure

    rest = "Uma explicação útil." if lang == "pt" else "It's a useful explanation."
    reply = TRUTH[lang] + " " + offer + " " + rest
    assert guard_invented_command_failure(reply, "omelette", lang) == TRUTH[lang] + " " + rest


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang,offer", [
    ("pt", "Diga sim."), ("en", "Say yes."),
    ("pt", "Quer que eu faça isso?"), ("en", "Do you want me to do that?"),
    ("pt", "Para continuar, diga sim."), ("en", "To continue, say yes."),
])
def test_v9_fake_offer_markers_without_quote_keep_other_sentences(live, stream, lang, offer):
    o, engine, model = live
    rest = "Posso ajudar com uma receita." if lang == "pt" else "I can help with a recipe."
    model.reply = TRUTH[lang] + " " + offer + " " + rest
    actual = TRUTH[lang] + " " + rest
    assert turn(o, lang=lang, stream=stream)[0] == actual
    assert engine.history[-1]["content"] == actual
    assert o._pending_action is None


@pytest.mark.parametrize("lang,suggestion", [("pt", "toca Veigh"), ("en", "play Veigh")])
@pytest.mark.parametrize("stager_result", [None, False, True])
def test_v9_imitated_offer_requires_successful_staging(lang, suggestion, stager_result):
    from src.core.hallucination import command_failure_scope, guard_invented_command_failure

    reply = expected(lang, suggestion)
    stager = Mock(return_value=stager_result) if stager_result is not None else None
    with command_failure_scope(stager):
        assert guard_invented_command_failure(reply, "some Veigh", lang) == expected(
            lang, suggestion if stager_result else None)
    if stager:
        stager.assert_called_once_with(suggestion, "SPOTIFY_PLAY", lang)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang,target", [
    ("pt", "a tela"), ("pt", "o som do Spotify"), ("pt", "a iluminação do teclado"),
    ("en", "the screen"), ("en", "the sound on Spotify"), ("en", "the keyboard lighting"),
])
@pytest.mark.parametrize("mechanism", ["direct", "function", "purpose"])
def test_control_failure_replaces_whole_reply(live, stream, lang, target, mechanism):
    o, engine, model = live
    opening = ({
        "direct": "Não consegui controlar", "function": "Não consegui acionar a função de controlar",
        "purpose": "Não consegui usar a ferramenta de dispositivo para controlar",
    } if lang == "pt" else {
        "direct": "I couldn't control", "function": "I couldn't trigger the function to control",
        "purpose": "I couldn't use the tool for device control to control",
    })[mechanism]
    model.reply = f"{opening} {target}. " + ("Use os botões." if lang == "pt" else "Use the buttons.")
    assert turn(o, lang=lang, stream=stream)[0] == TRUTH[lang]
    assert engine.history[-1]["content"] == TRUTH[lang]
    assert o._pending_action is None
    assert o.macos.mock_calls == []
    o.spotify.play_anything.assert_not_called()


@pytest.mark.parametrize("user_input,suggestion,allowed", [
    ("manda algo do VeIgH", "toca VEIGH no spotify", True),
    ("manda algo do matuê", "toca matue no spotify", True),
    ("manda algo do matue", "toca matuê no spotify", True),
    ("manda algo do matue\u0302", "toca MATUE no spotify", True),
    ("manda algo do xmatue", "toca matue no spotify", False),
    ("manda algo do matuezinho", "toca matue no spotify", False),
    ("quero o xsafari aberto", "abre o safari", False),
    ("quero o safariapp aberto", "abre o safari", False),
    ("quero o (SAFARI) aberto", "Cortex, abra o Safari, por favor!", True),
    ("quero o spot 5 aberto", "open spotify please", True),
    ("quero o spotify aberto", "open spot 5 please", True),
    ("minha playlist T(rap) por favor", "toca a playlist T(rap) no spotify", True),
    ("minha playlist trap por favor", "toca a playlist T(rap) no spotify", False),
    ("minha seleção de treino", "toca a playlist treino no spotify", True),
    ("minha seleção de treino", "toca a playlist festa no spotify", False),
    ("my late   night selection", "play my late night playlist on spotify", True),
    ("my late night selection", "play my workout playlist on spotify", False),
    ("um som do matuê", "toca rádio do matue no spotify", True),
    ("um som do veigh", "toca rádio do matue no spotify", False),
    ("some veigh please", "play veigh radio on spotify", True),
    ("some matue please", "play veigh radio on spotify", False),
    ("manda BACKSTAGE de MATUÊ", "toca backstage de matue no spotify", True),
    ("manda BACKSTAGE", "toca backstage de matue no spotify", False),
    ("quero o visual studio code aberto", "abre o Visual Studio Code", True),
    ("quero o studio aberto", "abre o Visual Studio Code", False),
    ("quero o pause aberto", "open pause on Spotify", False),
    ("quero o pausa aberto", "abre pausa no Spotify", False),
])
def test_argument_whole_words_and_shared_normalisation(live, user_input, suggestion, allowed):
    o, engine, model = live
    model.reply = f"Não consegui ajustar o som. Tenta dizer '{suggestion}'."
    assert turn(o, user_input)[0] == expected("pt", suggestion if allowed else None)
    assert bool(o._pending_action) == allowed


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("suggestions,offered", [
    ("'trava a tela' ou 'toca matue no spotify' ou 'toca veigh no spotify' ou 'pausa'",
     "toca veigh no spotify"),
    ('“toca matue no spotify” ou «abra a luz no teclado» ou ‘pausa’', "pausa"),
    ("'toca matue no spotify' ou 'trava a tela'", None),
])
def test_first_quote_passing_mode_and_argument_rules(live, stream, suggestions, offered):
    o, engine, model = live
    model.reply = f"Não consegui acionar o Spotify. Tenta dizer {suggestions}."
    assert turn(o, "manda um veigh aí", stream=stream)[0] == expected("pt", offered)
    assert bool(o._pending_action) == bool(offered)
    if offered:
        assert o._pending_action["args"]["text"] == offered
    o.spotify.play_anything.assert_not_called()
    o.spotify.pause.assert_not_called()
    assert o.macos.mock_calls == []


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("user_input,reply,suggestion", [
    ("manda um veigh aí", 'Não consegui executar o comando. Se você quiser ouvir música, tente dizer "toca veigh no spotify" ou "toque veigh agora".', "toca veigh no spotify"),
    ("abaixa a tela", 'Não consegui executar o comando. Se você quiser ajustar a tela, tente dizer "abraça a tela".', None),
    ("para esse som", 'Não consegui executar o comando. Se você quiser pausar a música, tente dizer "pausa a música".', "pausa a música"),
])
def test_generic_command_failure_live_replies(live, stream, user_input, reply, suggestion):
    o, engine, model = live
    model.reply = reply
    assert turn(o, user_input, stream=stream)[0] == expected("pt", suggestion)
    assert engine.history[-1]["content"] == expected("pt", suggestion)
    assert bool(o._pending_action) == bool(suggestion)
    o.spotify.play_anything.assert_not_called()
    o.spotify.pause.assert_not_called()
    assert o.macos.mock_calls == []


@pytest.mark.parametrize("verb", ["executar", "realizar", "completar", "cumprir", "fazer"])
@pytest.mark.parametrize("determiner", ["o", "esse", "este", "seu", "teu"])
@pytest.mark.parametrize("noun", ["comando", "pedido"])
def test_generic_pt_command_failure(live, verb, determiner, noun):
    o, engine, model = live
    model.reply = f"\nEu não consegui {verb} {determiner} {noun}."
    assert turn(o)[0] == TRUTH["pt"]
    assert o._pending_action is None


@pytest.mark.parametrize("verb", ["execute", "run", "carry out", "complete", "do"])
@pytest.mark.parametrize("determiner", ["the", "that", "this", "your"])
@pytest.mark.parametrize("noun", ["command", "request"])
def test_generic_en_command_failure(live, verb, determiner, noun):
    o, engine, model = live
    model.reply = f"I couldn't {verb} {determiner} {noun}."
    assert turn(o, lang="en")[0] == TRUTH["en"]
    assert o._pending_action is None


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang,word", [("pt", "sim"), ("en", "yes")])
def test_approval_dispatches_exact_stored_text_once(live, monkeypatch, stream, lang, word):
    o, engine, model = live
    assert turn(o, lang=lang, stream=stream)[0] == expected(lang, "toca matuê no spotify")
    calls = Mock(wraps=registry.run)
    monkeypatch.setattr(registry, "run", calls)
    assert turn(o, word, lang=lang, stream=stream) == ("Playing.", "SPOTIFY_PLAY")
    assert any(c.args[1] == "toca matuê no spotify" for c in calls.call_args_list)
    o.spotify.play_anything.assert_called_once_with("matuê", device_hint=None, lang=lang)
    assert model.call_count == 1
    turn(o, word, lang=lang, stream=stream)
    assert o.spotify.play_anything.call_count == 1


@pytest.mark.parametrize("condition", ["changed", "expired", "origin", "unrelated", "declined", "declined_route"])
@pytest.mark.parametrize("stream", [False, True])
def test_invalid_or_cancelled_approval_never_runs(live, monkeypatch, condition, stream):
    o, engine, model = live
    turn(o, stream=stream)
    assert o._pending_action is not None
    if condition == "changed":
        monkeypatch.setattr(registry, "simulate", lambda *a, **kw:
                            registry.RouteMatch("MAC_LOCK", "changed"))
    elif condition == "expired":
        action = json.loads(Path(o._PENDING_ACTION_FILE).read_text())
        action["staged_at"] = (datetime.now(timezone.utc) - timedelta(minutes=16)).isoformat()
        Path(o._PENDING_ACTION_FILE).write_text(json.dumps(action))
    elif condition == "unrelated":
        turn(o, "que horas são", stream=stream)
    elif condition == "declined":
        assert turn(o, "não", stream=stream)[1] == "AGENT_ACTION_CANCELLED"
    elif condition == "declined_route":
        monkeypatch.setattr(registry, "run", Mock(return_value=None))
    before = model.call_count
    reply, mode = turn(o, "sim", stream=stream, conversation="B" if condition == "origin" else "A")
    o.spotify.play_anything.assert_not_called()
    if condition in ("changed", "declined_route"):
        assert reply == TRUTH["pt"]
        assert model.call_count == before
    if condition == "origin":
        assert mode == "AGENT_ACTION_ELSEWHERE"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("reply", [
    "Não consegui usar a ferramenta de busca para achar o preço do Mac.",
    "Não consegui usar a ferramenta para encontrar informações sobre o teclado.",
    "I couldn't use the function to find an article about the screen.",
    "Não consegui acionar o comando de ajuste de busca sobre o brilho da tela.",
    "I couldn't use the search tool to find the price of the Mac.",
    "Não consegui encontrar informações sobre o brilho do Mac.",
    "Não consegui entender a pergunta sobre Spotify.",
    "I couldn't find information about the keyboard.",
    "I couldn't understand the question about music.",
    "Uma explicação. Não consegui pausar a música.",
    "Não consegui acionar a função de busca para essa pergunta.",
    "Não consegui encontrar informações sobre X.",
    "não consegui entender a pergunta",
    "I couldn't use the function to find that article.",
    "Não consegui acionar a função de ajustar.",
    "I couldn't activate the control to adjust.",
    "Não consegui acionar o comando de busca para isso.",
    "Não consegui acionar o controle de versão do projeto.",
    "Não consegui acionar o comando de ajuste de.",
    "I couldn't use the command for adjustment of version control in the project.",
    "I couldn't trigger the command for change of.",
    "Não consegui executar o código que você mandou.",
    "I couldn't execute the code you sent.",
    "Uma explicação. Não consegui executar o comando.",
    "Uma explicação.\nNão consegui executar o comando.",
    "Uma explicação. " + MECHANISM_PAUSE,
    "Uma explicação.\n" + MECHANISM_PAUSE,
    "Não consegui executar o comandou.",
    "I couldn't complete your requester.",
    "Uma explicação. " + V9_PLAY,
    "Uma explicação.\n" + V9_PLAY,
    "Não entendi esse comando, então nada foi feito para mudar a pergunta. Tenta dizer 'pausa'.",
    "Não entendi esse comando, então nada foi feitou. Tenta dizer 'pausa'.",
    "I didn't understand that command, so nothing was done about the question. Try saying 'pause'.",
    "I didn't understand that command, so nothing was donewrong. Try saying 'pause'.",
])
def test_knowledge_answers_untouched(live, stream, reply):
    o, engine, model = live
    model.reply = reply
    assert turn(o, stream=stream)[0] == reply
    assert o._pending_action is None


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang,purpose", [
    ("pt", "de busca sobre"),
    ("pt", "de pesquisa sobre"),
    ("pt", "para encontrar dados sobre"),
    ("pt", "para achar dados sobre"),
    ("pt", "para buscar dados sobre"),
    ("pt", "para pesquisar dados sobre"),
    ("pt", "para procurar dados sobre"),
    ("pt", "para descobrir dados sobre"),
    ("pt", "de informação sobre"),
    ("pt", "de informações sobre"),
    ("pt", "de informacao sobre"),
    ("pt", "de informacoes sobre"),
    ("pt", "para ler um artigo sobre"),
    ("pt", "para consultar o preço de"),
    ("pt", "para consultar uma notícia sobre"),
    ("en", "to search for data about"),
    ("en", "for web data about"),
    ("en", "for internet data about"),
    ("en", "to find data about"),
    ("en", "to look up data about"),
    ("en", "for information about"),
    ("en", "to read an article about"),
    ("en", "to check the price of"),
    ("en", "for news about"),
    ("pt", "de busca para silenciar o"),
    ("pt", "de pesquisa para iluminar o"),
    ("en", "for search to mute the"),
    ("en", "for information to illuminate the"),
])
def test_mechanism_knowledge_purpose_before_device_passes(live, stream, lang, purpose):
    o, engine, model = live
    opening = "Não consegui usar a ferramenta" if lang == "pt" else "I couldn't use the tool"
    model.reply = f"{opening} {purpose} Mac."
    assert turn(o, lang=lang, stream=stream)[0] == model.reply
    assert engine.history[-1]["content"] == model.reply
    assert o._pending_action is None


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang,reply,replaced", [
    ("pt", "Não consegui usar a ferramenta solicitada para o computador e pesquisar depois.", False),
    ("en", "I couldn't use the tool requested for the computer and search afterwards.", False),
    ("pt", "Não consegui usar a ferramenta solicitada para o teclado. A busca falhou.", False),
    ("en", "I couldn't use the tool requested for the keyboard. The search failed.", False),
    ("pt", "Não consegui usar a ferramenta do pesquisador para o computador.", False),
    ("en", "I couldn't use the tool from Newsroom for the computer.", False),
    ("pt", "Não consegui usar a ferramenta solicitada para silenciar o computador e pesquisar depois.", True),
    ("en", "I couldn't use the tool requested to mute the computer and search afterwards.", True),
    ("pt", "Não consegui usar a ferramenta solicitada para iluminar o teclado. A busca falhou.", True),
    ("en", "I couldn't use the tool requested to light up the keyboard. The search failed.", True),
    ("pt", "Não consegui usar a ferramenta do pesquisador para silenciar o computador.", True),
    ("en", "I couldn't use the tool from Newsroom to mute the computer.", True),
    ("pt", "Não consegui executar o comando de busca sobre o Mac.", True),
    ("en", "I couldn't execute the command to search for the Mac.", True),
])
def test_knowledge_exclusion_stays_before_device_in_mechanism_branch(live, stream, lang, reply, replaced):
    o, engine, model = live
    model.reply = reply
    assert turn(o, lang=lang, stream=stream)[0] == (TRUTH[lang] if replaced else reply)
    assert o._pending_action is None


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("reply", [PLAY, "Não consegui executar o comando.", MECHANISM_PAUSE,
                                  V9_PLAY, V9_PLAY_EN])
def test_rewording_exemption(live, stream, reply):
    o, engine, model = live
    model.reply = reply
    assert turn(o, "Traduza para português: I couldn't pause the music.", stream=stream)[0] == reply
    assert o._pending_action is None


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("reply", [PAUSE, "Não consegui executar o comando.", MECHANISM_PAUSE,
                                  V9_PLAY, V9_PLAY_EN])
def test_route_ran_untouched(live, stream, reply):
    o, engine, model = live
    o.spotify.pause.return_value = reply
    assert turn(o, "pausa", stream=stream)[0] == reply
    model.assert_not_called()
    assert o._pending_action is None


@pytest.mark.parametrize("reply", [PLAY, "Não consegui executar o comando.", MECHANISM_PAUSE,
                                  V9_PLAY, V9_PLAY_EN])
def test_tool_ran_untouched(live, reply):
    o, engine, model = live
    model.side_effect = [
        {"message": {"tool_calls": [{"function": {"name": "fake", "arguments": {}}}]}},
        {"message": {"content": reply}},
    ]
    executor = Mock(return_value="Tool failure.")
    assert engine.chat_with_tools("comando", [], executor, "pt") == (reply, "fake")
    executor.assert_called_once_with("fake", {})


@pytest.mark.parametrize("stream", [False, True])
def test_failure_words_in_user_input_do_not_trigger_guard(live, stream):
    o, engine, model = live
    model.reply = "Uma resposta normal."
    assert turn(o, "não consegui dormir ontem…", stream=stream)[0] == model.reply
    assert o._pending_action is None


def test_no_tool_guard_and_audit(live, monkeypatch):
    o, engine, model = live
    log = Mock()
    monkeypatch.setattr("src.core.engine.audit.log", log)
    assert turn(o, "comando de música do matuê")[0] == expected("pt", "toca matuê no spotify")
    swaps = [c for c in log.call_args_list if len(c.args) > 2 and c.args[2] == "invented_command_failure"]
    assert len(swaps) == 1
    assert swaps[0].kwargs["payload"] == {
        "original_head": PLAY[:120], "user_input": "comando de música do matuê",
        "suggestion": "toca matuê no spotify", "mode": "SPOTIFY_PLAY",
    }
    assert engine.history[-1]["content"] == expected("pt", "toca matuê no spotify")


@pytest.mark.parametrize("suggestion", [
    "diminua o brilho", "aumenta a luz do teclado", "volume 30", "mudo",
    "unmute", "modo escuro", "pausa", "continua a música", "próxima música",
    "música anterior", "abre o Safari",
])
def test_allowed_controls(live, suggestion):
    o, engine, model = live
    model.reply = f"Não consegui ajustar o volume. Tenta dizer '{suggestion}'."
    assert turn(o, "quero o safari aberto")[0] == expected("pt", suggestion)


@pytest.mark.parametrize("suggestion", [
    "manda email pro João", "me lembra de estudar amanhã", "cria uma nota teste",
    "desliga o ps5", "trava a tela", "dorme", "fecha o Safari", "tira um print",
    "gosta dessa música", "algo que não existe", "estuda matemática",
    "pausa no YouTube", "pause on the computer", "pausa no Spotify e fecha o Safari",
])
def test_disallowed_and_unroutable_suggestions(live, suggestion):
    o, engine, model = live
    model.reply = f"Não consegui acionar o Spotify. Tenta dizer '{suggestion}'."
    assert turn(o)[0] == TRUTH["pt"]
    assert o._pending_action is None


@pytest.mark.parametrize("path", ["/ask", "/ask_voice", "/ask_stream"])
@pytest.mark.parametrize("tool_hint", [False, True])
def test_api_surfaces_only_emit_and_keep_replacement(live, monkeypatch, path, tool_hint):
    from fastapi.testclient import TestClient
    import src.api as api

    o, engine, model = live
    monkeypatch.setattr(api, "orchestrator", o)
    monkeypatch.setattr(api, "SECRET_API_KEY", "test-fake")
    monkeypatch.setattr(api, "_ALLOWLIST", [])
    monkeypatch.setattr(api, "_daily_history_msgs", lambda *a: None)
    record = Mock()
    monkeypatch.setattr(api, "_record_voice_turn", record)
    monkeypatch.setattr(api, "_record_ask_turn", record)
    api.limiter.reset()
    client = TestClient(api.app)
    response = client.post(path, headers={"X-Cortex-Key": "test-fake"}, json={
        "user_input": "comando de música do matuê" if tool_hint else "solta um matuê aí",
        "language": "pt",
    })
    assert response.status_code == 200
    want = expected("pt", "toca matuê no spotify")
    if path == "/ask":
        assert response.json()["response"] == want
    elif path == "/ask_stream":
        events = [json.loads(line[6:]) for line in response.text.splitlines()
                  if line.startswith("data: ")]
        assert "".join(e.get("chunk", "") for e in events) == want
        assert events[-1]["status_mode"] == "AGENT_AWAITING_CONFIRMATION"
    else:
        assert response.text == want
    assert engine.history[-1]["content"] == want
    assert record.call_args is not None
    assert want in record.call_args.args
    assert o._pending_action["origin"]["channel"] == (
        "network" if path == "/ask" else "voice")


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang,reply,suggestion", [
    ("pt", PAUSE, "pausa"),
    ("pt", MECHANISM_PAUSE, "pausa no Spotify"),
    ("en", MECHANISM_PAUSE_EN, "pause on Spotify"),
])
def test_wrong_intent_suggestion_requires_approval(live, stream, lang, reply, suggestion):
    o, engine, model = live
    model.reply = reply
    assert turn(o, lang=lang, stream=stream)[0] == expected(lang, suggestion)
    assert o._pending_action["args"] == {
        "text": suggestion, "mode": "SPOTIFY_PAUSE", "lang": lang,
    }
    o.spotify.pause.assert_not_called()
    assert turn(o, "sim" if lang == "pt" else "yes", lang=lang,
                stream=stream) == ("Paused.", "SPOTIFY_PAUSE")
    o.spotify.pause.assert_called_once_with(lang=lang)
    o.macos.mute.assert_not_called()


# Task 154 candidates veto pure chat even with an empty library. Omit "um"
# here so these cases still exercise the live-stream guard, not dispatch.
@pytest.mark.parametrize("lang,reply,suggestion,user_input", [
    ("pt", PLAY, "toca matuê no spotify", "solta matuê aí"),
    ("pt", "Não consegui acionar a função de pausar a música.", None, "solta matuê aí"),
    ("pt", MECHANISM_PAUSE, "pausa no Spotify", "solta matuê aí"),
    ("pt", V9_PLAY, "toca veigh no spotify", "solta veigh aí"),
    ("pt", V9_PLAY, None, "silencia o computador"),
    ("en", V9_PLAY_EN, "play veigh on spotify", "give me some veigh"),
    ("en", V9_PLAY_EN, None, "mute the computer"),
])
def test_failure_stream_is_checked_before_first_emission(live, lang, reply, suggestion, user_input):
    o, engine, model = live
    finished = []

    def response(**kwargs):
        def chunks():
            for char in reply:
                yield {"message": {"content": char}}
            finished.append(True)
        return chunks()

    model.side_effect = response
    o.set_turn_origin("voice")
    try:
        frames = o.process_stream(user_input, language=lang)
        first = next(frames)
        assert finished == [True]
        assert first[0] == (reply if suggestion is None and reply in (V9_PLAY, V9_PLAY_EN)
                            else expected(lang, suggestion))
        assert all(not f[0] for f in frames)
    finally:
        o.set_turn_origin(None)


def test_ordinary_reply_still_streams_live(live):
    o, engine, model = live
    model.reply = "Uma resposta normal."
    frames = o.process_stream("solta matuê aí", language="pt")
    assert next(frames)[0] == "U"
    list(frames)


@pytest.mark.parametrize("chunks_factory,expected_reply", [
    (MagicMock, ""),
    (lambda: ["Model ", "answer."], "Model answer."),
], ids=["magicmock", "list"])
def test_non_iterator_stream_finishes(orch, chunks_factory, expected_reply):
    chunks = chunks_factory()
    orch.engine_analytical.chat_stream.side_effect = None
    orch.engine_analytical.chat_stream.return_value = chunks
    stream = orch.process_stream("uma ação desconhecida", language="pt")
    try:
        frames = list(islice(stream, 10))
        assert len(frames) < 10, "process_stream did not finish within 10 frames"
        assert frames[-1][1] is True
        assert "".join(frame[0] for frame in frames) == expected_reply
    finally:
        stream.close()
    orch.engine_analytical.chat_stream.assert_called_once()
    if isinstance(chunks, MagicMock):
        chunks.__iter__.assert_called_once_with()
        chunks.__next__.assert_not_called()
        chunks.close.assert_called_once_with()


@pytest.mark.parametrize("reply", [
    "Não consegui ajustar a iluminação da foto.",
    "Não consegui aumentar o rendimento.",
    "I couldn't open the discussion.",
    "I couldn't adjust the lighting in the painting.",
])
def test_non_control_failures_pass(live, reply):
    o, engine, model = live
    model.reply = reply
    assert turn(o)[0] == reply


@pytest.mark.parametrize("stream", [False, True])
def test_approval_keeps_staged_language(live, stream):
    o, engine, model = live
    turn(o, lang="en", stream=stream)
    assert turn(o, "sim", lang="pt", stream=stream) == ("Playing.", "SPOTIFY_PLAY")
    o.spotify.play_anything.assert_called_once_with("matuê", device_hint=None, lang="en")


def test_engine_without_orchestrator_never_offers_unstaged_action(live):
    o, engine, model = live
    assert engine.chat_with_tools("comando de música", [], Mock(), "pt") == (TRUTH["pt"], None)
    assert o._pending_action is None


@pytest.mark.parametrize("reply", [
    "\nEu   não\tconsegui pausar a música. Tenta dizer 'pausa'.",
    "I  couldn’t pause the music. Try saying 'pausa'.",
    json.dumps({"response": PLAY}, ensure_ascii=False),
])
def test_failure_stream_handles_spacing_and_sanitisation(live, reply):
    o, engine, model = live
    model.reply = reply
    suggestion = "toca matuê no spotify" if reply.startswith("{") else "pausa"
    assert turn(o, stream=True)[0] == expected("pt", suggestion)


def test_disconnected_stream_withdraws_offer(live, monkeypatch):
    import threading
    from src.core.staging_mixin import Delivery

    o, engine, model = live
    monkeypatch.setattr("src.core.staging_mixin.UNDELIVERED_GRACE_S", 0.01)
    withdrawn = threading.Event()
    original = o.withdraw_pending

    def withdraw(by):
        result = original(by)
        withdrawn.set()
        return result

    monkeypatch.setattr(o, "withdraw_pending", withdraw)
    o.set_turn_delivery(Delivery())
    frames = o.process_stream("solta um matuê aí", language="pt")
    try:
        assert next(frames)[0] == expected("pt", "toca matuê no spotify")
    finally:
        o.set_turn_delivery(None)
        frames.close()
    assert withdrawn.wait(1)
    assert not Path(o._PENDING_ACTION_FILE).exists()
    turn(o, "sim")
    o.spotify.play_anything.assert_not_called()
