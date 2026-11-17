"""Voice sentence endings must not change command ownership or arguments."""
from unittest.mock import MagicMock, call

import pytest

from tests.test_taught_facts import orch, saved  # isolated Palace + fake engines/integrations


ENDINGS = ("", ".", "!", "?", "…", ". ", "...", "?!….")
DOUBTFUL_YES = ("sim?", "Sim?", "ok?", "claro?", "sim…")
DECISIONS = tuple((text, None) for text in DOUBTFUL_YES) + (
    ("sim.", "EXECUTED"), ("Sim!", "EXECUTED"),
    ("não?", "CANCELLED"), ("não.", "CANCELLED"),
)
# All general pending actions share one gate; code/PDF grants also have
# their own exact-yes gate. Stub execution so no integration can run.
PENDING_KINDS = (
    "send_email", "create_event", "create_reminder", "mac_quit_app",
    "mac_sleep", "ps5_standby", "execute_plan", "run_refined_prompt",
    "read_file_at", "read_files", "write_answers_pdf", "read_code_project",
    "propose_code_change",
    "code_root_add", "run_code_tests", "apply_code_change",
    "pdf_merge", "pdf_split", "pdf_png", "pdf_images",
)


def turn(o, text, stream):
    if stream:
        frames = list(o.process_stream(text, language="pt"))
        assert frames[-1][1]
        return frames[-1][2]
    return o.process(text, language="pt")[1]


# Every Mac registry route: integration calls are real dispatch into fakes;
# context readers and staging handlers are replaced before they can use a Mac.
MAC = (
    ("Aumente o brilho da tela", "brightness_up", (), "MAC_BRIGHTNESS"),
    ("Diminua o brilho da tela", "brightness_down", (), "MAC_BRIGHTNESS"),
    ("brilho da tela 50", "set_brightness", (50,), "MAC_BRIGHTNESS"),
    ("aumente o brilho do teclado", "keyboard_brightness_up", (), "MAC_KEYBOARD_BRIGHTNESS"),
    ("diminua o brilho do teclado", "keyboard_brightness_down", (), "MAC_KEYBOARD_BRIGHTNESS"),
    ("Aumente o brilho do teclado para 50", "keyboard_brightness_set", (50,), "MAC_KEYBOARD_BRIGHTNESS"),
    ("liga a luz do teclado", "keyboard_brightness_set", (50,), "MAC_KEYBOARD_BRIGHTNESS"),
    ("apaga a luz do teclado", "keyboard_brightness_set", (0,), "MAC_KEYBOARD_BRIGHTNESS"),
    ("aumente o volume", "volume_up", (), "MAC_VOLUME"),
    ("Diminua o volume", "volume_down", (), "MAC_VOLUME"),
    ("Volume 30", "set_system_volume", (30,), "MAC_VOLUME"),
    ("mudo", "mute", (), "MAC_MUTE"),
    ("desmuta", "unmute", (), "MAC_UNMUTE"),
    ("Ativa o modo escuro", "enable_dark_mode", (), "MAC_DARK"),
    ("desativa o modo escuro", "disable_dark_mode", (), "MAC_DARK"),
    ("alterna o modo", "toggle_dark_mode", (), "MAC_DARK"),
    ("trava a tela", "lock_screen", (), "MAC_LOCK"),
    ("tira um print", "screenshot", (), "MAC_SCREENSHOT"),
    ("lista os apps", "list_apps", (), "MAC_APP_LIST"),
    ("Abre o Safari", "open_app", ("Safari",), "MAC_OPEN_APP"),
    ("fecha a janela do Safari", "close_window", ("Safari",), "MAC_CLOSE_WINDOW"),
    ("vai para o Safari", "switch_to_app", ("Safari",), "MAC_APP_SWITCH"),
)
SPOTIFY = (
    ("Toca Beatles no Spotify", "play_anything", ("Beatles",), "SPOTIFY_PLAY"),
    ("Pausa a música", "pause", (), "SPOTIFY_PAUSE"),
    ("continua a música", "resume", (), "SPOTIFY_RESUME"),
    ("próxima música", "next_track", (), "SPOTIFY_NEXT"),
    ("música anterior", "previous_track", (), "SPOTIFY_PREV"),
    ("que música está tocando", "now_playing", (), "SPOTIFY_NOW"),
    ("curte essa música", "like_current", (), "SPOTIFY_LIKE"),
    ("volume do spotify 30", "set_volume", (30,), "SPOTIFY_VOLUME"),
)


@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("ending", ENDINGS)
@pytest.mark.parametrize("integration,case", [("macos", c) for c in MAC] + [("spotify", c) for c in SPOTIFY])
def test_integration_routes(orch, stream, ending, integration, case):
    text, method, args, mode = case
    fake = getattr(orch, integration)
    getattr(fake, method).return_value = "Done."
    assert turn(orch, text + ending, stream) == mode
    expected = getattr(call, method)(*args, lang="pt")
    if method == "play_anything":
        expected = call.play_anything(*args, device_hint=None, lang="pt")
    assert fake.method_calls == [expected]
    for engine in (orch.engine_analytical, orch.engine_creative):
        engine.chat.assert_not_called()
        engine.chat_stream.assert_not_called()
        engine.chat_with_tools.assert_not_called()


HANDLERS = (
    ("encerra o Safari", "_handle_mac_quit_regex", ("Safari",), {}, "AGENT_AWAITING_CONFIRMATION"),
    ("dorme o mac", "_handle_mac_sleep_regex", (), {"phrased_as_shutdown": False}, "AGENT_AWAITING_CONFIRMATION"),
    ("responde esse email", "_handle_reply_email", (), {}, "MAC_REPLY_EMAIL"),
    ("resume essa tela", "_handle_summarise_screen", (), {}, "MAC_SUMMARISE"),
    ("o que é isso", "_handle_what_is_this", (), {}, "MAC_WHAT_IS"),
    ("para de ver minha tela", "_handle_screen_eyes_off", (), {}, "MAC_EYES_OFF"),
    ("volta a ver minha tela", "_handle_screen_eyes_on", (), {}, "MAC_EYES_ON"),
    ("estuda redes neurais", "_handle_add_study", ("redes neurais",), {}, "STUDY_QUEUED"),
    ("Adiciona na sua lista de estudos: redes neurais", "_handle_add_study", ("redes neurais",), {}, "STUDY_QUEUED"),
    ("comi 100g de arroz", "_handle_fitness_log_meal", ("100g de arroz",), {"raw_text": "comi 100g de arroz"}, "FITNESS_MEAL"),
    ("corri 1.5 km em 10 minutos", "_handle_fitness_log_run", ("1.5", "10"), {"raw_text": "corri 1.5 km em 10 minutos"}, "FITNESS_RUN_LOG"),
    ("tempo em Londres", "_handle_weather", ("Londres",), {"raw_text": "tempo em Londres"}, "WEATHER"),
    ("planeja: organizar a semana", "_handle_planner_request", ("organizar a semana",), {}, "AGENT_AWAITING_CONFIRMATION"),
)


@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("ending", ENDINGS)
@pytest.mark.parametrize("text,handler,args,kwargs,mode", HANDLERS)
def test_handler_routes(orch, stream, ending, text, handler, args, kwargs, mode):
    fake = MagicMock(return_value=("Done.", mode))
    setattr(orch, handler, fake)
    assert turn(orch, text + ending, stream) == mode
    fake.assert_called_once_with(*args, **kwargs)


@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("ending", ENDINGS)
@pytest.mark.parametrize("prefix", ("anota ", "Anota: "))
def test_note_payload(orch, stream, ending, prefix):
    assert turn(orch, prefix + "comprar pão" + ending, stream) == "NOTE_CREATE"
    orch.notes.create_note.assert_called_once_with(title="comprar pão", body="", folder=None)


@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("ending", ENDINGS)
def test_reminder_payload(orch, stream, ending):
    assert turn(orch, "Me lembra de ligar pro João amanhã às 10" + ending, stream) == "AGENT_AWAITING_CONFIRMATION"
    assert orch._pending_action["type"] == "create_reminder"
    assert orch._pending_action["args"]["title"] == "ligar pro João"
    assert orch._pending_action["args"]["due_iso"] is not None


@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("ending", ENDINGS)
def test_reflex(orch, stream, ending):
    assert turn(orch, "Que horas são" + ending, stream) == "REFLEX"


@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("payload", ("1.5", "v3.1", "https://example.com/a.pdf?q=1.5", "~/a.pdf", "e.g.", "use e.g. v3.1 at https://example.com/a.pdf and ~/a.pdf"))
@pytest.mark.parametrize("ending", ("", ".", "!", "…"))
def test_literals_in_payload(orch, stream, payload, ending):
    assert turn(orch, "anota " + payload + ending, stream) == "NOTE_CREATE"
    orch.notes.create_note.assert_called_once_with(title=payload, body="", folder=None)


@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("text", ("Tell me a story about a cloud…", "can you explain why clouds float?!"))
def test_model_and_memory_keep_original(orch, stream, text):
    assert turn(orch, text, stream) == "OFFLINE"
    engine = orch.engine_analytical
    method = engine.chat_stream if stream else engine.chat
    assert method.call_args.args[0] == text
    assert engine.memory._palace.list_documents(tipo="knowledge") == []


@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("text,accepted", (("sim.", True), ("yes!", True), ("sim?", False), ("sim…", False), ("sim!!", False), ("sim, mas…", False), ("sim...", False)))
@pytest.mark.parametrize("kind", ("code_root_add", "run_code_tests", "pdf_merge", "pdf_split", "pdf_png", "pdf_images", "apply_code_change"))
def test_exact_yes_unchanged(orch, stream, text, accepted, kind):
    orch._pending_action = {"type": kind, "args": {"lang": "pt"}}
    orch._persist_pending_action()
    orch._pending_approval_reply = MagicMock(return_value=None)
    orch._execute_pending_action = MagicMock(return_value=("Executed.", "EXECUTED"))
    mode = turn(orch, text, stream)
    assert (mode == "EXECUTED") is accepted
    assert orch._execute_pending_action.call_count == int(accepted)


@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("text,expected", DECISIONS)
@pytest.mark.parametrize("kind", PENDING_KINDS)
def test_pending_decisions_preserve_authority(orch, stream, text, expected, kind):
    orch._pending_action = {"type": kind, "args": {"lang": "pt"}}
    orch._persist_pending_action()
    orch._pending_approval_reply = MagicMock(return_value=None)
    orch._execute_pending_action = MagicMock(return_value=("Done.", "EXECUTED"))
    orch._cancel_pending_action = MagicMock(return_value=("Cancelled.", "CANCELLED"))
    mode = turn(orch, text, stream)
    if expected:
        assert mode == expected
    else:
        assert mode not in ("EXECUTED", "CANCELLED")
    assert orch._execute_pending_action.call_count == int(expected == "EXECUTED")
    assert orch._cancel_pending_action.call_count == int(expected == "CANCELLED")


@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("prefix", ("estuda ", "please study ", "adiciona na sua lista de estudos: "))
@pytest.mark.parametrize("ending", ("", ".", "?", "…"))
def test_study_force_flag_survives(orch, stream, prefix, ending):
    orch._handle_add_study = MagicMock(return_value=("Queued.", "STUDY_QUEUED"))
    assert turn(orch, prefix + "current affairs !!" + ending, stream) == "STUDY_QUEUED"
    orch._handle_add_study.assert_called_once_with("current affairs !!")


def test_helper_contract():
    # Local import lets all end-to-end regressions run on the unfixed tree.
    from src.core.dispatch.normalisation import strip_trailing_sentence_punctuation as strip
    for text in ("", ".", "!", "?", "…", "...", "?!", "1.5", "v3.1", "~/a.pdf", "e.g."):
        assert strip(text) == text
    for ending in ENDINGS:
        assert strip("volume 30" + ending) == "volume 30"
    assert strip('anota "olá!"') == 'anota "olá!"'


@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("ending", ENDINGS)
def test_study_approval_selector_is_literal(orch, stream, ending):
    orch._handle_approve_study = MagicMock(return_value=("Approved.", "STUDY_APPROVED"))
    assert turn(orch, "aprova estudo 1" + ending, stream) == "STUDY_APPROVED"
    orch._handle_approve_study.assert_called_once_with("1" + ending.strip())


@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("ending", ("", "."))
def test_real_study_queue(orch, tmp_study_queue, stream, ending):
    orch.study_queue = tmp_study_queue
    assert turn(orch, "Adiciona na sua lista de estudos: redes neurais" + ending, stream) == "STUDY_QUEUED"
    assert tmp_study_queue.list_pending()[0]["topic"] == "redes neurais"


@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("ending", ("", "."))
def test_review_fixture_without_queue_is_the_study_error(orch, stream, ending, caplog):
    assert not hasattr(orch, "study_queue")
    assert turn(orch, "Adiciona na sua lista de estudos: redes neurais" + ending, stream) == "ROUTE_ERROR"
    assert "has no attribute 'study_queue'" in caplog.text


@pytest.mark.parametrize("stream", (False, True))
def test_taught_memory_keeps_sentence_marks(orch, stream):
    text = "aprenda com isso: Cortex veio do córtex cerebral…"
    assert turn(orch, text, stream) == "OFFLINE"
    assert saved(orch).read_text().endswith(text + "\n")


@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("ending", ENDINGS)
@pytest.mark.parametrize("text,handler", (("projetos", "_handle_projects"), ("onde parei", "_handle_focus")))
def test_pre_registry_commands(orch, stream, ending, text, handler):
    def handle(routed, *args):
        return ("Owned.", "PRE_REGISTRY") if routed == text else None
    setattr(orch, handler, handle)
    assert turn(orch, text + ending, stream) == "PRE_REGISTRY"


@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("text,accepted", tuple((t, False) for t in DOUBTFUL_YES) + (("sim.", True), ("Sim!", True)))
def test_named_file_yes(orch, stream, text, accepted):
    orch._offer = {"kind": "find_pick", "paths": ["/fake/a.pdf"], "verb": "read", "lang": "pt"}
    orch._act_on_named = MagicMock(return_value=("Read.", "FILE_READ"))
    assert (turn(orch, text, stream) == "FILE_READ") is accepted
    assert orch._act_on_named.call_count == int(accepted)
    if accepted:
        assert orch._act_on_named.call_args.args[:2] == ("read", "/fake/a.pdf")


@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("text,accepted", tuple((t, False) for t in DOUBTFUL_YES) + (("sim.", True), ("Sim!", True)))
@pytest.mark.parametrize("kind", (None, "write_pdf", "pick"))
def test_unpending_yes_keeps_raw_reply(orch, stream, text, accepted, kind):
    orch._offer = {"kind": kind, "paths": ["/fake/a.pdf", "/fake/b.pdf"]} if kind else None
    orch._fresh_exam = MagicMock(return_value={"fake": "exam"})
    orch._stage_write_answers = MagicMock(return_value=("Staged.", "WRITE_STAGED"))
    mode = turn(orch, text, stream)
    expected = {None: "NOTHING_PENDING", "write_pdf": "WRITE_STAGED", "pick": "FILE_LIST"}[kind]
    assert (mode == expected) is accepted
    assert orch._stage_write_answers.call_count == int(accepted and kind == "write_pdf")


@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("text,expected", DECISIONS)
@pytest.mark.parametrize("weather", (False, True))
def test_pending_query_decisions(orch, stream, text, expected, weather):
    from datetime import datetime
    orch._pending_query = "tempo em Londres…" if weather else "pesquise nuvens"
    pending_text = orch._pending_query
    orch.gatekeeper.grant_lease.return_value = datetime(2026, 11, 7, 12)
    orch._handle_weather = MagicMock(return_value=("Weather.", "WEATHER"))
    orch._do_web_search = MagicMock(return_value=("Search.", "SEARCH"))
    mode = turn(orch, text, stream)
    if expected == "EXECUTED":
        assert mode == ("WEATHER" if weather else "AUTO_SEARCH")
        orch.gatekeeper.grant_lease.assert_called_once()
        if weather:
            orch._handle_weather.assert_called_once_with("Londres", raw_text="tempo em Londres")
        else:
            orch._do_web_search.assert_called_once()
    else:
        assert mode == ("APPROVAL_DENIED" if expected == "CANCELLED" else "AWAITING_APPROVAL")
        orch.gatekeeper.grant_lease.assert_not_called()
        orch._handle_weather.assert_not_called()
        orch._do_web_search.assert_not_called()
        assert orch._pending_query == (None if expected == "CANCELLED" else pending_text)


@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("text", ("lembra disso?", "lembra da reunião?"))
def test_remember_questions_do_not_become_reminders(orch, stream, text):
    orch._handle_reminder_regex = MagicMock(side_effect=AssertionError("question became a reminder"))
    assert turn(orch, text, stream) != "AGENT_AWAITING_CONFIRMATION"
    orch._handle_reminder_regex.assert_not_called()
