"""Task 167: real history/dispatch, fake model, isolated stores, no services."""
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from src.core.dispatch.heuristics import REAL_TOOL_HINT_RE, TOOL_HINT_RE
from src.core.engine import SKIP_FOLLOWUP_MARKER
from src.core.hallucination import (
    ACTION_HALLUCINATION_FALLBACK, CONFIRMATION_HISTORY_MODES,
    CONFIRMATION_HISTORY_STAND_IN, HALLUCINATION_FALLBACK, detect_invented_action,
)
from src.core.model_profile import PROFILES
from src.core.stream_reply import ReplyReplacement
from src.core.tool_risk import build_preview
from src.ui import routes
from src.ui.chat_store import ChatStore
from src.weekly_review import WeeklyStats, aggregate, render_markdown
from tests.test_code_proposal import setup as code_setup, approve  # noqa: F401
from tests.test_file_capability_guard import rig, mac_rig  # noqa: F401
from tests.test_loss_note import orch, taught_orch, answer  # noqa: F401


@pytest.fixture
def store(tmp_path, monkeypatch):
    store = ChatStore(db_path=str(tmp_path / "chat.db"))
    monkeypatch.setattr(routes, "chat_store", store)
    yield store
    if store._conn is not None:
        store._conn.close()


@pytest.mark.parametrize("mode", sorted(CONFIRMATION_HISTORY_MODES))
def test_real_preview_modes_only_change_model_history(store, code_setup, mode):
    o, model, _ = code_setup
    if mode in ("CODE_PROPOSAL", "AWAITING_CONFIRMATION"):
        reply, actual_mode = approve(code_setup)
        if mode == "AWAITING_CONFIRMATION":
            reply, actual_mode = o._process_core("sim, pode aplicar")
    elif mode == "REFINE_STAGED":
        model.return_value = {"message": {"content": "Explique as regras do tênis."}}
        reply, actual_mode = o._handle_refine("explique tênis")
    else:
        reply, actual_mode = o._process_core("me lembre de beber água amanhã às 15h")
    assert actual_mode == mode
    cid = store.create_conversation()["id"]
    store.append_turn(cid, "pedido", reply, status_mode=mode)
    store.append_turn(cid, "cancela", "Cancelado.", status_mode="AGENT_ACTION_CANCELLED")
    history = routes._conversation_history_msgs(cid, 5)
    assert history == [
        {"role": "user", "content": "pedido"},
        {"role": "assistant", "content": CONFIRMATION_HISTORY_STAND_IN},
        {"role": "user", "content": "cancela"},
        {"role": "assistant", "content": "Cancelado."},
    ]
    assert store.get_turns(cid)[0]["reply"] == reply
    assert reply not in [m["content"] for m in history]


@pytest.mark.parametrize("mode", ["OFFLINE", "AGENT_ACTION_DONE", "TOOL_SEND_EMAIL",
                                      "AWAITING_APPROVAL", "CODE_PROPOSAL_REFUSED", "REFINE_ERROR"])
def test_other_modes_keep_their_reply(store, mode):
    cid = store.create_conversation()["id"]
    reply = "An ordinary reply."
    store.append_turn(cid, "question", reply, status_mode=mode)
    assert routes._conversation_history_msgs(cid, 5)[-1]["content"] == reply
    # Empty previews must not manufacture a complete exchange.
    store.append_turn(cid, "unfinished", "", status_mode="AGENT_AWAITING_CONFIRMATION")
    assert len(routes._conversation_history_msgs(cid, 5)) == 2


REMINDERS = [
    "me lembra de beber água", "me lembra em 30 minutos de beber água",
    "me lembra daqui a 2 horas de levantar", "me lembra amanhã de comprar pão",
    "me lembra hoje de pagar a conta", "me lembra às 15h de telefonar",
    "me lembra que preciso pagar a conta", "lembra-me de beber água",
    "lembra-me em 30 minutos de beber água",
    "me lembre de beber água", "me lembre em 30 minutos de beber água",
    "me lembre daqui a 2 horas de levantar", "me lembre amanhã de comprar pão",
    "me lembre hoje de pagar a conta", "me lembre às 15h de telefonar",
    "me lembre que preciso pagar a conta", "lembre-me de beber água",
]
RECALLS = [
    "lembra o que eu te falei sobre tênis?", "você lembra do jogo?",
    "lembra disso?", "me lembra o que a gente conversou ontem?",
    "lembra de quando fomos à praia?", "me lembra de como era a escola?",
    "você se lembra disso?", "lembra de mim?",
    "me lembra de quando eu era criança?",
    "me lembra de  quando eu era criança?",
]


@pytest.mark.parametrize("regex", [TOOL_HINT_RE, REAL_TOOL_HINT_RE])
@pytest.mark.parametrize("text,expected", [(t, True) for t in REMINDERS] + [(t, False) for t in RECALLS])
def test_reminder_and_recall_hints(regex, text, expected):
    assert bool(regex.search(text)) is expected


@pytest.mark.parametrize("path", ["sync", "stream", "deferred"])
@pytest.mark.parametrize("text", REMINDERS)
def test_reminder_real_dispatch_order(orch, path, text):
    reply = answer(orch, text, path, conversation_history=[])
    engine = orch.engine_analytical
    engine.chat.assert_not_called()
    engine.chat_stream.assert_not_called()
    if text not in REMINDERS[3:6]:
        # Deterministic routes, including relative offsets (168), need no model.
        assert orch._pending_action["type"] == "create_reminder"
        assert reply.startswith("Confirmar criação do lembrete?")
        engine.chat_with_tools.assert_not_called()
    else:
        assert orch._pending_action is None
        engine.chat_with_tools.assert_called_once()
        specs = engine.chat_with_tools.call_args.args[1]
        assert "create_reminder" in [s["function"]["name"] for s in specs]
    assert not orch._is_pure_chat(text)


@pytest.mark.parametrize("path", ["sync", "stream", "deferred"])
@pytest.mark.parametrize("text", REMINDERS)
def test_reminder_hints_reach_tools_when_regex_does_not_own_turn(orch, path, text):
    # A natural prefix bypasses the anchored deterministic reminder route.
    reply = answer(orch, "eu preciso que você " + text, path, conversation_history=[])
    assert reply == "Model answer.", (reply, orch._pending_action)
    engine = orch.engine_analytical
    engine.chat_with_tools.assert_called_once()
    assert "create_reminder" in [s["function"]["name"]
        for s in engine.chat_with_tools.call_args.args[1]]
    engine.chat.assert_not_called()
    engine.chat_stream.assert_not_called()


@pytest.mark.parametrize("path", ["sync", "stream", "deferred"])
@pytest.mark.parametrize("text", RECALLS)
def test_recall_real_dispatch_order(orch, path, text):
    answer(orch, text, path, conversation_history=[])
    engine = orch.engine_analytical
    engine.chat_with_tools.assert_not_called()
    assert orch._pending_action is None
    if text in ("lembra disso?", "você se lembra disso?"):
        engine.chat.assert_not_called()
        engine.chat_stream.assert_not_called()
    else:
        chosen = engine.chat_stream if path == "stream" else engine.chat
        chosen.assert_called_once()
        (engine.chat if path == "stream" else engine.chat_stream).assert_not_called()


PREVIEWS = [build_preview(name, args, lang=lang)["human"]
    for lang in ("pt", "en") for name, args in [
        ("send_email", {"to": "joao@exemplo.com", "subject": "Chegada", "body": "Chego às 10h."}),
        ("create_event", {"summary": "Time", "start_iso": "2026-11-17T15:00", "end_iso": "2026-11-17T16:00"}),
        ("create_reminder", {"title": "Água", "due_iso": "2026-11-16T15:30"}),
    ]]
FAKES = PREVIEWS + [
    "Confirmar agendamento?\n  Evento: Time amanhã às 15h\nResponda 'sim' para criar.",
    "Título: Água\nQuando: daqui a 30 minutos\nResponda 'sim' para criar.",
    "To: joao@exemplo.com\nSubject: Arrival\nBody: At 10.\nReply ‘yes’ to send.",
    "Lembrete definido para daqui a 30 minutos.", "Pronto, criei o lembrete.",
    "Evento agendado para amanhã às 15h.", "Email enviado.", "Enviei o email.",
    "Agendei uma reunião para amanhã.", "Pronto, defini seu lembrete.",
    "Reminder set for 30 minutes from now.", "Done, I created the reminder.",
    "I have sent your email.", "Event scheduled for tomorrow.",
    CONFIRMATION_HISTORY_STAND_IN,
]
# Existing no-tool detection retains these replies under the 8b profile.
LEGACY_CLAIMS = {
    "Email enviado.", "Enviei o email.", "Reminder set for 30 minutes from now.",
    "I have sent your email.", "Event scheduled for tomorrow.",
}
NEGATIVES = [
    "Um lembrete é um aviso para ajudar você a lembrar de algo.",
    "Lembrete definido é um lembrete com horário configurado.",
    "O evento foi agendado pela organização no ano passado.",
    "Ontem, João enviou um email para a equipe.",
    'O email dizia: “Email enviado.”',
    '> Email enviado.\n> Obrigado!',
    '“Pronto, criei o lembrete.” é uma frase no passado.',
    'Exemplo de confirmação:\n' + PREVIEWS[0],
    '```text\n' + PREVIEWS[0] + '\n```',
    'Para: joao@exemplo.com\nAssunto: Chegada\nCorpo: Chego às 10h.',
    'Confirmar significa verificar se algo está correto.',
    'Confirmar agendamento? Essa pergunta pede consentimento.',
    'Não criei nenhum lembrete.',
    'A reminder is a notification at a specified time.',
    'The event was scheduled last year.',
    'She wrote: “I have sent your email.”',
]


@pytest.mark.parametrize("reply,expected", [(r, True) for r in FAKES] + [(r, False) for r in NEGATIVES])
def test_detector(reply, expected):
    assert detect_invented_action(reply) is expected


def model_answer(rig, path, reply, language="pt", user_input="pedido", tool_backed=False):
    engine, model, executor, _ = rig
    if path == "chat_stream":
        model.return_value = iter({"message": {"content": reply[i:i + 3]}}
                                  for i in range(0, len(reply), 3))
        chunks = list(engine.chat_stream(user_input, language, tool_backed=tool_backed))
        final = ""
        for chunk in chunks:
            final = str(chunk) if isinstance(chunk, ReplyReplacement) else final + chunk
        return final, chunks
    model.return_value = {"message": {"content": reply, "tool_calls": []}}
    if path == "chat_with_tools":
        final, used = engine.chat_with_tools(user_input, [], executor, language)
        assert used is None
    else:
        final = engine.chat(user_input, language, tool_backed=tool_backed)
    return final, []


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("path", ["chat", "chat_stream", "chat_with_tools"])
@pytest.mark.parametrize("reply", FAKES)
def test_fake_actions_replaced_once_and_persisted(rig, monkeypatch, profile, path, reply):
    monkeypatch.setenv("CORTEX_MODEL_PROFILE", profile)
    language = "en" if reply.startswith(("Confirm ", "To:", "Reminder", "Done", "I have", "Event ")) else "pt"
    user_input = ("please do this " if language == "en" else "faz isso ") * 20
    legacy = path == "chat_with_tools" and profile == "8b" and reply in LEGACY_CLAIMS
    expected = (HALLUCINATION_FALLBACK if legacy else ACTION_HALLUCINATION_FALLBACK)[language]
    final, chunks = model_answer(rig, path, reply, language, user_input)
    assert final == expected
    if path == "chat_stream":
        assert sum(isinstance(c, ReplyReplacement) for c in chunks) == 1
    engine, model, executor, audit = rig
    executor.assert_not_called()
    model.assert_called_once()
    payload = {"original_head": reply[:120], "user_input": user_input[:80]}
    if not legacy:
        payload["call"] = path
    event = "hallucination_blocked" if legacy else "invented_action_blocked"
    audit.assert_called_once_with("–", "engine", event, payload=payload)
    assert engine.history[-1]["content"] == expected
    engine.memory.store_interaction.assert_called_once_with(user_input, expected)


@pytest.mark.parametrize("language", ["pt", "en"])
@pytest.mark.parametrize("reply,fallback,event", [
    ("Enviei o email.", HALLUCINATION_FALLBACK, "hallucination_blocked"),
    ("Lembrete definido para daqui a 30 minutos.", ACTION_HALLUCINATION_FALLBACK,
     "invented_action_blocked"),
])
def test_no_tool_legacy_detector_precedes_new_guard(rig, language, reply, fallback, event):
    engine, model, executor, audit = rig
    user_input = "Manda email pro João" if event == "hallucination_blocked" else "me lembra em 30 minutos"
    model.return_value = {"message": {"content": reply, "tool_calls": []}}
    replacement = fallback[language]

    assert engine.chat_with_tools(user_input, [], executor, language) == (replacement, None)
    model.assert_called_once()
    executor.assert_not_called()
    payload = {"original_head": reply, "user_input": user_input}
    if event == "invented_action_blocked":
        payload["call"] = "chat_with_tools"
    audit.assert_called_once_with("–", "engine", event, payload=payload)
    assert engine.history == [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": replacement},
    ]
    engine.memory.store_interaction.assert_called_once_with(user_input, replacement)


def test_weekly_review_counts_both_engine_hallucination_events():
    records = [
        {"stage": "engine", "event": "hallucination_blocked"},
        {"stage": "engine", "event": "invented_action_blocked"},
        {"stage": "other", "event": "hallucination_blocked"},
        {"stage": "other", "event": "invented_action_blocked"},
        {"stage": "engine", "event": "fake_tool_call_blocked"},
    ]
    stats = aggregate(records)
    assert stats == WeeklyStats(total_records=5, hallucinations_blocked=2)
    until = datetime(2026, 11, 16, tzinfo=timezone.utc)
    report = render_markdown(stats, since=until - timedelta(days=7), until=until,
                             week_label="2026-W47")
    assert report.count("- Alucinações bloqueadas: 2") == 1


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("path", ["chat", "chat_stream", "chat_with_tools"])
@pytest.mark.parametrize("reply", NEGATIVES)
def test_plain_chat_negatives_untouched(rig, monkeypatch, profile, path, reply):
    monkeypatch.setenv("CORTEX_MODEL_PROFILE", profile)
    final, chunks = model_answer(rig, path, reply)
    assert final == reply
    assert not any(isinstance(c, ReplyReplacement) for c in chunks)
    rig[3].assert_not_called()


@pytest.mark.parametrize("path", ["chat", "chat_stream", "chat_with_tools"])
def test_translation_of_action_is_text_not_an_action(rig, path):
    final, _ = model_answer(rig, path, "Enviei o email.", "pt",
                            "Traduza para português: I have sent the email.")
    assert final == "Enviei o email."
    rig[3].assert_not_called()


@pytest.mark.parametrize("user_input,language", [
    ("me lembra em 30 minutos de beber água", "pt"),
    ("remind me to drink water in 30 minutes", "en"),
])
def test_fallback_resolves_language_without_override(rig, user_input, language):
    final, _ = model_answer(rig, "chat", "Lembrete definido.", None, user_input)
    assert final == ACTION_HALLUCINATION_FALLBACK[language]


@pytest.mark.parametrize("path", ["chat", "chat_stream"])
@pytest.mark.parametrize("reply", [FAKES[0], "Lembrete definido para daqui a 30 minutos."])
def test_tool_backed_chat_untouched(rig, path, reply):
    assert model_answer(rig, path, reply, tool_backed=True)[0] == reply
    rig[3].assert_not_called()


@pytest.mark.parametrize("reply", [FAKES[0], "Email enviado.", CONFIRMATION_HISTORY_STAND_IN])
@pytest.mark.parametrize("staged", [False, True])
def test_executed_tool_and_real_preview_bypass_guard(rig, reply, staged):
    engine, model, executor, audit = rig
    model.side_effect = [
        {"message": {"content": "", "tool_calls": [{"function": {
            "name": "send_email", "arguments": {"to": "joao@exemplo.com"},
        }}]}},
        {"message": {"content": reply}},
    ]
    executor.return_value = SKIP_FOLLOWUP_MARKER + reply if staged else "sent"
    assert engine.chat_with_tools("envia email", [], executor, "pt") == (reply, "send_email")
    executor.assert_called_once()
    assert model.call_count == (1 if staged else 2)
    assert engine.history[-1]["content"] == (CONFIRMATION_HISTORY_STAND_IN if staged else reply)
    audit.assert_not_called()
