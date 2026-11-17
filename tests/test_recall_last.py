"""Subjectless recall uses this channel's earlier user words, without the 8B."""
from unittest.mock import MagicMock

import pytest

from src.core.dispatch import registry
from src.core.dispatch.normalisation import strip_vocatives, strip_courtesy
from src.core.orchestrator import CortexOrchestrator
from src.core.redaction import redact_text
from tests.test_loss_note import orch, answer, PATHS, taught_orch  # noqa: F401


PT_FORMS = ["lembra?", "lembra"] + [
    f"{prefix}lembra {subject}{end}"
    for prefix in ("", "você ", "voce ", "se ", "você se ", "voce se ")
    for subject in ("disso", "disto", "daquilo", "dessa", "desse")
    for end in ("?", "")
    # No-question-mark bare forms already belong to REMINDER_TRIGGER_RE.
    if prefix or end
]
EN_FORMS = [f"{prefix}remember {subject}{end}"
            for prefix in ("do you ", "you ", "")
            for subject in ("this", "that") for end in ("?", "")
            if prefix or subject == "that"]


def no_model_or_retrieval(o):
    for e in (o.engine_analytical, o.engine_creative):
        e.chat.assert_not_called()
        e.chat_stream.assert_not_called()
        e.chat_with_tools.assert_not_called()
    o._needs_realtime.assert_not_called()
    assert o.engine_analytical.memory._palace.list_documents(tipo="knowledge") == []


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("text", PT_FORMS + EN_FORMS)
def test_all_forms_quote_last_user_without_model(orch, path, text):
    english = text in EN_FORMS
    sentence = "My match is on Saturday." if english else "No sábado vou jogar tênis."
    history = [
        {"role": "user", "content": "Older words."},
        {"role": "user", "content": sentence},
        {"role": "assistant", "content": "Assistant text must not be recalled."},
    ]
    orch.engine_analytical.memory.retrieve_relevant_context = MagicMock(side_effect=AssertionError("retrieval"))
    actual = answer(orch, "CoRtEx, " + text.upper(), path, conversation_history=history)
    assert actual == (f'I do: you said “{sentence}”.' if english else f'Lembro: você disse “{sentence}”.')
    no_model_or_retrieval(orch)
    orch.engine_analytical.memory.retrieve_relevant_context.assert_not_called()


@pytest.mark.parametrize("path", PATHS)
def test_skips_recall_and_empty_teaching_redacts_then_truncates(orch, path):
    sentence = "Minha senha é abcDEF987. " + "texto " * 80
    history = [{"role": "user", "content": sentence}]
    history += [{"role": "user", "content": t} for t in (
        "cortex, lembra disso?", "you remember that", "aprenda com isso", "learn this")]
    quoted = redact_text(sentence)[:299].rstrip() + "…"
    assert answer(orch, "você lembra disso?", path, conversation_history=history) == f'Lembro: você disse “{quoted}”.'
    no_model_or_retrieval(orch)


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("language", ["pt", "en"])
def test_empty_conversation_does_not_leak_global_history(orch, path, language):
    orch.engine_analytical.history = [{"role": "user", "content": "Another channel."}]
    reply = answer(orch, "lembra?", path, language=language, conversation_history=[])
    assert reply == ("Remember what? Tell me the subject." if language == "en" else "Lembrar do quê? Me diz o assunto.")
    no_model_or_retrieval(orch)


@pytest.mark.parametrize("path", PATHS)
def test_voice_uses_engine_global_history(orch, path):
    orch.engine_analytical.history = [{"role": "user", "content": "Voz anterior."}]
    assert answer(orch, "lembra?", path) == 'Lembro: você disse “Voz anterior.”.'
    no_model_or_retrieval(orch)


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("has_subject", [False, True])
def test_recall_is_literal_and_streams_once(orch, path, has_subject):
    # A pending study/crash notice must not extend the contract's literal reply.
    for name in ("_maybe_announce_pending", "_maybe_announce_abandoned"):
        setattr(orch, name, getattr(CortexOrchestrator, name).__get__(orch))
    orch.studies_review = MagicMock()
    orch.studies_review.list_pending.side_effect = AssertionError("unrelated study notice")
    orch._abandoned_effects_lock = MagicMock()
    orch._abandoned_effects_lock.__enter__.side_effect = AssertionError("unrelated crash notice")
    history = [{"role": "user", "content": "Minha frase."}] if has_subject else []
    history += [{"role": "user", "content": t} for t in ("lembra?", "aprenda com isso")]
    expected = ('I do: you said “Minha frase.”.' if has_subject
                else "Remember what? Tell me the subject.")
    if path == "sync":
        assert orch.process("lembra?", language="en", conversation_history=history) == (expected, "RECALL_LAST")
    else:
        if path == "deferred":
            orch._is_pure_chat = lambda text: False
        assert list(orch.process_stream("lembra?", language="en", conversation_history=history)) == [
            (expected, True, "RECALL_LAST", "en")]
    no_model_or_retrieval(orch)


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("text", [
    "você lembra do meu cachorro?", "do you remember my dog?",
    "me lembra disso amanhã", "lembra de comprar pão", "lembra disso",
    "lembra disto", "lembra daquilo", "lembra dessa", "lembra desse",
    "me lembra de pagar amanhã", "anota isso", "toque remember that",
    "estuda remember that", "remember that tomorrow", "please remember that",
])
def test_subjects_reminders_and_routes_keep_ownership(orch, path, text):
    route_text = strip_courtesy(strip_vocatives(text))
    routes = registry.DISPATCH_ROUTES + registry.POST_PLANNER_ROUTES
    match = next((r for r in routes if r.regex.match(route_text)), None)
    if match:
        names = {
            "REMINDER_TRIGGER_RE": "_handle_reminder_regex",
            "NOTE_CREATE_RE": "_handle_note_create",
            "STUDY_RE": "_handle_add_study",
        }
        handler = names.get(match.name)
        if handler:
            setattr(orch, handler, MagicMock(return_value=("Owned route.", "EXISTING_ROUTE")))
            assert answer(orch, text, path, conversation_history=[]) == "Owned route."
            return
    reply = answer(orch, text, path, conversation_history=[])
    assert reply not in ("Lembrar do quê? Me diz o assunto.", "Remember what? Tell me the subject.")
    assert not reply.startswith(("Lembro:", "I do:"))
