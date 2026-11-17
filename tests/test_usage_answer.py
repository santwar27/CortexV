"""141a: saved UI turn counts, deterministic routing and provenance boundaries."""
import json
from unittest.mock import MagicMock

import pytest

from tests.test_taught_facts import orch  # noqa: F401
from tests.test_ui_routes import client, ui_app  # noqa: F401


PT_SCOPE = "A contagem cobre apenas as chamadas ao modelo registradas pelo código."
EN_SCOPE = "The count covers only model calls recorded by the code."
PT = "A última resposta usou 2.913 tokens de entrada e 148 de saída, em 2 chamadas ao modelo cortex-lite.\n" + PT_SCOPE
EN = "The last answer used 2,913 input tokens and 148 output tokens, in 2 calls to model cortex-lite.\n" + EN_SCOPE
PT_UNKNOWN = "Não tenho a contagem da última resposta."
EN_UNKNOWN = "I don't have the count for the last answer."
QUESTION = "quantos tokens você gasta para responder essa pergunta?"
USAGE = {"prompt_eval_count": 2913, "eval_count": 148, "calls": 2,
         "models": ["cortex-lite"], "online": []}
POSITIVES = [
    (QUESTION, "pt"), ("quantos tokens você gastou?", "pt"),
    ("quantos tokens voce usou?", "pt"), ("quantos tokens você gasta?", "pt"),
    ("quantos tokens essa resposta usou?", "pt"), ("quantos tokens foram?", "pt"),
    ("quantos tokens você gastou nessa resposta?", "pt"),
    ("Cortex, por favor, QUANTOS TOKENS VOCÊ USOU NA ÚLTIMA RESPOSTA?", "pt"),
    ("how many tokens did you use?", "en"),
    ("how many tokens did that answer use?", "en"),
    ("how many tokens do you spend?", "en"),
    ("how many tokens did your last answer use?", "en"),
    ("Please, how many input/output tokens did you use?", "en"),
]
NEGATIVES = [
    "o que é um token?", "qual o token da API?", "explain tokens", "gera um token",
    "quantos tokens cabem na janela?", "token do GitHub",
    "manda um email sobre tokens", "send an email mentioning tokens",
    "quantos tokens você gastou e manda por email",
    "how many tokens did you use and send it by email",
    "quantos tokens você usa na janela de contexto?",
    'Explique a frase "quantos tokens você gastou?"',
    '"quantos tokens você gastou?"', "“how many tokens did you use?”",
    "> how many tokens did you use?", "`quantos tokens foram?`",
    "````\nhow many tokens did you use?\n````",
]


@pytest.fixture
def usage_orch(orch, monkeypatch):
    import ollama
    from src.ui.chat_store import store

    # Fail closed on accidental raw model calls, including the classifier.
    for name in ("chat", "generate"):
        monkeypatch.setattr(ollama, name, MagicMock(side_effect=AssertionError("no Ollama")))
    for engine in (orch.engine_analytical, orch.engine_creative):
        engine.max_history = 10
    cid = store.create_conversation()["id"]
    orch.set_turn_origin("ui", cid)
    yield orch, store, cid
    orch.set_turn_origin(None)


def save(store, cid, usage=USAGE, mode="OFFLINE"):
    return store.append_turn(conversation_id=cid, user="previous question",
                             reply="previous answer", status_mode=mode, usage=usage)


def run(orch, text, stream=False, **kwargs):
    if stream:
        frames = list(orch.process_stream(text, **kwargs))
        assert frames[-1][1] is True
        return "".join(f[0] for f in frames), frames[-1][2]
    return orch.process(text, **kwargs)


def assert_no_model(orch):
    import ollama

    ollama.chat.assert_not_called()
    ollama.generate.assert_not_called()
    orch._needs_realtime.assert_not_called()
    for engine in (orch.engine_analytical, orch.engine_creative):
        for name in ("chat", "chat_stream", "chat_with_tools"):
            getattr(engine, name).assert_not_called()


@pytest.mark.parametrize("text,lang", POSITIVES)
@pytest.mark.parametrize("stream", [False, True])
def test_exact_saved_counts_without_model(usage_orch, text, lang, stream):
    from src.core import turn_metrics

    o, store, cid = usage_orch
    save(store, cid)
    # An open current bucket must never be the referent.
    turn_metrics.reset()
    turn_metrics.add(999, 888, "wrong-current-model")
    try:
        assert run(o, text, stream) == (PT if lang == "pt" else EN, "USAGE_PREVIOUS")
        assert turn_metrics.current()["calls"] == 1
        assert_no_model(o)
    finally:
        turn_metrics.collect()


@pytest.mark.parametrize("text", NEGATIVES)
def test_detector_declines_non_usage_and_mixed_requests(text):
    from src.core.usage_answer import detect_usage_question

    assert detect_usage_question(text) is None


@pytest.mark.parametrize("text", NEGATIVES)
@pytest.mark.parametrize("stream", [False, True])
def test_negative_routing_unchanged(usage_orch, text, stream):
    o, store, cid = usage_orch
    save(store, cid)
    reply, mode = run(o, text, stream)
    assert mode == "OFFLINE"
    assert reply == "Model answer."
    assert any(getattr(engine, name).called
               for engine in (o.engine_analytical, o.engine_creative)
               for name in ("chat", "chat_stream", "chat_with_tools"))


@pytest.mark.parametrize("stream", [False, True])
def test_conversation_isolation_and_singular(usage_orch, stream):
    o, store, first = usage_orch
    second = store.create_conversation()["id"]
    save(store, first)
    save(store, second, {**USAGE, "prompt_eval_count": 1, "eval_count": 1, "calls": 1})
    for cid, expected in [(first, PT), (second,
            "A última resposta usou 1 token de entrada e 1 de saída, em 1 chamada ao modelo cortex-lite.\n" + PT_SCOPE)]:
        o.set_turn_origin("ui", cid)
        assert run(o, QUESTION, stream) == (expected, "USAGE_PREVIOUS")
    assert_no_model(o)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("usage,kind", [
    ({**USAGE, "calls": 0, "prompt_eval_count": 0, "eval_count": 0, "models": []}, "zero"),
    (None, "unknown"), ({}, "unknown"),
    ({"calls": 0}, "unknown"), ({**USAGE, "calls": 0}, "unknown"),
])
def test_latest_turn_is_not_skipped(usage_orch, stream, lang, usage, kind):
    o, store, cid = usage_orch
    save(store, cid)
    save(store, cid, usage, mode="OFFLINE")
    expected = ({"pt": "A última resposta foi dada pelo código, sem usar o modelo (0 tokens).",
                 "en": "The last answer was given by code, without using the model (0 tokens)."}
                if kind == "zero" else {"pt": PT_UNKNOWN, "en": EN_UNKNOWN})[lang]
    question = QUESTION if lang == "pt" else "how many tokens did you use?"
    assert run(o, question, stream) == (expected, "USAGE_PREVIOUS")
    assert_no_model(o)


@pytest.mark.parametrize("mode", ["USAGE_PREVIOUS", "SELF_MODEL", "SELF_PLAN", "SELF_AVATAR"])
@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("stream", [False, True])
def test_code_only_previous_turn_without_usage_is_zero(usage_orch, mode, lang, stream):
    o, store, cid = usage_orch
    save(store, cid, usage=None, mode=mode)
    question = QUESTION if lang == "pt" else "how many tokens did you use?"
    zero = ("A última resposta foi dada pelo código, sem usar o modelo (0 tokens)."
            if lang == "pt" else
            "The last answer was given by code, without using the model (0 tokens).")
    assert run(o, question, stream) == (zero, "USAGE_PREVIOUS")
    assert_no_model(o)


@pytest.mark.parametrize("mode", ["OFFLINE", "SELF_OTHER", "ATTACHMENT"])
@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("stream", [False, True])
def test_other_previous_mode_without_usage_stays_unknown(usage_orch, mode, lang, stream):
    o, store, cid = usage_orch
    save(store, cid, usage=None, mode=mode)
    question = QUESTION if lang == "pt" else "how many tokens did you use?"
    assert run(o, question, stream) == (
        PT_UNKNOWN if lang == "pt" else EN_UNKNOWN, "USAGE_PREVIOUS")
    assert_no_model(o)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang", ["pt", "en"])
def test_no_previous_turn(usage_orch, stream, lang):
    o, store, cid = usage_orch
    other = store.create_conversation()["id"]
    save(store, other)
    question = QUESTION if lang == "pt" else "how many tokens did you use?"
    expected = ("Ainda não há uma resposta anterior nesta conversa." if lang == "pt"
                else "There is no previous answer in this conversation yet.")
    assert run(o, question, stream) == (expected, "USAGE_PREVIOUS")
    assert_no_model(o)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("origin", [None, "terminal", "voice", "ui"])
@pytest.mark.parametrize("lang", ["pt", "en"])
def test_channel_without_ui_conversation(usage_orch, stream, origin, lang):
    o, store, cid = usage_orch
    save(store, cid)
    o.set_turn_origin(origin)
    expected = (("Ainda não há uma resposta anterior nesta conversa." if lang == "pt"
                 else "There is no previous answer in this conversation yet.")
                if origin in {"terminal", "voice"} else
                ("Não consigo identificar a conversa deste turno." if lang == "pt"
                 else "I can't identify this turn's conversation."))
    assert run(o, QUESTION if lang == "pt" else "how many tokens do you spend?", stream) == (
        expected, "USAGE_PREVIOUS")
    assert_no_model(o)


@pytest.mark.parametrize("lang", ["pt", "en"])
def test_multiple_models_and_english_singular(usage_orch, lang):
    o, store, cid = usage_orch
    save(store, cid, {**USAGE, "models": ["cortex-criativo", "cortex-lite"]})
    expected = ("A última resposta usou 2.913 tokens de entrada e 148 de saída, em 2 chamadas nos modelos cortex-criativo e cortex-lite.\n" + PT_SCOPE
                if lang == "pt" else
                "The last answer used 2,913 input tokens and 148 output tokens, in 2 calls to models cortex-criativo and cortex-lite.\n" + EN_SCOPE)
    assert run(o, QUESTION if lang == "pt" else "how many tokens did you use?")[0] == expected
    save(store, cid, {**USAGE, "calls": 1, "prompt_eval_count": 1, "eval_count": 1})
    assert run(o, "how many tokens did you use?")[0] == (
        "The last answer used 1 input token and 1 output token, in 1 call to model cortex-lite.\n" + EN_SCOPE)
    assert_no_model(o)


@pytest.mark.parametrize("stream", [False, True])
def test_attachment_and_retrieved_text_are_not_user_intent(usage_orch, stream):
    from src.core.prompt_safety import fence

    o, store, cid = usage_orch
    save(store, cid)
    o._answer_attachment = MagicMock(return_value=("Attachment answer.", "ATTACHMENT"))
    for label in ("PERGUNTA", "USER QUERY"):
        text = fence(QUESTION) + f"\n{label}: summarise this"
        assert run(o, text, stream) == ("Attachment answer.", "ATTACHMENT")


@pytest.mark.parametrize("stream", [False, True])
def test_pending_and_command_owners_win(usage_orch, stream):
    o, store, cid = usage_orch
    save(store, cid)
    o._answer_pdf_split_question = MagicMock(return_value=("Pending owner.", "PDF_OWNER"))
    assert run(o, QUESTION, stream) == ("Pending owner.", "PDF_OWNER")
    o._answer_pdf_split_question.return_value = None
    o._handle_send_email_structured = MagicMock(return_value=("Email owner.", "EMAIL_OWNER"))
    assert run(o, "manda email para test@example.com, assunto: tokens, corpo: " + QUESTION,
               stream) == ("Email owner.", "EMAIL_OWNER")
    assert_no_model(o)


def test_store_read_failure_is_unknown(usage_orch, monkeypatch):
    o, store, _ = usage_orch
    monkeypatch.setattr(store, "get_turns", MagicMock(side_effect=OSError("unavailable")))
    assert run(o, QUESTION) == (PT_UNKNOWN, "USAGE_PREVIOUS")
    assert_no_model(o)


@pytest.mark.parametrize("stream", [False, True])
def test_fixed_reply_has_no_announcements_and_uses_question_language(usage_orch, stream):
    o, store, cid = usage_orch
    save(store, cid)
    # Exercise the real announcement guards instead of the fixture's stubs.
    del o._maybe_announce_pending
    del o._maybe_announce_abandoned
    if stream:
        assert list(o.process_stream("how many tokens did you use?", language="pt")) == [
            (EN, True, "USAGE_PREVIOUS", "en")]
    else:
        assert run(o, "how many tokens did you use?", language="pt") == (EN, "USAGE_PREVIOUS")
    assert_no_model(o)


@pytest.mark.parametrize("lang", ["pt", "en"])
def test_real_ui_saves_answer_and_next_question_references_it(
        usage_orch, ui_app, client, monkeypatch, lang):
    from src.ui import routes

    o, store, cid = usage_orch
    monkeypatch.setattr(routes, "chat_store", store)
    ui_app.state.orchestrator = o
    save(store, cid)
    question = QUESTION if lang == "pt" else "how many tokens did you use?"
    zero = ("A última resposta foi dada pelo código, sem usar o modelo (0 tokens)."
            if lang == "pt" else
            "The last answer was given by code, without using the model (0 tokens).")
    for expected in (PT if lang == "pt" else EN, zero):
        response = client.post("/ui/chat/stream", headers={"X-Cortex-Key": "test-key-xyz"},
                               json={"text": question, "conversation_id": cid})
        assert response.status_code == 200
        frames = [json.loads(line[6:]) for line in response.text.splitlines()
                  if line.startswith("data: ")]
        assert frames[-1]["turn"]["reply"] == expected
        last = store.get_turns(cid)[-1]
        assert last["reply"] == expected
        assert last["status_mode"] == "USAGE_PREVIOUS"
        assert last["usage"] is None
    assert_no_model(o)
