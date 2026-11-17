"""Explicit teaching: literal user text, deterministic receipt, no model I/O."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import threading
from unittest.mock import MagicMock

import pytest

from src.core.knowledge_base import KnowledgeBase
from src.core.memory import CortexMemory, unverified_speaker
from src.core.orchestrator import CortexOrchestrator
from src.memory.palace import Palace


PT = "Guardei o que você me ensinou: “{}”"
EN = "I kept what you taught me: “{}”"
SENTENCE = "deixa eu te dizer algo, aprenda com isso. Cortex veio do córtex cerebral."
TAUGHT_TITLE = "# Pedro — o que ele ensinou ao Cortex"
TAUGHT_FRAMING = (
    '(Palavras do Pedro para o Cortex: “você”, “seu”, “sua”, “teu”, “tua”, '
    '“you”, “your” = o Cortex; “eu”, “meu”, “minha”, “I”, “my” = o Pedro.)'
)
TAUGHT_NOTE = (
    "Cortex itself already handled saving the user's sentence; a fixed receipt "
    "will be added after your reply. Do not call any tool for it. Do not say "
    "you saved, stored, noted, or could not save anything. In the user's "
    "sentence, você/seu/sua/teu/tua/you/your means you, Cortex; eu/meu/minha/I/my "
    "means Pedro. Just reply briefly and naturally to what the user said."
)


@pytest.fixture
def orch(tmp_path, monkeypatch):
    # Real in-memory Palace, fake embedding and 8B; never construct a default KB.
    palace = Palace(db_path=":memory:", embed_fn=lambda text: [1.0, 0.0])
    memory = CortexMemory.__new__(CortexMemory)
    memory._palace = palace
    memory.knowledge = KnowledgeBase(palace=palace, approved_dir=str(tmp_path / "approved"))
    o = CortexOrchestrator.__new__(CortexOrchestrator)
    for name in ("engine_analytical", "engine_creative"):
        engine = MagicMock()
        engine.memory = memory
        engine.chat.return_value = "Model answer."
        engine.chat_stream.side_effect = lambda *a, **kw: iter(["Model ", "answer."])
        engine.chat_with_tools.return_value = ("Model answer.", None)
        setattr(o, name, engine)
    o.gatekeeper = MagicMock()
    o.gatekeeper.evaluate_gate.return_value.mode = "OFFLINE"
    o._pending_action = None
    o._pending_query = None
    o._pending_engine = None
    o._intent_cache = OrderedDict()
    o.skill_registry = None
    o.lease_hours = 4
    o._needs_realtime = MagicMock(return_value=False)
    o._maybe_announce_pending = lambda reply, mode: reply
    o._maybe_announce_abandoned = lambda reply, mode: reply
    for name in ("spotify", "macos", "notes", "reader", "writer", "vision"):
        setattr(o, name, MagicMock())
    monkeypatch.setenv("CORTEX_SEMANTIC_ROUTER", "0")
    yield o
    palace.close()


def saved(orch):
    return Path(orch.engine_analytical.memory.knowledge.approved_dir) / "pedro_ensinou.md"


def answer(orch, text, stream=False, language=None):
    if stream:
        frames = list(orch.process_stream(text, language=language))
        assert frames[-1][1] is True
        assert all(not frame[1] for frame in frames[:-1])
        return "".join(frame[0] for frame in frames)
    return orch.process(text, language=language)[0]


TRIGGERS = [
    "aprenda com isso", "aprende com isso", "aprenda isso", "aprende isso",
    "guarda isso", "guarde isso", "memoriza isso", "memorize isso",
    "lembre-se disso", "lembra disso", "não esquece disso", "não esqueça disso",
    "nao esqueca disso", "NAO ESQUECE DISSO", "remember this", "learn this",
    "keep this in mind",
]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("trigger", TRIGGERS)
@pytest.mark.parametrize("position,separator", [("start", ":"), ("middle", ","), ("end", ""), ("start", ".")])
def test_every_trigger_keeps_literal_text(orch, stream, trigger, position, separator):
    fact = "Cortex came from the cerebral cortex"
    sentence = {"start": f"{trigger}{separator} {fact}",
                "middle": f"Pedro said, {trigger}{separator} {fact}",
                "end": f"{fact}, {trigger}{separator}"}[position]
    english = trigger in ("remember this", "learn this", "keep this in mind")
    # These exact start forms already belong to reminders / WRITE_RE. The
    # contract's route-precedence rule takes priority over teaching here.
    if position == "start" and (trigger == "lembra disso" or
                               (separator == "." and trigger in ("guarda isso", "guarde isso"))):
        handler = "_handle_reminder_regex" if trigger == "lembra disso" else "_handle_write"
        setattr(orch, handler, MagicMock(return_value=("Owned route.", "EXISTING_ROUTE")))
        assert answer(orch, "CoRtEx, " + sentence, stream) == "Owned route."
        assert not saved(orch).exists()
        return
    reply = answer(orch, "  CoRtEx, " + sentence + "  ", stream)
    assert saved(orch).read_text().endswith(sentence + "\n")
    assert reply == "Model answer.\n\n" + (EN if english else PT).format(sentence)


@pytest.mark.parametrize("text", [
    "você lembra disso?", "aprenda com isso: Cortex veio do cérebro?",
    "não precisa guardar isso", "não aprenda isso: uma ideia",
    "não memorize isso: uma ideia", "nao guarde isso: uma ideia",
    "don't remember this: a thought", "do not learn this: a thought",
    "I cannot remember this: a thought", "I can't keep this in mind: a thought",
    "não precisa dizer aprenda com isso", "anota isso", "anota que Cortex veio do cérebro",
])
@pytest.mark.parametrize("stream", [False, True])
def test_questions_negations_and_notes_do_not_keep(orch, text, stream):
    answer(orch, text, stream)
    assert not saved(orch).exists()


@pytest.mark.parametrize("text,handler,mode", [
    ("anota que aprenda com isso: o nome veio do cérebro", "_handle_note_create", "NOTE_CREATE"),
    ("toque learn this", None, "SPOTIFY_PLAY"),
    ("estuda learn this", "_handle_add_study", "STUDIER_ADD"),
    ("cria uma nota chamada learn this", "_handle_note_create", "NOTE_CREATE"),
])
@pytest.mark.parametrize("stream", [False, True])
def test_existing_routes_own_their_phrases(orch, text, handler, mode, stream):
    if handler:
        setattr(orch, handler, MagicMock(return_value=("Owned route.", mode)))
    else:
        orch.spotify.play_anything.return_value = "Owned route."
    if stream:
        frames = list(orch.process_stream(text))
        assert frames[-1][2] == mode
        reply = "".join(f[0] for f in frames)
    else:
        reply, actual_mode = orch.process(text)
        assert actual_mode == mode
    assert reply == "Owned route."
    assert not saved(orch).exists()


@pytest.mark.parametrize("text,handler", [
    ("creative learn this: the name came from the brain", None),
    ("estuda learn this", "_handle_add_study"),
    ("planeja: learn this", "_handle_planner_request"),
])
@pytest.mark.parametrize("stream", [False, True])
def test_owned_phrases_still_do_not_teach_when_handler_falls_through(orch, text, handler, stream):
    if handler:
        setattr(orch, handler, MagicMock(return_value=None))
    assert answer(orch, text, stream) == "Model answer."
    assert not saved(orch).exists()


@pytest.mark.parametrize("stream", [False, True])
def test_explicit_search_keeps_its_existing_route(orch, stream):
    orch.gatekeeper.evaluate_gate.return_value.mode = "ONLINE_REQUEST"
    orch._do_web_search = MagicMock(return_value=("Existing search.", "ONLINE_RAG"))
    assert answer(orch, "search learn this: the name Cortex", stream) == "Existing search."
    assert not saved(orch).exists()


@pytest.mark.parametrize("wake", ["Cortex ", "CORTEX, ", "cortex: ", "Cortex. ", "Cortex — "])
def test_only_leading_wake_word_is_removed(orch, wake):
    sentence = "aprenda isso:  Pedro disse Cortex, e Cortex ficou."
    assert answer(orch, "  " + wake + sentence + "  ").endswith(PT.format(sentence))
    assert saved(orch).read_text().endswith(sentence + "\n")


def test_atomic_shape_duplicates_index_and_reindex(orch, monkeypatch):
    import os
    kb = orch.engine_analytical.memory.knowledge
    index = MagicMock(wraps=kb.index_file)
    monkeypatch.setattr(kb, "index_file", index)
    replace = os.replace
    replacements = []

    def observe(source, destination):
        assert Path(source).parent == Path(destination).parent
        assert Path(source).read_text().startswith(TAUGHT_TITLE + "\n")
        replacements.append((source, destination))
        replace(source, destination)

    monkeypatch.setattr(os, "replace", observe)
    assert PT.format(SENTENCE) in answer(orch, "Cortex " + SENTENCE)
    first = saved(orch).read_bytes()
    assert answer(orch, "cortex, " + SENTENCE).endswith("Isso eu já tinha guardado.")
    assert saved(orch).read_bytes() == first
    second = "remember this: Pedro chose the name Cortex."
    answer(orch, second)
    content = saved(orch).read_text()
    assert content.startswith(TAUGHT_TITLE + "\n\n## ")
    dates = [line[3:] for line in content.splitlines() if line.startswith("## ")]
    assert len(dates) == 2
    assert all(datetime.fromisoformat(date).utcoffset().total_seconds() == 0 for date in dates)
    assert index.call_count == len(replacements) == 2
    assert set(saved(orch).parent.iterdir()) == {saved(orch)}
    assert kb.reindex_all() == 1
    chunks = kb._palace.list_documents(tipo="knowledge")
    assert len(chunks) == 2
    assert all(not kb._STUDY_CHUNK_RE.match(c["texto"]) for c in chunks)
    ctx = orch.engine_analytical.memory.retrieve_relevant_context("Cortex name")
    assert SENTENCE in ctx and "[KNOWLEDGE]" in ctx and "[STUDY]" not in ctx


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("text,language,receipt", [
    ("aprenda com isso", None, "O que eu devo guardar? Diga, por exemplo: aprenda com isso: <o fato>."),
    ("cortex, deixa eu te dizer algo, aprenda com isso.", None, "O que eu devo guardar? Diga, por exemplo: aprenda com isso: <o fato>."),
    ("learn this", None, "What should I keep? Say, for example: learn this: <the fact>."),
    ("aprenda com isso", "en", "What should I keep? Say, for example: learn this: <the fact>."),
    ("aprenda isso: " + "a" * 1000, None, "É grande demais para eu guardar de uma vez — mande em partes menores."),
    ("learn this: " + "a" * 1000, None, "That's too long to keep at once — send it in smaller parts."),
    ("aprenda isso: minha senha é abc.DEF987", None, "Isso parece ter um dado sensível, então não guardei."),
    ("learn this: my password is abc.DEF987", None, "That looks like it has sensitive data, so I didn't keep it."),
])
def test_rejected_teaching(orch, stream, text, language, receipt):
    reply = answer(orch, text, stream, language)
    assert reply.endswith(receipt)
    assert not saved(orch).exists()
    if "<" in receipt:
        assert reply == receipt
        orch.engine_analytical.chat.assert_not_called()
        orch.engine_analytical.chat_stream.assert_not_called()
        orch.engine_analytical.chat_with_tools.assert_not_called()
        orch._needs_realtime.assert_not_called()


@pytest.mark.parametrize("stream,pure", [(False, True), (True, True), (True, False)])
def test_unverified_speaker_never_teaches(orch, stream, pure):
    if not pure:
        orch._is_pure_chat = lambda text: False
    with unverified_speaker():
        reply = answer(orch, SENTENCE, stream)
    assert reply == "Model answer."
    assert not saved(orch).exists()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("language", ["pt", "en"])
@pytest.mark.parametrize("failure", ["write", "index_false", "index_raise"])
def test_failures_have_truthful_receipts(orch, monkeypatch, stream, language, failure):
    import os
    kb = orch.engine_analytical.memory.knowledge
    if failure == "write":
        monkeypatch.setattr(os, "replace", MagicMock(side_effect=OSError("disk full")))
        receipt = {"pt": "Não consegui guardar — nada mudou.", "en": "I couldn't keep it — nothing changed."}[language]
    else:
        monkeypatch.setattr(kb, "index_file", MagicMock(return_value=False, side_effect=RuntimeError("index failed") if failure == "index_raise" else None))
        receipt = {"pt": "Guardei no arquivo, mas não consegui indexar agora.", "en": "I saved it to the file, but couldn't index it right now."}[language]
    reply = answer(orch, SENTENCE, stream, language)
    assert reply == "Model answer.\n\n" + receipt
    assert saved(orch).exists() == (failure != "write")
    assert not kb._palace.list_documents(tipo="knowledge")
    assert list(saved(orch).parent.glob("*.tmp")) == []


@pytest.mark.parametrize("stream,pure", [(False, True), (True, True), (True, False)])
def test_model_failure_still_sends_confirmation_last(orch, stream, pure):
    orch.engine_analytical.chat.side_effect = RuntimeError("fake 8B down")
    orch.engine_analytical.chat_stream.side_effect = RuntimeError("fake 8B down")
    if not pure:
        orch._is_pure_chat = lambda text: False
    reply = answer(orch, SENTENCE, stream)
    assert reply.endswith("\n\n" + PT.format(SENTENCE))
    assert "fake 8B down" not in reply


@pytest.mark.parametrize("pure", [False, True])
def test_stream_receipt_is_its_own_last_chunk(orch, pure):
    if not pure:
        orch._is_pure_chat = lambda text: False
    frames = list(orch.process_stream("cortex, " + SENTENCE))
    assert "".join(frame[0] for frame in frames[:-1]) == "Model answer."
    assert frames[-1] == ("\n\n" + PT.format(SENTENCE), True, "OFFLINE", "pt")


def test_confirmation_follows_existing_announcements(orch):
    orch._maybe_announce_pending = lambda reply, mode: reply + "\n\nPending study."
    orch._maybe_announce_abandoned = lambda reply, mode: reply + "\n\nAbandoned action."
    reply = answer(orch, SENTENCE)
    assert reply == "Model answer.\n\nPending study.\n\nAbandoned action.\n\n" + PT.format(SENTENCE)


def test_multiline_markdown_is_one_sentence_chunk(orch):
    sentence = ("aprenda isso: linhas exatas\n## um título que Pedro disse\n"
                "## 2026-10-29T12:00:00.123456+00:00\n\n"
                "```\n# Topic: literal\nFim.")
    answer(orch, sentence)
    docs = orch.engine_analytical.memory._palace.list_documents(tipo="knowledge")
    assert len(docs) == 1
    assert sentence in docs[0]["texto"]
    assert not KnowledgeBase.is_study_chunk(docs[0]["texto"])
    assert answer(orch, sentence).endswith("Isso eu já tinha guardado.")


def test_internal_whitespace_and_newlines_are_literal(orch):
    sentence = "learn this: Two  spaces, café.\r\nAnother\tline."
    assert answer(orch, "Cortex: " + sentence, language="en").endswith(EN.format(sentence))
    assert saved(orch).read_bytes().endswith(sentence.encode() + b"\n")
    assert answer(orch, sentence, language="en").endswith("I had already kept that.")


@pytest.mark.parametrize("failure", ["write", "replace"])
def test_failed_append_preserves_previous_file(orch, monkeypatch, failure):
    import os
    import tempfile
    answer(orch, SENTENCE)
    before = saved(orch).read_bytes()
    kb = orch.engine_analytical.memory.knowledge
    index = MagicMock(wraps=kb.index_file)
    monkeypatch.setattr(kb, "index_file", index)
    if failure == "write":
        monkeypatch.setattr(tempfile, "mkstemp", MagicMock(side_effect=OSError("disk full")))
    else:
        monkeypatch.setattr(os, "replace", MagicMock(side_effect=OSError("disk full")))
    reply = answer(orch, "learn this: Another fact.")
    assert reply.endswith("I couldn't keep it — nothing changed.")
    assert saved(orch).read_bytes() == before
    index.assert_not_called()
    assert set(saved(orch).parent.iterdir()) == {saved(orch)}


def test_keep_is_audited(orch):
    import json
    from src.core.audit import audit
    answer(orch, SENTENCE)
    events = [json.loads(line) for line in Path(audit.path).read_text().splitlines()]
    assert any(event["stage"] == "memory" and event["payload"].get("status") == "kept"
               for event in events)


def test_partial_model_failure_still_finishes_with_receipt(orch):
    def broken(*args, **kwargs):
        yield "A partial answer."
        raise RuntimeError("fake stream failure")
    orch.engine_analytical.chat_stream.side_effect = broken
    frames = list(orch.process_stream(SENTENCE))
    assert frames[0][0] == "A partial answer."
    assert frames[-1] == ("\n\n" + PT.format(SENTENCE), True, "INTERNAL_ERROR", "pt")


def test_concurrent_teaching_does_not_lose_sentences(orch):
    from src.core.taught_facts import keep_taught_fact
    kb = orch.engine_analytical.memory.knowledge
    sentences = [f"learn this: fact number {i}" for i in range(8)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda text: keep_taught_fact(text, kb, "en"), sentences))
    assert all(result.status == "kept" for result in results)
    content = saved(orch).read_text()
    assert all(content.count(text + "\n") == 1 for text in sentences)
    assert len(kb._palace.list_documents(tipo="knowledge")) == len(sentences)


def test_indexed_and_retrieved_teaching_has_pedros_point_of_view(orch):
    sentences = [
        "aprenda com isso: O seu nome veio de várias inspirações, e uma delas "
        "foi do personagem da Marvel, professor Xavier.",
        "learn this: your name is Cortex, and my name is Pedro.",
    ]
    for sentence in sentences:
        answer(orch, sentence)
    memory = orch.engine_analytical.memory
    for chunk in memory._palace.list_documents(tipo="knowledge"):
        text = chunk["texto"]
        assert text.startswith(TAUGHT_TITLE + "\n\n")
        assert any(text.endswith(TAUGHT_FRAMING + "\n\n" + s) for s in sentences)
    context = memory.retrieve_relevant_context("de onde veio o seu nome?")
    assert "[KNOWLEDGE]" in context
    assert TAUGHT_TITLE in context and TAUGHT_FRAMING in context
    assert sentences[0] in context


def test_framed_file_round_trips_literal_crlf_and_markdown(orch):
    from src.core.taught_facts import taught_chunks, taught_sections

    sentences = [
        "aprenda isso: seu nome é Cortex.\r\n## Um título\r\n"
        "<!-- cortex:taught-chars=5 -->\r\n\r\n```md\r\n# literal\r\n```",
        "learn this: my dog is Bolinha.\n"
        "## 2026-10-29T12:00:00.123456+00:00\n\n" + TAUGHT_FRAMING,
    ]
    for sentence in sentences:
        answer(orch, sentence)
    content = saved(orch).read_bytes().decode("utf-8")
    assert content.startswith(TAUGHT_TITLE + "\n")
    sections = taught_sections(content)
    assert [sentence for _, sentence in sections] == sentences
    for (heading, sentence), chunk in zip(sections, taught_chunks(content)):
        assert (heading + "\n") in content
        assert TAUGHT_FRAMING + "\n\n" + sentence + "\n" in content
        assert chunk == f"{TAUGHT_TITLE}\n\n{heading}\n\n{TAUGHT_FRAMING}\n\n{sentence}"
    before = saved(orch).read_bytes()
    assert answer(orch, sentences[0]).endswith("Isso eu já tinha guardado.")
    assert saved(orch).read_bytes() == before
    kb = orch.engine_analytical.memory.knowledge
    assert kb.reindex_all() == 1
    assert {d["texto"] for d in kb._palace.list_documents(tipo="knowledge")} == set(taught_chunks(content))


@pytest.mark.parametrize("title", ["# Pedro — o que ele me ensinou", "# Unexpected title"])
def test_unexpected_teaching_title_still_refuses_write(orch, title):
    from src.core.taught_facts import taught_sections

    saved(orch).write_text(title + "\n", encoding="utf-8")
    before = saved(orch).read_bytes()
    with pytest.raises(ValueError, match="Unexpected teaching file title"):
        taught_sections(before.decode("utf-8"))
    assert answer(orch, SENTENCE).endswith("Não consegui guardar — nada mudou.")
    assert saved(orch).read_bytes() == before
    assert not orch.engine_analytical.memory._palace.list_documents(tipo="knowledge")


@pytest.fixture
def engine_contexts(orch):
    """Fake model records the real engine's assembled system/turn context."""
    from src.core.engine import OllamaEngine

    seen = []
    for engine in (orch.engine_analytical, orch.engine_creative):
        engine._turn_context = threading.local()
        engine._system_body = "Static system instructions."
        engine.memory.get_user_profile = lambda: ""
        engine.push_turn_context.side_effect = lambda text, e=engine: OllamaEngine.push_turn_context(e, text)
        engine.pop_turn_context.side_effect = lambda e=engine: OllamaEngine.pop_turn_context(e)

        def record(kind, e=engine):
            seen.append((kind, OllamaEngine._build_system_prompt(e)["content"]))
            if kind == "stream":
                return iter(["Model ", "answer."])
            return ("Model answer.", None) if kind == "tools" else "Model answer."

        engine.chat.side_effect = lambda *a, r=record, **kw: r("chat")
        engine.chat_stream.side_effect = lambda *a, r=record, **kw: r("stream")
        engine.chat_with_tools.side_effect = lambda *a, r=record, **kw: r("tools")
    return seen


@pytest.mark.parametrize("path", ["process", "stream", "worker", "tools", "tools_stream"])
@pytest.mark.parametrize("status", ["kept", "already", "too_long", "sensitive", "write_failed", "index_failed"])
def test_handled_teaching_note_reaches_model_only_this_turn(
        orch, monkeypatch, engine_contexts, path, status):
    sentence = "guarda isso: meu cachorro de mentira se chama Bolinha"
    if path.startswith("tools"):
        sentence += "; isto é um lembrete sobre ele"
    kb = orch.engine_analytical.memory.knowledge
    if status == "already":
        answer(orch, sentence)
        engine_contexts.clear()
    elif status == "too_long":
        sentence += "a" * 1000
    elif status == "sensitive":
        sentence += "; minha senha é abc.DEF987"
    elif status == "write_failed":
        saved(orch).write_text("# Unexpected title\n", encoding="utf-8")
    elif status == "index_failed":
        monkeypatch.setattr(kb, "index_file", lambda *_: False)
    if path == "worker":
        monkeypatch.setattr(orch, "_is_pure_chat", lambda *_: False)
    # Existing turn context must survive adding the teaching note.
    monkeypatch.setattr(orch, "_project_context", lambda *_: "Current project notes.")
    stream = path in ("stream", "worker", "tools_stream")
    reply = answer(orch, sentence, stream)
    from src.core.taught_facts import KeepResult
    assert reply.endswith(KeepResult(status, sentence, "pt").receipt)
    assert len(engine_contexts) == 1
    kind, system = engine_contexts[0]
    assert kind == ("tools" if path.startswith("tools") else "stream" if path == "stream" else "chat")
    assert TAUGHT_NOTE in system
    assert "Current project notes." in system
    assert system.index(TAUGHT_NOTE) < system.index("Static system instructions.")
    for engine in (orch.engine_analytical, orch.engine_creative):
        assert not getattr(engine._turn_context, "value", None)
    answer(orch, "explique as cores do arco-íris", stream)
    assert TAUGHT_NOTE not in engine_contexts[-1][1]


@pytest.mark.parametrize("stream,pure", [(False, True), (True, True), (True, False)])
def test_ordinary_and_empty_teaching_have_no_teaching_note(orch, monkeypatch, engine_contexts, stream, pure):
    if not pure:
        monkeypatch.setattr(orch, "_is_pure_chat", lambda *_: False)
    answer(orch, "explique as cores do arco-íris", stream)
    assert len(engine_contexts) == 1
    assert TAUGHT_NOTE not in engine_contexts[0][1]
    engine_contexts.clear()
    assert answer(orch, "aprenda com isso", stream).startswith("O que eu devo guardar?")
    assert engine_contexts == []
    for engine in (orch.engine_analytical, orch.engine_creative):
        assert not getattr(engine._turn_context, "value", None)
