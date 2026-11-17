"""
process_stream — the streaming entry point the voice pipeline uses.

Regression coverage for the bug where the reflex layer was bypassed on
the streaming path: the voice client only ever calls /ask_stream ->
process_stream, whose _is_pure_chat shortcut routed trivial queries
("que horas são", "oi") straight to the LLM instead of the reflex. The
LLM then hallucinated ("não posso fornecer a hora atual"). The reflex
must run at the top of process_stream too, guarded by pending state.

s62 (C8): the tuple grew a 4th element — the turn's reply language —
so the SSE layer can announce it and the voice listener stops keeping a
hand-maintained PT word list to re-derive what the server already knows
(M5-A4/M10-A3). Each of the three paths (reflex shortcut, pure-chat
stream, deferral to process()) must carry it. The engine-error final
must be a SPEAKABLE localised sentence, never the raw exception
("[ENGINE ERROR: ...]" would be read aloud by the TTS — M10-A5).

No real LLM/Ollama — the engine is a MagicMock; a reflex hit must never
touch it.
"""
from unittest.mock import MagicMock

import threading

import pytest


class TestProcessStreamReflex:
    @pytest.fixture
    def orch(self):
        from src.core.orchestrator import CortexOrchestrator
        o = CortexOrchestrator.__new__(CortexOrchestrator)
        o.engine_analytical = MagicMock()
        o.gatekeeper = MagicMock()
        o._pending_action = None
        o._pending_query = None
        return o

    @pytest.mark.parametrize("text", [
        "que horas são",
        "Que horas são?",
        "que horas são agora",
        "me diz as horas",
        "que horas que são",
        "oi",
        "que dia é hoje",
    ])
    def test_reflex_short_circuits_before_llm(self, orch, text):
        chunks = list(orch.process_stream(text))
        assert len(chunks) == 1, f"expected single emission for {text!r}"
        reply, is_final, mode, lang = chunks[0]
        assert is_final is True
        assert mode == "REFLEX", f"{text!r} should be REFLEX, got {mode}"
        assert reply.strip(), "reflex reply must be non-empty"
        assert lang == "pt", "PT reflex reply must report lang=pt"
        # The whole point: a reflex hit must not hit the engine.
        orch.engine_analytical.chat_stream.assert_not_called()

    def test_reflex_skipped_when_pending_action(self, orch):
        # A staged action (awaiting sim/não) takes precedence — a
        # reflex-looking word must flow through the full pipeline, not
        # short-circuit. process() is patched to stand in for it.
        orch._pending_action = {"some": "action"}
        with pytest.MonkeyPatch().context() as mp:
            orch.process = MagicMock(return_value=("staged reply", "AWAITING_APPROVAL"))
            chunks = list(orch.process_stream("oi"))
        assert len(chunks) == 1
        # Language (4th element) is covered by TestProcessStreamLanguage;
        # here the mocked process() never sets _reply_lang, so only the
        # routing part of the tuple is meaningful.
        assert chunks[0][:3] == ("staged reply", True, "AWAITING_APPROVAL")
        orch.process.assert_called_once()

    def test_non_reflex_falls_through_to_pure_chat(self, orch):
        # A real question (not a reflex pattern) must still stream from
        # the engine — the reflex guard must not swallow normal chat.
        orch._is_pure_chat = MagicMock(return_value=True)
        orch.engine_analytical.chat_stream = MagicMock(return_value=iter(["Resp", "osta"]))
        # _resolve_lang is module-level; the call should just pass text through.
        chunks = list(orch.process_stream("me explica o que é RAG"))
        # streamed chunks then a final OFFLINE marker, all carrying the lang
        assert chunks[0] == ("Resp", False, None, "pt")
        assert chunks[-1] == ("", True, "OFFLINE", "pt")
        orch.engine_analytical.chat_stream.assert_called_once()


class TestProcessStreamLanguage:
    """C8 (M5-A4/M10-A3): every tuple carries the turn's reply language,
    on all three paths — the listener's voice choice depends on it."""

    @pytest.fixture
    def orch(self):
        from src.core.orchestrator import CortexOrchestrator
        o = CortexOrchestrator.__new__(CortexOrchestrator)
        o.engine_analytical = MagicMock()
        o.gatekeeper = MagicMock()
        o._pending_action = None
        o._pending_query = None
        return o

    def test_en_reflex_reports_en(self, orch):
        chunks = list(orch.process_stream("what time is it"))
        reply, is_final, mode, lang = chunks[0]
        assert mode == "REFLEX"
        assert lang == "en"

    def test_reflex_lang_follows_reply_not_input_heuristic(self, orch):
        # "oi" reads as EN to the input heuristic (no accents/keywords) but
        # the reflex reply is PT ("Oi! Em que posso ajudar?"). The announced
        # language must match the WORDS the TTS will read — reply wins
        # (the s53 lesson, now enforced server-side). Even an explicit EN
        # override can't make PT words speakable by an EN voice.
        chunks = list(orch.process_stream("oi", language="en"))
        reply, is_final, mode, lang = chunks[0]
        assert mode == "REFLEX"
        assert lang == "pt"

    def test_language_override_wins_on_pure_chat(self, orch):
        # Pure chat GENERATES in the requested language, so the client's
        # explicit override (e.g. an iOS Shortcut bound to EN) wins there —
        # same _resolve_lang rule as process().
        orch._is_pure_chat = MagicMock(return_value=True)
        orch.engine_analytical.chat_stream = MagicMock(return_value=iter(["Hi"]))
        chunks = list(orch.process_stream("me explica o que é RAG", language="en"))
        assert chunks[0] == ("Hi", False, None, "en")
        assert chunks[-1] == ("", True, "OFFLINE", "en")
        # And the engine was actually asked for that language.
        _, kwargs = orch.engine_analytical.chat_stream.call_args
        assert kwargs.get("response_language") == "en"

    def test_pure_chat_sets_reply_lang_on_instance(self, orch):
        # The shortcut bypasses _process_core, which is what normally sets
        # _reply_lang; the shortcut must set it itself so the turn state
        # stays coherent for anything reading it after the stream.
        orch._is_pure_chat = MagicMock(return_value=True)
        orch.engine_analytical.chat_stream = MagicMock(return_value=iter(["x"]))
        list(orch.process_stream("me explica o que é RAG"))
        assert orch._reply_lang == "pt"

    def test_deferral_reports_process_core_language(self, orch):
        # The deferral path must report what _process_core decided (it sees
        # pending state the shortcut-level resolve can't), not re-guess.
        # conversation_history is now handed through on purpose: the deferral
        # runs process() in a worker thread, and the UI 2c scope is
        # thread-local, so the binding has to be re-opened THERE.
        def fake_process(text, image_b64=None, language=None,
                         conversation_history=None):
            orch._reply_lang = "en"  # what _process_core would have done
            return ("done it", "MAC_OPEN")
        orch.process = fake_process
        orch._is_pure_chat = MagicMock(return_value=False)
        chunks = list(orch.process_stream("abre o spotify"))
        assert chunks == [("done it", True, "MAC_OPEN", "en")]

    def test_empty_input_carries_lang(self, orch):
        chunks = list(orch.process_stream("", language="en"))
        assert chunks == [("[SYSTEM ERROR] Empty input.", True, "INPUT_ERROR", "en")]


class TestProcessStreamEngineError:
    """C8 (M10-A5): an engine death mid-stream must yield a SPEAKABLE
    localised sentence — the raw exception text would be fed to the TTS."""

    @pytest.fixture
    def orch(self):
        from src.core.orchestrator import CortexOrchestrator
        o = CortexOrchestrator.__new__(CortexOrchestrator)
        o.engine_analytical = MagicMock()
        o.gatekeeper = MagicMock()
        o._pending_action = None
        o._pending_query = None
        o._is_pure_chat = MagicMock(return_value=True)
        return o

    @pytest.mark.parametrize("text,want_lang", [
        ("me explica o que é RAG", "pt"),
        ("explain what RAG is", "en"),
    ])
    def test_engine_error_is_speakable_and_localised(self, orch, text, want_lang):
        orch.engine_analytical.chat_stream = MagicMock(side_effect=RuntimeError("boom"))
        chunks = list(orch.process_stream(text))
        assert len(chunks) == 1
        reply, is_final, mode, lang = chunks[0]
        assert is_final is True
        assert mode == "INTERNAL_ERROR"
        assert lang == want_lang
        # Speakable: no raw marker, no exception text.
        assert "[ENGINE ERROR" not in reply
        assert "boom" not in reply
        assert reply.strip().endswith(".")

    def test_engine_error_mid_stream_after_chunks(self, orch):
        # Death AFTER some chunks flowed: the partial chunks pass through
        # and the final is still the speakable error, not the raw text.
        def gen(*a, **k):
            yield "Os agentes "
            raise RuntimeError("socket reset")
        orch.engine_analytical.chat_stream = gen
        chunks = list(orch.process_stream("me explica o que é RAG"))
        assert chunks[0] == ("Os agentes ", False, None, "pt")
        reply, is_final, mode, lang = chunks[-1]
        assert mode == "INTERNAL_ERROR"
        assert "socket reset" not in reply

    def test_engine_error_is_audited(self, orch, monkeypatch):
        # The technical detail must survive somewhere — the audit log,
        # since it no longer rides in the spoken reply.
        from src.core import orchestrator as orch_mod
        logged = []
        monkeypatch.setattr(orch_mod.audit, "log",
                            lambda *a, **k: logged.append((a, k)))
        orch.engine_analytical.chat_stream = MagicMock(
            side_effect=RuntimeError("ollama exploded"))
        list(orch.process_stream("me explica o que é RAG"))
        stream_errors = [(a, k) for a, k in logged if "stream_error" in a]
        assert stream_errors, "engine stream error must be audit-logged"
        assert "ollama exploded" in str(stream_errors[0][1].get("payload"))


class TestProcessStreamLatencyLogging:
    """M7-A3/M11-A5: the streaming shortcut exits must emit the same
    `orchestrator/response` + latency_ms record process() emits — the
    latency_report (the swap runbook's measuring instrument) only reads
    that event, so the voice path was invisible and the reported p95 came
    from a text-only population."""

    @pytest.fixture
    def orch(self):
        from src.core.orchestrator import CortexOrchestrator
        o = CortexOrchestrator.__new__(CortexOrchestrator)
        o.engine_analytical = MagicMock()
        o.gatekeeper = MagicMock()
        o._pending_action = None
        o._pending_query = None
        return o

    def _capture(self, monkeypatch):
        events = []
        def fake_log(trace, stage, event, payload=None, latency_ms=None):
            events.append({"stage": stage, "event": event,
                           "payload": payload or {}, "latency_ms": latency_ms})
        monkeypatch.setattr("src.core.orchestrator.audit.log", fake_log)
        return events

    def _responses(self, events):
        return [e for e in events
                if e["stage"] == "orchestrator" and e["event"] == "response"]

    def test_reflex_exit_logs_latency(self, orch, monkeypatch):
        events = self._capture(monkeypatch)
        list(orch.process_stream("que horas são"))
        resp = self._responses(events)
        assert len(resp) == 1
        assert resp[0]["payload"]["mode"] == "REFLEX"
        assert resp[0]["latency_ms"] is not None
        assert resp[0]["latency_ms"] >= 0

    def test_pure_chat_exit_logs_latency(self, orch, monkeypatch):
        events = self._capture(monkeypatch)
        orch._is_pure_chat = MagicMock(return_value=True)
        orch.engine_analytical.chat_stream = MagicMock(
            return_value=iter(["Res", "posta"]))
        list(orch.process_stream("me explica o que é RAG"))
        resp = self._responses(events)
        assert len(resp) == 1
        assert resp[0]["payload"]["mode"] == "OFFLINE"
        assert resp[0]["latency_ms"] is not None

    def test_engine_error_exit_logs_latency(self, orch, monkeypatch):
        events = self._capture(monkeypatch)
        orch._is_pure_chat = MagicMock(return_value=True)
        orch.engine_analytical.chat_stream = MagicMock(
            side_effect=RuntimeError("ollama died"))
        list(orch.process_stream("me explica o que é RAG"))
        resp = self._responses(events)
        assert len(resp) == 1
        assert resp[0]["payload"]["mode"] == "INTERNAL_ERROR"

    def test_deferral_path_does_not_double_log(self, orch, monkeypatch):
        # The deferral branch goes through process(), which logs its own
        # response — process_stream must NOT add a second record for the
        # same turn (it would double-count in the report).
        events = self._capture(monkeypatch)
        orch._is_pure_chat = MagicMock(return_value=False)
        orch.process = MagicMock(return_value=("done", "MAC_OPEN"))
        list(orch.process_stream("abre o spotify"))
        assert self._responses(events) == []  # mocked process() logs nothing


class TestWebSearchStreaming:
    """_do_web_search streams its synthesis token by token when a token
    emitter is installed (the /ui/chat/stream case), and stays a single
    blocking call otherwise (CLI/voice) — so a 20s ONLINE_RAG answer types out
    instead of landing all at once."""

    @pytest.fixture
    def orch(self):
        from src.core.orchestrator import CortexOrchestrator
        o = CortexOrchestrator.__new__(CortexOrchestrator)
        o.engine_analytical = MagicMock()
        o.web_searcher = MagicMock()
        o.web_searcher.search.return_value = (
            "Source 1 (placar): Flamengo 2 - 1 Palmeiras — 18/07.\n[url]"
        )
        return o

    def test_blocking_when_no_emitter(self, orch):
        orch.engine_analytical.chat.return_value = "Foi Palmeiras 2 a 1."
        reply, mode = orch._do_web_search(orch.engine_analytical, "resultado do jogo")
        assert mode == "ONLINE_RAG"
        assert reply == "Foi Palmeiras 2 a 1."
        orch.engine_analytical.chat.assert_called_once()
        orch.engine_analytical.chat_stream.assert_not_called()

    def test_streams_and_accumulates_when_emitter_installed(self, orch):
        orch.engine_analytical.chat_stream.return_value = iter(
            ["Foi ", "Palmeiras ", "2 a 1."])
        pieces = []
        orch.set_token_emitter(pieces.append)
        try:
            reply, mode = orch._do_web_search(
                orch.engine_analytical, "resultado do jogo")
        finally:
            orch.set_token_emitter(None)
        # streamed piece by piece …
        assert pieces == ["Foi ", "Palmeiras ", "2 a 1."]
        # … and still returns the full reply for the buffer / blocking callers
        assert reply == "Foi Palmeiras 2 a 1."
        assert mode == "ONLINE_RAG"
        orch.engine_analytical.chat.assert_not_called()

    def test_synthesis_replacement_is_final_not_an_emitted_token(self, orch):
        from src.core.stream_reply import ReplyReplacement

        replacement = ReplyReplacement("fixed")
        orch.engine_analytical.chat_stream.return_value = iter(["a", "b", replacement])
        pieces = []
        orch.set_token_emitter(pieces.append)
        try:
            reply, mode = orch._synthesise(
                orch.engine_analytical, "web prompt", "en", "ONLINE_RAG")
        finally:
            orch.set_token_emitter(None)

        assert pieces == ["a", "b"]
        assert reply == "fixed"
        assert reply is replacement
        assert mode == "ONLINE_RAG"
        orch.engine_analytical.chat_stream.assert_called_once_with(
            "web prompt", response_language="en", tool_backed=True)


class TestLocalDocumentSynthesisStreams:
    """LOCAL_RAG streams too — it never did, in either surface.

    The web path checked for a token emitter and streamed the synthesis; the
    local-document path, one screen below it in the same file, ended in a
    blocking engine.chat(). So "resume esse documento" sat silent through the
    read AND the whole synthesis while the identical web path streamed. Both
    now go through _synthesise, which is also the single place the voice path
    will be taught about streaming.
    """

    @pytest.fixture
    def orch(self):
        from src.core.orchestrator import CortexOrchestrator
        o = CortexOrchestrator.__new__(CortexOrchestrator)
        o.engine_analytical = MagicMock()
        o.reader = MagicMock()
        o.reader.read_document.return_value = "Relatório trimestral. Receita subiu."
        return o

    def test_blocking_when_no_emitter(self, orch):
        orch.engine_analytical.chat.return_value = "A receita subiu."
        reply, mode = orch._handle_read(
            orch.engine_analytical, "relatorio.pdf", "resume", "pt")
        assert (reply, mode) == ("A receita subiu.", "LOCAL_RAG")
        orch.engine_analytical.chat_stream.assert_not_called()

    def test_streams_and_accumulates_when_emitter_installed(self, orch):
        orch.engine_analytical.chat_stream.return_value = iter(
            ["A ", "receita ", "subiu."])
        pieces = []
        orch.set_token_emitter(pieces.append)
        try:
            reply, mode = orch._handle_read(
                orch.engine_analytical, "relatorio.pdf", "resume", "pt")
        finally:
            orch.set_token_emitter(None)
        assert pieces == ["A ", "receita ", "subiu."]
        # … and the full text still comes back for the buffer / chat store
        assert (reply, mode) == ("A receita subiu.", "LOCAL_RAG")
        orch.engine_analytical.chat.assert_not_called()

    def test_a_read_error_never_reaches_the_model(self, orch):
        # The error string is returned as-is, before any synthesis — the
        # streaming branch must not swallow that early exit.
        orch.reader.read_document.return_value = "[SYSTEM ERROR] no such file"
        orch.set_token_emitter(lambda _p: None)
        try:
            reply, mode = orch._handle_read(
                orch.engine_analytical, "sumiu.pdf", None, "pt")
        finally:
            orch.set_token_emitter(None)
        assert mode == "FILE_ERROR"
        orch.engine_analytical.chat_stream.assert_not_called()
        orch.engine_analytical.chat.assert_not_called()


class TestBothRagPathsShareOneSynthesis:
    """Anti-drift: the two RAG paths must not grow separate streaming rules
    again. Pinning the shared helper is what makes teaching the voice path
    (part 2) a one-place change instead of two that can diverge."""

    def test_web_and_document_paths_both_call_the_helper(self, monkeypatch):
        from src.core.orchestrator import CortexOrchestrator
        o = CortexOrchestrator.__new__(CortexOrchestrator)
        o.engine_analytical = MagicMock()
        o.web_searcher = MagicMock()
        o.web_searcher.search.return_value = "Source 1: algo.\n[url]"
        o.reader = MagicMock()
        o.reader.read_document.return_value = "conteudo"

        seen = []
        monkeypatch.setattr(
            CortexOrchestrator, "_synthesise",
            lambda self, engine, injected, lang, mode: (seen.append(mode) or ("r", mode)))

        o._do_web_search(o.engine_analytical, "quem ganhou")
        o._handle_read(o.engine_analytical, "doc.pdf", None, "pt")
        assert seen == ["ONLINE_RAG", "LOCAL_RAG"]


class TestMarketPriceEmission:
    @pytest.fixture
    def setup(self, monkeypatch):
        from src.core.orchestrator import CortexOrchestrator
        from src.tools.web_search import WebSearcher
        monkeypatch.setenv("CORTEX_SEARCH_COUNTRY", "GB")
        o = CortexOrchestrator.__new__(CortexOrchestrator)
        o.web_searcher = MagicMock()
        o.web_searcher.get_market.return_value = WebSearcher().get_market()
        o.web_searcher.search.return_value = "Source 1: costs ₹1,09,999 in India."
        engine = MagicMock()
        events = []
        monkeypatch.setattr("src.core.orchestrator.audit.log",
                            lambda *a, **kw: events.append((a, kw)))
        return o, engine, events

    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize("lang,query,reply,expected", [
        ("pt", "quanto custa o Pixel 10 Pro?",
         "O Pixel 10 Pro custa ₹1,09,999 no Reino Unido.",
         "Não achei o preço no Reino Unido."),
        ("pt", "quanto custa o Pixel 10 Pro?",
         "O Pixel 10 Pro custa ₹1,09,999 (256GB) na Índia e £999 (128GB) no Reino Unido.",
         "Não achei o preço no Reino Unido."),
        ("en", "how much is the Pixel?", "The Pixel costs INR 999 in the UK.",
         "I couldn't find the price in the United Kingdom."),
        ("pt", "quanto é ₹1,09,999 em libras?", "Custa ₹1,09,999.", "Custa ₹1,09,999."),
        ("pt", "quanto custa?", "Custa £999.", "Custa £999."),
        ("pt", "quais as cores?", "  Tem duas cores.\nE 256GB!  ",
         "  Tem duas cores.\nE 256GB!  "),
    ])
    def test_web_paths_guard_before_delivery(self, setup, stream, lang, query, reply, expected):
        o, engine, events = setup
        engine.chat.return_value = reply
        engine.chat_stream.return_value = iter(reply)  # worst chunk boundaries
        emitted = []
        if stream:
            o.set_token_emitter(emitted.append)
        final, mode = o._do_web_search(engine, query, lang)
        assert (final, mode) == (expected, "ONLINE_RAG")
        if stream:
            assert "".join(emitted) == final
            if "₹" not in expected:
                assert all("₹" not in p for p in emitted)
        guard_events = [(a, kw) for a, kw in events if a[2] == "market_price_guard"]
        if reply != expected:
            assert len(guard_events) == 1
            assert guard_events[0][1]["payload"] == {
                "reason": "foreign_price_removed", "currencies": ["INR"], "count": 1,
            }
        else:
            assert guard_events == []

    def test_foreign_chunk_holdback_and_statuses(self, setup):
        o, engine, events = setup
        emitted, statuses = [], []
        o.set_token_emitter(emitted.append)
        o.set_status_emitter(lambda *args: statuses.append(args))

        def chunks(*args, **kw):
            assert [s[0] for s in statuses] == ["web_search", "thinking"]
            yield "O Pixel custa ₹1,"
            assert emitted == []
            o._emit_status("thinking", "still synthesising")
            yield "09,999 no Reino"
            assert emitted == []
            yield " Unido. Mais nada."

        engine.chat_stream.side_effect = chunks
        final, _ = o._do_web_search(engine, "quanto custa?")
        assert final == "Mais nada. Não achei o preço no Reino Unido."
        assert "".join(emitted) == final
        assert all("₹" not in part for part in emitted)
        assert statuses[-1] == ("thinking", "still synthesising")

    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize("query,reply,expected,currency", [
        ("quanto custa o iPhone 17?",
         "O iPhone 17 tem 256 GB e custa US$ 1.199. No Reino Unido sai por £1,199.",
         "No Reino Unido sai por £1,199.", "USD"),
        ("quanto custa?", "custa ₹1,09,999. É um bom celular.",
         "É um bom celular. Não achei o preço no Reino Unido.", "INR"),
        ("quanto custa em rupees?", "Custa US$ 1.199. Custa ₹1,09,999.",
         "Custa ₹1,09,999.", "USD"),
        ("quanto custa em rupias?", "custa ₹1,09,999. É um bom celular.",
         "custa ₹1,09,999. É um bom celular.", None),
    ])
    def test_gap_only_when_no_allowed_money_remains(
        self, setup, stream, query, reply, expected, currency,
    ):
        o, engine, events = setup
        engine.chat.return_value = reply
        engine.chat_stream.return_value = iter(reply)
        emitted = []
        if stream:
            o.set_token_emitter(emitted.append)
        final, mode = o._do_web_search(engine, query, "pt")
        assert (final, mode) == (expected, "ONLINE_RAG")
        assert "".join(emitted) == (expected if stream else "")
        guard_events = [kw for a, kw in events if a[2] == "market_price_guard"]
        assert guard_events == ([{"payload": {
            "reason": "foreign_price_removed", "currencies": [currency], "count": 1,
        }}] if currency else [])

    def test_decimal_holdback_partial_flush_and_sentence_latency(self, setup):
        o, engine, _ = setup
        emitted = []
        o.set_token_emitter(emitted.append)

        def chunks(*args, **kw):
            yield "Custa £1."
            assert emitted == []
            yield "199 no Reino Unido. "
            assert "".join(emitted) == "Custa £1.199 no Reino Unido."
            yield "Mais: ₹999"  # no final punctuation: must still be checked

        engine.chat_stream.side_effect = chunks
        final, _ = o._do_web_search(engine, "quanto custa?")
        assert final == "Custa £1.199 no Reino Unido."
        assert "".join(emitted) == final

    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize("scope", ["no_market", "local"])
    def test_out_of_scope_is_untouched(self, setup, stream, scope):
        o, engine, events = setup
        reply = "Custa ₹999 no Reino Unido."
        engine.chat.return_value = reply
        engine.chat_stream.return_value = iter([reply])
        emitted = []
        if stream:
            o.set_token_emitter(emitted.append)
        if scope == "no_market":
            o.web_searcher.get_market.return_value = None
            final, _ = o._do_web_search(engine, "quanto custa?")
        else:
            from src.core.market_price_guard import web_market_scope
            with web_market_scope(o.web_searcher.get_market(), "quanto custa?"):
                final, _ = o._synthesise(engine, "document", "pt", "LOCAL_RAG")
        assert final == reply
        if stream:
            assert "".join(emitted) == reply
        else:
            assert not emitted
        assert not any(a[2] == "market_price_guard" for a, _ in events)

    @pytest.mark.parametrize("fail", [False, True])
    def test_market_scope_resets_after_web_synthesis(self, setup, fail):
        o, engine, _ = setup
        reply = "Costs ₹999."
        engine.chat.side_effect = RuntimeError("engine failed") if fail else None
        engine.chat.return_value = reply
        if fail:
            with pytest.raises(RuntimeError):
                o._do_web_search(engine, "price?", "en")
        else:
            o._do_web_search(engine, "price?", "en")
        engine.chat.side_effect = None
        # A later caller without _do_web_search gets no stale market/query.
        assert o._synthesise(engine, "prompt", "en", "ONLINE_RAG")[0] == reply

    def test_other_guards_keep_replacement_semantics_with_market(self, setup):
        from src.core.stream_reply import ReplyReplacement
        o, engine, _ = setup
        replacement = ReplyReplacement("A corrected answer.")
        engine.chat_stream.return_value = iter(["An unfinished sentence", replacement])
        emitted = []
        o.set_token_emitter(emitted.append)
        final, _ = o._do_web_search(engine, "price?", "en")
        assert final is replacement
        assert emitted == []  # never flush superseded partial prose

    def test_foreign_replacement_is_checked_without_becoming_a_token(self, setup):
        from src.core.stream_reply import ReplyReplacement
        o, engine, events = setup
        engine.chat_stream.return_value = iter([
            "An unfinished sentence", ReplyReplacement("Costs INR 999."),
        ])
        emitted = []
        o.set_token_emitter(emitted.append)
        final, _ = o._do_web_search(engine, "price?", "en")
        assert isinstance(final, ReplyReplacement)
        assert final == "I couldn't find the price in the United Kingdom."
        assert emitted == []
        assert len([a for a, _ in events if a[2] == "market_price_guard"]) == 1

    def test_single_audit_event_aggregates_removed_sentences(self, setup):
        o, engine, events = setup
        engine.chat_stream.return_value = iter(["INR 999. ", "EUR 899. ", "USD 999."])
        emitted = []
        o.set_token_emitter(emitted.append)
        final, _ = o._do_web_search(engine, "price?", "en")
        assert "".join(emitted) == final == "I couldn't find the price in the United Kingdom."
        guard_events = [kw for a, kw in events if a[2] == "market_price_guard"]
        assert guard_events == [{"payload": {
            "reason": "foreign_price_removed", "currencies": ["EUR", "INR", "USD"], "count": 3,
        }}]


class TestDeferredPathStreams:
    """The deferral runs the pipeline in a worker thread so a RAG synthesis
    streams out of process_stream — which is the ONLY entry point the voice
    pipeline uses. Before this, ONLINE_RAG was silent by voice for the whole
    search plus synthesis (~47s p95) and the mitigation was a spoken filler.
    """

    @pytest.fixture
    def orch(self):
        from src.core.orchestrator import CortexOrchestrator
        o = CortexOrchestrator.__new__(CortexOrchestrator)
        o.engine_analytical = MagicMock()
        o.engine_creative = MagicMock()
        o._is_pure_chat = MagicMock(return_value=False)
        o._pending_action = None
        o._pending_query = None
        return o

    def _streaming_process(self, orch, pieces, final=None, mode="ONLINE_RAG"):
        """A process() that streams like _synthesise does: through whatever
        token emitter is installed on ITS OWN thread."""
        def fake(text, image_b64=None, language=None, conversation_history=None):
            orch._reply_lang = "pt"
            emit = orch._token_emitter()
            for p in pieces:
                emit(p)
            return (final if final is not None else "".join(pieces), mode)
        return fake

    def test_synthesis_tokens_come_out_as_chunks(self, orch):
        orch.process = self._streaming_process(orch, ["Foi ", "Palmeiras ", "2 a 1."])
        chunks = list(orch.process_stream("quem ganhou o jogo"))
        assert [c[0] for c in chunks[:-1]] == ["Foi ", "Palmeiras ", "2 a 1."]
        assert all(c[1] is False for c in chunks[:-1])
        # the final tuple owes nothing: the consumer already has every word
        assert chunks[-1] == ("", True, "ONLINE_RAG", "pt")

    def test_web_replacement_reaches_final_and_delivery_outcome(self, orch):
        from src.core.staging_mixin import Delivery
        from src.core.stream_reply import ReplyReplacement

        replacement = ReplyReplacement("fixed")
        orch.engine_analytical.chat_stream.return_value = iter(["a", "b", replacement])
        orch.web_searcher = MagicMock()
        orch.web_searcher.search.return_value = "Source 1 (T):\nevidence\n[url]"
        orch.studies_review = MagicMock()
        orch.studies_review.list_pending.return_value = []

        def core(text, image_b64=None, language=None):
            orch._reply_lang = "en"
            return orch._do_web_search(orch.engine_analytical, text, lang="en")

        orch._process_core = core
        delivery = Delivery()
        emitted = []
        orch.set_turn_delivery(delivery)
        orch.set_token_emitter(emitted.append)
        try:
            frames = list(orch.process_stream("search for news", language="en"))
        finally:
            orch.set_token_emitter(None)
            orch.set_turn_delivery(None)

        assert emitted == ["a", "b"]
        assert frames[:-1] == [("a", False, None, "en"), ("b", False, None, "en")]
        assert frames[-1] == ("fixed", True, "ONLINE_RAG", "en")
        assert isinstance(frames[-1][0], ReplyReplacement)
        # The delivery stores the authoritative worker result. Transports
        # clear their live accumulator on the same typed final tuple.
        assert delivery.finished.is_set()
        assert delivery.outcome == ("fixed", "ONLINE_RAG")
        stored = []
        for chunk, _final, _mode, _lang in frames:
            if isinstance(chunk, ReplyReplacement):
                stored.clear()
            stored.append(chunk)
        assert "".join(stored) == "fixed"

    def test_a_non_streaming_path_is_untouched(self, orch):
        # Tools, dispatch and reflex never call the emitter — they must still
        # land as one final chunk, exactly as before.
        def fake(text, image_b64=None, language=None, conversation_history=None):
            orch._reply_lang = "pt"
            return ("Abrindo o Spotify.", "MAC_OPEN")
        orch.process = fake
        assert list(orch.process_stream("abre o spotify")) == [
            ("Abrindo o Spotify.", True, "MAC_OPEN", "pt")]

    def test_an_appended_postscript_still_arrives(self, orch):
        # _maybe_announce_pending appends "💡 Por sinal…" AFTER the synthesis
        # streamed. That tail is exactly what the final tuple still owes.
        orch.process = self._streaming_process(
            orch, ["Resposta."], final="Resposta.\n\n💡 Por sinal: 1 estudo.")
        chunks = list(orch.process_stream("quem ganhou o jogo"))
        assert chunks[0][0] == "Resposta."
        assert chunks[-1] == ("\n\n💡 Por sinal: 1 estudo.", True, "ONLINE_RAG", "pt")
        # and reassembling the stream reproduces the reply exactly once
        assert "".join(c[0] for c in chunks) == "Resposta.\n\n💡 Por sinal: 1 estudo."

    def test_a_rewritten_reply_is_reported_not_hidden(self, orch, monkeypatch):
        # If some future step rewrites the reply after it streamed, correct
        # CONTENT wins over avoiding a duplicate — and the audit says so,
        # because nothing else would ever surface it.
        events = []
        from src.core import orchestrator as orch_mod
        monkeypatch.setattr(orch_mod.audit, "log",
                            lambda *a, **k: events.append((a, k)))
        orch.process = self._streaming_process(
            orch, ["texto cru"], final="TEXTO LIMPO")
        chunks = list(orch.process_stream("quem ganhou o jogo"))
        assert chunks[-1] == ("TEXTO LIMPO", True, "ONLINE_RAG", "pt")
        assert any("stream_reply_diverged" in a for a, _k in events)

    def test_an_exception_in_the_worker_reaches_the_caller(self, orch):
        def boom(text, image_b64=None, language=None, conversation_history=None):
            raise RuntimeError("pipeline died")
        orch.process = boom
        with pytest.raises(RuntimeError, match="pipeline died"):
            list(orch.process_stream("quem ganhou o jogo"))


class TestThreadLocalsCrossTheWorkerBoundary:
    """The three silent failures the worker thread could cause.

    set_token_emitter, set_status_emitter and the UI 2c history override are
    ALL thread-local. Running process() in another thread drops every one of
    them, and none would raise — the turn would just quietly answer with the
    global history, with no live-activity feed, and with the UI's token side
    channel dead. Each gets its own pin.
    """

    @pytest.fixture
    def orch(self):
        from src.core.orchestrator import CortexOrchestrator
        o = CortexOrchestrator.__new__(CortexOrchestrator)
        o.engine_analytical = MagicMock()
        o.engine_creative = MagicMock()
        o._is_pure_chat = MagicMock(return_value=False)
        o._pending_action = None
        o._pending_query = None
        return o

    def test_the_conversation_history_reaches_the_worker(self, orch):
        seen = {}

        def fake(text, image_b64=None, language=None, conversation_history=None):
            orch._reply_lang = "pt"
            seen["history"] = conversation_history
            seen["thread"] = threading.current_thread().name
            return ("ok", "OFFLINE")
        orch.process = fake
        history = [{"role": "user", "content": "antes"}]
        list(orch.process_stream("abre o spotify", conversation_history=history))
        # handed through, so process() can re-open the scope on ITS thread …
        assert seen["history"] == history
        # … and it really is another thread, or none of this would matter
        assert seen["thread"] != threading.current_thread().name

    def test_the_status_emitter_survives_the_hop(self, orch):
        stages = []

        def fake(text, image_b64=None, language=None, conversation_history=None):
            orch._reply_lang = "pt"
            orch._emit_status("web_search", "quem ganhou")
            return ("ok", "ONLINE_RAG")
        orch.process = fake
        orch.set_status_emitter(lambda s, d="": stages.append((s, d)))
        try:
            list(orch.process_stream("quem ganhou o jogo"))
        finally:
            orch.set_status_emitter(None)
        assert stages == [("web_search", "quem ganhou")]

    def test_the_outer_token_emitter_still_receives(self, orch):
        # The /ui route's side channel predates this and must keep working
        # byte for byte: it is what sets its `streamed` flag.
        side = []
        orch.process = None

        def fake(text, image_b64=None, language=None, conversation_history=None):
            orch._reply_lang = "pt"
            emit = orch._token_emitter()
            emit("um ")
            emit("dois")
            return ("um dois", "ONLINE_RAG")
        orch.process = fake
        orch.set_token_emitter(side.append)
        try:
            chunks = list(orch.process_stream("quem ganhou o jogo"))
        finally:
            orch.set_token_emitter(None)
        assert side == ["um ", "dois"]                      # side channel
        assert [c[0] for c in chunks[:-1]] == ["um ", "dois"]  # and the generator

    def test_the_workers_cleanup_does_not_disarm_the_caller(self, orch):
        # The worker clears ITS emitters in a finally. Thread-locals mean that
        # must not reach the caller's — the /ui route installs its emitters
        # once and drives the whole turn through them, so wiping them here
        # would kill the live feed for every later step of the same request.
        def fake(text, image_b64=None, language=None, conversation_history=None):
            orch._reply_lang = "pt"
            return ("ok", "OFFLINE")
        orch.process = fake
        mine_status, mine_token = (lambda *a, **k: None), (lambda *a, **k: None)
        orch.set_status_emitter(mine_status)
        orch.set_token_emitter(mine_token)
        try:
            list(orch.process_stream("abre o spotify"))
            assert orch._status_emitter() is mine_status
            assert orch._token_emitter() is mine_token
        finally:
            orch.set_status_emitter(None)
            orch.set_token_emitter(None)


class TestConversationScope:
    """UI 2c: a UI conversation's recent turns are bound as the engines'
    short-term history for the turn only — without core depending on the ui
    storage layer (the route hands in a pre-loaded message list), without
    clobbering the global voice/CLI history, and cleared afterwards."""

    def _engines(self):
        from src.core.engine import OllamaEngine
        made = []
        for _ in range(2):
            e = OllamaEngine.__new__(OllamaEngine)
            try:
                OllamaEngine.__init__(e)
            except Exception:
                pass
            made.append(e)
        return made

    def _orch(self):
        from src.core.orchestrator import CortexOrchestrator
        o = CortexOrchestrator.__new__(CortexOrchestrator)
        o.engine_analytical, o.engine_creative = self._engines()
        o.gatekeeper = MagicMock()
        o._pending_action = None
        o._pending_query = None
        o._is_pure_chat = MagicMock(return_value=True)
        o._shadow_classify = MagicMock()
        o._shadow_semantic_route = MagicMock()
        return o

    def test_scope_binds_both_engines_and_unbinds(self):
        o = self._orch()
        convo = [{"role": "user", "content": "meu nome é Pedro"},
                 {"role": "assistant", "content": "Prazer, Pedro."}]
        with o._conversation_scope(convo):
            assert o.engine_analytical.history == convo
            assert o.engine_creative.history == convo
        # cleared on both — back to the global thread
        assert getattr(o.engine_analytical._history_override, "value", None) is None
        assert getattr(o.engine_creative._history_override, "value", None) is None

    def test_scope_noop_when_history_is_none(self):
        # None = no conversation context (voice/CLI) → global history, no bind.
        o = self._orch()
        with o._conversation_scope(None):
            assert getattr(o.engine_analytical._history_override,
                           "value", None) is None

    def test_scope_binds_empty_for_a_fresh_conversation(self):
        # [] = a conversation with no turns yet (its first message). It must
        # bind an EMPTY, isolated override — NOT fall through to the global
        # thread — so the first turn can't leak another conversation's history.
        o = self._orch()
        o.engine_analytical._global_history.append(
            {"role": "user", "content": "outra conversa"})
        with o._conversation_scope([]):
            assert o.engine_analytical.history == []      # isolated, not global
            assert o.engine_creative.history == []
        assert getattr(o.engine_analytical._history_override,
                       "value", None) is None

    def test_process_stream_binds_loaded_history(self):
        o = self._orch()
        ea = o.engine_analytical
        seen = {}

        def rec(text, response_language=None):
            # What the LLM would see for this turn: the conversation's thread.
            seen["history"] = list(ea.history)
            return iter(["ok"])

        ea.chat_stream = rec
        convo = [{"role": "user", "content": "meu nome é Pedro"},
                 {"role": "assistant", "content": "Prazer, Pedro."}]
        chunks = list(o.process_stream("qual é meu nome",
                                       conversation_history=convo))
        assert seen["history"] == convo
        # streamed then finalised
        assert chunks[-1][1] is True
        # override cleared after the turn — global thread restored
        assert getattr(ea._history_override, "value", None) is None

    def test_process_stream_no_history_leaves_global(self):
        """No conversation_id (voice/CLI) → no override; the engine runs on its
        global thread exactly as before."""
        o = self._orch()
        ea = o.engine_analytical
        ea._global_history.append({"role": "user", "content": "global-prev"})
        seen = {}

        def rec(text, response_language=None):
            seen["history"] = list(ea.history)
            return iter(["ok"])

        ea.chat_stream = rec
        list(o.process_stream("qual é meu nome"))
        assert seen["history"] == [{"role": "user", "content": "global-prev"}]
        assert getattr(ea._history_override, "value", None) is None
