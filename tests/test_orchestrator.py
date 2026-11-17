"""
Orchestrator regex + pure routing logic.

Covers: language detection (PT vs EN), the Spotify play+device-hint
extraction, the offline-hint fast-path for _needs_realtime, and the
realtime-marker veto. No LLM, no Ollama, no Spotify network calls.
"""
import pytest


class TestLanguageDetection:
    """_detect_lang must mirror the input language even without accents."""

    @pytest.fixture
    def detect(self):
        from src.core.orchestrator import _detect_lang
        return _detect_lang

    @pytest.mark.parametrize("text", [
        "me explica RAG em detalhes",
        "explica machine learning",
        "conta sobre transformers",
        "fala sobre o brasil",
        "mostra o resumo",
        "preciso de ajuda agora",
        "gostaria de entender quantization",
        "sou junior dev",
        "o que é RAG?",
        "também quero saber",
        "então me explica",
        "obrigado pela ajuda",
        "que horas são",
        "Estamos em 2026.",  # year reflex reply must select the PT voice
        # "quem" added s69 (M10-A11): the only PT question word that was
        # missing from the pattern — this input read as EN before.
        "quem descobriu o brasil?",
        # Informal/abbreviated PT (2026-06-02): the user drops accents and
        # uses chat shorthand, especially on the laptop. These read as EN
        # to the old heuristic → PT question got an English reply.
        "me fala sobre vc",
        "vc sabe quem descobriu o brasil",
        "tb quero saber",
        "tbm quero",
        "pq o ceu e azul",
        "qto custa",
        "blz qual a hora",
        "obg pela ajuda",
        "vlw",
        "pra que serve isso",
        "agr nao",
        "mds que isso",
        # s84: the remembering family read as EN and got English replies.
        "lembra daquele dia?",
        "lembra de mim?",
        # s84: everyday PT from the user's real sentences that read as EN.
        "toque matue no spotify",
        "aumenta o volume",
        "abaixa o brilho",
        "modo escuro",
        "como fazer gelo?",
        "como funciona o attention mechanism?",
        "tudo bem?",
        "comi 200g de frango e 2 ovos",
        "volume maximo",
        # s88 (Codex lote 8): no keyword matched, so they were English.
        "me ajuda",
        "pode continuar",
        "continua",
        "posso perguntar uma coisa",
    ])
    def test_detects_pt(self, detect, text):
        assert detect(text) == "pt", f"expected pt for: {text}"

    @pytest.mark.parametrize("text", [
        "explain machine learning",
        "tell me about transformers",
        "show me the summary",
        "I need help now",
        "what is RAG?",
        "how does attention work",
        "about machine learning",
        "the show is on",
        "what time is it",
        # EN sentences containing words that look like PT shortcuts — must
        # NOT false-positive to PT (these are the collision traps the
        # informal-PT list was kept clear of).
        "i need to go now",
        "add milk to the list",
        "no problem at all",
        "i want to learn python",
        # s84 collision traps for the everyday-PT additions: "os", "nas",
        # "do" and "volume" were kept out on purpose.
        "update the OS",
        "what is a NAS?",
        "do you know?",
        "volume 50",
        "Play daisies by Justin Bieber on Spotify",
        "please continue",
        "can you help me",
    ])
    def test_detects_en(self, detect, text):
        assert detect(text) == "en", f"expected en for: {text}"


class TestSpotifyPlayRegex:
    """Spotify play command extraction + device hint capture."""

    @pytest.fixture
    def cls(self):
        from src.core.orchestrator import CortexOrchestrator
        return CortexOrchestrator

    @pytest.mark.parametrize("text, expected_query, expected_hint", [
        ("toque TETO no spotify no meu iphone", "TETO", "iphone"),
        ("toca matuê", "matuê", None),
        ("play queen on my mac", "queen", "mac"),
        ("coloque WIU no meu iphone", "WIU", "iphone"),
        ("ponha jorge ben no celular", "jorge ben", "celular"),
        ("toque queen na caixinha", "queen", "caixinha"),
        ("play taylor swift on my macbook", "taylor swift", "macbook"),
        ("toque a música queen no spotify", "queen", None),
        ("play bruno mars on homepod", "bruno mars", "homepod"),
        ("toca mc cabelinho no spotify", "mc cabelinho", None),
    ])
    def test_play_with_optional_device_hint(self, cls, text, expected_query, expected_hint):
        m = cls._SPOTIFY_PLAY_RE.match(text)
        assert m, f"play regex failed to match: {text}"
        assert m.group(1).strip() == expected_query

        h = cls._SPOTIFY_DEVICE_HINT_RE.search(text)
        got_hint = h.group(1).lower() if h else None
        assert got_hint == expected_hint

    # ---------- Polite/permissive verb forms (sessão 10 bloco F) ----------
    # The original bug: "ola cortex, pode tocar matue no spotify?"
    # combined a vocative AND a polite verb AND a trailing "?", all
    # three of which the regex rejected. Vocatives are handled
    # upstream by strip_vocatives (see test_dispatch_normalisation).
    # These cases cover the verb + punctuation layers on the regex
    # itself, assuming the vocative is already gone.
    @pytest.mark.parametrize("text, expected_query", [
        ("pode tocar matue", "matue"),
        ("podes tocar matue", "matue"),
        ("poderia tocar matue no spotify", "matue"),
        ("podia colocar matue no celular", "matue"),
        ("pode pôr queen", "queen"),
        ("pode por bohemian rhapsody", "bohemian rhapsody"),
        ("pode tocar matue no spotify?", "matue"),
        ("quero ouvir matue", "matue"),
        ("queria ouvir djonga", "djonga"),
        ("gostaria de ouvir queen", "queen"),
        ("can you play matue", "matue"),
        ("could you play matue on spotify", "matue"),
        ("can you put on bohemian rhapsody", "bohemian rhapsody"),
    ])
    def test_polite_verb_forms_match(self, cls, text, expected_query):
        m = cls._SPOTIFY_PLAY_RE.match(text)
        assert m, f"polite play regex failed to match: {text}"
        assert m.group(1).strip() == expected_query

    # ---------- Trailing punctuation (sessão 10 bloco F) ----------
    @pytest.mark.parametrize("text, expected_query", [
        ("toca matue?", "matue"),
        ("toca matue!", "matue"),
        ("toca matue.", "matue"),
        ("toca matue no spotify?", "matue"),
        ("play queen on my mac?", "queen"),
        ("toca matue???", "matue"),  # multiple trailing punct
    ])
    def test_trailing_punctuation_tolerated(self, cls, text, expected_query):
        m = cls._SPOTIFY_PLAY_RE.match(text)
        assert m, f"trailing-punct regex failed to match: {text}"
        assert m.group(1).strip() == expected_query

    # Negative: things that should still NOT match (avoid false
    # positives from the broader verb set).
    @pytest.mark.parametrize("text", [
        "ouve isso aqui",            # "ouvir" is verb-of-result, not standalone
        "tocou matue ontem",         # past tense, not a command
        "fala matue",                # different verb
        "compra matue",              # different verb
        "matue",                     # no verb
    ])
    def test_non_play_phrases_do_not_match(self, cls, text):
        assert not cls._SPOTIFY_PLAY_RE.match(text), f"false positive: {text}"


class TestOfflineFastPath:
    """_OFFLINE_HINT_RE + _REALTIME_MARKER_RE together control the
    LLM-classifier short-circuit in _needs_realtime."""

    @pytest.fixture
    def cls(self):
        from src.core.orchestrator import CortexOrchestrator
        return CortexOrchestrator

    @pytest.mark.parametrize("text", [
        "me explica o que é RAG",
        "explica machine learning",
        "explique o que é um pointer em C",
        "como funciona a transformação de Fourier",
        "how does attention work in transformers",
        "qual a diferença entre TCP e UDP",
        "define entropy",
        "how do quantised LLMs work",
        # 2026-05-27: extended fast-path — "o que é / what is / what's"
        # cover the most common definitional prefixes that were
        # previously falling through to the 1-2s LLM classifier.
        "o que é RAG",
        "o que significa quantização",
        "que é entropy",
        "what is Alan Turing",
        "what is a transformer",
        "what's a closure",
        "what does GIL mean",
    ])
    def test_offline_hint_matches_stable_concepts(self, cls, text):
        assert cls._OFFLINE_HINT_RE.match(text), text
        assert not cls._REALTIME_MARKER_RE.search(text), text

    @pytest.mark.parametrize("text", [
        "me explica as eleições atuais",
        "explain the latest news on AI",
        "explica o jogo de hoje",
        "explica o resultado de hoje",
        "explain what's trending now",
        # The new "o que é / what is" prefixes still get vetoed when a
        # realtime marker is present.
        "o que é o último jogo do Flamengo",
        "what is the latest news",
    ])
    def test_realtime_marker_vetoes_fast_path(self, cls, text):
        # The offline-hint pattern may match the verb, but the realtime
        # marker must veto the fast-path decision.
        assert cls._REALTIME_MARKER_RE.search(text), text

    @pytest.mark.parametrize("text", [
        "qual o clima em Londres",
        "quem ganhou o jogo",
        "preço do bitcoin",
    ])
    def test_no_offline_hint_match(self, cls, text):
        """Queries without a definitional/explanatory prefix should
        not match the fast-path — they go to the LLM classifier."""
        assert not cls._OFFLINE_HINT_RE.match(text), text


class TestMathOnlyFastPath:
    """_MATH_ONLY_RE — second offline fast-path for pure arithmetic."""

    @pytest.fixture
    def cls(self):
        from src.core.orchestrator import CortexOrchestrator
        return CortexOrchestrator

    @pytest.mark.parametrize("text", [
        "2+2",
        "2 + 2",
        "15 * 7",
        "15*7?",
        "(3+1)/2",
        "(3 + 1) / 2 =",
        "  100 - 50  ",
        "3.14 * 2",
        "1,5 + 2,5",  # PT-BR decimal comma
        "2^10",
    ])
    def test_math_only_matches_pure_arithmetic(self, cls, text):
        assert cls._MATH_ONLY_RE.match(text), text

    @pytest.mark.parametrize("text", [
        "what is 2+2",
        "quanto é 2+2",
        "calcule 2+2",
        "explain 2+2",
        "qual o clima",
        "abc",
        "2+2 dollars",
    ])
    def test_math_only_rejects_natural_language(self, cls, text):
        """Pattern must not catch arithmetic embedded in a question —
        those should still go through the LLM classifier (or another
        fast-path) to preserve correct routing for things like
        'calcule', 'quanto é', or mixed-language math questions."""
        assert not cls._MATH_ONLY_RE.match(text), text


# ----------------------------------------------------------------------
# Reminder regex-first dispatch (2026-05-22)
# ----------------------------------------------------------------------


class TestReminderRegex:
    """_REMINDER_TRIGGER_RE, _REMINDER_BODY_RE, _parse_reminder_due."""

    @pytest.fixture
    def cls(self):
        from src.core.orchestrator import CortexOrchestrator
        return CortexOrchestrator

    @pytest.mark.parametrize("text,expected_body", [
        ("me lembre da ida ao lago amanhã às 4 horas",   "ida ao lago amanhã às 4 horas"),
        ("me lembre de comprar leite hoje às 18h",        "comprar leite hoje às 18h"),
        ("lembre-me de pagar a conta amanhã",             "pagar a conta amanhã"),
        ("lembra de ligar pro médico",                    "ligar pro médico"),
        ("cria um lembrete pra reunião amanhã às 9h",     "reunião amanhã às 9h"),
        ("criar lembrete de academia",                    "academia"),
        ("set a reminder for the dentist tomorrow",       "the dentist tomorrow"),
        ("remind me to buy milk",                         "buy milk"),
        # s84: the most common Brazilian phrasing had no route at all
        ("me lembra de ligar pra mãe amanhã",             "ligar pra mãe amanhã"),
        ("me lembra que tenho dentista hoje às 15h",      "tenho dentista hoje às 15h"),
        # a question that carries a time still books (S3 asks first)
        ("lembra da reunião amanhã às 10?",               "reunião amanhã às 10?"),
    ])
    def test_trigger_captures_body(self, cls, text, expected_body):
        m = cls._REMINDER_TRIGGER_RE.match(text)
        assert m is not None, f"no match for: {text!r}"
        assert m.group("body").strip() == expected_body

    @pytest.mark.parametrize("text", [
        "toca matuê",
        "qual o clima hoje",
        "me explica RAG",
        "adicione ao calendário",  # calendar, not reminder
    ])
    def test_trigger_does_not_overmatch(self, cls, text):
        assert cls._REMINDER_TRIGGER_RE.match(text) is None, text

    @pytest.mark.parametrize("body,expected_title,expected_time_fragment", [
        ("ida ao lago amanhã às 4 horas",   "ida ao lago",      "amanhã às 4 horas"),
        ("comprar leite hoje às 18h",        "comprar leite",    "hoje às 18h"),
        ("pagar a conta amanhã",             "pagar a conta",    "amanhã"),
        ("reunião amanhã às 9h",             "reunião",          "amanhã às 9h"),
        ("ligar pro médico",                 "ligar pro médico", None),  # no time expr
    ])
    def test_body_split(self, cls, body, expected_title, expected_time_fragment):
        m = cls._REMINDER_BODY_RE.match(body)
        if expected_time_fragment is None:
            assert m is None or m.group("title").strip() == expected_title
        else:
            assert m is not None, f"no match for: {body!r}"
            assert m.group("title").strip() == expected_title
            assert m.group("time_expr").strip() == expected_time_fragment

    @pytest.mark.parametrize(
        "time_expr,days_offset,expected_hour,expected_min,expected_assumed", [
        # M1-A12c: the parser now also reports WHEN it guessed the hour
        # ("assumed") so the confirmation preview can say so.
        ("amanhã às 4 horas",   1, 16, 0, "16:00"),  # 4 ≤ 6 → assume tarde
        ("amanhã às 9h",        1,  9, 0, None),     # 9 → manhã literal
        ("hoje às 18h",         0, 18, 0, None),
        ("hoje às 18h30",       0, 18, 30, None),
        ("amanhã às 4 da tarde",1, 16, 0, None),     # explicit period
        ("amanhã às 8 da manhã",1,  8, 0, None),
        ("amanhã",              1,  9, 0, "09:00"),  # day only → default
    ])
    def test_parse_reminder_due(self, cls, time_expr, days_offset,
                                expected_hour, expected_min, expected_assumed):
        from datetime import datetime, timezone
        now = datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc)
        result, assumed = cls._parse_reminder_due(time_expr, now)
        assert result is not None, f"None for {time_expr!r}"
        from datetime import datetime as DT
        dt = DT.fromisoformat(result)
        from datetime import timedelta
        expected_date = (now + timedelta(days=days_offset)).date()
        assert dt.date() == expected_date, f"{time_expr!r}: date mismatch"
        assert dt.hour == expected_hour, f"{time_expr!r}: hour mismatch"
        assert dt.minute == expected_min, f"{time_expr!r}: minute mismatch"
        assert assumed == expected_assumed, f"{time_expr!r}: assumed mismatch"

    def test_parse_reminder_due_none_for_empty(self, cls):
        from datetime import datetime, timezone
        now = datetime(2026, 5, 22, 12, 0, tzinfo=timezone.utc)
        assert cls._parse_reminder_due("", now) == (None, None)
        assert cls._parse_reminder_due(None, now) == (None, None)


# ----------------------------------------------------------------------
# Calendar regex-first dispatch (2026-05-22)
# ----------------------------------------------------------------------


class TestCalendarRegex:
    """_CALENDAR_TRIGGER_RE + _handle_calendar_regex + _calendar_end_iso.

    Mirrors TestReminderRegex. The body parser is shared with reminders
    (_REMINDER_BODY_RE) — calendar-specific tests focus on trigger
    detection, fallthrough on missing time, end_iso default."""

    @pytest.fixture
    def cls(self):
        from src.core.orchestrator import CortexOrchestrator
        return CortexOrchestrator

    @pytest.mark.parametrize("text,expected_body", [
        ("adicione no meu calendario a ida ao lago amanha as 4 horas",
         "a ida ao lago amanha as 4 horas"),
        ("adicione ao meu calendario do google a ida ao lago amanha as 4 horas",
         "a ida ao lago amanha as 4 horas"),
        ("adicione no calendário reunião amanhã às 9h",
         "reunião amanhã às 9h"),
        ("marca no calendar a reunião amanhã às 9h",
         "a reunião amanhã às 9h"),
        ("coloca na agenda jantar hoje às 20h",
         "jantar hoje às 20h"),
        ("agenda evento reunião com Bruno amanhã às 14h",
         "reunião com Bruno amanhã às 14h"),
        ("agende um evento jantar amanhã às 20h",
         "jantar amanhã às 20h"),
        ("novo evento: ida ao lago amanhã às 4h",
         "ida ao lago amanhã às 4h"),
        ("criar evento reunião amanhã às 10h",
         "reunião amanhã às 10h"),
        ("add to my calendar lake trip tomorrow at 4pm",
         "lake trip tomorrow at 4pm"),
        ("schedule a meeting tomorrow at 4pm",
         "a meeting tomorrow at 4pm"),
        ("create event lake trip tomorrow at 4",
         "lake trip tomorrow at 4"),
    ])
    def test_trigger_captures_body(self, cls, text, expected_body):
        m = cls._CALENDAR_TRIGGER_RE.match(text)
        assert m is not None, f"no match for: {text!r}"
        assert m.group("body").strip() == expected_body

    @pytest.mark.parametrize("text", [
        "toca matuê",
        "qual o clima hoje",
        "me explica RAG",
        "me lembre da ida ao lago amanhã",      # reminder, not calendar
        "lembre-me de comprar leite",            # reminder
        "meus eventos próximos 7 dias",          # list query
        "o que tem na minha agenda hoje",        # list query
        "what's on my calendar today",           # list query
    ])
    def test_trigger_does_not_overmatch(self, cls, text):
        assert cls._CALENDAR_TRIGGER_RE.match(text) is None, text

    def test_calendar_end_iso_default_one_hour(self, cls):
        from datetime import datetime, timezone
        start = datetime(2026, 5, 22, 16, 0, tzinfo=timezone.utc).isoformat()
        end = cls._calendar_end_iso(start)
        end_dt = datetime.fromisoformat(end)
        start_dt = datetime.fromisoformat(start)
        assert (end_dt - start_dt).total_seconds() == 3600

    def test_calendar_end_iso_custom_hours(self, cls):
        from datetime import datetime, timezone
        start = datetime(2026, 5, 22, 9, 0, tzinfo=timezone.utc).isoformat()
        end = cls._calendar_end_iso(start, default_hours=3)
        end_dt = datetime.fromisoformat(end)
        start_dt = datetime.fromisoformat(start)
        assert (end_dt - start_dt).total_seconds() == 10800

    def test_calendar_end_iso_preserves_timezone(self, cls):
        from datetime import datetime, timezone, timedelta
        # Non-UTC offset to confirm we don't drop tz info
        offset = timezone(timedelta(hours=-3))
        start = datetime(2026, 5, 22, 16, 0, tzinfo=offset).isoformat()
        end = cls._calendar_end_iso(start)
        assert end.endswith("-03:00") or "-0300" in end

    @pytest.fixture
    def orchestrator(self, tmp_path, monkeypatch):
        # Sandbox the pending-action file so tests don't trample real state.
        from src.core.orchestrator import CortexOrchestrator
        monkeypatch.setattr(
            CortexOrchestrator, "_PENDING_ACTION_FILE",
            str(tmp_path / "pending_action.json"),
        )
        # __new__ avoids full __init__ (which boots Ollama, gatekeeper, etc.)
        # We only need the regex methods + the _pending_action attribute.
        orch = CortexOrchestrator.__new__(CortexOrchestrator)
        orch._pending_action = None
        return orch

    def test_handle_stages_pending_action(self, orchestrator):
        from datetime import datetime, timezone
        now = datetime(2026, 5, 22, 12, 0, tzinfo=timezone.utc)
        result = orchestrator._handle_calendar_regex(
            "a ida ao lago amanha as 4 horas", now,
        )
        assert result is not None
        reply, mode = result
        assert mode == "AGENT_AWAITING_CONFIRMATION"
        assert orchestrator._pending_action is not None
        assert orchestrator._pending_action["type"] == "create_event"
        args = orchestrator._pending_action["args"]
        assert "ida ao lago" in args["summary"].lower()
        # 4 horas → 1-6 range, no period qualifier → assume PM → 16:00
        start_dt = datetime.fromisoformat(args["start_iso"])
        assert start_dt.hour == 16
        assert start_dt.minute == 0
        # end_iso = start + 1h default
        end_dt = datetime.fromisoformat(args["end_iso"])
        assert (end_dt - start_dt).total_seconds() == 3600

    def test_handle_returns_none_for_body_without_time(self, orchestrator):
        from datetime import datetime, timezone
        now = datetime(2026, 5, 22, 12, 0, tzinfo=timezone.utc)
        # Calendar events need an explicit start time. Without one we
        # bail out so the LLM can disambiguate (vs reminders, where a
        # title-only body is still a valid undated reminder).
        result = orchestrator._handle_calendar_regex("ida ao lago", now)
        assert result is None
        assert orchestrator._pending_action is None

    def test_handle_returns_none_for_empty_body(self, orchestrator):
        from datetime import datetime, timezone
        now = datetime(2026, 5, 22, 12, 0, tzinfo=timezone.utc)
        assert orchestrator._handle_calendar_regex("", now) is None
        assert orchestrator._handle_calendar_regex("   ", now) is None

    def test_confirmation_prompt_shape(self, orchestrator):
        from datetime import datetime, timezone
        now = datetime(2026, 5, 22, 12, 0, tzinfo=timezone.utc)
        reply, _ = orchestrator._handle_calendar_regex(
            "reunião amanhã às 9h", now,
        )
        assert "Confirmar criação do evento" in reply
        assert "reunião" in reply.lower()
        assert "sim" in reply.lower() and ("não" in reply.lower() or "nao" in reply.lower())

    def test_reminder_and_calendar_dispatch_are_disjoint(self, cls):
        """A query that matches the reminder trigger must NOT also match
        the calendar trigger (and vice versa). The orchestrator dispatches
        them sequentially and the regexes are designed to be disjoint."""
        reminder_only = [
            "me lembre de comprar leite amanhã às 10h",
            "lembre-me de pagar a conta amanhã",
            "remind me to buy milk",
        ]
        calendar_only = [
            "adicione no meu calendario reunião amanhã às 9h",
            "agenda evento jantar amanhã às 20h",
            "schedule a meeting tomorrow at 4pm",
        ]
        for text in reminder_only:
            assert cls._REMINDER_TRIGGER_RE.match(text), text
            assert cls._CALENDAR_TRIGGER_RE.match(text) is None, text
        for text in calendar_only:
            assert cls._CALENDAR_TRIGGER_RE.match(text), text
            assert cls._REMINDER_TRIGGER_RE.match(text) is None, text

    # --- content-first trigger (2026-06-02) -------------------------------

    @pytest.mark.parametrize("text,expected_body", [
        ("marca reunião com Bruno amanhã às 15h no calendário",
         "reunião com Bruno amanhã às 15h"),
        ("coloca esses eventos na agenda", "esses eventos"),
        ("adicione esses jogos ao meu calendário", "esses jogos"),
        ("adiciona dentista dia 20 às 9h no calendário do google",
         "dentista dia 20 às 9h"),
    ])
    def test_content_first_rescues_keyword_first_miss(self, cls, text, expected_body):
        # These put the content BEFORE the calendar keyword, so keyword-first
        # misses them entirely (they used to fall to the flaky 8B). The
        # content-first fallback rescues them and captures a clean body.
        assert cls._CALENDAR_TRIGGER_RE.match(text) is None, text
        m = cls._CALENDAR_TRIGGER_CONTENT_RE.match(text)
        assert m is not None, f"content-first missed: {text!r}"
        assert m.group("body").strip() == expected_body

    @pytest.mark.parametrize("text", [
        "adiciona leite na lista de compras",
        "marca o ponto no trabalho",
        "coloca a roupa na máquina",
        "põe o mac pra dormir",
        "toca um som no spotify",
        "what's on my calendar today",
    ])
    def test_content_first_does_not_overmatch(self, cls, text):
        assert cls._CALENDAR_TRIGGER_CONTENT_RE.match(text) is None, text

    # --- vague-reference guard (2026-06-02) -------------------------------

    @pytest.mark.parametrize("body", [
        "esses jogos",
        "essas reuniões",
        "aqueles compromissos",
        "esse evento",
    ])
    def test_vague_body_asks_for_specifics(self, orchestrator, body):
        from datetime import datetime, timezone
        now = datetime(2026, 5, 22, 12, 0, tzinfo=timezone.utc)
        result = orchestrator._handle_calendar_regex(body, now)
        assert result is not None, body
        reply, mode = result
        assert mode == "CALENDAR_CLARIFY"
        assert "data e hora" in reply.lower()
        # Must NOT stage a guessed event.
        assert orchestrator._pending_action is None

    def test_non_vague_dateless_body_still_falls_through(self, orchestrator):
        from datetime import datetime, timezone
        now = datetime(2026, 5, 22, 12, 0, tzinfo=timezone.utc)
        # No demonstrative → not the "pull from context" signature → let the
        # LLM handle it (return None), unchanged from before.
        assert orchestrator._handle_calendar_regex("reunião de equipe", now) is None
        assert orchestrator._pending_action is None

    def test_vague_body_with_a_date_still_stages(self, orchestrator):
        from datetime import datetime, timezone
        now = datetime(2026, 5, 22, 12, 0, tzinfo=timezone.utc)
        # The guard only fires when no date parses. A demonstrative that DOES
        # carry a concrete time stages normally (never reaches the guard).
        result = orchestrator._handle_calendar_regex("esse jantar amanhã às 20h", now)
        assert result is not None
        _, mode = result
        assert mode == "AGENT_AWAITING_CONFIRMATION"
        assert orchestrator._pending_action["type"] == "create_event"


# ----------------------------------------------------------------------
# Language override (added 2026-05-26)
# ----------------------------------------------------------------------


class TestLanguageOverride:
    """`/ask` accepts an optional `language` field. The orchestrator's
    `_resolve_lang` prefers the client's tag over the regex heuristic.
    Both real-world inputs Pedro reported as broken on the iOS PT
    shortcut are covered."""

    def test_detect_lang_misses_mixed_input(self):
        # "play michael jackson no spotify" — EN verb, PT preposition,
        # no accent, no PT keyword the regex looks for. Heuristic says EN.
        from src.core.orchestrator import _detect_lang
        assert _detect_lang("play michael jackson no spotify") == "en"

    def test_detect_lang_catches_unaccented_demonstratives(self):
        # s76 (inovação #3): the PT demonstratives esse/essa/este/esta/isto
        # were added to the heuristic, so "esta me ouvindo?" now correctly
        # reads as PT (it used to miss → EN). The override path still exists
        # for clients that know their language.
        from src.core.orchestrator import _detect_lang
        assert _detect_lang("esta me ouvindo?") == "pt"
        # A phrase with NO listed PT cue still misses — the heuristic is
        # best-effort, the override is the real fix. (The example used to be
        # "manda ai pls"; "manda" joined the list in s84.)
        assert _detect_lang("rolou ai pls") == "en"

    def test_resolve_lang_override_wins(self):
        from src.core.orchestrator import _resolve_lang
        # Override beats the heuristic when valid.
        assert _resolve_lang("play michael jackson no spotify", "pt") == "pt"
        assert _resolve_lang("esta me ouvindo?", "pt") == "pt"
        # And EN override on PT-detected text.
        assert _resolve_lang("que horas são", "en") == "en"

    def test_resolve_lang_falls_back_on_none(self):
        from src.core.orchestrator import _resolve_lang
        assert _resolve_lang("play michael jackson no spotify", None) == "en"
        assert _resolve_lang("que horas são", None) == "pt"

    def test_resolve_lang_falls_back_on_invalid(self):
        from src.core.orchestrator import _resolve_lang
        # Garbage override is ignored — heuristic kicks in.
        assert _resolve_lang("play michael jackson no spotify", "xx") == "en"
        assert _resolve_lang("que horas são", "lixo") == "pt"
        assert _resolve_lang("que horas são", "") == "pt"

    def test_api_normalises_aliases(self):
        from src.api import _normalise_language
        # Canonical
        assert _normalise_language("pt") == "pt"
        assert _normalise_language("en") == "en"
        # Common aliases
        assert _normalise_language("pt-br") == "pt"
        assert _normalise_language("PT-BR") == "pt"
        assert _normalise_language("portuguese") == "pt"
        assert _normalise_language("português") == "pt"
        assert _normalise_language("en-GB") == "en"
        assert _normalise_language("en_us") == "en"
        assert _normalise_language("English") == "en"
        # Garbage → None (server falls back to heuristic)
        assert _normalise_language("xx") is None
        assert _normalise_language("klingon") is None
        assert _normalise_language(None) is None
        assert _normalise_language("") is None
        assert _normalise_language("   ") is None


# ----------------------------------------------------------------------
# macOS app-quit — S3 stage + confirm flow (sessão 11 batch 2)
# ----------------------------------------------------------------------


class TestMacQuitRegex:
    """_handle_mac_quit_regex stages an S3 mac_quit_app pending action;
    _execute_pending_action runs it on 'sim'. Built via __new__ so we
    skip the heavy __init__ (Ollama, gatekeeper, etc.)."""

    @pytest.fixture
    def orchestrator(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock
        from src.core.orchestrator import CortexOrchestrator
        monkeypatch.setattr(
            CortexOrchestrator, "_PENDING_ACTION_FILE",
            str(tmp_path / "pending_action.json"),
        )
        orch = CortexOrchestrator.__new__(CortexOrchestrator)
        orch._pending_action = None
        orch.macos = MagicMock()
        orch.macos.quit_app_outcome.return_value = (True, "Fechando Notion.")
        return orch

    def test_handle_stages_pending_action(self, orchestrator):
        reply, mode = orchestrator._handle_mac_quit_regex("Notion")
        assert mode == "AGENT_AWAITING_CONFIRMATION"
        staged = dict(orchestrator._pending_action)
        staged.pop("staged_at", None)  # TTL stamp (M1-A9), time-dependent
        staged.pop("id", None)  # the action's identity (s88), random
        assert staged == {
            "type": "mac_quit_app",
            "args": {"app_name": "Notion"},
            "origin": None,
        }
        assert "Fechar o Notion?" in reply
        assert "não salvo" in reply

    def test_confirm_executes_quit(self, orchestrator):
        orchestrator._handle_mac_quit_regex("Notion")
        reply, mode = orchestrator._execute_pending_action()
        orchestrator.macos.quit_app_outcome.assert_called_once_with("Notion", lang="pt")
        assert mode == "AGENT_ACTION_DONE"
        assert reply == "Fechando Notion."
        # Pending slot is cleared after execution.
        assert orchestrator._pending_action is None

    def test_cancel_does_not_quit(self, orchestrator):
        orchestrator._handle_mac_quit_regex("Notion")
        reply, mode = orchestrator._cancel_pending_action()
        orchestrator.macos.quit_app_outcome.assert_not_called()
        assert mode == "AGENT_ACTION_CANCELLED"
        assert "fechamento do Notion" in reply
        assert orchestrator._pending_action is None


# =====================================================================
# ONLINE_RAG synthesis prompt — anti-hallucination hedging (s34)
# =====================================================================
# Backlog (s33): with live sports/fixture queries the 8B invented
# specific opponents/dates/times that were NOT in the Tavily snippets.
# The synthesis prompt must instruct the model to hedge/admit the gap
# rather than fill specifics from memory. These tests pin the prompt
# shape deterministically (no live model/network needed).

class TestOnlineRagHedging:

    @pytest.fixture
    def orchestrator(self):
        from src.core.orchestrator import CortexOrchestrator
        return CortexOrchestrator.__new__(CortexOrchestrator)

    def _capture_injected(self, orchestrator, web_result="some snippet text"):
        """Run _do_web_search with fakes; return the prompt sent to the LLM."""
        captured = {}

        class _FakeSearcher:
            def search(self, query):
                return web_result

            def get_market(self):
                return None

        class _FakeEngine:
            def chat(self, injected, response_language="pt", *, tool_backed=False):
                captured["prompt"] = injected
                captured["lang"] = response_language
                return "reply"

        orchestrator.web_searcher = _FakeSearcher()
        # _SEARCH_STRIP_RE is a class attribute, available without __init__.
        reply, mode = orchestrator._do_web_search(
            _FakeEngine(), "horários dos próximos jogos da seleção", lang="pt",
        )
        assert mode == "ONLINE_RAG"
        return captured["prompt"]

    def test_prompt_forbids_inventing_specifics(self, orchestrator):
        prompt = self._capture_injected(orchestrator)
        low = prompt.lower()
        # Names the failure-mode facts so the instruction is concrete.
        assert "do not invent" in low
        for fact in ("fixtures", "dates", "scores", "opponents"):
            assert fact in low, fact

    def test_prompt_tells_model_to_admit_the_gap(self, orchestrator):
        prompt = self._capture_injected(orchestrator)
        # Both language-specific hedge phrasings are offered to the model.
        assert "Não encontrei essa informação nos resultados." in prompt
        assert "I couldn't find that in the results." in prompt

    @pytest.mark.parametrize("country", [None, "GB"])
    def test_web_search_passes_searchers_market(self, orchestrator, monkeypatch, country):
        from unittest.mock import ANY, MagicMock, patch
        from src.core.orchestrator import build_rag_prompt
        from src.tools.web_search import WebSearcher

        monkeypatch.delenv("CORTEX_SEARCH_COUNTRY", raising=False)
        if country:
            monkeypatch.setenv("CORTEX_SEARCH_COUNTRY", country)
        market = WebSearcher(cost_tracker=MagicMock()).get_market()
        orchestrator.web_searcher = MagicMock()
        orchestrator.web_searcher.search.return_value = "frozen evidence"
        orchestrator.web_searcher.get_market.return_value = market
        orchestrator._synthesise = MagicMock(return_value=("reply", "ONLINE_RAG"))
        engine = MagicMock()
        with patch("src.core.orchestrator.build_rag_prompt", wraps=build_rag_prompt) as build:
            assert orchestrator._do_web_search(engine, "compare A and B", "en") == (
                "reply", "ONLINE_RAG",
            )
        orchestrator.web_searcher.get_market.assert_called_once_with()
        build.assert_called_once_with("frozen evidence", "compare A and B", market=market)
        orchestrator._synthesise.assert_called_once_with(
            engine, ANY, "en", "ONLINE_RAG",
        )
        assert ("PRICES:" in orchestrator._synthesise.call_args.args[1]) == bool(market)

    def test_prompt_anchors_todays_date_next_to_the_data(self, orchestrator):
        """Bug s73 (Copa 2026): 'próximo jogo' picked the SECOND fixture from
        correct snippets. The date lives in the engine system prompt, but too
        far from the data — the injected prompt must restate it adjacent."""
        from datetime import datetime
        prompt = self._capture_injected(orchestrator)
        today = datetime.now().astimezone().strftime("%Y-%m-%d")
        assert "TODAY'S DATE:" in prompt
        assert today in prompt
        # The anchor must come BEFORE the data it disambiguates.
        assert prompt.index("TODAY'S DATE:") < prompt.index("LIVE WEB DATA:")

    def test_prompt_instructs_temporal_selection_for_next_event(self, orchestrator):
        """When several dated events are listed, 'next' = earliest future one."""
        prompt = self._capture_injected(orchestrator)
        low = prompt.lower()
        assert "next/upcoming event" in low
        assert "próximo" in low
        assert "earliest" in low

    def test_query_proximo_is_rewritten_to_primeiro_a_partir_de_hoje(self, orchestrator):
        """The part that actually flips the s73 answer: the 8B reads 'próximo
        jogo' as 'the second item in the list'. No instruction fixes it — the
        word itself must be rewritten before injection."""
        prompt = self._capture_injected(orchestrator)
        # _capture_injected sends "horários dos próximos jogos da seleção" —
        # PLURAL, which must NOT be rewritten (a list has no wrong pick).
        assert "próximos jogos" in prompt
        from src.core.orchestrator import rewrite_next_event_query
        assert rewrite_next_event_query("qual o proximo jogo do brasil") == \
            "qual o primeiro (contando a partir de hoje) jogo do brasil"
        assert rewrite_next_event_query("qual a próxima partida do brasil") == \
            "qual a primeira (contando a partir de hoje) partida do brasil"

    def test_rewrite_leaves_time_units_and_plurals_alone(self):
        from src.core.orchestrator import rewrite_next_event_query
        for untouched in (
            "o que acontece na próxima semana",
            "me lembra disso no próximo mês",
            "na próxima segunda-feira",
            "quais os próximos jogos do brasil",
            "qual a capital da França",
        ):
            assert rewrite_next_event_query(untouched) == untouched

    def test_snippets_are_persisted_to_audit(self, orchestrator, monkeypatch):
        """Bug s73: snippets weren't persisted anywhere — diagnosing a wrong
        ONLINE_RAG answer required a paid replay. They must land in audit."""
        seen = []
        from src.core import orchestrator as orch_mod
        monkeypatch.setattr(
            orch_mod.audit, "log",
            lambda trace, stage, event, payload=None, latency_ms=None:
                seen.append((stage, event, payload)),
        )
        self._capture_injected(orchestrator, web_result="Source 1: Brasil x Marrocos 13/06")
        snippet_events = [e for e in seen if e[0] == "web_search" and e[1] == "snippets"]
        assert len(snippet_events) == 1
        payload = snippet_events[0][2]
        assert "Marrocos" in payload["context"]
        assert payload["query"]  # the actual query sent to the provider

    def test_snippets_audit_payload_is_capped(self, orchestrator, monkeypatch):
        """One event must not dwarf the audit file (rotation is 10MB)."""
        seen = []
        from src.core import orchestrator as orch_mod
        monkeypatch.setattr(
            orch_mod.audit, "log",
            lambda trace, stage, event, payload=None, latency_ms=None:
                seen.append((stage, event, payload)),
        )
        self._capture_injected(orchestrator, web_result="x" * 50_000)
        payload = [e for e in seen if e[1] == "snippets"][0][2]
        assert len(payload["context"]) <= 3000
        assert len(payload["query"]) <= 200

    def test_search_error_short_circuits_without_llm(self, orchestrator):
        """A [SYSTEM ...] error from the searcher must not reach the LLM."""
        class _ErrSearcher:
            def search(self, query):
                return "[SYSTEM] search backend down"

        class _BoomEngine:
            def chat(self, *a, **k):
                raise AssertionError("LLM must not be called on search error")

        orchestrator.web_searcher = _ErrSearcher()
        reply, mode = orchestrator._do_web_search(
            _BoomEngine(), "qualquer coisa", lang="pt",
        )
        assert mode == "SEARCH_ERROR"
        assert reply.startswith("[SYSTEM")


# =====================================================================
# Live activity feed — status emitter (s36)
# =====================================================================
# The orchestrator emits semantic pipeline stages ("web_search",
# "thinking", "reading", …) to an optional callback the streaming /ui
# endpoint installs. Default None → no-op, so every non-UI caller is
# unaffected. These tests pin the mechanism without a live model.

class TestStatusEmitter:

    @pytest.fixture
    def orchestrator(self):
        from src.core.orchestrator import CortexOrchestrator
        return CortexOrchestrator.__new__(CortexOrchestrator)

    def test_emit_status_is_noop_without_listener(self, orchestrator):
        # __new__ instance has no _status_local at all; the getattr guard
        # must make this a safe no-op (no AttributeError).
        orchestrator._emit_status("thinking")
        orchestrator.set_status_emitter(None)  # explicit clear, still no-op
        orchestrator._emit_status("web_search", "detail")

    def test_emit_status_forwards_stage_and_detail(self, orchestrator):
        seen = []
        orchestrator.set_status_emitter(lambda stage, detail: seen.append((stage, detail)))
        orchestrator._emit_status("web_search", "clima londres")
        orchestrator._emit_status("thinking")
        assert seen == [("web_search", "clima londres"), ("thinking", "")]

    def test_emit_status_swallows_listener_errors(self, orchestrator):
        # A broken UI feed must never break a reply.
        def boom(stage, detail):
            raise RuntimeError("listener exploded")
        orchestrator.set_status_emitter(boom)
        orchestrator._emit_status("thinking")  # must not raise

    def test_status_emitter_is_thread_local(self, orchestrator):
        # An emitter installed on the main thread must NOT be visible to a
        # second thread — this is what stops a concurrent voice turn from
        # cross-talking into a UI turn's feed.
        import threading as _t
        seen_main = []
        orchestrator.set_status_emitter(lambda s, d: seen_main.append(s))
        other_saw = []

        def other_thread():
            orchestrator._emit_status("web_search", "leak?")  # no emitter here
            other_saw.append(getattr(orchestrator._status_local, "emitter", None))

        t = _t.Thread(target=other_thread)
        t.start(); t.join()
        assert other_saw == [None]   # other thread saw no emitter
        assert seen_main == []       # nothing leaked into main's list

    def test_web_search_emits_search_then_thinking(self, orchestrator):
        seen = []
        orchestrator.set_status_emitter(lambda stage, detail: seen.append(stage))

        class _FakeSearcher:
            def search(self, query):
                return "live snippet"

            def get_market(self):
                return None

        class _FakeEngine:
            def chat(self, injected, response_language="pt", *, tool_backed=False):
                return "answer"

        orchestrator.web_searcher = _FakeSearcher()
        orchestrator._do_web_search(_FakeEngine(), "clima em londres", lang="pt")
        assert seen == ["web_search", "thinking"]


# =====================================================================
# Cognitive Arbiter — shadow mode wiring (s39)
# =====================================================================
# The arbiter scaffold classifies every turn and logs the route to the
# audit WITHOUT changing routing. _shadow_classify centralises it so it
# runs once per turn across all entry points (process via _process_core;
# process_stream's reflex/pure-chat shortcuts that bypass _process_core).

class TestArbiterShadow:

    @pytest.fixture
    def orchestrator(self):
        from src.core.orchestrator import CortexOrchestrator
        return CortexOrchestrator.__new__(CortexOrchestrator)

    def test_shadow_noop_when_flag_off(self, orchestrator, monkeypatch):
        orchestrator._arbiter_shadow = False
        logged = []
        monkeypatch.setattr("src.core.orchestrator.audit.log",
                            lambda *a, **k: logged.append(a))
        orchestrator._shadow_classify("oi")
        assert logged == []

    def test_shadow_noop_when_attr_missing(self, orchestrator, monkeypatch):
        # __new__ instance without the attribute at all → getattr guard.
        logged = []
        monkeypatch.setattr("src.core.orchestrator.audit.log",
                            lambda *a, **k: logged.append(a))
        orchestrator._shadow_classify("oi")  # must not raise
        assert logged == []

    def test_shadow_logs_route_when_on(self, orchestrator, monkeypatch):
        orchestrator._arbiter_shadow = True
        logged = []
        monkeypatch.setattr("src.core.orchestrator.audit.log",
                            lambda *a, **k: logged.append((a, k)))
        orchestrator._shadow_classify("qual a estratégia de marketing?")
        assert len(logged) == 1
        args, kwargs = logged[0]
        assert args[1] == "arbiter" and args[2] == "classify"
        payload = kwargs["payload"]
        assert "route" in payload and "confidence" in payload
        assert payload["input"] == "qual a estratégia de marketing?"

    def test_shadow_swallows_classifier_errors(self, orchestrator, monkeypatch):
        orchestrator._arbiter_shadow = True
        def boom(_):
            raise RuntimeError("classifier broke")
        monkeypatch.setattr("src.core.orchestrator.arbiter.classify", boom)
        # must never crash the live path
        orchestrator._shadow_classify("oi")


class TestAssumedTimeNote:
    """M1-A12c: when the time parser GUESSED the hour (PM assumption on
    1-6h, or the 09:00 day-only default), the confirmation preview says
    so — the resolved time alone let an assumed 16:00 sail through on a
    reflexive 'sim'."""

    @pytest.fixture
    def orchestrator(self, tmp_path, monkeypatch):
        from src.core.orchestrator import CortexOrchestrator
        monkeypatch.setattr(
            CortexOrchestrator, "_PENDING_ACTION_FILE",
            str(tmp_path / "pending_action.json"),
        )
        orch = CortexOrchestrator.__new__(CortexOrchestrator)
        orch._pending_action = None
        return orch

    def _now(self):
        from datetime import datetime, timezone
        return datetime(2026, 5, 22, 12, 0, tzinfo=timezone.utc)

    def test_calendar_assumed_pm_notes_it(self, orchestrator):
        reply, mode = orchestrator._handle_calendar_regex(
            "a ida ao lago amanha as 4 horas", self._now())
        assert mode == "AGENT_AWAITING_CONFIRMATION"
        assert "assumi 16:00" in reply

    def test_calendar_explicit_period_has_no_note(self, orchestrator):
        reply, mode = orchestrator._handle_calendar_regex(
            "a ida ao lago amanha as 4 da tarde", self._now())
        assert mode == "AGENT_AWAITING_CONFIRMATION"
        assert "assumi" not in reply

    def test_reminder_day_only_notes_default(self, orchestrator):
        reply, mode = orchestrator._handle_reminder_regex(
            "pagar o aluguel amanhã", self._now())
        assert mode == "AGENT_AWAITING_CONFIRMATION"
        assert "assumi 09:00" in reply

    def test_reminder_explicit_hour_has_no_note(self, orchestrator):
        reply, mode = orchestrator._handle_reminder_regex(
            "pagar o aluguel amanhã às 18h", self._now())
        assert mode == "AGENT_AWAITING_CONFIRMATION"
        assert "assumi" not in reply

    def test_reminder_question_mark_does_not_hide_the_time(self, orchestrator):
        # s84: "lembra da reunião amanhã às 10?" — the "?" sat after the hour,
        # the end-anchored time regex missed it, and the reminder was staged
        # with no due date and "?" in its title.
        reply, mode = orchestrator._handle_reminder_regex(
            "reunião amanhã às 10?", self._now())
        assert mode == "AGENT_AWAITING_CONFIRMATION"
        args = orchestrator._pending_action["args"]
        assert args["title"] == "reunião"
        assert args["due_iso"] is not None

    def test_note_en(self, orchestrator):
        orchestrator._reply_lang = "en"
        reply, mode = orchestrator._handle_calendar_regex(
            "a ida ao lago amanha as 4 horas", self._now())
        assert "I assumed 16:00" in reply


class TestLeaseHoursEnv:
    """M1-A12b: a malformed CORTEX_LEASE_HOURS must not crash the boot."""

    def test_malformed_env_falls_back_to_default(self, monkeypatch):
        from src.core.orchestrator import _lease_hours_from_env
        monkeypatch.setenv("CORTEX_LEASE_HOURS", "4h")
        assert _lease_hours_from_env() == 4
        monkeypatch.setenv("CORTEX_LEASE_HOURS", "")
        assert _lease_hours_from_env() == 4

    def test_valid_env_is_respected(self, monkeypatch):
        from src.core.orchestrator import _lease_hours_from_env
        monkeypatch.setenv("CORTEX_LEASE_HOURS", "8")
        assert _lease_hours_from_env() == 8

    def test_unset_env_defaults_to_4(self, monkeypatch):
        from src.core.orchestrator import _lease_hours_from_env
        monkeypatch.delenv("CORTEX_LEASE_HOURS", raising=False)
        assert _lease_hours_from_env() == 4
