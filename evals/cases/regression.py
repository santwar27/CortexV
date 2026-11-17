"""Hard regressions — specific historical bugs that must stay fixed.

Distinct from the other category files: these are tagged 'regression'
and reference the JOURNAL entry that introduced them. The eval
framework lets us guarantee these don't come back as we evolve
the dispatch chain.
"""
from evals.core import EvalCase, EvalKind


CASES: list[EvalCase] = [
    # Sessão 10 bloco F — Matuê hallucination
    EvalCase(
        name="matue_original_bug_routes_correctly",
        kind=EvalKind.DISPATCH,
        category="regression",
        input="ola cortex, pode tocar matue no spotify?",
        expected_mode="SPOTIFY_PLAY",
        expected_capture="matue",
        description=(
            "The exact input that produced a hallucinated track list. "
            "Must dispatch to SPOTIFY_PLAY, not fall through to LLM."
        ),
        tags=["regression", "jou-s10-blocoF"],
    ),

    # Sessão 11 bloco H — hallucination detector
    EvalCase(
        name="matue_hallucination_caught",
        kind=EvalKind.DETECTOR,
        category="regression",
        detector_input="Tocando 'BACKSTAGE' de Matuê agora.",
        expected_flagged=True,
        description=(
            "The detector must continue to flag the canonical Llama-8B "
            "narration shape we observed in production."
        ),
        tags=["regression", "jou-s11-blocoH"],
    ),

    # Sessão 12 bloco B — JSON-ish leak sanitiser. Pattern the user saw:
    # `{play "Funk do Brasil" playlist on Spotify}`
    EvalCase(
        name="json_pseudo_tool_call_stripped",
        kind=EvalKind.SANITISER,
        category="regression",
        sanitiser_input=(
            '{play "Funk do Brasil" playlist on Spotify}\n'
            "Boa escolha, Pedro."
        ),
        expected_not_contains="{play",
        expected_contains="Boa escolha",
        description=(
            "Sanitiser must remove brace-wrapped pseudo tool calls but "
            "preserve the prose that follows."
        ),
        tags=["regression", "jou-s12-blocoB"],
    ),

    # Sessão 10 bloco F — vocative strip
    EvalCase(
        name="vocative_doesnt_break_dispatch",
        kind=EvalKind.DISPATCH,
        category="regression",
        input="ei cortex, modo escuro",
        expected_mode="MAC_DARK",
        description=(
            "Vocative 'ei cortex,' must be stripped so the mac dark "
            "dispatch fires; previously this fell through to LLM."
        ),
        tags=["regression", "jou-s10-blocoF"],
    ),

    # Task 111 — the user's 2026-10-28 Mac volume default.
    EvalCase(
        name="volume_50_is_mac",
        kind=EvalKind.DISPATCH,
        category="regression",
        input="volume 50",
        expected_mode="MAC_VOLUME",
        expected_capture="50",
        description=(
            "Bare 'volume N' controls the Mac. Spotify volume requires "
            "Spotify to be named explicitly."
        ),
        tags=["regression", "jou-s10-blocoG", "conflict"],
    ),

    # ---------------------------------------------------------------
    # Sessão 13 bloco F — bugs from real chat logs the user shared
    # ---------------------------------------------------------------

    # Bug 2 — ack prefixes desanchoravam dispatch.
    EvalCase(
        name="ack_prefix_ok_dot",
        kind=EvalKind.DISPATCH,
        category="regression",
        input="ok. toque matue no spotify no meu iphone",
        expected_mode="SPOTIFY_PLAY",
        expected_capture="matue",
        description=(
            "User input from real chat log: ack prefix 'ok.' was "
            "breaking the ^-anchored SPOTIFY_PLAY_RE. strip_vocatives "
            "now removes 'ok.' before dispatch."
        ),
        tags=["regression", "jou-s13-blocoF", "bug2"],
    ),
    EvalCase(
        name="ack_prefix_beleza",
        kind=EvalKind.DISPATCH,
        category="regression",
        input="beleza, pausa",
        expected_mode="SPOTIFY_PAUSE",
        tags=["regression", "jou-s13-blocoF", "bug2"],
    ),
    EvalCase(
        name="ack_prefix_show",
        kind=EvalKind.DISPATCH,
        category="regression",
        input="show, próxima",
        expected_mode="SPOTIFY_NEXT",
        tags=["regression", "jou-s13-blocoF", "bug2"],
    ),
    EvalCase(
        name="ack_prefix_got_it",
        kind=EvalKind.DISPATCH,
        category="regression",
        input="got it, play queen",
        expected_mode="SPOTIFY_PLAY",
        expected_capture="queen",
        tags=["regression", "jou-s13-blocoF", "bug2"],
    ),

    # Bug 3 — PAUSE didn't cover "pare" / "para de tocar".
    EvalCase(
        name="pause_pare_imperative",
        kind=EvalKind.DISPATCH,
        category="regression",
        input="pare",
        expected_mode="SPOTIFY_PAUSE",
        description="3rd-person imperative 'pare' is what most users actually say.",
        tags=["regression", "jou-s13-blocoF", "bug3"],
    ),
    EvalCase(
        name="pause_para_de_tocar",
        kind=EvalKind.DISPATCH,
        category="regression",
        input="para de tocar",
        expected_mode="SPOTIFY_PAUSE",
        tags=["regression", "jou-s13-blocoF", "bug3"],
    ),
    EvalCase(
        name="pause_pare_de_tocar_a_musica",
        kind=EvalKind.DISPATCH,
        category="regression",
        input="pare de tocar a música",
        expected_mode="SPOTIFY_PAUSE",
        tags=["regression", "jou-s13-blocoF", "bug3"],
    ),
    # Negative: "para" + non-music context must NOT trigger pause.
    EvalCase(
        name="pause_para_mim_not_triggered",
        kind=EvalKind.REGEX,
        category="regression",
        regex_dotted="src.core.dispatch.spotify.SPOTIFY_PAUSE_RE",
        regex_input="para mim",
        expected_regex_match=False,
        description="'para mim' is conversational, not a pause command.",
        tags=["regression", "jou-s13-blocoF", "bug3", "false_positive_guard"],
    ),

    # Bug 4 — nested JSON wrap escaped the sanitiser.
    EvalCase(
        name="nested_json_response_unwrapped",
        kind=EvalKind.SANITISER,
        category="regression",
        sanitiser_input=(
            '{ "response": { "text": "Oi lá, sô! Tudo bão.", '
            '"context": "chill", "confidence": 0.9 } }'
        ),
        expected_contains="Oi lá, sô! Tudo bão.",
        expected_not_contains='"context"',
        description=(
            "Real Llama 8B output from chat log. Sanitiser must "
            "extract the 'text' field from the nested envelope."
        ),
        tags=["regression", "jou-s13-blocoF", "bug4"],
    ),
    EvalCase(
        name="nested_json_output_text_unwrapped",
        kind=EvalKind.SANITISER,
        category="regression",
        sanitiser_input='{"output": {"text": "Boa noite, Pedro."}}',
        expected_contains="Boa noite, Pedro.",
        expected_not_contains='"output"',
        tags=["regression", "jou-s13-blocoF", "bug4"],
    ),
    EvalCase(
        name="nested_json_answer_flat",
        kind=EvalKind.SANITISER,
        category="regression",
        sanitiser_input='{"answer": "42"}',
        expected_contains="42",
        expected_not_contains='"answer"',
        tags=["regression", "jou-s13-blocoF", "bug4"],
    ),
    # Negative: inline JSON in prose must NOT be unwrapped.
    EvalCase(
        name="inline_json_in_prose_preserved",
        kind=EvalKind.SANITISER,
        category="regression",
        sanitiser_input=(
            'Aqui vai um exemplo de JSON inline: {"key": "value"}. '
            'Funciona assim.'
        ),
        expected_contains='{"key": "value"}',
        description=(
            "When prose contains an inline JSON example the sanitiser "
            "must leave it alone — only WHOLE-RESPONSE JSON envelopes "
            "get unwrapped."
        ),
        tags=["regression", "jou-s13-blocoF", "bug4", "false_positive_guard"],
    ),

    # Bug 5 — known issue, KNOWLEDGE-injection confuses 8B on complex
    # multi-intent queries. Documented for the next bloco that
    # tackles system prompt + KNOWLEDGE handling.
    EvalCase(
        name="known_issue_calendar_jogos_complex",
        kind=EvalKind.DISPATCH,
        category="regression",
        input=(
            "cortex, estou na inglaterra e sou brasileiro, entao quero "
            "saber os horarios dos proximos jogos da seleção brasileira "
            "e adicionar ao meu calendario, me ajuda."
        ),
        # Routing-wise this IS correct: TOOL_HINT_RE picks up
        # "calendário" so it heads into chat_with_tools. The bug was
        # downstream — Llama 8B with [KNOWLEDGE] perfis injection asked
        # irrelevant clarifying questions instead of acting. FIXED in
        # sessão 14: chat_with_tools now suppresses [KNOWLEDGE] in the
        # ACT path (verified by pytest test_chat_with_tools_suppresses_
        # knowledge — eval can't run the 8B). This case still pins the
        # routing so a future dispatch change can't silently break it.
        expected_mode="OFFLINE",
        expected_matcher="fallthrough",
        description=(
            "Bug 5 (multi-intent web_search + create_event): KNOWLEDGE "
            "injection used to derail the 8B into irrelevant questions. "
            "Fixed by suppressing KNOWLEDGE in the ACT path; this eval "
            "pins the routing decision (chat_with_tools)."
        ),
        tags=["regression", "jou-s13-blocoF", "jou-s14", "bug5"],
    ),

    # Sessão 20 live bug — the user asked Cortex to search & schedule the
    # Brasileira's next games. The 8B emitted a raw `{"object": {...}}`
    # JSON envelope as the user-facing reply (status_mode=OFFLINE, no
    # tool actually called — the model "imagined" the event as a JSON
    # payload). Every keyed regex in the sanitiser missed it because
    # the wrapper key was "object", not "kw"/"name". Sessão 20 added a
    # generic raw-JSON-object guard: any pure JSON object as the WHOLE
    # reply is now replaced with an honest "couldn't process, rephrase"
    # message. The dispatch-layer fix is separate (planner regression
    # below).
    EvalCase(
        name="raw_object_wrapper_leak_replaced_with_refusal",
        kind=EvalKind.SANITISER,
        category="regression",
        sanitiser_input=(
            '{"object": {"summary": "Proximos jogos da Seleção '
            'Brasileira", "start_iso": "2026-06-13T19:00:00+00:00", '
            '"end_iso": "2026-06-13T22:00:00+00:00"}}'
        ),
        expected_contains="Não consegui completar",
        expected_not_contains='"object"',
        description=(
            "Exact live leak from sessão 20: an entire calendar-event "
            "payload smuggled into the reply as a JSON envelope. Must "
            "be replaced with a refusal, NOT leaked to TTS / iPhone."
        ),
        tags=["regression", "jou-s20", "sanitiser-leak"],
    ),
    EvalCase(
        name="raw_function_wrapper_leak_refused",
        kind=EvalKind.SANITISER,
        category="regression",
        sanitiser_input='{"function": {"name": "send_email", "to": "x"}}',
        expected_contains="Não consegui completar",
        expected_not_contains='"function"',
        description=(
            "Variant: wrapper key is 'function' instead of 'object' — "
            "the generic raw-JSON guard catches both."
        ),
        tags=["regression", "jou-s20"],
    ),
]
