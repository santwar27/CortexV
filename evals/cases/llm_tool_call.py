"""Tool-call behaviour cases — real engine.chat_with_tools() (sessão C).

These check the part of the pipeline `chat()` can't: when a prompt is
an ACTION ("send an email", "schedule a meeting"), does the model pick
the RIGHT tool from TOOL_SPECS? The runner exposes the real specs and
runs the real tool-calling loop, but with a side-effect-free stub
executor — nothing is sent, scheduled, or read; we only observe which
tool the model selected (and optionally its args).

Same contract as LLM_BEHAVIOR: opt-in (`make eval-llm`), never in
`make check`, and `drift_tolerant=True` — an 8B's tool selection wobbles
run to run, so a miss is DRIFT (visible, non-fatal), not a failure.
Model-agnostic: zero model name here; the runner reads CORTEX_MODEL.

Arg substring checks are kept deliberately light and accent-free — they
exist to catch gross arg mistakes, not to pin exact phrasing an 8B can't
guarantee.
"""
from evals.core import EvalCase, EvalKind


def _tool(name, prompt, *, tool=None, no_tool=False, args=None,
          language="pt", desc="", tags=None):
    return EvalCase(
        name=name,
        kind=EvalKind.LLM_TOOL_CALL,
        category="llm_tool_call",
        input=prompt,
        expected_tool=tool,
        expect_no_tool=no_tool,
        expected_arg_contains=dict(args or {}),
        language=language,
        description=desc,
        tags=list(tags or []),
    )


CASES: list[EvalCase] = [
    _tool(
        "read_document_pt",
        "abre o documento roadmap.md e me diz o que tem nele",
        tool="read_document",
        args={"filename": "roadmap"},
        desc="Reading the user's own vault → read_document, filename echoed.",
    ),
    _tool(
        "create_event_pt",
        "agenda uma reunião amanhã às 10h sobre o projeto",
        tool="create_event",
        desc="Scheduling → create_event (args vary too much on an 8B to pin).",
    ),
    _tool(
        "send_email_pt",
        "envia um email para maria@empresa.com avisando que vou chegar atrasado",
        tool="send_email",
        args={"to": "maria"},
        desc="Composing mail → send_email, recipient carried into `to`.",
    ),
    _tool(
        "list_events_pt",
        "o que eu tenho na agenda essa semana?",
        tool="list_events",
        desc="Asking about the agenda → read-only list_events.",
    ),
    _tool(
        "web_search_realtime_pt",
        "qual o preço do bitcoin hoje?",
        tool="web_search",
        args={"query": "bitcoin"},
        desc="Real-time, changes over time → web_search with the topic in query.",
    ),
    _tool(
        "stable_knowledge_no_tool_pt",
        "o que é fotossíntese?",
        no_tool=True,
        desc=(
            "Stable knowledge — the specs explicitly say answer directly, "
            "NOT web_search. The model should reply in chat, no tool."
        ),
        tags=["anti_overtooling"],
    ),
    _tool(
        "chitchat_no_tool_pt",
        "oi, tudo bem?",
        no_tool=True,
        desc="Plain greeting must not trigger any tool.",
        tags=["anti_overtooling"],
    ),
    _tool(
        "write_document_pt",
        "salva 'comprar leite e pão' no arquivo notas.txt",
        tool="write_document",
        args={"filename": "notas"},
        desc="Explicit save verb + filename → write_document with filename carried.",
    ),
    _tool(
        "create_reminder_pt",
        "me lembra de ligar pro joão amanhã às 9h",
        tool="create_reminder",
        desc="'Me lembra' phrasing → create_reminder (args vary too much to pin).",
    ),
    _tool(
        "polite_email_pt",
        "pode mandar um email pra ana@empresa.com avisando que vou atrasar?",
        tool="send_email",
        args={"to": "ana"},
        desc=(
            "Polite/indirect form ('pode mandar') must still trigger "
            "send_email — refusing politely would be a regression."
        ),
    ),
    _tool(
        "polite_email_absence_pt",
        "dá pra mandar um email pra ana@empresa.com avisando que não vou hoje?",
        tool="send_email",
        desc="Indirect 'dá pra mandar' request must select send_email.",
    ),
    _tool(
        "favor_email_late_pt",
        "me faz um favor, manda um email pra ana@empresa.com avisando que vou atrasar",
        tool="send_email",
        desc="A favour phrased as an action must select send_email.",
    ),
    _tool(
        "polite_reminder_rent_pt",
        "consegue me lembrar de pagar o aluguel dia 5?",
        tool="create_reminder",
        desc="Polite 'consegue me lembrar' request must select create_reminder.",
    ),
    _tool(
        "general_knowledge_engine_no_tool_pt",
        "como funciona um motor de combustão interna?",
        no_tool=True,
        desc=(
            "Non-fotossíntese variant of the stable-knowledge guard — "
            "general engineering knowledge should be answered directly."
        ),
        tags=["anti_overtooling"],
    ),
    _tool(
        "find_file_resume_pdf_pt",
        "cadê meu currículo em pdf",
        tool="find_file",
        args={"query": "curr"},
        desc="Locate a file via Spotlight; query is the key term, without pdf.",
    ),
    _tool(
        "list_emails_recent_pt",
        "mostra meus 3 emails mais recentes",
        tool="list_emails",
        args={"count": "3"},
        desc="Reading recent mail → list_emails, with the requested count.",
    ),
    _tool(
        "remind_rent_not_event_pt",
        "me lembra de pagar o aluguel amanhã às 9h",
        tool="create_reminder",
        desc="A reminder to do a task belongs in Reminders, not Calendar.",
    ),
    _tool(
        "schedule_meeting_not_reminder_pt",
        "marca uma reunião amanhã às 14h sobre o projeto Atlas",
        tool="create_event",
        desc="Scheduling a meeting belongs in Calendar, not Reminders.",
    ),
    _tool(
        "save_project_note_pt",
        "guarda 'o projeto Atlas foi aprovado' no arquivo projeto.txt",
        tool="write_document",
        args={"filename": "projeto"},
        desc="Explicitly storing text → vault write, not document read.",
    ),
    _tool(
        "read_project_note_pt",
        "leia meu documento projeto.txt e resuma o conteúdo",
        tool="read_document",
        args={"filename": "projeto"},
        desc="Reading a named vault document → read_document, not write or find.",
    ),
    _tool(
        "web_search_exchange_rate_pt",
        "qual a cotação do euro hoje?",
        tool="web_search",
        args={"query": "euro"},
        desc="Today's exchange rate needs fresh data, not a memory answer.",
    ),
    _tool(
        "stable_moon_phases_no_tool_pt",
        "por que a Lua tem fases?",
        no_tool=True,
        desc="Stable science → direct answer, without web_search.",
        tags=["anti_overtooling"],
    ),
    _tool(
        "send_email_en",
        "Send an email to alex@example.com saying the report is ready.",
        tool="send_email",
        args={"to": "alex"},
        language="en",
        desc="An English action must select the same mail tool as Portuguese.",
    ),
    _tool(
        "file_in_passing_no_tool_pt",
        "Terminei de revisar relatorio.txt hoje. Me conta uma piada.",
        no_tool=True,
        desc="A passing filename mention is not a request to read, find or save.",
        tags=["anti_overtooling"],
    ),
    _tool(
        "email_in_passing_no_tool_pt",
        "Recebi um email de parabéns hoje. Me conta uma piada.",
        no_tool=True,
        desc="A passing email mention is not an inbox or send request.",
        tags=["anti_overtooling"],
    ),
]
