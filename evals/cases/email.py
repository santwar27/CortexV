"""Gmail dispatch behaviour.

LIST_EMAILS_RE was originally narrow: only "meus emails" / "my inbox"
matched. Real users say "lista meus 3 últimos emails" or "o que tem na
minha inbox?" — both fell through to the LLM, which then mis-picked
list_events (Calendar) and produced a hallucinated "couldn't access
Gmail" reply (sessão 19 live smoke). These cases lock the expanded
regex so a future tightening can't silently regress the natural
shapes.
"""
from evals.core import EvalCase, EvalKind


def _list_emails(name, text, *, desc=""):
    return EvalCase(
        name=name,
        kind=EvalKind.DISPATCH,
        category="email",
        input=text,
        expected_mode="LIST_EMAILS",
        expected_matcher="LIST_EMAILS_RE",
        description=desc,
    )


CASES: list[EvalCase] = [
    _list_emails("bare_pt",          "meus emails"),
    _list_emails("bare_recent_pt",   "meus emails recentes"),
    _list_emails("verb_lista_pt",    "lista meus emails"),
    _list_emails("verb_mostra_pt",   "mostra meus emails"),
    _list_emails("with_count_pt",    "lista meus 3 ultimos emails"),
    _list_emails("with_count_accent_pt", "liste meus últimos 3 emails"),
    _list_emails("with_count_after_modifier_pt", "meus 5 emails"),
    _list_emails("question_pt",      "meus ultimos emails?"),
    _list_emails("inbox_question_pt","o que tem na minha inbox?",
                 desc="Inbox-anchored alternate branch."),
    _list_emails("verb_me_da_pt",    "me dá meus emails"),
    _list_emails("caixa_de_entrada_pt", "caixa de entrada"),
    # EN
    _list_emails("bare_en",          "my inbox"),
    _list_emails("verb_show_en",     "show my inbox"),
    _list_emails("verb_show_me_en",  "show me my last 3 emails"),
    _list_emails("inbox_word_en",    "inbox"),

    # Anti-collision — must NOT match LIST_EMAILS_RE. simulate_dispatch
    # falls through to mode=OFFLINE when no regex fires; the real
    # orchestrator then routes via chat_with_tools (TOOL_HINT_RE).
    EvalCase(
        name="send_email_freeform_does_not_steal_list",
        kind=EvalKind.DISPATCH,
        category="email",
        input="envia um email pra joão sobre atraso",
        expected_mode="OFFLINE",
        expected_matcher="fallthrough",
        description=(
            "Free-form send_email (no field markers) must reject the list "
            "regex AND the structured regex so the orchestrator routes "
            "via chat_with_tools."
        ),
    ),

    # ----- send_email STRUCTURED form (sessão 20 bloco B) -----
    # The user types explicit field markers; the orchestrator extracts
    # fields LITERALLY via the regex's named groups, bypassing the LLM.
    # Fixes the live bug where the 8B paraphrased subject ("eu te amo" →
    # "Í eu te amo") and dropped the "h" from body ("milhões" → "milões").
    EvalCase(
        name="send_email_structured_pt_pedro_live",
        kind=EvalKind.DISPATCH,
        category="email",
        input=(
            "mandar um email para: lara@example.com - assunto: "
            "eu te amo -- Corpo: I love you mil milhões"
        ),
        expected_mode="SEND_EMAIL_STRUCTURED",
        expected_matcher="SEND_EMAIL_STRUCTURED_RE",
        description=(
            "Exact live input from sessão 20 — the regex must catch it "
            "so the LLM never gets the chance to transcribe-wrong."
        ),
        tags=["regression", "jou-s20"],
    ),
    EvalCase(
        name="send_email_structured_en",
        kind=EvalKind.DISPATCH,
        category="email",
        input=(
            "send an email to bob@example.com - subject: hi - "
            "body: hello there"
        ),
        expected_mode="SEND_EMAIL_STRUCTURED",
        expected_matcher="SEND_EMAIL_STRUCTURED_RE",
    ),
    EvalCase(
        name="send_email_structured_pt_comma",
        kind=EvalKind.DISPATCH,
        category="email",
        input="manda email pra a@b.co, assunto: x, corpo: y",
        expected_mode="SEND_EMAIL_STRUCTURED",
        expected_matcher="SEND_EMAIL_STRUCTURED_RE",
        description="Comma separator instead of dash — must still parse.",
    ),
    EvalCase(
        name="send_email_freeform_misses_structured",
        kind=EvalKind.DISPATCH,
        category="email",
        input="envia email pra joão dizendo que vou chegar atrasado",
        expected_mode="OFFLINE",
        expected_matcher="fallthrough",
        description=(
            "Free-form (no labelled fields) must NOT match the structured "
            "regex — it relies on chat_with_tools where the LLM extracts "
            "args with its own (imperfect) judgement."
        ),
    ),

    # ----- Contact book (sessão 21) -----
    EvalCase(
        name="contact_save_pt",
        kind=EvalKind.DISPATCH,
        category="email",
        input="salva o email da lara como lara@example.com",
        expected_mode="CONTACT_SAVE",
        expected_matcher="CONTACT_SAVE_RE",
    ),
    EvalCase(
        name="contact_save_pt_compound_name",
        kind=EvalKind.DISPATCH,
        category="email",
        input="guarda o email da maria julia como mj@x.com",
        expected_mode="CONTACT_SAVE",
        expected_matcher="CONTACT_SAVE_RE",
        description="Multi-token name should be captured whole.",
    ),
    EvalCase(
        name="contact_save_en",
        kind=EvalKind.DISPATCH,
        category="email",
        input="save the email for anna as anna@x.com",
        expected_mode="CONTACT_SAVE",
        expected_matcher="CONTACT_SAVE_RE",
    ),
    EvalCase(
        name="contact_lookup_pt",
        kind=EvalKind.DISPATCH,
        category="email",
        input="qual o email da lara?",
        expected_mode="CONTACT_LOOKUP",
        expected_matcher="CONTACT_LOOKUP_RE",
    ),
    EvalCase(
        name="contact_lookup_en",
        kind=EvalKind.DISPATCH,
        category="email",
        input="what's lara's email?",
        expected_mode="CONTACT_LOOKUP",
        expected_matcher="CONTACT_LOOKUP_RE",
    ),
    EvalCase(
        name="send_by_name_pt",
        kind=EvalKind.DISPATCH,
        category="email",
        input="manda email pra lara - assunto: oi - corpo: tudo bem?",
        expected_mode="SEND_EMAIL_BY_NAME",
        expected_matcher="SEND_EMAIL_BY_NAME_RE",
        description=(
            "Send-by-name routes to the contact-resolver path, which "
            "then stages the same _pending_action as structured."
        ),
    ),
    EvalCase(
        name="send_email_address_form_beats_byname",
        kind=EvalKind.DISPATCH,
        category="email",
        input=(
            "mandar um email para: lara@example.com - assunto: oi "
            "- corpo: tudo bem?"
        ),
        expected_mode="SEND_EMAIL_STRUCTURED",
        expected_matcher="SEND_EMAIL_STRUCTURED_RE",
        description=(
            "Anti-collision: when the target is an email (with the "
            "':' marker), STRUCTURED must win — not BY_NAME."
        ),
    ),
]
