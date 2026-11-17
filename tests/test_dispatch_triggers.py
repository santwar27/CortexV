"""Dispatch trigger regexes — quick unit coverage for the bits the
eval DISPATCH cases can't easily express (named groups, capture values).

The eval suite already locks the routing decisions per natural phrase;
this file pins the structural details: which numeric group fires
when, what the trailing-noun branch returns, etc.
"""
import pytest

from src.core.dispatch.triggers import (
    LIST_EMAILS_RE,
    SEND_EMAIL_STRUCTURED_RE,
)


class TestListEmailsCount:
    """LIST_EMAILS_RE exposes the user-given count via one of two named
    groups — `count_pre` ("3 últimos emails") or `count_post`
    ("últimos 3 emails"). Only one fires per match; the orchestrator
    OR-s them when deciding the count to pass to Gmail."""

    @pytest.mark.parametrize("text,expected", [
        ("lista meus 3 ultimos emails", "3"),
        ("liste meus últimos 3 emails", "3"),
        ("meus 5 emails", "5"),
        ("show me my last 3 emails", "3"),
        ("meus últimos 10 emails", "10"),
    ])
    def test_count_captured(self, text, expected):
        m = LIST_EMAILS_RE.match(text)
        assert m is not None, f"{text!r} should match"
        got = m.groupdict().get("count_pre") or m.groupdict().get("count_post")
        assert got == expected

    @pytest.mark.parametrize("text", [
        "meus emails",
        "meus ultimos emails?",
        "o que tem na minha inbox?",
        "inbox",
        "show my inbox",
    ])
    def test_no_count_when_user_omits(self, text):
        m = LIST_EMAILS_RE.match(text)
        assert m is not None
        got = m.groupdict().get("count_pre") or m.groupdict().get("count_post")
        assert got is None, f"expected no count, got {got!r}"


class TestSendEmailStructuredGroups:
    """SEND_EMAIL_STRUCTURED_RE bypass for the 8B paraphrase bug
    (sessão 20 live: 'eu te amo' → 'Í eu te amo'; 'milhões' → 'milões').
    Named groups extract verbatim — the whole point is no LLM in the
    path."""

    def test_pedro_live_input_extracts_clean(self):
        text = (
            "mandar um email para: lara@example.com - assunto: "
            "eu te amo -- Corpo: I love you mil milhões"
        )
        m = SEND_EMAIL_STRUCTURED_RE.match(text)
        assert m is not None
        # The h in "milhões" MUST survive — that's the whole point.
        assert m.group("to") == "lara@example.com"
        assert m.group("subject").strip() == "eu te amo"
        assert m.group("body").strip() == "I love you mil milhões"

    def test_en_dash_separators(self):
        m = SEND_EMAIL_STRUCTURED_RE.match(
            "send an email to bob@example.com - subject: hi - body: hello there"
        )
        assert m is not None
        assert m.group("to") == "bob@example.com"
        assert m.group("subject").strip() == "hi"
        assert m.group("body").strip() == "hello there"

    def test_comma_separators(self):
        m = SEND_EMAIL_STRUCTURED_RE.match(
            "manda email pra a@b.co, assunto: x, corpo: y"
        )
        assert m is not None
        assert m.group("to") == "a@b.co"

    @pytest.mark.parametrize("text", [
        "envia email pra joão sobre atraso",          # no field markers
        "send email to bob",                          # no field markers
        "lista meus emails",                          # list, not send
    ])
    def test_does_not_match_free_form(self, text):
        # Free-form (no labelled fields) MUST fall through so the LLM
        # tool path stays the route for unstructured requests.
        assert SEND_EMAIL_STRUCTURED_RE.match(text) is None


class TestSearchStripRe:
    """The provider query must not keep dangling medium words (s73: the
    Copa bug trace shows the Tavily query starting with 'online qual o ...'
    because only the verb 'pesquise' was stripped)."""

    @pytest.mark.parametrize("text,expected", [
        ("pesquise online qual o proximo jogo", "qual o proximo jogo"),
        ("pesquise na internet sobre grafeno", "grafeno"),
        ("search on the web for graphene", "graphene"),
        ("busca na web clima amanhã", "clima amanhã"),
        # No medium word — behaviour unchanged.
        ("pesquise sobre grafeno", "grafeno"),
        ("look up for graphene", "graphene"),
    ])
    def test_strips_verb_and_medium_word(self, text, expected):
        from src.core.dispatch.triggers import SEARCH_STRIP_RE
        assert SEARCH_STRIP_RE.sub("", text) == expected

    def test_online_mid_query_is_kept(self):
        # Only the LEADING trigger words are stripped — 'online' as part
        # of the actual subject must survive.
        from src.core.dispatch.triggers import SEARCH_STRIP_RE
        assert SEARCH_STRIP_RE.sub("", "pesquise sobre cursos online de python") == \
            "cursos online de python"
