"""Task 116: condolences, unverified news, and Pedro's taught pronouns."""
import pytest

from src.core import prompts


OLD_APOLOGY = "- Apologize gratuitously ('sinto muito', 'I'm sorry but').\n"
NEW_APOLOGY = (
    "- Apologize gratuitously for yourself ('I'm sorry but', 'sinto muito, mas'). "
    "Condolence is not an apology — see NEWS.\n"
)
TAUGHT_SENTENCE = (
    "In [KNOWLEDGE] chunks titled '# Pedro — o que ele ensinou ao Cortex', "
    "Pedro speaks TO Cortex: 'você/seu/sua' ('you/your') means Cortex, so "
    "answer about yourself in the first person ('Meu nome veio…'), never "
    "about Pedro. "
)


@pytest.fixture
def v6():
    body = getattr(prompts, "SYSTEM_PROMPT_V6", None)
    assert body is not None, "SYSTEM_PROMPT_V6 must exist"
    return body


def test_v6_remains_selectable(v6):
    assert prompts.resolve_prompt_version("6") == 6
    assert prompts.get_prompt_body(6) == v6


def test_latest_version_is_thirteen():
    assert prompts.LATEST_PROMPT_VERSION == 13


def test_v5_remains_selectable():
    assert prompts.resolve_prompt_version("5") == 5
    assert prompts.get_prompt_body(prompts.resolve_prompt_version("5")) == prompts.SYSTEM_PROMPT_V5


def test_condolence_is_excluded_from_apology_ban(v6):
    do_not = v6.split("DO NOT:\n", 1)[1].split("\n\n", 1)[0]
    assert NEW_APOLOGY.rstrip() in do_not
    assert OLD_APOLOGY.rstrip() not in do_not
    assert "'sinto muito'" not in do_not


@pytest.fixture
def news(v6):
    return v6.split("NEWS:\n", 1)[1].split("\n\n", 1)[0]


def test_sad_news_starts_with_brief_personal_condolence(news):
    assert "Sad personal news (death, illness, loss)" in news
    assert "one short, warm condolence to Pedro" in news
    assert "'Sinto muito, Pedro.' / 'I'm sorry, Pedro.'" in news
    assert "optionally add 1–2 sentences about the person or thing" in news
    assert "No lecture, lists or offers of help" in news


def test_present_news_needs_evidence_to_be_confirmed(news):
    assert "User-reported present news (today/recent days)" in news
    assert "unconfirmed by tool results or PAST CONTEXT: take their word" in news
    assert "never confirm as known" in news
    for phrase in ("'sim'", "'yes'", "'that's right'", "'é verdade'"):
        assert phrase in news
    assert "or invent details (date, cause, age)" in news


def test_news_has_literal_wrong_right_pair(news):
    assert "WRONG: Ele faleceu hoje, sim." in news
    assert (
        "RIGHT: Sinto muito, Pedro. Ele marcou muita gente — faz sentido "
        "que tenha marcado você também."
    ) in news


def test_news_stays_short(news):
    assert len(news.split()) <= 90


def test_taught_pronouns_are_in_existing_knowledge_paragraph(v6):
    paragraph = v6.split("If PAST CONTEXT from prior conversations", 1)[1].split("\n\n", 1)[0]
    assert TAUGHT_SENTENCE in paragraph
    assert v6.count(TAUGHT_SENTENCE) == 1


def test_only_authorized_sections_change(v6, news):
    # Undo the three permitted edits: every other byte must still be V5,
    # including the upstream music note, NEVER SIMULATE, and TOOL ARG FIDELITY.
    restored = v6.replace(NEW_APOLOGY, OLD_APOLOGY, 1)
    restored = restored.replace("NEWS:\n" + news + "\n\n", "", 1)
    restored = restored.replace(TAUGHT_SENTENCE, "", 1)
    assert restored == prompts.SYSTEM_PROMPT_V5
