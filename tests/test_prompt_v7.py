"""117b: only the always-on NEWS guidance leaves V6; dated memory stays."""
import pytest

from src.core import prompts


@pytest.fixture
def v7():
    return prompts.SYSTEM_PROMPT_V7


def test_latest_is_live_and_v7_and_v6_remain_selectable():
    assert prompts.LATEST_PROMPT_VERSION == 13
    assert prompts.resolve_prompt_version(None) == 13
    assert prompts.resolve_prompt_version("7") == 7
    assert prompts.resolve_prompt_version("6") == 6
    assert prompts.get_prompt_body(6) == prompts.SYSTEM_PROMPT_V6


def test_v7_is_exactly_v6_without_news_and_pointer(v7):
    v6 = prompts.SYSTEM_PROMPT_V6
    news = "NEWS:\n" + v6.split("NEWS:\n", 1)[1].split("\n\n", 1)[0] + "\n\n"
    expected = v6.replace(news, "", 1).replace(
        "'). Condolence is not an apology — see NEWS.\n", "').\n", 1)
    assert v7 == expected
    assert "NEWS" not in v7
    assert "PAST CONTEXT conversations predate today" not in v7
    assert "- Apologize gratuitously for yourself ('I'm sorry but', 'sinto muito, mas').\n" in v7


def test_v7_keeps_knowledge_and_fits_existing_cap(v7):
    assert "Pedro speaks TO Cortex: 'você/seu/sua' ('you/your') means Cortex" in v7
    assert "When PAST CONTEXT contains a chunk tagged [KNOWLEDGE]" in v7
    assert len(v7) // 4 < 1850
