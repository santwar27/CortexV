"""Named old-code scopes exempt only hunks wholly contained within them."""
import difflib

import pytest

from src.core import code_proposals
from src.core.staging_mixin import StagingMixin
from tests.test_code_proposal import setup

OLD = ('def total(prices):\n    """Sum all prices."""\n    result = 0\n'
       '    for p in prices[1:]:\n        result += p\n    return result\n\n'
       'def count(prices):\n    result = len(prices)\n    return result\n')
NEW = OLD.replace('    result = 0\n    for p in prices[1:]:\n        result += p\n    return result',
                  '    return sum(prices)')
REQUEST = {"pt": "conserta o total, que pula o primeiro preço",
           "en": "fix total, which skips the first price"}


def outside(old, new, request, lang):
    old_lines, new_lines = old.splitlines(), new.splitlines()
    opcodes = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False).get_opcodes()
    return StagingMixin._code_change_outside(request, old_lines, new_lines, opcodes, lang)


def warning(lang, lines, deleted=False):
    noun = "linhas" if lang == "pt" else "lines"
    state = ("removidas" if deleted else "mudadas") if lang == "pt" else ("removed" if deleted else "changed")
    return [f"{noun} {lines} ({state})"]


@pytest.mark.parametrize("lang", ["pt", "en"])
def test_measured_total_fix(lang):
    assert outside(OLD, NEW, REQUEST[lang], lang) == []


@pytest.mark.parametrize("lang", ["pt", "en"])
def test_other_function_and_crossing_hunk_still_flagged(lang):
    new = OLD.replace('    result = len(prices)\n    return result', '    return 0')
    assert outside(OLD, new, REQUEST[lang], lang) == warning(lang, "9–10")
    # The hunk crosses two functions without containing the requested name.
    old = 'def total(prices):\n    left = 1\n\ndef count(prices):\n    right = 2\n'
    new = 'def total(prices):\n    left = 3\n\ndef count(prices):\n    right = 4\n'
    new = new.replace('\n\ndef count(prices):', '\n# boundary\ndef count(values):')
    assert outside(old, new, REQUEST[lang], lang) == warning(lang, "2–5")


@pytest.mark.parametrize("lang", ["pt", "en"])
def test_class_and_method_scope(lang):
    old = 'class Cart:\n    def add(self):\n        return 1\n\n    def clear(self):\n        return 2\n'
    new = old.replace('return 1', 'return 3')
    verb = "conserta" if lang == "pt" else "fix"
    assert outside(old, new, verb + " Cart", lang) == []
    assert outside(old, new, verb + " add", lang) == []
    sibling = old.replace('return 2', 'return 4')
    assert outside(old, sibling, verb + " add", lang) == [
        "linha 6 (mudada)" if lang == "pt" else "line 6 (changed)"]


@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("definition", ["def total", "async def total"])
def test_nested_function_scope_and_decorators(lang, definition):
    old = f'class Wrapper:\n    @outer\n    @inner\n    {definition}(prices):\n        return 1\n'
    for new in (old.replace('@outer', '@other'), old.replace('return 1', 'return 2'),
                old.replace('    @outer\n', '')):
        assert outside(old, new, REQUEST[lang], lang) == []


@pytest.mark.parametrize("lang", ["pt", "en"])
def test_nested_decorated_class_scope(lang):
    old = 'def factory():\n    @outer\n    @inner\n    class Cart:\n        value = 1\n    return Cart\n'
    for new in (old.replace('@outer', '@other'), old.replace('value = 1', 'value = 2')):
        assert outside(old, new, "Cart", lang) == []


@pytest.mark.parametrize("lang", ["pt", "en"])
def test_top_level_comments_and_unparseable_old_keep_textual_rule(lang):
    old = '# keep\n# these\n# comments\n' + OLD
    assert outside(old, OLD, REQUEST[lang], lang) == warning(lang, "1–3", deleted=True)
    broken = OLD + '\ndef broken(:\n'
    assert outside(broken, NEW + '\ndef broken(:\n', REQUEST[lang], lang) == warning(lang, "3–6")
    # The original textual anchor still exempts an unparseable hunk.
    assert outside(broken, broken.replace('def total(', 'def subtotal('), REQUEST[lang], lang) == []
    # Anchor names remain case sensitive, and a name in NEW alone grants no old scope.
    assert outside(OLD, NEW, "TOTAL", lang) == warning(lang, "3–6")
    old = 'def count(prices):\n    return 1\n\nsentinel = 0\n'
    new = old.replace('return 1', 'return 2') + '\ndef total(prices):\n    return 0\n'
    assert outside(old, new, REQUEST[lang], lang) == [
        "linha 2 (mudada)" if lang == "pt" else "line 2 (changed)"]


def test_route_total_fix_has_no_outside_warning(setup):
    o, llm, path = setup
    path = path.rename(path.with_name("cart.py"))
    path.write_text(OLD)
    llm.return_value = {"message": {"content": "FILE: cart.py\n```python\n" + NEW + "```"}}
    ask = "muda em ~/cortex_sandbox/cart.py: " + REQUEST["pt"]
    assert o._process_core(ask)[1] == "AGENT_AWAITING_CONFIRMATION"
    reply, mode = o._process_core("sim")
    assert mode == "CODE_PROPOSAL"
    assert "mudou fora do que você pediu" not in reply
    pid = o._pending_action["args"]["id"]
    assert code_proposals.load(pid)["outside"] == []
    assert path.read_text() == OLD
    llm.assert_called_once()
