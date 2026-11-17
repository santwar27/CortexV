"""Two dispatch regexes took seconds to refuse an input (s88, Codex lote 9).

Measured before the fix: FITNESS_CLEAR_TODAY_RE on "limpa " + 24 × "todas
as " + "hoje" took 8.5s (exponential: each "todas as" read two ways), and
MATH_ONLY_RE on "1" + 2000 spaces + "x" took 10s (cubic). Run in a child
process so a regression fails on the clock instead of hanging the suite.
"""
import subprocess
import sys

import pytest

ROOT = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))


@pytest.mark.parametrize("code", [
    "from src.core.dispatch.fitness import FITNESS_CLEAR_TODAY_RE as R\n"
    "assert R.search('limpa ' + 'todas as ' * 200 + 'hoje') is None",
    "from src.core.dispatch.heuristics import MATH_ONLY_RE as R\n"
    "assert R.match('1' + ' ' * 20000 + 'x') is None",
])
def test_refused_in_well_under_a_second(code):
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, timeout=5)


def test_math_still_reads_math():
    from src.core.dispatch.heuristics import MATH_ONLY_RE as R
    for s in ("2+2", " 15 * 7 ", "(3+1)/2 = ", "2^10?", "3,5 + 1 ?  "):
        assert R.match(s), s
    for s in ("2+2 é quanto", "x+1", "?"):
        assert not R.match(s), s
