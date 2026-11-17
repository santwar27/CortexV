"""
i18n lint (s61, Tier 1) — a NEW dispatch handler can't silently regress the
spoken-language localisation (s60/s61) without this failing.

Rule: in every `_handle_*` method of the handler mixins (src/core/*_mixin.py),
each `return` that yields a user-facing reply must produce that reply through
the localisation helper `self._t(...)` / `t(...)` — NOT a bare string/f-string
literal. Replies built elsewhere (a variable, or a delegated call like
`entities.last_contact_reply(lang=...)`) are fine; the lint only flags raw
literals, which are exactly the "hardcoded PT" the s60 work removed.

Static (AST) — no execution, runs in `make check`. If a legitimately
language-neutral literal ever needs to ship, add it to `_ALLOW` with a reason.
"""
import ast
import re
from pathlib import Path

import pytest

# A "real word" = 3+ consecutive letters. Distinguishes translatable prose
# ("Anotado", "Opening") from structure/punctuation ("\n", ": ", " — ", "%.")
# and from interpolated names, which carry no literal prose of their own.
_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ]{3,}")

_MIXINS = sorted((Path(__file__).resolve().parent.parent / "src" / "core").glob("*_mixin.py"))

# Intentional exceptions (module stem, literal prefix, reason). Empty today:
# every _handle_* reply is localised. Tool-results that the LLM reframes live
# in _execute_tool* (NOT _handle_*), so they're out of scope by construction.
_ALLOW: list[tuple[str, str]] = []


def _reply_node(ret: ast.Return):
    """The user-facing element of a handler return: the first item of a
    `(reply, mode)` tuple, or the bare value otherwise. None for `return`/
    `return None`."""
    v = ret.value
    if v is None:
        return None
    if isinstance(v, ast.Tuple) and v.elts:
        return v.elts[0]
    return v


def _is_localised_call(node: ast.AST) -> bool:
    """True if `node` is a call to `self._t(...)` or `t(...)`."""
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    if isinstance(f, ast.Attribute) and f.attr == "_t":
        return True
    if isinstance(f, ast.Name) and f.id == "t":
        return True
    return False


def _has_untranslated_prose(node: ast.AST) -> bool:
    """True if `node` contains a raw word-bearing string literal that is NOT
    inside a localisation call. Punctuation/separators ('\\n', ': ', ' — '),
    interpolated names, and `self._t(...)`/delegated calls don't count — so an
    f-string that merely assembles already-localised pieces + data is fine, but
    a bare prose literal ('Anotado: ...') is flagged."""
    if _is_localised_call(node):
        return False  # the prose lives inside _t(pt, en) → localised
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return bool(_WORD_RE.search(node.value))
    if isinstance(node, ast.JoinedStr):  # f-string: only its literal text parts
        return any(
            isinstance(v, ast.Constant) and isinstance(v.value, str)
            and _WORD_RE.search(v.value)
            for v in node.values
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _has_untranslated_prose(node.left) or _has_untranslated_prose(node.right)
    return False  # Name / opaque call (e.g. "\n".join(...), delegated) → skip


def _violations_in(path: Path) -> list[str]:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    out: list[str] = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef) or not fn.name.startswith("_handle_"):
            continue
        for ret in (n for n in ast.walk(fn) if isinstance(n, ast.Return)):
            reply = _reply_node(ret)
            if reply is None or not _has_untranslated_prose(reply):
                continue  # localised, a variable, or pure structure/data → OK
            snippet = ast.get_source_segment(src, reply) or ""
            if any(path.stem == stem and snippet.lstrip("f\"'").startswith(pref)
                   for stem, pref in _ALLOW):
                continue
            out.append(f"{path.name}:{ret.lineno} {fn.name}: bare reply prose "
                       f"{snippet[:60]!r} — wrap in self._t(pt, en).")
    return out


@pytest.mark.parametrize("path", _MIXINS, ids=lambda p: p.name)
def test_handler_replies_are_localised(path):
    violations = _violations_in(path)
    assert not violations, "Unlocalised handler reply(s):\n" + "\n".join(violations)


def _expr(code: str) -> ast.AST:
    return ast.parse(code, mode="eval").body


class TestProseDetector:
    @pytest.mark.parametrize("code", [
        'f\'Anotado: "{title}".\'',          # bare PT prose f-string
        '"Pausado."',                         # bare PT constant
        '"Olá " + name',                      # prose in a concat operand
        'f"Opening {app}."',                  # bare EN prose
    ])
    def test_flags_untranslated(self, code):
        assert _has_untranslated_prose(_expr(code)) is True

    @pytest.mark.parametrize("code", [
        'self._t("Pausado.", "Paused.")',     # localised
        't(lang, "a", "b")',                  # localised (module fn)
        'reply',                               # variable
        'f"{c.name}: {x}."',                  # name + punctuation only
        'header + "\\n" + "\\n".join(lines)',  # localised var + structure/data
        'entities.last_contact_reply(q, lang=lang)',  # delegated call
        'None',
    ])
    def test_passes_localised_or_structural(self, code):
        assert _has_untranslated_prose(_expr(code)) is False


def test_lint_actually_scans_handlers():
    # Guard the guard: make sure the glob found the mixins and the walker sees
    # _handle_ methods (so a silent no-op can't make the lint vacuously pass).
    assert any(p.name == "notes_mixin.py" for p in _MIXINS)
    src = (Path(__file__).resolve().parent.parent / "src/core/notes_mixin.py")
    names = [n.name for n in ast.walk(ast.parse(src.read_text()))
             if isinstance(n, ast.FunctionDef) and n.name.startswith("_handle_")]
    assert "_handle_note_create" in names
