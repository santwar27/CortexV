"""
M2-A7 / M8-A6 — keep_alive on every Ollama call site.

Ollama's keep_alive is last-call-wins: a bare ollama.generate/chat resets
the model unload timer to 5 min, so the engine's 30m warm-model win was
intermittently undone by the realtime classifier, the planner, the nightly
Studier, and the deep/dream scaffolds. Every site now passes
keep_alive=_KEEP_ALIVE (the value lives once in ollama_client.KEEP_ALIVE).

These tests pin the single source of truth AND — via AST, the permanent
net the review asked for — that NO ollama.generate/chat call in src/ is
missing the kwarg. A future raw call that forgets it fails here.

Task 166a adds the shared THINK flag with the same explicit per-site seam,
so Gemma cannot silently spend a call's answer budget on thinking.
"""
import ast
import importlib
import pathlib

import pytest

from src.core import ollama_client


class TestKeepAliveConstant:
    def test_default_is_30m(self, monkeypatch):
        import importlib
        monkeypatch.delenv("CORTEX_KEEP_ALIVE", raising=False)
        reloaded = importlib.reload(ollama_client)
        try:
            assert reloaded.KEEP_ALIVE == "30m"
        finally:
            importlib.reload(ollama_client)

    def test_reads_env(self, monkeypatch):
        import importlib
        monkeypatch.setenv("CORTEX_KEEP_ALIVE", "99m")
        reloaded = importlib.reload(ollama_client)
        try:
            assert reloaded.KEEP_ALIVE == "99m"
        finally:
            monkeypatch.delenv("CORTEX_KEEP_ALIVE", raising=False)
            importlib.reload(ollama_client)

    def test_engine_shares_the_same_value(self):
        # engine.py imports KEEP_ALIVE as _KEEP_ALIVE — one knob, not two.
        from src.core import engine
        assert engine._KEEP_ALIVE == ollama_client.KEEP_ALIVE


class TestThinkConstant:
    @pytest.mark.parametrize("value,expected", [
        (None, False), ("", False), ("0", False), ("false", False),
        ("no", False), ("off", False), ("2", False), ("enabled", False),
        ("1", True), ("true", True), ("yes", True), ("on", True),
        (" TrUe ", True), (" YES\t", True), (" ON ", True),
    ])
    def test_env_parsing(self, monkeypatch, value, expected):
        try:
            with monkeypatch.context() as env:
                if value is None:
                    env.delenv("CORTEX_THINK", raising=False)
                else:
                    env.setenv("CORTEX_THINK", value)
                assert importlib.reload(ollama_client).THINK is expected
        finally:
            importlib.reload(ollama_client)

    def test_engine_shares_the_same_value(self):
        from src.core import engine
        assert engine._THINK is ollama_client.THINK


def _is_ollama_call(node: ast.Call) -> str | None:
    """Return 'generate'/'chat' if the call is ollama.<fn>(...), else None."""
    f = node.func
    if isinstance(f, ast.Attribute) and f.attr in ("generate", "chat"):
        if isinstance(f.value, ast.Name) and f.value.id == "ollama":
            return f.attr
    return None


class TestEveryCallSiteHasKeepAlive:
    """Permanent net: every ollama.generate/chat call in src/ must carry a
    keep_alive keyword (direct kwarg, or via the keep_alive-bound lambda the
    injected planner sites use). Forgetting it silently downgrades the
    warm-model timer again — the exact M2-A7/M8-A6 bug."""

    def test_no_bare_ollama_calls(self):
        src = pathlib.Path(__file__).resolve().parent.parent / "src"
        offenders = []
        for py in src.rglob("*.py"):
            tree = ast.parse(py.read_text(), filename=str(py))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and _is_ollama_call(node):
                    has_ka = any(k.arg == "keep_alive" for k in node.keywords)
                    if not has_ka:
                        rel = py.relative_to(src)
                        offenders.append(f"{rel}:{node.lineno} ollama.{_is_ollama_call(node)}")
        assert not offenders, (
            "ollama call without keep_alive (downgrades the warm timer, "
            "M2-A7/M8-A6) — pass keep_alive=_KEEP_ALIVE:\n" + "\n".join(offenders)
        )


class TestEveryCallSiteHasThink:
    """Gemma thinks by default: every direct call must opt in or out explicitly."""

    def test_no_bare_ollama_calls(self):
        src = pathlib.Path(__file__).resolve().parent.parent / "src"
        offenders = []
        for py in src.rglob("*.py"):
            tree = ast.parse(py.read_text(), filename=str(py))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and _is_ollama_call(node):
                    if not any(k.arg == "think" for k in node.keywords):
                        offenders.append(
                            f"{py.relative_to(src)}:{node.lineno} ollama.{_is_ollama_call(node)}"
                        )
        assert not offenders, (
            "ollama call without think — pass the shared THINK value "
            "(False for the realtime classifier):\n" + "\n".join(offenders)
        )
