"""Approved project inventory, deterministic context, and no real model."""
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.core import code_roots, planner
from src.core.prompt_safety import OPEN, CLOSE
from src.tools.local_reader import LocalFileReader
from tests.test_dispatch_parity import _build_real_orch
from tests.test_code_explain import SYSTEM

ASK = "lê ~/cortex_sandbox/loja e explica como apply_discount é usado em add_item"
SOURCES = {
    "cart.py": "from discounts import apply_discount\nclass Cart:\n    def add_item(self, price):\n        return apply_discount(price)\n",
    "discounts.py": "def apply_discount(price):\n    return price / 2\n",
    "models.py": "class Product:\n    pass\n",
    "tests/test_cart.py": "def test_cart():\n    assert True\n",
}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CORTEX_CODE_ROOTS_FILE", raising=False)
    root = tmp_path / "cortex_sandbox"
    root.mkdir()
    assert code_roots.add(str(root))[0]
    o = _build_real_orch(tmp_path)
    o._PENDING_ACTION_FILE = str(tmp_path / "pending.json")
    o.background, o._reply_lang = False, "pt"
    o.reader = LocalFileReader(str(tmp_path / "vault"))
    o.reader.read_code = Mock(wraps=o.reader.read_code)
    llm = Mock(return_value={"message": {"content": "fixed answer"}})
    monkeypatch.setattr("src.core.staging_mixin.ollama.chat", llm)
    monkeypatch.setattr("src.core.doc_retrieval._default_embed", Mock(side_effect=AssertionError("embedding")))
    def put(name, content="pass\n"):
        path = root / "loja" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path
    for name, content in SOURCES.items():
        put(name, content)
    return o, llm, put


def body(llm):
    llm.assert_called_once()
    call = llm.call_args.kwargs
    assert len(call["messages"]) == 2 and "tools" not in call
    return call["messages"][1]["content"].split(OPEN + "\n")[1].split("\n" + CLOSE)[0]


@pytest.mark.parametrize("stream", [False, True])
def test_discovery_and_one_complete_preview(setup, stream):
    o, llm, put = setup
    for name in (".venv/x.py", "__pycache__/y.py", "node_modules/z.py", "notes.md",
                 "venv/a.py", "env/a.py", "site-packages/a.py", "build/a.py", "dist/a.py",
                 "a/b/c/d/e/f/deep.py"):
        put(name)
    outside = Path.home() / "outside.py"
    outside.write_text("pass\n")
    put("fora.py").unlink()
    (put("notes.md").parent / "fora.py").symlink_to(outside)
    ask = "lê ~/cortex_sandbox/loja e explica como o desconto é aplicado no carrinho"
    if stream:
        *_, (reply, _, mode, _) = o.process_stream(ask, language="pt")
    else:
        reply, mode = o._process_core(ask)
    real = str((Path.home() / "cortex_sandbox/loja").resolve())
    assert mode == "AGENT_AWAITING_CONFIRMATION"
    assert o._pending_action["type"] == "read_code_project"
    assert o._pending_action["args"] == dict(root=real, files=sorted(SOURCES),
        question=ask.replace("~/cortex_sandbox/loja", "loja"), lang="pt")
    assert reply == (f"Posso abrir e ler o projeto loja? São 4 arquivos .py em {real}: "
        "cart.py, discounts.py, models.py e tests/test_cart.py.\n"
        "Só leitura — nada é alterado e nenhum comando roda.\n"
        "Responda 'sim' para ler ou 'não' para cancelar.")
    llm.assert_not_called()
    o.reader.read_code.assert_not_called()


@pytest.mark.parametrize("question,order", [
    (ASK, ["cart.py", "discounts.py", "models.py", "tests/test_cart.py"]),
    (ASK.replace("apply_discount é usado em add_item", "models apply_discount"),
     ["models.py", "discounts.py", "cart.py", "tests/test_cart.py"]),
])
def test_map_code_notes_and_one_emission(setup, monkeypatch, question, order):
    o, llm, _ = setup
    log, note = Mock(), Mock()
    monkeypatch.setattr("src.core.staging_mixin.audit.log", log)
    monkeypatch.setattr("src.core.prompt_safety.note_false_read", note)
    assert o._process_core(question)[1] == "AGENT_AWAITING_CONFIRMATION"
    emitted = []
    o.set_token_emitter(emitted.append)
    reply, mode = o._process_core("sim")
    text = body(llm)
    assert text.startswith("MAP: cart.py (4 linhas) — define: Cart, Cart.add_item\n"
        "MAP: discounts.py (2 linhas) — define: apply_discount\n"
        "MAP: models.py (2 linhas) — define: Product\n"
        "MAP: tests/test_cart.py (2 linhas) — define: test_cart\n\n")
    assert text.split("\n\n", 1)[1] == "\n\n".join("FILE: " + name + "\n" +
        "\n".join(f"L{i:<3} {line}" for i, line in enumerate(SOURCES[name].splitlines(), 1)) for name in order)
    assert reply == "fixed answer\n\n(Li inteiros: " + ", ".join(order) + ".)"
    assert emitted == [reply] and mode == "LOCAL_RAG" and len(text) <= 6000
    assert llm.call_args.kwargs["messages"][0]["content"] == SYSTEM
    assert llm.call_args.kwargs["messages"][1]["content"].endswith("PEDIDO DO PEDRO: " + question.replace("~/cortex_sandbox/loja", "loja"))
    note.assert_called_once_with("fixed answer", "code")
    events = [c for c in log.call_args_list if c.args == ("–", "code", "project")]
    assert len(events) == 1
    assert events[0].kwargs["payload"] == dict(files=4, whole=4, partial=0, map_only=0,
        chars=len(text), ms=events[0].kwargs["payload"]["ms"])


def test_budget_prioritises_last_definition_and_reports_omissions(setup):
    o, llm, put = setup
    for name in SOURCES:
        put(name).unlink()
    for i in range(30):
        put(f"f{i:02}.py", f"def {'wanted' if i == 29 else 'step_' + str(i)}():\n    return 1\n" + "# padding\n" * 37)
    o._process_core("lê ~/cortex_sandbox/loja e explica wanted")
    reply, _ = o._process_core("sim")
    text = body(llm)
    assert text.count("MAP:") == 30 and len(text) <= 6000
    assert text.split("\n\n")[1].startswith("FILE: f29.py\nL1   def wanted")
    assert text.count("FILE:") < 15
    assert "Li inteiros: f29.py" in reply and "Só vi no mapa:" in reply


@pytest.mark.parametrize("count,phrase", [(41, "tem 41 arquivos .py — é grande demais"), (0, "não tem arquivos .py que eu possa ler")])
def test_too_big_or_empty(setup, count, phrase):
    o, llm, put = setup
    for name in SOURCES:
        put(name).unlink()
    put("notes.md")
    for i in range(count):
        put(f"f{i}.py")
    reply, mode = o._process_core(ASK)
    assert phrase in reply and mode == "FILE_LIST" and o._pending_action is None
    llm.assert_not_called()


@pytest.mark.parametrize("change", ["delete", "escape"])
def test_approved_inventory_rechecked(setup, change):
    o, llm, put = setup
    o._process_core(ASK)
    path = put("models.py")
    path.unlink()
    if change == "escape":
        outside = Path.home() / "outside.py"
        outside.write_text("secret = 123\n")
        path.symlink_to(outside)
    put("added.py", "def added():\n    pass\n")
    reply, _ = o._process_core("sim")
    text = body(llm)
    assert "(Não consegui ler: models.py.)" in reply
    assert "MAP: models.py" in text and "FILE: models.py" not in text
    assert "added.py" not in text and "secret" not in text
    assert o.reader.read_code.call_count == 3


def test_unchanged_routes_and_authorised_root_itself(setup):
    o, llm, put = setup
    assert o._process_core("~/cortex_sandbox/loja")[1] == "FILE_LIST"
    assert o._pending_action is None
    folder = Path.home() / "documents"
    folder.mkdir()
    for name in ("one.md", "two.md"):
        (folder / name).write_text("notes")
    o._process_core("~/documents read them")
    assert o._pending_action["type"] == "read_files"
    o._process_core("não")
    o._process_core("lê ~/cortex_sandbox/loja/cart.py")
    assert o._pending_action["type"] == "read_file_at"
    assert o._process_core("sim") == ("fixed answer", "LOCAL_RAG")
    assert body(llm).startswith("FILE: cart.py\n")
    o._process_core("lê ~/cortex_sandbox")
    assert o._pending_action["type"] == "read_code_project"
    assert o._pending_action["args"]["files"] == ["loja/" + name for name in sorted(SOURCES)]


def test_never_tool_plan_macro_or_background(setup, monkeypatch):
    o, _, _ = setup
    name, args = "read_code_project", {"root": str(Path.home() / "cortex_sandbox/loja"), "files": sorted(SOURCES)}
    o.engine_analytical.chat_with_tools.side_effect = lambda text, specs, execute, **kw: (execute(name, args), name)
    assert "ERROR" in o._process_core("procura um documento sobre astronomia")[0]
    monkeypatch.setattr(planner, "decompose", lambda *a, **kw: planner.Plan("read", [planner.Step(id="p1", tool=name, args=args)]))
    assert o.process("planeja: ler meu projeto")[1] == "AGENT_STAGE_ERROR"
    assert "ERROR" in o._skill_macro_stage_step(name, args)
    assert "ERROR" in o._execute_tool_unstaged(name, args)
    assert "ERROR" in o._gate_background_action(name, args)
    assert o._pending_action is None


@pytest.mark.parametrize("lang,offline", [("pt", False), ("en", True)])
def test_exact_partial_ranges_language_and_fail_soft(setup, lang, offline):
    import re
    from itertools import groupby
    o, llm, put = setup
    put("models.py", "\n".join(f"def extra_{i}():\n    return 'some longer content here'\n" for i in range(150)))
    preview, _ = o._process_core("read ~/cortex_sandbox/loja models", language=lang)
    assert ("Reply 'yes'" if lang == "en" else "Responda 'sim'") in preview
    if offline:
        llm.side_effect = RuntimeError("offline")
    emitted = []
    o.set_token_emitter(emitted.append)
    reply, mode = o._process_core("sim")
    text = body(llm)
    assert len(text) <= 6000 and text.count("FILE:") == 1
    assert "extra_11, …" in text and "extra_12" not in text.split("\n\n")[0]
    shown = [int(n) for n in re.findall(r"^L(\d+) ", text, re.M)]
    groups = [[n for _, n in g] for _, g in groupby(enumerate(shown), lambda pair: pair[1] - pair[0])]
    ranges = ", ".join(str(g[0]) if len(g) == 1 else f"{g[0]}–{g[-1]}" for g in groups)
    expected = (f"(Read in part: models.py (lines {ranges}). Only in the map: cart.py, discounts.py, tests/test_cart.py.)"
                if lang == "en" else f"(Li em parte: models.py (linhas {ranges}). Só vi no mapa: cart.py, discounts.py, tests/test_cart.py.)")
    assert reply == ("(I couldn't answer right now)" if offline else "fixed answer") + "\n\n" + expected
    assert emitted == [reply] and mode == "LOCAL_RAG"


def test_map_alone_cannot_overflow_the_model_window(setup):
    o, llm, put = setup
    for name in SOURCES:
        put(name).unlink()
    for i in range(40):
        put(f"{i}_" + "long_name_" * 20 + ".py")
    assert o._process_core(ASK)[1] == "AGENT_AWAITING_CONFIRMATION"
    reply, mode = o._process_core("sim")
    assert "mapa do projeto é grande demais" in reply and mode == "LOCAL_RAG"
    llm.assert_not_called()
