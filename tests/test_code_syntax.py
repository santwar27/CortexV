"""In-process syntax checks: fixed replies, no execution or source writes."""
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.core import code_roots
from src.tools.local_reader import LocalFileReader
from tests.test_dispatch_parity import _build_real_orch

ASK = "checa a sintaxe em ~/cortex_sandbox/loja"
NOTE = "\n(Só conferi a escrita do Python — nada foi executado.)"
NOTE_EN = "\n(I only checked how the Python is written — nothing was run.)"


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CORTEX_CODE_ROOTS_FILE", raising=False)
    root = tmp_path / "cortex_sandbox"
    root.mkdir()
    assert code_roots.add(str(root))[0]
    o = _build_real_orch(tmp_path)
    o._PENDING_ACTION_FILE = str(tmp_path / "pending.json")
    o.background, o._reply_lang = False, "en"
    o.set_turn_origin("terminal")
    o.reader = LocalFileReader(str(tmp_path / "vault"))
    llm = Mock(side_effect=AssertionError("model called"))
    monkeypatch.setattr("src.core.staging_mixin.ollama.chat", llm)
    subprocesses = [Mock(side_effect=AssertionError("subprocess called")) for _ in range(2)]
    for name, mock in zip(("Popen", "run"), subprocesses):
        monkeypatch.setattr("subprocess." + name, mock)
    def put(name, content="pass\n"):
        path = root / "loja" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path
    put("cart.py")
    yield o, llm, put
    llm.assert_not_called()
    for mock in subprocesses:
        mock.assert_not_called()


@pytest.mark.parametrize("stream", [False, True])
def test_folder_errors_exact_and_untouched(setup, stream, monkeypatch):
    o, _, put = setup
    paths = [put("cart.py"), put("discounts.py", "def f(:\n"),
             put("tests/test_cart.py", "def f():\npass\n")]
    before = [p.read_bytes() for p in paths]
    log = Mock()
    monkeypatch.setattr("src.core.staging_mixin.audit.log", log)
    if stream:
        *_, (reply, _, mode, _) = o.process_stream(ASK)
    else:
        reply, mode = o._process_core(ASK)
    assert (reply, mode) == ("Erros de sintaxe em 2 de 3 arquivos .py de loja:\n"
        "- discounts.py, linha 1: invalid syntax\n"
        "- tests/test_cart.py, linha 2: expected an indented block after function definition on line 1" + NOTE,
        "CODE_SYNTAX_ERRORS")
    assert [p.read_bytes() for p in paths] == before
    assert not list(paths[0].parent.rglob("__pycache__"))
    assert o._pending_action is None and o._reply_lang == "pt"
    events = [c for c in log.call_args_list if c.args == ("–", "code", "syntax")]
    assert len(events) == 1 and events[0].kwargs == {"payload": dict(files=3, errors=2, not_checked=0)}


@pytest.mark.parametrize("prefix,tail,en", [
    ("checa a sintaxe em", "", False), ("  CONFERE A SINTAXE DO", ".?!  ", False),
    ("confere a sintaxe de", "", False), ("checa a sintaxe da", "", False),
    ("check the syntax of", "!", True), ("check the syntax in", "", True)])
@pytest.mark.parametrize("kind", ["folder", "file", "error"])
def test_exact_success_and_single_error(setup, prefix, tail, en, kind):
    o, _, put = setup
    path = "~/cortex_sandbox/loja"
    if kind != "folder":
        path += "/cart.py"
    if kind == "error":
        put("cart.py", "def f(:\n")
        expected = ("Syntax error in cart.py, line 1: invalid syntax" if en else
                    "Erro de sintaxe em cart.py, linha 1: invalid syntax")
    elif kind == "file":
        expected = "Syntax ok in cart.py." if en else "Sintaxe ok em cart.py."
    else:
        expected = ("Syntax ok: 1 .py files in loja, no errors." if en else
                    "Sintaxe ok: 1 arquivos .py em loja, nenhum erro.")
    o.set_turn_origin("voice")
    assert o._process_core(prefix + " " + path + tail) == (
        expected + (NOTE_EN if en else NOTE), "CODE_SYNTAX_ERRORS" if kind == "error" else "CODE_SYNTAX_OK")


def test_top_level_code_never_runs(setup):
    o, _, put = setup
    marker = Path.home() / "marker"
    path = put("cart.py", f'import os\nopen({str(marker)!r}, "w")\nos.system("touch {marker}")\n')
    before = path.read_bytes()
    assert o._process_core(ASK)[1] == "CODE_SYNTAX_OK"
    assert not marker.exists() and path.read_bytes() == before


@pytest.mark.parametrize("case,expected,mode", [
    ("two", "Uma pasta ou um arquivo .py por vez: checa a sintaxe em ~/pasta.", "CODE_SYNTAX_REFUSED"),
    ("md", "Só confiro a sintaxe de arquivos .py de uma pasta autorizada para código.", "CODE_SYNTAX_REFUSED"),
    ("outside_file", "não posso conferir outside.py: unsupported format — .py only inside a folder authorised for code.", "FILE_ERROR"),
    ("outside_folder", "não posso conferir a pasta outside: ela não está autorizada para código.", "FILE_ERROR"),
    ("missing", "não achei missing.py na pasta empty.", "FILE_ERROR"),
    ("large", "o projeto loja tem 41 arquivos .py — é grande demais para eu ler de uma vez; me aponta uma subpasta ou um arquivo.", "FILE_LIST"),
    ("empty", "a pasta loja não tem arquivos .py que eu possa ler.", "FILE_LIST")])
@pytest.mark.parametrize("en", [False, True])
def test_refusals(setup, case, expected, mode, en):
    o, _, put = setup
    ask = ASK
    if case == "two":
        put("two.py")
        ask += " e ~/cortex_sandbox/loja/two.py"
    elif case == "md":
        put("notes.md")
        ask += "/notes.md"
    elif case == "outside_file":
        (Path.home() / "outside.py").write_text("pass\n")
        ask = "checa a sintaxe de ~/outside.py"
    elif case == "outside_folder":
        (Path.home() / "outside").mkdir()
        ask = "checa a sintaxe da ~/outside"
    elif case == "missing":
        (Path.home() / "empty").mkdir()
        ask = "checa a sintaxe em ~/empty/missing.py"
    elif case == "large":
        for i in range(40):
            put(f"f{i}.py")
    elif case == "empty":
        put("cart.py").unlink()
    if en:
        ask = "check the syntax of " + ask.split(" ", 4)[4]
        expected = {
            "two": "One folder or one .py file at a time: check the syntax of ~/folder.",
            "md": "I only check the syntax of .py files in a folder authorised for code.",
            "outside_file": "I can't check outside.py: unsupported format — .py only inside a folder authorised for code.",
            "outside_folder": "I can't check the folder outside: it isn't authorised for code.",
            "missing": "I couldn't find missing.py in the folder empty.",
            "large": "the project loja has 41 .py files — too big to read at once; point me to a subfolder or a file.",
            "empty": "the folder loja has no .py files I can read.",
        }[case]
    assert o._process_core(ask) == (expected, mode)
    assert o._pending_action is None


def test_truncation_and_null_bytes(setup):
    o, _, put = setup
    put("cart.py", "#" + "x" * 15001)
    put("null.py", "\0")
    assert o._process_core(ASK) == ("Erros de sintaxe em 1 de 1 arquivo .py de loja:\n"
        "- null.py, linha ?: source code string cannot contain null bytes\n"
        "(Não consegui conferir: cart.py.)" + NOTE, "CODE_SYNTAX_ERRORS")


def test_english_folder_diagnostics_and_value_error(setup, monkeypatch):
    o, _, put = setup
    put("z.py", "#" + "x" * 15001)
    compiler = Mock(side_effect=ValueError("source code string cannot contain null bytes"))
    monkeypatch.setattr("src.core.staging_mixin.compile", compiler, raising=False)
    assert o._process_core("check the syntax in ~/cortex_sandbox/loja") == (
        "Syntax errors in 1 of 1 .py file in loja:\n"
        "- cart.py, line ?: source code string cannot contain null bytes\n"
        "(I couldn't check: z.py.)" + NOTE_EN, "CODE_SYNTAX_ERRORS")
    compiler.assert_called_once_with("pass\n", "cart.py", "exec", dont_inherit=True)


@pytest.mark.parametrize("failure", ["gate", "read", "recursion", "memory"])
@pytest.mark.parametrize("en", [False, True])
def test_not_checked_alone_is_not_error(setup, monkeypatch, failure, en):
    o, _, _ = setup
    if failure == "gate":
        gate = Mock(side_effect=[(True, str(Path.home() / "cortex_sandbox/loja/cart.py")), (False, "revoked")])
        monkeypatch.setattr("src.tools.file_finder.is_readable_path", gate)
    elif failure == "read":
        o.reader.read_code = Mock(return_value="[SYSTEM ERROR] unreadable")
    else:
        monkeypatch.setattr("src.core.staging_mixin.compile", Mock(side_effect=(
            RecursionError if failure == "recursion" else MemoryError)), raising=False)
    log = Mock()
    monkeypatch.setattr("src.core.staging_mixin.audit.log", log)
    ask = "check the syntax of ~/cortex_sandbox/loja" if en else ASK
    expected = ("I couldn't check the syntax of cart.py." if en else
                "Não consegui conferir a sintaxe de cart.py.")
    assert o._process_core(ask + "/cart.py") == (
        expected + (NOTE_EN if en else NOTE), "CODE_SYNTAX_NOT_CHECKED")
    log.assert_called_once_with("–", "code", "syntax", payload=dict(files=1, errors=0, not_checked=1))


@pytest.mark.parametrize("en", [False, True])
def test_single_truncated_file_is_not_checked(setup, monkeypatch, en):
    o, _, put = setup
    put("big.py", "#" + "x" * 15001)
    log = Mock()
    monkeypatch.setattr("src.core.staging_mixin.audit.log", log)
    ask = "check the syntax of ~/cortex_sandbox/loja" if en else ASK
    expected = ("I couldn't check the syntax of big.py." if en else
                "Não consegui conferir a sintaxe de big.py.")
    assert o._process_core(ask + "/big.py") == (
        expected + (NOTE_EN if en else NOTE), "CODE_SYNTAX_NOT_CHECKED")
    log.assert_called_once_with("–", "code", "syntax", payload=dict(files=1, errors=0, not_checked=1))


@pytest.mark.parametrize("en", [False, True])
@pytest.mark.parametrize("checked,error", [(0, False), (1, False), (1, True)])
def test_folder_counts_only_checked_files(setup, monkeypatch, en, checked, error):
    o, _, put = setup
    if checked:
        put("cart.py", "def f(:\n" if error else "pass\n")
    else:
        put("cart.py", "#" + "x" * 15001)
    put("big.py", "#" + "x" * 15001)
    log = Mock()
    monkeypatch.setattr("src.core.staging_mixin.audit.log", log)
    if checked == 0:
        expected = ("I couldn't check any of the 2 .py files in loja." if en else
                    "Não consegui conferir nenhum dos 2 arquivos .py de loja.")
        mode, errors, not_checked, names = "CODE_SYNTAX_NOT_CHECKED", 0, 2, "big.py, cart.py"
    else:
        if error:
            expected = ("Syntax errors in 1 of 1 .py file in loja:\n"
                        "- cart.py, line 1: invalid syntax" if en else
                        "Erros de sintaxe em 1 de 1 arquivo .py de loja:\n"
                        "- cart.py, linha 1: invalid syntax")
        else:
            expected = ("Syntax ok: 1 .py file in loja, no errors." if en else
                        "Sintaxe ok: 1 arquivo .py em loja, nenhum erro.")
        mode, errors, not_checked, names = (
            "CODE_SYNTAX_ERRORS" if error else "CODE_SYNTAX_OK", int(error), 1, "big.py")
    expected += (f"\n(I couldn't check: {names}.)" if en else
                 f"\n(Não consegui conferir: {names}.)")
    ask = "check the syntax of ~/cortex_sandbox/loja" if en else ASK
    assert o._process_core(ask) == (expected + (NOTE_EN if en else NOTE), mode)
    log.assert_called_once_with("–", "code", "syntax", payload=dict(
        files=2, errors=errors, not_checked=not_checked))


@pytest.mark.parametrize("ask,action", [
    ("lê ~/cortex_sandbox/loja/cart.py e explica", "read_file_at"),
    ("lê ~/cortex_sandbox/loja e explica", "read_code_project"),
    (ASK + "/cart.py e explica", "read_file_at"),
    ("me checa a sintaxe em ~/cortex_sandbox/loja/cart.py", "read_file_at"),
    (ASK + "/cart.py,", "read_file_at")])
def test_unchanged_paths(setup, ask, action):
    o, _, _ = setup
    assert o._process_core(ask)[1] == "AGENT_AWAITING_CONFIRMATION"
    assert o._pending_action["type"] == action
