"""Only a terminal grant makes Python files eligible for confirmed reads."""
import json
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from src.tools.file_finder import is_readable_path, readable_files_in
from src.tools.local_reader import LocalFileReader

PY_REFUSAL = "unsupported format — .py only inside a folder authorised for code"
OTHER_REFUSAL = "unsupported format — only .pdf, .txt, .md and .docx"


@pytest.fixture
def registry(tmp_path, monkeypatch):
    from src.core import code_roots  # HOME must be resolved AFTER this import.
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("CORTEX_CODE_ROOTS_FILE", raising=False)
    monkeypatch.setattr(code_roots.audit, "log", Mock())
    return code_roots, home


def put(path, text="total = 2 + 3\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return str(path.resolve())


def test_registry_rules_and_cli(registry, capsys):
    cr, home = registry
    file = home / "file.py"
    put(file)
    (home / ".ssh").mkdir()
    (home / "escape").symlink_to(home.parent, target_is_directory=True)
    for path, reason in [
        (home / "missing", "não existe"), (file, "não é uma pasta"),
        (home, "é a sua pasta pessoal inteira — escolha uma pasta dentro dela"),
        (home.parent, "fica fora da sua pasta pessoal"),
        (home / ".ssh", "é um lugar protegido (credenciais, chaves, dados de apps)"),
        (home / "escape", "fica fora da sua pasta pessoal"),
    ]:
        assert cr.main(["add", str(path)]) == 1
        assert capsys.readouterr().out == f"Não autorizei {path}: {reason}.\n"
    assert cr.roots() == []
    assert cr.main(["list"]) == 0
    assert capsys.readouterr().out == "Nenhuma pasta autorizada para código.\n"
    root = home / "cortex_sandbox"
    root.mkdir()
    real = str(root.resolve())
    link = home / "alias"
    link.symlink_to(root, target_is_directory=True)
    assert cr.main(["add", str(link)]) == 0
    assert capsys.readouterr().out == f"Pasta autorizada para o Cortex ler código: {real}\n"
    cr.audit.log.assert_called_once_with("–", "code_roots", "add", payload={"root": real})
    assert cr.roots() == [real] and not cr.inside(real)
    saved = home / ".cortex/code_roots.json"
    assert saved.stat().st_mode & 0o777 == 0o600
    record = json.loads(saved.read_text())[0]
    assert record["root"] == real
    assert datetime.fromisoformat(record["added_at"]).utcoffset() == timezone.utc.utcoffset(None)
    assert cr.main(["add", str(root)]) == 0
    assert capsys.readouterr().out == f"Essa pasta já estava autorizada: {real}\n"
    assert cr.main(["list"]) == 0
    assert capsys.readouterr().out == f"Pastas autorizadas para código:\n- {real}\n"
    cr.audit.log.reset_mock()
    assert cr.main(["remove", str(link)]) == 0
    assert capsys.readouterr().out == f"Autorização removida: {real}\n"
    cr.audit.log.assert_called_once_with("–", "code_roots", "remove", payload={"root": real})
    assert cr.main(["remove", str(root)]) == 1
    assert capsys.readouterr().out == f"Essa pasta não estava autorizada: {root}\n"
    assert cr.roots() == []


def test_gate_and_revocation(registry):
    cr, home = registry
    root = home / "cortex_sandbox"
    good = put(root / "calc/prices.py")
    assert cr.add(str(root))[0]
    assert is_readable_path(good) == (True, good)
    assert is_readable_path(put(root / "upper.PY"))[0]
    outside = put(home / "outra/x.py")
    link = root / "calc/link.py"
    link.symlink_to(outside)
    for path in (outside, str(link), put(home / "cortex_sandbox_old/x.py")):
        assert is_readable_path(path) == (False, PY_REFUSAL)
    for suffix in ("sh", "pyc", "js"):
        assert is_readable_path(put(root / f"calc/run.{suffix}")) == (False, OTHER_REFUSAL)
    assert "protected" in is_readable_path(put(root / ".git/x.py"))[1]
    assert "absolute" in is_readable_path("prices.py")[1]
    assert is_readable_path(str(root / "missing.py")) == (False, "no such file")
    put(root / ".cloud.py.icloud", "stub")
    assert "not downloaded" in is_readable_path(str(root / "cloud.py"))[1]
    assert cr.remove(str(root))[0]
    assert is_readable_path(good) == (False, PY_REFUSAL)


@pytest.mark.parametrize("broken", ["{", "{}", '[{"root": "/"}]'])
def test_corrupt_registry_fails_closed_and_add_repairs(registry, broken):
    cr, home = registry
    root = home / "cortex_sandbox"
    path = put(root / "x.py")
    assert cr.add(str(root))[0]
    (home / ".cortex/code_roots.json").write_text(broken)
    assert cr.roots() == []
    assert is_readable_path(path) == (False, PY_REFUSAL)
    assert cr.add(str(root))[0]
    assert is_readable_path(path)[0]


def test_override_unreadable_and_fail_soft_audit(registry, monkeypatch):
    cr, home = registry
    root = home / "cortex_sandbox"
    root.mkdir()
    saved = home / "custom.json"
    monkeypatch.setenv("CORTEX_CODE_ROOTS_FILE", str(saved))
    cr.audit.log.side_effect = OSError("audit unavailable")
    assert cr.add(str(root))[0] and saved.is_file()
    assert not (home / ".cortex/code_roots.json").exists()
    assert cr.remove(str(root))[0]
    saved.unlink()
    saved.mkdir()  # Opening a directory as JSON fails on every test host.
    assert cr.roots() == []


def test_code_reader_keeps_document_dispatch_and_cap(tmp_path):
    reader = LocalFileReader(str(tmp_path / "vault"))
    path = put(tmp_path / "prices.py")
    assert reader.read_code(path) == "total = 2 + 3\n"
    assert reader.read_path(path).startswith("[SYSTEM ERROR] Unsupported file format")
    for length in (15000, 15001):
        put(tmp_path / "prices.py", "x" * length)
        notice = "\n\n[SYSTEM NOTICE: Document truncated due to local memory constraints.]"
        assert reader.read_code(path) == "x" * 15000 + (notice if length > 15000 else "")


@pytest.mark.parametrize("stream,multiple", [(False, False), (True, False), (False, True)])
def test_confirmed_routes_send_fenced_code_to_engine(registry, monkeypatch, stream, multiple):
    from tests.test_dispatch_parity import _build_real_orch
    from src.core.prompt_safety import OPEN, CLOSE
    bare = Mock(return_value={"message": {"content": "resposta"}})
    monkeypatch.setattr("src.core.staging_mixin.ollama.chat", bare)
    cr, home = registry
    folder = home / "cortex_sandbox/calc"
    paths = [put(folder / "prices.py")]
    if multiple:
        paths.append(put(folder / "tax.py", "tax = 7\n"))
    assert cr.add(str(folder.parent))[0]
    orch = _build_real_orch(home)
    orch._PENDING_ACTION_FILE = str(home / "pending.json")
    orch.background = False
    orch.reader = LocalFileReader(str(home / "vault"))
    read = Mock(wraps=orch.reader.read_code)
    monkeypatch.setattr(orch.reader, "read_code", read)
    monkeypatch.setattr(orch.reader, "read_path", Mock(side_effect=AssertionError("wrong reader")))
    assert [p for p, _ in readable_files_in(str(folder))] == paths
    reply, mode = orch._process_core("o que tem em ~/cortex_sandbox/calc?")
    assert mode == "FILE_LIST" and "prices.py" in reply
    text = "lê ~/cortex_sandbox/calc/prices.py"
    if multiple:
        text += " e ~/cortex_sandbox/calc/tax.py"
    text += " e explica o total"
    if stream:
        *_, (_reply, _final, mode, _lang) = orch.process_stream(text, None)
    else:
        _reply, mode = orch._process_core(text)
    assert mode == "AGENT_AWAITING_CONFIRMATION"
    assert orch._pending_action["type"] == ("read_files" if multiple else "read_file_at")
    args = orch._pending_action["args"]
    assert (args["paths"] if multiple else [args["path"]]) == paths
    read.assert_not_called()
    orch.engine_analytical.chat.assert_not_called()
    orch.engine_analytical.chat_stream.assert_not_called()
    _reply, mode = orch._execute_pending_action()
    assert mode == "LOCAL_RAG"
    assert [call.args[0] for call in read.call_args_list] == paths
    bare.assert_called_once()
    prompt = bare.call_args.kwargs["messages"][1]["content"]
    fenced = prompt.split(OPEN + "\n", 1)[1].split("\n" + CLOSE, 1)[0]
    assert "total = 2 + 3" in fenced
    if multiple:
        assert "tax = 7" in fenced
