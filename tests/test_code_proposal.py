"""Whole-file proposals: fake model, real gate, and byte-identical sources."""
import difflib
import hashlib
import json
import re
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.core import code_roots, planner
from src.core.ollama_client import THINK
from src.core.prompt_safety import fence, guard_line
from src.tools.local_reader import LocalFileReader
from tests.test_dispatch_parity import _build_real_orch

OLD = 'def normalise_label(x):\n    return x.strip()\n\na = normalise_label("A")\nb = normalise_label("B")\n'
NEW = OLD.replace("normalise_label", "clean_label")
REQUEST = "renomeia normalise_label para clean_label"
ASK = "muda em ~/cortex_sandbox/labels.py: " + REQUEST
TYPED = "Pedir mudança de código só digitando, no terminal do Cortex ou na tela."


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CORTEX_CODE_ROOTS_FILE", raising=False)
    monkeypatch.delenv("CORTEX_CODE_PROPOSALS_DIR", raising=False)
    root = tmp_path / "cortex_sandbox"
    root.mkdir()
    assert code_roots.add(str(root))[0]
    path = root / "labels.py"
    path.write_text(OLD)
    o = _build_real_orch(tmp_path)
    o._PENDING_ACTION_FILE = str(tmp_path / "pending.json")
    o.background, o._reply_lang = False, "pt"
    o.set_turn_origin("terminal")
    o.reader = LocalFileReader(str(tmp_path / "vault"))
    o.reader.read_code = Mock(wraps=o.reader.read_code)
    llm = Mock(return_value={"message": {"content": "FILE: labels.py\n```python\n" + NEW + "```\nRenomeei a função."}})
    monkeypatch.setattr("src.core.staging_mixin.ollama.chat", llm)
    return o, llm, path


def approve(setup, answer=None, ask=ASK):
    o, llm, _ = setup
    if answer is not None:
        llm.return_value = {"message": {"content": answer}}
    assert o._process_core(ask)[1] == "AGENT_AWAITING_CONFIRMATION"
    return o._process_core("sim")


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("prefix,change,lang", [("muda em", REQUEST, "pt"),
    ("  MUDE EM", REQUEST, "pt"), ("muda no", REQUEST, "pt"), ("muda na", REQUEST, "pt"),
    ("muda em", "conserta o off-by-one do last_index", "pt"),
    ("change in", "rename normalise_label to clean_label", "en"),
    ("change", "rename normalise_label to clean_label", "en")])
def test_route_preview(setup, stream, prefix, change, lang):
    o, llm, path = setup
    ask = f"{prefix} ~/cortex_sandbox/labels.py: {change}"
    if stream:
        *_, (reply, _, mode, _) = o.process_stream(ask)
    else:
        reply, mode = o._process_core(ask)
    assert mode == "AGENT_AWAITING_CONFIRMATION"
    assert o._pending_action["type"] == "propose_code_change"
    assert o._pending_action["args"] == dict(path=str(path), request=change, lang=lang)
    assert reply == (f"May I read 'labels.py' and ask the model for a proposed change?\nRequest: {change}\n"
        f"It is in {path.parent}. Nothing is changed in the file and no command runs.\n"
        "Reply 'yes' to continue or 'no' to cancel." if lang == "en" else
        f"Posso ler 'labels.py' e pedir ao modelo uma proposta de mudança?\nPedido: {change}\n"
        f"Está em {path.parent}. Nada é alterado no arquivo e nenhum comando roda.\n"
        "Responda 'sim' para continuar ou 'não' para cancelar.")
    llm.assert_not_called()
    o.reader.read_code.assert_not_called()


@pytest.mark.parametrize("case,expected,mode", [
    ("voice", TYPED, "CODE_PROPOSAL_REFUSED"),
    ("empty", "Me diz o que mudar: muda em ~/cortex_sandbox/labels.py: o que mudar.", "CODE_PROPOSAL_REFUSED"),
    ("folder", "Uma mudança por vez, em UM arquivo .py: muda em ~/pasta/arquivo.py: o que mudar.", "CODE_PROPOSAL_REFUSED"),
    ("two", "Uma mudança por vez, em UM arquivo .py: muda em ~/pasta/arquivo.py: o que mudar.", "CODE_PROPOSAL_REFUSED"),
    ("md", "Só proponho mudança em arquivo .py de uma pasta autorizada para código.", "CODE_PROPOSAL_REFUSED"),
    ("outside", "não posso mudar outside.py: unsupported format — .py only inside a folder authorised for code.", "FILE_ERROR")])
def test_refused_before_read(setup, case, expected, mode):
    o, llm, path = setup
    ask = ASK
    if case == "voice":
        o.set_turn_origin("voice")
    elif case == "empty":
        ask = ASK.split(":")[0] + ":"
    elif case == "folder":
        ask = "muda em ~/cortex_sandbox: renomeia"
    elif case == "two":
        (path.parent / "two.py").write_text("pass\n")
        ask += " e ~/cortex_sandbox/two.py"
    elif case == "md":
        path.with_suffix(".md").write_text("notes")
        ask = ASK.replace("labels.py", "labels.md")
    elif case == "outside":
        (path.parent.parent / "outside.py").write_text("pass\n")
        ask = ASK.replace("cortex_sandbox/labels.py", "outside.py")
    assert o._process_core(ask) == (expected, mode)
    assert o._pending_action is None
    llm.assert_not_called()
    o.reader.read_code.assert_not_called()


def test_proposal_exact_reply_call_store_and_audit(setup, monkeypatch):
    from src.core import code_proposals
    o, llm, path = setup
    log, emitted = Mock(), []
    monkeypatch.setattr("src.core.staging_mixin.audit.log", log)
    o.set_token_emitter(emitted.append)
    reply, mode = approve(setup)
    diff = "\n".join(difflib.unified_diff(OLD.splitlines(), NEW.splitlines(), fromfile="a/labels.py", tofile="b/labels.py", n=3, lineterm=""))
    pid = re.search(r"Proposta ([0-9a-f]{16}) guardada", reply)[1]
    assert reply == (f"Proposta — nada foi aplicado.\nArquivo: {path} (−3 +3 linhas).\n```diff\n{diff}\n```\n"
                     f"(Proposta {pid} guardada. O arquivo continua como estava.)\n"
                     "Aplicar esta mudança em labels.py? Só aplico se o arquivo não mudar até lá.\n"
                     "Responda 'sim' (exatamente essa palavra) para aplicar ou 'não' para cancelar.")
    assert o._pending_action["type"] == "apply_code_change"
    assert mode == "CODE_PROPOSAL" and emitted == [] and path.read_bytes() == OLD.encode()
    record = code_proposals.load(pid)
    assert record == dict(version=1, created_at=record["created_at"], path=str(path), root=str(path.parent),
        rel="labels.py", request=REQUEST, lang="pt", origin="terminal", original_sha256=hashlib.sha256(OLD.encode()).hexdigest(),
        original_chars=len(OLD), new_text=NEW, new_sha256=hashlib.sha256(NEW.encode()).hexdigest(), diff=diff, outside=[], risky=[])
    assert record["created_at"].endswith("Z")
    assert (Path.home() / ".cortex/code_proposals" / (pid + ".json")).stat().st_mode & 0o777 == 0o600
    assert (Path.home() / ".cortex/code_proposals").stat().st_mode & 0o777 == 0o700
    llm.assert_called_once()
    call = llm.call_args.kwargs
    fmt = 'For each file you change, write a line `FILE: <path>` and then the COMPLETE new content of that file in a ```python block; then at most two sentences in Brazilian Portuguese.'
    assert call == dict(model=o.engine_analytical.model_name, messages=[
        dict(role="system", content=o._CODE_SYSTEM.format(fmt=fmt, guard=guard_line("en"))),
        dict(role="user", content=fence("FILE: labels.py\n" + OLD) + "\n\nPEDIDO DO PEDRO: " + REQUEST)],
        options={"temperature": 0.0}, keep_alive=call["keep_alive"], think=THINK)
    events = [c.kwargs["payload"] for c in log.call_args_list if c.args == ("–", "code", "propose")]
    assert len(events) == 1 and events[0] == dict(outcome="proposed", id=pid, removed=3, added=3, outside=[], risky=[], chars=len(OLD), ms=events[0]["ms"])


def test_outside_and_risk(setup):
    o, llm, path = setup
    path.rename(path.with_name("config.py"))
    path = path.with_name("config.py")
    path.write_text("# keep\n# these\n# comments\ndef port(PORT):\n    return None\n")
    reply, _ = approve(setup, '```python\ndef port(PORT):\n    return 8000\n```', "muda em ~/cortex_sandbox/config.py: sem PORT, o padrão é 8000")
    assert "Atenção: mudou fora do que você pediu: linhas 1–3 (removidas)." in reply
    llm.return_value = {"message": {"content": '```py\n' + path.read_text() + 'os.system("curl x")\n```'}}
    reply, _ = o._process_core("muda em ~/cortex_sandbox/config.py: sem PORT, o padrão é 8000")
    assert 'Atenção: a linha nova 6 usa rede, shell, apaga arquivo ou executa código: os.system("curl x")' in reply


@pytest.mark.parametrize("answer,expected", [
    ("```py\npass\n```\n```py\npass\n```", "A resposta trouxe mais de um bloco de código; só confiro UM arquivo inteiro."),
    ("Renomeei.", "Não saiu uma proposta que eu possa conferir: o modelo não devolveu o arquivo labels.py inteiro."),
    ("```diff\n-x\n+y\n```", "Não saiu uma proposta que eu possa conferir: o modelo não devolveu o arquivo labels.py inteiro."),
    *[(f"FILE: {label}\n```python\npass\n```", f"A resposta mexia em outro arquivo ({label}); só aceito mudança em labels.py.") for label in ("other.py", "../x.py", "/tmp/x.py")],
    ("```python\ndef x(:\n```", "O código proposto não é Python válido (linha 1: invalid syntax)."),
    ("```python\n#" + "x" * 2200 + "\n```", f"A proposta ficou grande demais para conferir (2202 caracteres para um arquivo de {len(OLD)})."),
    ("```python\n" + OLD + "```", "O modelo devolveu labels.py sem nenhuma mudança."),
    ("", "(não consegui responder agora)"),
])
def test_fixed_refusals(setup, answer, expected, monkeypatch):
    o, _, path = setup
    log = Mock()
    monkeypatch.setattr("src.core.staging_mixin.audit.log", log)
    reply, mode = approve(setup, answer)
    assert reply == expected + " Nada foi aplicado." and mode == "CODE_PROPOSAL_REFUSED"
    assert path.read_bytes() == OLD.encode() and not (Path.home() / ".cortex/code_proposals").exists()
    assert len([c for c in log.call_args_list if c.args == ("–", "code", "propose")]) == 1


@pytest.mark.parametrize("answer", [
    "FILE: other.py\nAqui está o arquivo inteiro:\n```python\n" + NEW + "```",
    "FILE: labels.py\n```python\n" + NEW + "```\nFILE: other.py",
    "  > # **FILE: `other.py`**\n```python\n" + NEW + "```",
    "FILE: labels.py\nFILE: 'other.py'\nFILE: third.py\n```python\n" + NEW + "```",
])
@pytest.mark.parametrize("lang", ["pt", "en"])
def test_every_outside_file_label_must_match(setup, answer, lang):
    _, llm, path = setup
    ask = ASK if lang == "pt" else "change ~/cortex_sandbox/labels.py: rename normalise_label to clean_label"
    reply, mode = approve(setup, answer, ask)
    assert reply == ("A resposta mexia em outro arquivo (other.py); só aceito mudança em labels.py. Nada foi aplicado."
                     if lang == "pt" else
                     "The reply changed another file (other.py); I only accept a change to labels.py. Nothing was applied.")
    assert mode == "CODE_PROPOSAL_REFUSED"
    assert path.read_bytes() == OLD.encode() and not (Path.home() / ".cortex/code_proposals").exists()
    llm.assert_called_once()


@pytest.mark.parametrize("answer,new", [
    ("**FILE: labels.py**\n```python\n" + NEW + "```", NEW),
    ("  > # __FILE: 'labels.py'__\n```python\n" + NEW + "```", NEW),
    ("```python\n# FILE: x.py\n" + NEW + "```", "# FILE: x.py\n" + NEW),
])
def test_formatted_file_labels_and_code_comments(setup, answer, new):
    from src.core import code_proposals
    _, llm, path = setup
    reply, mode = approve(setup, answer)
    assert mode == "CODE_PROPOSAL" and reply.startswith("Proposta — nada foi aplicado.")
    pid = re.search(r"Proposta ([0-9a-f]{16}) guardada", reply)[1]
    assert code_proposals.load(pid)["new_text"] == new
    assert path.read_bytes() == OLD.encode()
    llm.assert_called_once()


def test_large_exception_and_voice_approval(setup):
    o, llm, path = setup
    path.write_text("#" + "x" * 4000)
    assert approve(setup) == ("labels.py tem 4001 caracteres — grande demais para eu propor mudança no arquivo inteiro (até 4000). Nada foi aplicado.", "CODE_PROPOSAL_REFUSED")
    llm.assert_not_called()
    path.write_text(OLD)
    llm.side_effect = RuntimeError("offline")
    assert o._process_core(ASK) == ("(não consegui responder agora) Nada foi aplicado.", "CODE_PROPOSAL_REFUSED")
    o._approved_read_paths = set()
    o.set_turn_origin("ui", "A")
    o._process_core(ASK)
    llm.reset_mock()
    o.set_turn_origin("voice")
    o.reader.read_code.reset_mock()
    assert o._process_core("sim") == (TYPED + " Nada foi aplicado.", "CODE_PROPOSAL_REFUSED")
    llm.assert_not_called()
    o.reader.read_code.assert_not_called()
    assert path.read_bytes() == OLD.encode()


def test_store_validation_override_and_failed_save(setup, monkeypatch):
    from src.core import code_proposals
    o, _, path = setup
    monkeypatch.setenv("CORTEX_CODE_PROPOSALS_DIR", str(path.parent.parent / "store"))
    pid = code_proposals.save({"version": 1, "new_text": "pass\n"})
    assert code_proposals.load(pid)["new_text"] == "pass\n"
    assert code_proposals.load("../x") is None and code_proposals.load("0" * 16) is None
    (path.parent.parent / "store" / (pid + ".json")).write_text('{}')
    assert code_proposals.load(pid) is None
    monkeypatch.setattr(code_proposals, "save", Mock(side_effect=OSError("full")))
    o._remember_read_approval(str(path))
    reply, mode = o._process_core(ASK)
    assert mode == "CODE_PROPOSAL" and "```diff\n" in reply
    assert reply.endswith("(Não consegui guardar esta proposta. O arquivo continua como estava.)")
    assert o._pending_action is None and path.read_bytes() == OLD.encode()


def test_never_tool_plan_macro_background_and_old_routes(setup, monkeypatch):
    o, _, path = setup
    name, args = "propose_code_change", dict(path=str(path), request=REQUEST)
    o.engine_analytical.chat_with_tools.side_effect = lambda text, specs, execute, **kw: (execute(name, args), name)
    assert "ERROR" in o._process_core("procura um documento sobre astronomia")[0]
    monkeypatch.setattr(planner, "decompose", lambda *a, **kw: planner.Plan("change", [planner.Step(id="p1", tool=name, args=args)]))
    assert o.process("planeja: mudar meu código")[1] == "AGENT_STAGE_ERROR"
    for method in (o._skill_macro_stage_step, o._execute_tool_unstaged, o._gate_background_action):
        assert "ERROR" in method(name, args)
    assert o._pending_action is None
    for ask in ("lê ~/cortex_sandbox/labels.py e explica", "o que muda em ~/cortex_sandbox/labels.py?"):
        o._process_core(ask)
        assert o._pending_action["type"] == "read_file_at"
        o._process_core("não")


@pytest.mark.parametrize("failure", ["missing", "revoked", "md_link", "error", "truncated"])
def test_read_rechecked_before_model(setup, failure):
    o, llm, path = setup
    o._process_core(ASK)
    if failure == "missing":
        path.unlink()
    elif failure == "revoked":
        code_roots.remove(str(path.parent))
    elif failure == "md_link":
        path.unlink()
        target = path.parent.parent / "notes.md"
        target.write_text("secret")
        path.symlink_to(target)
    else:
        o.reader.read_code.return_value = ("[SYSTEM ERROR] broken" if failure == "error" else
            "pass\n\n[SYSTEM NOTICE: Document truncated due to local memory constraints.]")
    reply, mode = o._process_core("sim")
    assert reply.endswith("Nada foi aplicado.") and mode in ("FILE_ERROR", "CODE_PROPOSAL_REFUSED")
    llm.assert_not_called()
    if failure in ("missing", "revoked", "md_link"):
        o.reader.read_code.assert_not_called()
    assert not (Path.home() / ".cortex/code_proposals").exists()


def test_english_normalisation_nested_label_and_missing_path(setup):
    from src.core import code_proposals
    o, llm, path = setup
    path.parent.joinpath("nested").mkdir()
    nested = path.parent / "nested/labels.py"
    nested.write_text(OLD.rstrip("\n") + "\n\n")
    answer = 'FILE: `nested/labels.py`\n```\n' + NEW.replace("\n", "\r\n") + '```\nI renamed it.'
    reply, mode = approve(setup, answer, "change ~/cortex_sandbox/nested/labels.py: rename normalise_label to clean_label")
    assert mode == "CODE_PROPOSAL" and reply.startswith("Proposal — nothing was applied.")
    assert "I renamed" not in reply and "Warning:" not in reply
    pid = re.search(r"Proposal ([0-9a-f]{16}) saved", reply)[1]
    record = code_proposals.load(pid)
    assert record["new_text"] == NEW.rstrip("\n") + "\n\n" and record["rel"] == "nested/labels.py"
    assert "in English." in llm.call_args.kwargs["messages"][0]["content"]
    assert llm.call_args.kwargs["messages"][1]["content"].endswith("PEDRO'S REQUEST: rename normalise_label to clean_label")
    o._process_core("change ~/cortex_sandbox/missing.py: rename x")
    assert o._reply_lang == "en" and o._pending_action is None
