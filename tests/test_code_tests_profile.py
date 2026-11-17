"""Fixed test-command proposals: every execution boundary must stay unused."""
import json
import shlex
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.core import code_roots, planner, tool_risk
from src.tools.local_reader import LocalFileReader
from tests.test_dispatch_parity import _build_real_orch

ASK = "roda os testes em ~/cortex_sandbox/loja"
TYPED = "Rodar os testes só digitando, no terminal do Cortex ou na tela."
REFUSED = "O comando guardado não confere com o perfil fixo — peça de novo. Nada foi rodado."


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CORTEX_CODE_ROOTS_FILE", raising=False)
    root = tmp_path / "cortex_sandbox"
    folder = root / "loja"
    (folder / "tests").mkdir(parents=True)
    for rel in ("cart.py", "tests/test_cart.py"):
        (folder / rel).write_text("raise AssertionError('project code ran')\n")
    assert code_roots.add(str(root))[0]
    o = _build_real_orch(tmp_path)
    o._PENDING_ACTION_FILE = str(tmp_path / "pending.json")
    o.background, o._reply_lang = False, "en"
    o.set_turn_origin("terminal")
    o.reader = LocalFileReader(str(tmp_path / "vault"))
    forbidden = []
    for target in ("ollama.chat", "subprocess.Popen", "subprocess.run", "os.system", "os.execv", "os.fork",
                   "src.core.staging_mixin.run_effect"):
        mock = Mock(side_effect=AssertionError(target + " called"))
        monkeypatch.setattr(target, mock)
        forbidden.append(mock)
    before = {p: p.read_bytes() for p in folder.rglob("*.py")}
    yield o, folder
    for mock in forbidden:
        mock.assert_not_called()
    assert all(p.read_bytes() == data for p, data in before.items() if p.exists())
    assert not list(folder.rglob("__pycache__"))


def test_profile(setup):
    from src.core import code_commands as commands
    _, folder = setup
    cwd = str(folder)
    profile = commands.tests_profile(cwd)
    assert profile == dict(kind="tests", argv=[sys.executable, "-I", "-B", "-m", "unittest", "discover",
        "-s", cwd, "-t", cwd, "-p", "test*.py"], cwd=cwd,
        env={"PATH": "/usr/bin:/bin", "LANG": "en_US.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"},
        timeout_s=30, network=False, max_output_bytes=65536)
    assert commands.validate(profile)[0]
    assert commands.validate(commands.tests_profile(str(folder.parent)))[0]
    assert commands.render(profile) == shlex.join(profile["argv"])
    assert "subprocess" not in Path(commands.__file__).read_text()
    link = folder.parent / "link"
    link.symlink_to(folder, target_is_directory=True)
    for changes in ({"argv": ["/other/python", *profile["argv"][1:]]},
                    {"argv": profile["argv"] + ["extra"]}, {"argv": profile["argv"][:-1]},
                    {"argv": [sys.executable, "-c", "raise AssertionError()"]},
                    {"cwd": str(folder.parent)}, {"cwd": str(link)}, {"cwd": "relative"},
                    {"timeout_s": 31}, {"network": True}, {"network": 0}, {"max_output_bytes": 1},
                    {"env": {**profile["env"], "EXTRA": "1"}}, {"extra": 1}):
        ok, reason = commands.validate({**profile, **changes})
        assert not ok and reason
    assert not commands.validate(commands.tests_profile(str(link)))[0]
    for malformed in (None, {}, {k: v for k, v in profile.items() if k != "env"}):
        assert not commands.validate(malformed)[0]
    code_roots.remove(str(folder.parent))
    assert not commands.validate(profile)[0]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("prefix", ["roda os testes em", "  RODE OS TESTES DO", "run the tests in"])
def test_route_preview_and_exact_approval(setup, monkeypatch, stream, prefix):
    from src.core import code_commands as commands
    o, folder = setup
    en = prefix.startswith("run")
    lang = "en" if en else "pt"
    o.set_turn_origin("ui", "A") if stream else o.set_turn_origin("terminal")
    def turn(text):
        if stream:
            *_, (reply, _, mode, _) = o.process_stream(text)
            return reply, mode
        return o._process_core(text)
    reply, mode = turn(prefix + " ~/cortex_sandbox/loja.?!  ")
    profile = commands.tests_profile(str(folder))
    command = shlex.join(profile["argv"])
    assert mode == "AGENT_AWAITING_CONFIRMATION"
    assert o._pending_action["type"] == "run_code_tests"
    assert o._pending_action["args"] == dict(profile=profile, lang=lang)
    assert json.loads(Path(o._PENDING_ACTION_FILE).read_text()) == o._pending_action
    expected = (f"Run the tests of loja?\nCommand: {command}\nIn the folder: {folder}\nNo network; stops after 30 s.\n"
        "Reply 'yes' (exactly that word) to run them or 'no' to cancel." if en else
        f"Rodar os testes de loja?\nComando: {command}\nNa pasta: {folder}\nSem internet; para em 30 s.\n"
        "Responda 'sim' (exatamente essa palavra) para rodar ou 'não' para cancelar.")
    preview = tool_risk.build_preview("run_code_tests", dict(profile=profile), lang=lang)
    assert reply == preview["human"] == expected
    assert preview["llm"] == "[SYSTEM ERROR] run_code_tests requires the typed phrase."
    assert preview["cancel_label"] == ("tests of loja" if en else "os testes de loja")
    assert tool_risk.TOOL_RISK["run_code_tests"] == tool_risk.S3
    log = Mock()
    monkeypatch.setattr("src.core.staging_mixin.audit.log", log)
    from src.core.effect_runner import run_effect
    from src.core.code_runner import TestRun
    monkeypatch.setattr("src.core.staging_mixin.run_effect", run_effect)
    monkeypatch.setattr("src.core.code_runner.run_tests", lambda *a, **kw: TestRun("passed", 0, 0.4, "OK\n", 3, False))
    assert turn("yes" if en else "sim") == (
        "Ran the tests of loja in a copy of the folder, without network: they passed (in 0.4 s).\n\nOutput (the end):\n```\nOK\n```" if en else
        "Rodei os testes de loja numa cópia da pasta, sem internet: passaram (em 0,4 s).\n\nSaída (o fim):\n```\nOK\n```", "CODE_TESTS_PASSED")
    assert o._pending_action is None
    assert [c.kwargs["payload"] for c in log.call_args_list if c.args == ("–", "code", "tests_check")] == [{"outcome": "checked"}]


@pytest.mark.parametrize("case,expected,mode", [
    ("voice", TYPED, "CODE_TESTS_REFUSED"), ("background", TYPED, "CODE_TESTS_REFUSED"),
    ("two", "Uma pasta por vez: roda os testes em ~/pasta.", "CODE_TESTS_REFUSED"),
    ("missing", "não achei missing na pasta loja. Nela tem: cart.py.", "FILE_ERROR"),
    ("file", "Me aponta a pasta do projeto: roda os testes em ~/pasta.", "CODE_TESTS_REFUSED"),
    ("outside", "não posso rodar testes em outside: a pasta não está autorizada para código.", "FILE_ERROR"),
    ("empty", "não achei testes em loja (arquivos test*.py).", "CODE_TESTS_REFUSED"),
    ("large", "o projeto loja tem 41 arquivos .py — é grande demais para eu ler de uma vez; me aponta uma subpasta ou um arquivo.", "FILE_LIST")])
def test_refusals(setup, case, expected, mode):
    o, folder = setup
    ask = ASK
    if case == "voice":
        o.set_turn_origin("voice")
    elif case == "background":
        o.background = True
    elif case == "two":
        ask += " e ~/cortex_sandbox/loja/cart.py"
    elif case in ("file", "missing"):
        ask += "/cart.py" if case == "file" else "/missing"
    elif case == "outside":
        (Path.home() / "outside").mkdir()
        ask = "roda os testes em ~/outside"
    elif case == "empty":
        (folder / "tests/test_cart.py").unlink()
    elif case == "large":
        for i in range(39):
            (folder / f"f{i}.py").touch()
    assert o._process_core(ask) == (expected, mode)
    assert o._pending_action is None


@pytest.mark.parametrize("word", ["ok", "pode", "sim, pode"])
@pytest.mark.parametrize("en", [False, True])
def test_loose_approval_and_cancel(setup, word, en):
    o, _ = setup
    assert o._process_core(ASK.replace("roda os testes em", "run the tests of") if en else ASK)[1] == "AGENT_AWAITING_CONFIRMATION"
    before = Path(o._PENDING_ACTION_FILE).read_bytes()
    assert o._process_core(word) == ("To run the tests, reply exactly 'yes' (or 'no' to cancel)." if en else
        "Para rodar os testes, responda exatamente 'sim' (ou 'não' para cancelar).", "AWAITING_CONFIRMATION")
    assert Path(o._PENDING_ACTION_FILE).read_bytes() == before
    assert o._process_core("não")[1] == "AGENT_ACTION_CANCELLED"
    assert o._pending_action is None


@pytest.mark.parametrize("origin", [("voice", None), ("ui", "B")])
def test_approval_elsewhere(setup, origin):
    o, _ = setup
    o.set_turn_origin("ui", "A")
    o._process_core(ASK)
    before = Path(o._PENDING_ACTION_FILE).read_bytes()
    o.set_turn_origin(*origin)
    assert o._process_core("sim")[1] == "AGENT_ACTION_ELSEWHERE"
    assert Path(o._PENDING_ACTION_FILE).read_bytes() == before


@pytest.mark.parametrize("change", ["argv", "cwd", "revoked"])
def test_tampered_pending(setup, change, monkeypatch):
    o, folder = setup
    o._process_core(ASK)
    path = Path(o._PENDING_ACTION_FILE)
    pending = json.loads(path.read_text())
    if change == "revoked":
        code_roots.remove(str(folder.parent))
    else:
        pending["args"]["profile"][change] = [sys.executable, "-c", "pass"] if change == "argv" else str(folder.parent)
        path.write_text(json.dumps(pending))
    log = Mock()
    monkeypatch.setattr("src.core.staging_mixin.audit.log", log)
    assert o._process_core("sim") == (REFUSED, "CODE_TESTS_REFUSED")
    assert [c.kwargs["payload"] for c in log.call_args_list if c.args == ("–", "code", "tests_check")] == [{"outcome": "refused"}]


def test_never_tool_and_old_route(setup, monkeypatch):
    o, _ = setup
    name, args = "run_code_tests", {}
    o.engine_analytical.chat_with_tools.side_effect = lambda text, specs, execute, **kw: (execute(name, args), name)
    assert "ERROR" in o._process_core("procura um documento sobre astronomia")[0]
    monkeypatch.setattr(planner, "decompose", lambda *a, **kw: planner.Plan("tests", [planner.Step(id="p1", tool=name, args=args)]))
    assert o.process("planeja: rodar testes")[1] == "AGENT_STAGE_ERROR"
    for method in (o._skill_macro_stage_step, o._execute_tool_unstaged, o._gate_background_action):
        assert "ERROR" in method(name, args)
    assert o._pending_action is None
    assert o._process_core(ASK + "/cart.py e explica")[1] == "AGENT_AWAITING_CONFIRMATION"
    assert o._pending_action["type"] == "read_file_at"
