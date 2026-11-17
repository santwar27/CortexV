"""Typed approval runs contained tests; fixed replies keep output away from the model."""
import os
from unittest.mock import Mock

import pytest

from src.core import code_commands, code_roots, code_runner, code_sandbox, effect_store
from src.core.staging_mixin import Delivery
from src.tools.local_reader import LocalFileReader
from src.ui import routes
from src.ui.chat_store import ChatStore
from tests.test_claim_remove_failure import _turn
from tests.test_dispatch_parity import _build_real_orch
from tests.test_ui_routes import client, ui_app

MARKERS = {"pt": "\n\nSaída (o fim):\n```\n", "en": "\n\nOutput (the end):\n```\n"}
NOTES = {"pt": "(A saída foi cortada: mostrei só o fim.)",
         "en": "(The output was cut: I showed only the end.)"}
CASES = [
    ("passed", 0, "PASSED",
     "Rodei os testes de loja numa cópia da pasta, sem internet: passaram (em 0,4 s).",
     "Ran the tests of loja in a copy of the folder, without network: they passed (in 0.4 s)."),
    ("failed", 1, "FAILED",
     "Rodei os testes de loja numa cópia da pasta, sem internet: falharam (código de saída 1, em 0,4 s).",
     "Ran the tests of loja in a copy of the folder, without network: they failed (exit code 1, in 0.4 s)."),
    ("failed", 5, "NONE",
     "Rodei o unittest em loja numa cópia da pasta, mas ele não achou nenhum teste para rodar (código de saída 5). Os arquivos test*.py precisam estar na pasta ou numa subpasta com __init__.py.",
     "Ran unittest in loja in a copy of the folder, but it found no tests to run (exit code 5). The test*.py files must be in the folder or in a subfolder with __init__.py."),
    ("timeout", None, "TIMEOUT",
     "Os testes de loja passaram de 30 s — parei tudo. Não sei o resultado.",
     "The tests of loja went past 30 s — I stopped everything. I don't know the result."),
    ("stopped", None, "STOPPED",
     "Parei os testes de loja a seu pedido — nada ficou rodando.",
     "Stopped the tests of loja as you asked — nothing is left running."),
    ("unavailable", None, "REFUSED",
     "Não rodei os testes de loja: a contenção do Mac (sandbox-exec) não está disponível agora. Nada foi rodado.",
     "I didn't run the tests of loja: the Mac's containment (sandbox-exec) isn't available right now. Nothing was run."),
    ("too_big", None, "REFUSED",
     "Não rodei os testes de loja: a pasta é grande demais para copiar (mais de 2.000 arquivos ou 50 MB). Nada foi rodado.",
     "I didn't run the tests of loja: the folder is too big to copy (over 2,000 files or 50 MB). Nothing was run."),
    ("error", None, "ERROR",
     "Não consegui rodar os testes de loja (PermissionError). Nada ficou rodando.",
     "I couldn't run the tests of loja (PermissionError). Nothing is left running."),
]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CORTEX_CODE_ROOTS_FILE", raising=False)
    folder = tmp_path / "cortex_sandbox" / "loja"
    folder.mkdir(parents=True)
    (folder / "test_calc.py").write_text("raise AssertionError('unexpected execution')\n")
    assert code_roots.add(str(folder.parent))[0]
    store = effect_store.EffectStore(str(tmp_path / "effects.db"))
    monkeypatch.setattr(effect_store, "store", store)
    o = _build_real_orch(tmp_path)
    o._PENDING_ACTION_FILE = str(tmp_path / "pending.json")
    o.background, o._reply_lang = False, "pt"
    o.set_turn_origin("ui", "conversation")
    o.reader = LocalFileReader(str(tmp_path / "vault"))
    chat = Mock(side_effect=AssertionError("ollama.chat called"))
    monkeypatch.setattr("ollama.chat", chat)
    yield o, folder, store
    chat.assert_not_called()
    for engine in (o.engine_analytical, o.engine_creative):
        engine.chat.assert_not_called()
        engine.chat_with_tools.assert_not_called()
    if store._conn is not None:
        store._conn.close()
    if store._owner_fd is not None:
        os.close(store._owner_fd)


def stage(o, stream=False, lang="pt"):
    ask = "run the tests in" if lang == "en" else "roda os testes em"
    assert _turn(o, ask + " ~/cortex_sandbox/loja", stream, lang)[1] == "AGENT_AWAITING_CONFIRMATION"
    delivery = Delivery()
    o.set_turn_delivery(delivery)
    return delivery


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("output,truncated", [("", False), ("END\n\n", False), ("END\n\n", True)])
@pytest.mark.parametrize("status,code,mode,pt,en", CASES)
def test_fixed_result(setup, monkeypatch, stream, lang, output, truncated, status, code, mode, pt, en):
    o, folder, store = setup
    delivery = stage(o, stream, lang)
    profile = code_commands.tests_profile(str(folder))
    assert o._pending_action["args"]["profile"] == profile
    def run(received, cancelled):
        assert received == profile and callable(cancelled) and not cancelled()
        [effect] = store.in_states(["executando"])
        assert effect.action == "run_code_tests"
        return code_runner.TestRun(status, code, .4, output, len(output), truncated, "PermissionError")
    fake = Mock(side_effect=run)
    monkeypatch.setattr(code_runner, "run_tests", fake)
    log = Mock()
    monkeypatch.setattr("src.core.staging_mixin.audit.log", log)
    ran = status in ("passed", "failed", "timeout", "stopped")
    expected = en if lang == "en" else pt
    if ran and output:
        expected += MARKERS[lang] + output.rstrip("\n") + "\n```"
        if truncated:
            expected += "\n" + NOTES[lang]
    if mode == "FAILED":
        expected += ("\n\nTo have me explain the failure, say 'explain the failure'." if lang == "en" else
                     "\n\nPara eu explicar a falha, diga 'explica a falha'.")
    assert _turn(o, "yes" if lang == "en" else "sim", stream, lang) == (expected, "CODE_TESTS_" + mode)
    fake.assert_called_once()
    [effect] = store.recent()
    assert effect.state == ("feito" if ran else "falhou")
    assert effect.action == "run_code_tests"
    assert effect.args == {"folder": str(folder), "command": code_commands.render(profile)}
    assert effect.origin == {"channel": "ui", "conversation": "conversation", "turn": delivery.id}
    assert [c.kwargs["payload"] for c in log.call_args_list
            if c.args == ("–", "code", "tests_check")] == [{"outcome": "checked"}]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("case", ["stop", "before", "raise", "unrecorded"])
def test_stop_and_uncertainty(setup, monkeypatch, stream, case):
    o, _, store = setup
    delivery = stage(o, stream)
    def run(profile, cancelled):
        assert not cancelled()
        if case == "raise":
            raise RuntimeError("boom")
        delivery.cancelled.set()
        assert cancelled()
        return code_runner.TestRun("stopped", None, .4, "", 0, False)
    fake = Mock(side_effect=run)
    monkeypatch.setattr(code_runner, "run_tests", fake)
    if case == "before":
        delivery.cancelled.set()
    elif case == "unrecorded":
        monkeypatch.setattr(store, "create", lambda *a, **kw: None)
    reply, mode = _turn(o, "sim", stream)
    if case == "stop":
        assert (reply, mode) == (CASES[4][3], "CODE_TESTS_STOPPED")
    elif case == "before":
        assert (reply, mode) == ("Parei antes de executar — não fiz nada.", "AGENT_ACTION_CANCELLED")
    elif case == "unrecorded":
        assert (reply, mode) == ("Não consegui registrar essa ação antes de executar — não fiz nada.", "AGENT_ACTION_ERROR")
    else:
        assert (reply, mode) == ("Não sei se deu certo (boom). Confere antes de pedir de novo — não vou repetir sozinho.", "AGENT_ACTION_ERROR")
    assert fake.call_count == (1 if case in ("stop", "raise") else 0)
    effects = store.recent()
    if case == "unrecorded":
        assert effects == []
    else:
        assert len(effects) == 1
        assert effects[0].state == {"stop": "feito", "before": "cancelado", "raise": "desconhecido"}[case]


@pytest.mark.parametrize("lang", ["pt", "en"])
def test_history_keeps_only_headline(tmp_path, monkeypatch, lang):
    from src.core.staging_mixin import CODE_TESTS_OUTPUT
    assert CODE_TESTS_OUTPUT == MARKERS
    store = ChatStore(str(tmp_path / "chat.db"))
    monkeypatch.setattr(routes, "chat_store", store)
    cid = store.create_conversation()["id"]
    headline = CASES[1][4 if lang == "en" else 3]
    # Even another language's marker inside output must not leak any of the block.
    reply = headline + MARKERS[lang] + "PRIVATE" + MARKERS["pt" if lang == "en" else "en"] + "END\n```\n" + NOTES[lang]
    ordinary = "ordinary" + MARKERS[lang] + "unchanged\n```"
    store.append_turn(cid, "oi", ordinary, status_mode="CHAT")
    saved = store.append_turn(cid, "sim", reply, status_mode="CODE_TESTS_FAILED")
    assert routes._conversation_history_msgs(cid, 5) == [
        {"role": "user", "content": "oi"}, {"role": "assistant", "content": ordinary},
        {"role": "user", "content": "sim"}, {"role": "assistant", "content": headline}]
    assert store.get_turns(cid)[1] == saved and saved["reply"] == reply
    store._conn.close()


@pytest.mark.parametrize("lang,label", [("pt", "rodar os testes"), ("en", "run the tests")])
def test_effect_label(setup, client, ui_app, monkeypatch, lang, label):
    o, _, store = setup
    ui_app.state.orchestrator = o
    stage(o, lang=lang)
    monkeypatch.setattr(code_runner, "run_tests", lambda *a, **kw: code_runner.TestRun("passed", 0, .4, "OK\n", 3, False))
    _turn(o, "yes" if lang == "en" else "sim", False, lang)
    response = client.get("/ui/effects", headers={"X-Cortex-Key": "test-key-xyz"}, params={"lang": lang})
    assert response.status_code == 200
    [item] = response.json()["effects"]
    assert item["id"] == store.recent()[0].id and item["label"] == label


@pytest.mark.parametrize("case", ["passed", "failed", "none"])
def test_real_approval(setup, case):
    if not code_sandbox.available():
        pytest.skip("sandbox-exec unavailable here")
    o, folder, store = setup
    (folder / "calc.py").write_text("def add(a, b): return a + b\n")
    (folder / "test_calc.py").unlink()
    test = folder / "test_calc.py"
    if case == "none":
        (folder / "tests").mkdir()
        test = folder / "tests" / "test_x.py"
    test.write_text("import unittest\nfrom calc import add\nclass Calc(unittest.TestCase):\n"
                    f"    def test_add(self): self.assertEqual(add(1, 2), {4 if case == 'failed' else 3})\n")
    before = {p.relative_to(folder): p.read_bytes() for p in folder.rglob("*") if p.is_file()}
    stage(o)
    reply, mode = _turn(o, "sim", False)
    assert mode == "CODE_TESTS_" + {"passed": "PASSED", "failed": "FAILED", "none": "NONE"}[case]
    headline, output = reply.split(MARKERS["pt"], 1)
    if case == "none":
        assert headline == CASES[2][3]
    else:
        assert headline.startswith("Rodei os testes de loja numa cópia da pasta, sem internet: " + (
            "passaram (em " if case == "passed" else "falharam (código de saída 1, em "))
        assert ("OK" if case == "passed" else "FAILED") in output
    assert output.split("\n\nPara eu explicar a falha,", 1)[0].endswith("\n```")
    assert store.recent()[0].state == "feito"
    assert {p.relative_to(folder): p.read_bytes() for p in folder.rglob("*") if p.is_file()} == before
