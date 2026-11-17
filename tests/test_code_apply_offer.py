"""Successful applies stage the existing test preview; only typed approval runs it."""
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.core import code_commands, code_runner
from tests.test_claim_remove_failure import _turn
from tests.test_code_apply_write import success
from tests.test_code_proposal import OLD, NEW, REQUEST, setup  # noqa: F401


@pytest.fixture
def rig(setup, monkeypatch):
    fake = Mock(return_value=code_runner.TestRun("passed", 0, .4, "", 0, False))
    monkeypatch.setattr(code_runner, "run_tests", fake)
    return (*setup, fake)


def project(rig, rel="loja/cart.py", tests="loja/tests/test_cart.py"):
    _, llm, original, _ = rig
    root = original.parent
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if path != original:
        original.rename(path)
    llm.return_value = {"message": {"content": f"FILE: {path.name}\n```python\n{NEW}```"}}
    if tests:
        test = root / tests
        test.parent.mkdir(parents=True, exist_ok=True)
        test.write_text("raise AssertionError('project code ran')\n")
    return path


def proposal(rig, path, stream=False, lang="pt"):
    # Same read/proposal approval cycle as test_code_apply_approval.proposal.
    o, llm, _, fake = rig
    before = path.read_bytes()
    ask = f"muda em {path}: {REQUEST}" if lang == "pt" else f"change {path}: rename normalise_label to clean_label"
    assert _turn(o, ask, stream, lang)[1] == "AGENT_AWAITING_CONFIRMATION"
    assert _turn(o, "sim", stream, lang)[1] == "CODE_PROPOSAL"
    assert o._pending_action["type"] == "apply_code_change"
    assert path.read_bytes() == before
    llm.assert_called_once()
    fake.assert_not_called()
    return o._pending_action["args"]


def preview(folder, lang="pt"):
    command = code_commands.render(code_commands.tests_profile(str(folder)))
    return (f"Run the tests of {folder.name}?\nCommand: {command}\nIn the folder: {folder}\nNo network; stops after 30 s.\n"
            "Reply 'yes' (exactly that word) to run them or 'no' to cancel." if lang == "en" else
            f"Rodar os testes de {folder.name}?\nComando: {command}\nNa pasta: {folder}\nSem internet; para em 30 s.\n"
            "Responda 'sim' (exatamente essa palavra) para rodar ou 'não' para cancelar.")


def pending(o, folder, lang="pt"):
    assert o._pending_action["type"] == "run_code_tests"
    assert o._pending_action["args"] == dict(profile=code_commands.tests_profile(str(folder)), lang=lang)
    assert json.loads(Path(o._PENDING_ACTION_FILE).read_text()) == o._pending_action


def no_pending(o):
    assert o._pending_action is None and not Path(o._PENDING_ACTION_FILE).exists()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang", ["pt", "en"])
def test_offer_then_exact_approval(rig, stream, lang):
    o, llm, _, fake = rig
    path = project(rig)
    args = proposal(rig, path, stream, lang)
    # The apply's language wins even if this confirmation turn uses another.
    assert _turn(o, "sim", stream, "en" if lang == "pt" else "pt") == (
        success(args["id"], path, lang)[0] + "\n\n" + preview(path.parent, lang), "AGENT_ACTION_DONE")
    pending(o, path.parent, lang)
    assert path.read_bytes() == NEW.encode()
    fake.assert_not_called()
    assert _turn(o, "sim", stream, lang) == (
        "Ran the tests of loja in a copy of the folder, without network: they passed (in 0.4 s)." if lang == "en" else
        "Rodei os testes de loja numa cópia da pasta, sem internet: passaram (em 0,4 s).", "CODE_TESTS_PASSED")
    fake.assert_called_once()
    assert fake.call_args.args == (code_commands.tests_profile(str(path.parent)),)
    assert callable(fake.call_args.kwargs["cancelled"])
    no_pending(o)
    llm.assert_called_once()


@pytest.mark.parametrize("test_rel,folder_rel", [
    ("loja/tests/test_cart.py", "loja"),
    ("loja/sub/tests/test_x.py", "loja/sub"),
    ("test_root.py", "."),
])
def test_nearest_folder_including_root(rig, test_rel, folder_rel):
    o, _, original, fake = rig
    path = project(rig, "loja/sub/x.py", test_rel)
    args = proposal(rig, path)
    folder = original.parent / folder_rel
    assert _turn(o, "sim", False) == (
        success(args["id"], path)[0] + "\n\n" + preview(folder), "AGENT_ACTION_DONE")
    pending(o, folder)
    fake.assert_not_called()


@pytest.mark.parametrize("case", ["none", "large"])
def test_silently_no_offer(rig, case):
    o, _, original, fake = rig
    path = project(rig, "labels.py", None) if case == "none" else project(rig)
    # Tests above the authorised root must never produce an offer.
    (original.parent.parent / "test_outside.py").write_text("pass\n")
    if case == "large":
        for i in range(o._CODE_PROJECT_MAX_FILES - 1):
            (path.parent / f"f{i}.py").touch()
    args = proposal(rig, path)
    assert _turn(o, "sim", False) == success(args["id"], path)
    assert path.read_bytes() == NEW.encode()
    no_pending(o)
    fake.assert_not_called()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("end", ["desfaz", "não"])
def test_undo_or_cancel_offer(rig, stream, end):
    o, _, _, fake = rig
    path = project(rig)
    original = OLD.replace("\n", "\r\n").encode()
    path.write_bytes(original)
    args = proposal(rig, path, stream)
    assert _turn(o, "sim", stream)[1] == "AGENT_ACTION_DONE"
    pending(o, path.parent)
    result = _turn(o, end, stream)
    if end == "desfaz":
        assert result == (f"Pronto, desfiz: a mudança em {path} — o arquivo voltou a ser o original (proposta {args['id']}).", "UNDO_DONE")
        assert path.read_bytes() == original
    else:
        assert result == ("Cancelado. Não vou prosseguir com os testes de loja.", "AGENT_ACTION_CANCELLED")
        assert path.read_bytes() == NEW.encode()
    no_pending(o)
    fake.assert_not_called()


@pytest.mark.parametrize("case", ["changed", "raise"])
def test_no_offer_without_success(rig, monkeypatch, case):
    o, _, _, fake = rig
    path = project(rig)
    proposal(rig, path)
    if case == "changed":
        path.write_bytes(b"# another edit\r\n")
        expected = ("cart.py mudou desde a proposta — peça uma proposta nova. Nada foi aplicado.", "CODE_APPLY_REFUSED")
    else:
        monkeypatch.setattr(o, "_write_code_change", Mock(side_effect=OSError("full")))
        expected = ("Não sei se deu certo (full). Confere antes de pedir de novo — não vou repetir sozinho.", "AGENT_ACTION_ERROR")
    before = path.read_bytes()
    assert _turn(o, "sim", False) == expected
    assert path.read_bytes() == before
    no_pending(o)
    fake.assert_not_called()


@pytest.mark.parametrize("mode", ["FILE_LIST", "CODE_TESTS_REFUSED"])
def test_stage_refusal_keeps_apply_reply(rig, monkeypatch, mode):
    o, _, _, fake = rig
    path = project(rig)
    args = proposal(rig, path)
    stage = Mock(return_value=("refused", mode))
    monkeypatch.setattr(o, "_stage_code_tests", stage)
    assert _turn(o, "sim", False) == success(args["id"], path)
    stage.assert_called_once_with(str(path.parent))
    no_pending(o)
    fake.assert_not_called()


@pytest.mark.parametrize("stream", [False, True])
def test_strict_and_typed_offer_approval(rig, stream):
    o, _, _, fake = rig
    path = project(rig)
    proposal(rig, path, stream)
    assert _turn(o, "sim", stream)[1] == "AGENT_ACTION_DONE"
    pending(o, path.parent)
    before = Path(o._PENDING_ACTION_FILE).read_bytes()
    for word in ("ok", "sim, pode"):
        assert _turn(o, word, stream) == (
            "Para rodar os testes, responda exatamente 'sim' (ou 'não' para cancelar).", "AWAITING_CONFIRMATION")
        assert Path(o._PENDING_ACTION_FILE).read_bytes() == before
    o.set_turn_origin("voice")
    assert _turn(o, "sim", stream) == (
        "Essa confirmação está esperando onde você digitou o pedido — responde lá digitando.", "AGENT_ACTION_ELSEWHERE")
    assert Path(o._PENDING_ACTION_FILE).read_bytes() == before
    fake.assert_not_called()
