"""Typed approval checks the stored proposal; every turn leaves source bytes alone."""
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.core import code_proposals, code_roots, planner, tool_risk
from tests.test_code_proposal import ASK, NEW, TYPED, setup  # noqa: F401

QUESTION = {
    "pt": "Aplicar esta mudança em labels.py? Só aplico se o arquivo não mudar até lá.\n"
          "Responda 'sim' (exatamente essa palavra) para aplicar ou 'não' para cancelar.",
    "en": "Apply this change to labels.py? I only apply it if the file does not change before then.\n"
          "Reply 'yes' (exactly that word) to apply it or 'no' to cancel.",
}


def turn(setup, text, stream=False, lang="pt", write=False):
    from contextlib import nullcontext
    from unittest.mock import patch
    from src.core.effect_runner import run_effect

    o, _, path = setup
    before = path.read_bytes() if path.exists() else None
    with patch("src.core.staging_mixin.run_effect", run_effect) if write else nullcontext():
        if stream:
            *_, (reply, _, mode, _) = o.process_stream(text, language=lang)
            result = reply, mode
        else:
            result = o._process_core(text, language=lang)
    assert (path.read_bytes() if path.exists() else None) == (NEW.encode() if write else before)
    return result


def proposal(setup, stream=False, lang="pt"):
    ask = ASK if lang == "pt" else "change ~/cortex_sandbox/labels.py: rename normalise_label to clean_label"
    assert turn(setup, ask, stream, lang)[1] == "AGENT_AWAITING_CONFIRMATION"
    reply, mode = turn(setup, "sim", stream, lang)
    assert mode == "CODE_PROPOSAL"
    assert setup[0]._pending_action is not None
    return reply, setup[0]._pending_action["args"]


@pytest.fixture
def audit_check(monkeypatch):
    log = Mock()
    monkeypatch.setattr("src.core.staging_mixin.audit.log", log)
    monkeypatch.setattr("src.core.staging_mixin.run_effect", Mock(side_effect=AssertionError("effect ran")))
    return lambda: [c.kwargs["payload"] for c in log.call_args_list
                    if c.args == ("–", "code", "apply_check")]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("origin", [("terminal", None), ("ui", "A")])
@pytest.mark.parametrize("lang", ["pt", "en"])
def test_question_preview_and_binding(setup, stream, origin, lang):
    o, _, path = setup
    o.set_turn_origin(*origin)
    reply, args = proposal(setup, stream, lang)
    assert reply.endswith(QUESTION[lang])
    record = code_proposals.load(args["id"])
    assert args == dict(id=args["id"], lang=lang, **{k: record[k] for k in
                       ("path", "original_sha256", "new_sha256", "diff")})
    pending = json.loads(Path(o._PENDING_ACTION_FILE).read_text())
    assert pending == o._pending_action and pending["type"] == "apply_code_change"
    assert pending["origin"] == o._turn_origin()
    expected = (f"Aplicar a proposta {args['id']} em {path}?\n```diff\n{record['diff']}\n```\n"
                "Só aplico se o arquivo não mudou desde a proposta.\n" if lang == "pt" else
                f"Apply proposal {args['id']} to {path}?\n```diff\n{record['diff']}\n```\n"
                "I only apply it if the file has not changed since the proposal.\n") + QUESTION[lang].splitlines()[-1]
    preview = tool_risk.build_preview("apply_code_change", args, lang=lang)
    assert o._human_confirmation_prompt(pending, lang=lang) == preview["human"] == expected
    assert preview["llm"] == "[SYSTEM ERROR] apply_code_change requires a typed proposal."
    assert preview["cancel_label"] == ("a mudança em labels.py" if lang == "pt" else "change to labels.py")
    assert tool_risk.TOOL_RISK["apply_code_change"] == tool_risk.S3
    long_diff = record["diff"] * 1000
    assert f"```diff\n{long_diff}\n```" in tool_risk.build_preview(
        "apply_code_change", {**args, "diff": long_diff}, lang=lang)["human"]


def test_question_ignores_cwd_studies(setup, tmp_path, monkeypatch):
    """A populated checkout must not add Pedro's daily study announcement."""
    checkout = tmp_path / "checkout"
    pending = checkout / "data/knowledge/pending"
    approved = checkout / "data/knowledge/approved"
    pending.mkdir(parents=True)
    approved.mkdir()
    briefing = pending / "sentinel.md"
    briefing.write_text("# Topic: Private pending study\n", encoding="utf-8")
    marker = checkout / "data/last_announce.txt"
    marker.write_text("2000-01-01", encoding="utf-8")
    monkeypatch.chdir(checkout)

    reply, _ = proposal(setup, stream=True)

    assert reply.endswith(QUESTION["pt"])
    assert briefing.read_text(encoding="utf-8") == "# Topic: Private pending study\n"
    assert list(approved.iterdir()) == []
    assert marker.read_text(encoding="utf-8") == "2000-01-01"


@pytest.mark.parametrize("word,lang", [("sim", "pt"), ("Sim.", "pt"), ("yes", "en")])
@pytest.mark.parametrize("stream", [False, True])
def test_exact_approval(setup, audit_check, word, lang, stream):
    o, llm, path = setup
    _, args = proposal(setup, stream, lang)
    expected = (f"Apliquei a proposta {args['id']} em {path} (−3 +3 linhas). Antes de gravar, guardei uma cópia exata do original. Para voltar ao original, diga 'desfaz'."
                if lang == "pt" else f"Applied proposal {args['id']} to {path} (−3 +3 lines). Before writing, I saved an exact copy of the original. To go back to the original, say 'undo'.")
    assert turn(setup, word, stream, lang, write=True) == (expected, "AGENT_ACTION_DONE")
    assert o._pending_action is None and not Path(o._PENDING_ACTION_FILE).exists()
    assert audit_check() == [dict(id=args["id"], outcome="checked")]
    llm.assert_called_once()


@pytest.mark.parametrize("word", ["ok", "pode", "beleza", "sim, pode"])
@pytest.mark.parametrize("lang", ["pt", "en"])
def test_loose_approval(setup, audit_check, word, lang):
    o, _, _ = setup
    proposal(setup, lang=lang)
    before = Path(o._PENDING_ACTION_FILE).read_bytes()
    strict = ("Para aplicar uma mudança de código, responda exatamente 'sim' (ou 'não' para cancelar)."
              if lang == "pt" else "To apply a code change, reply exactly 'yes' (or 'no' to cancel).")
    assert turn(setup, word, lang=lang) == (strict, "AWAITING_CONFIRMATION")
    assert Path(o._PENDING_ACTION_FILE).read_bytes() == before and audit_check() == []
    assert turn(setup, "sim", lang=lang, write=True)[1] == "AGENT_ACTION_DONE"


@pytest.mark.parametrize("origin", [("voice", None), ("ui", "B")])
@pytest.mark.parametrize("end", ["não", "qual é a capital da Austrália?"])
def test_elsewhere_cancel_and_drop(setup, audit_check, origin, end):
    o, _, _ = setup
    o.set_turn_origin("ui", "A")
    proposal(setup)
    before = Path(o._PENDING_ACTION_FILE).read_bytes()
    o.set_turn_origin(*origin)
    assert turn(setup, "sim") == (
        "Essa confirmação está esperando onde você digitou o pedido — responde lá digitando.", "AGENT_ACTION_ELSEWHERE")
    assert Path(o._PENDING_ACTION_FILE).read_bytes() == before
    o.set_turn_origin("ui", "A")
    result = turn(setup, end)
    if end == "não":
        assert result[1] == "AGENT_ACTION_CANCELLED"
    assert o._pending_action is None and audit_check() == []


@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("failure", ["edited", "deleted", "revoked", "artifact", "swapped", "expired",
                                     "path", "original_sha256", "new_sha256", "diff", "syntax"])
def test_rechecks(setup, audit_check, failure, lang):
    o, _, path = setup
    _, args = proposal(setup, lang=lang)
    expected = (f"Não achei a proposta {args['id']} como ela foi mostrada — peça de novo." if lang == "pt" else
                f"I couldn't find proposal {args['id']} as it was shown — please ask again.")
    reason = "proposal"
    if failure in ("edited", "deleted"):
        path.write_bytes(b"# changed\r\n") if failure == "edited" else path.unlink()
        expected = ("labels.py mudou desde a proposta — peça uma proposta nova." if lang == "pt" else
                    "labels.py changed since the proposal — ask for a new one.")
        reason = "changed"
    elif failure == "revoked":
        code_roots.remove(str(path.parent))
        why = "unsupported format — .py only inside a folder authorised for code"
        expected = f"não posso mudar labels.py: {why}." if lang == "pt" else f"I can't change labels.py: {why}."
        reason = "path"
    elif failure == "artifact":
        (Path.home() / ".cortex/code_proposals" / (args["id"] + ".json")).write_text("{}")
    elif failure in ("swapped", "syntax"):
        record = code_proposals.load(args["id"])
        record["new_text"] = "def broken(:\n"
        record["diff"] = "another diff"
        args["id"] = code_proposals.save(record)
        if failure == "syntax":
            args["diff"] = record["diff"]
            expected = "A proposta guardada não é Python válido." if lang == "pt" else "The stored proposal is not valid Python."
            reason = "syntax"
        else:
            expected = (f"Não achei a proposta {args['id']} como ela foi mostrada — peça de novo." if lang == "pt" else
                        f"I couldn't find proposal {args['id']} as it was shown — please ask again.")
    elif failure == "expired":
        o._pending_action["staged_at"] = "2000-01-01T00:00:00+00:00"
    else:
        args[failure] = "tampered"
    o._persist_pending_action()
    reply, mode = turn(setup, "sim", lang=lang)
    if failure == "expired":
        assert (reply, mode) == (("Tinha uma ação antiga aguardando confirmação e descartei por segurança (passou de 15 minutos). Me pede de novo se ainda quiser."
            if lang == "pt" else "There was an old action awaiting confirmation and I discarded it for safety (it was over 15 minutes old). Ask me again if you still want it."), "STALE_ACTION_DISCARDED")
        assert audit_check() == []
    else:
        assert (reply, mode) == (expected + (" Nada foi aplicado." if lang == "pt" else " Nothing was applied."), "CODE_APPLY_REFUSED")
        assert audit_check() == [dict(id=args["id"], outcome=reason)]
    assert o._pending_action is None


@pytest.mark.parametrize("failure", ["save", "refused"])
def test_no_artifact_no_question(setup, monkeypatch, failure):
    o, llm, _ = setup
    if failure == "save":
        monkeypatch.setattr(code_proposals, "save", Mock(side_effect=OSError("full")))
    else:
        llm.return_value = {"message": {"content": "not code"}}
    turn(setup, ASK)
    reply, _ = turn(setup, "sim")
    assert QUESTION["pt"] not in reply and o._pending_action is None


def test_direct_runner_typed_guard(setup, audit_check):
    o, _, path = setup
    _, args = proposal(setup)
    before = path.read_bytes()
    o.set_turn_origin("voice")
    assert o._run_action("apply_code_change", args) == (TYPED + " Nada foi aplicado.", "CODE_APPLY_REFUSED")
    assert audit_check() == [dict(id=args["id"], outcome="origin")]
    assert path.read_bytes() == before


def test_never_tool_plan_macro_background(setup, monkeypatch, audit_check):
    o, _, path = setup
    name, args = "apply_code_change", dict(id="0" * 16, path=str(path))
    before = path.read_bytes()
    o.engine_analytical.chat_with_tools.side_effect = lambda text, specs, execute, **kw: (execute(name, args), name)
    assert "ERROR" in turn(setup, "procura um documento sobre astronomia")[0]
    monkeypatch.setattr(planner, "decompose", lambda *a, **kw: planner.Plan("change", [planner.Step(id="p1", tool=name, args=args)]))
    assert o.process("planeja: mudar meu código")[1] == "AGENT_STAGE_ERROR"
    for method in (o._skill_macro_stage_step, o._execute_tool_unstaged, o._gate_background_action):
        assert "ERROR" in method(name, args)
    assert o._pending_action is None and audit_check() == [] and path.read_bytes() == before
