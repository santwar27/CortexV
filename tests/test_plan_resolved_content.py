"""A plan's approval covers only the content its preview actually showed."""
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.core import planner
from src.core.audit import audit
from src.core.staging_mixin import Delivery
from tests.test_dispatch_parity import _build_real_orch
from tests.test_pending_origin import _turn

CONTENT = "Notícias do projeto X: entrega na sexta.\nConfira com a equipa."


def _plan(extra=None, body="{p1}"):
    steps = [planner.Step("p1", "web_search", {"query": "notícias do projeto X"}),
             planner.Step("p2", "send_email", {"to": "ana@exemplo.com", "body": body},
                          ["p1"])]
    return planner.Plan("pesquisar e enviar", steps + (extra or []))


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    o = _build_real_orch(tmp_path)
    o.background = False
    ran = Mock(return_value=CONTENT)
    monkeypatch.setattr(o, "_execute_tool_unstaged", ran)
    summary = Mock(side_effect=lambda r, **kw: planner._deterministic_summary(
        r.plan, r.results, kw["lang"]))
    monkeypatch.setattr(planner, "summarise", summary)

    def stage(plan=None, lang="pt"):
        monkeypatch.setattr(planner, "decompose", lambda *a, **kw: plan or _plan())
        reply, mode = _turn(o, "planeja: pesquisar e enviar", channel=None, lang=lang)
        assert mode == "AGENT_AWAITING_CONFIRMATION"
        return reply, dict(o._pending_action)
    return o, ran, summary, stage


@pytest.mark.parametrize("lang,stream", [("pt", False), ("en", True)])
def test_first_yes_shows_resolved_content_and_new_identity(rig, lang, stream):
    o, ran, summary, stage = rig
    preview, original = stage(lang=lang)
    assert "body: {p1}" in preview and CONTENT not in preview
    reply, mode = _turn(o, "sim", stream=stream)
    assert mode == "AGENT_AWAITING_CONFIRMATION"
    assert [c.args[0] for c in ran.call_args_list] == ["web_search"]
    pending = json.loads(Path(o._PENDING_ACTION_FILE).read_text())
    assert pending == o._pending_action and pending["type"] == "execute_plan"
    assert pending["id"] != original["id"]
    assert original["origin"] is None
    assert pending["origin"] == {"channel": "ui", "conversation": "A"}
    assert pending["staged_at"] and pending["args"]["lang"] == lang
    assert pending["args"]["objetivo"] == original["args"]["objetivo"]
    email, = pending["args"]["passos"]
    assert email["id"] == "p2" and email["args"]["body"] == CONTENT
    assert email["depende_de"] == []
    phrase = ("Antes de enviar, confira o conteúdo:" if lang == "pt" else
              "Before sending, check the content:")
    rest = planner.Plan(_plan().objetivo, [planner.Step(**email)])
    assert reply == (planner._deterministic_summary(_plan(), [
        planner.StepResult("p1", True, CONTENT)], lang) + "\n\n" + phrase +
        "\n" + planner.render_preview(rest, lang))
    summary.assert_not_called()
    events = [json.loads(line) for line in Path(audit.path).read_text().splitlines()]
    paused = [e for e in events if e["stage"] == "planner" and e["event"] == "paused"]
    assert [e["payload"] for e in paused] == [{"step_id": "p2", "tool": "send_email"}]
    assert _turn(o, "sim", conversation="B")[1] == "AGENT_ACTION_ELSEWHERE"
    assert json.loads(Path(o._PENDING_ACTION_FILE).read_text()) == pending
    assert ran.call_count == 1


@pytest.mark.parametrize("answer,mode,sends", [
    ("sim", "AGENT_ACTION_DONE", 1), ("não", "AGENT_ACTION_CANCELLED", 0)])
def test_second_answer_sends_exact_preview_or_cancels(rig, answer, mode, sends):
    o, ran, _, stage = rig
    stage()
    reply, _ = _turn(o, "sim")
    assert CONTENT in reply
    assert _turn(o, answer)[1] == mode
    emails = [c for c in ran.call_args_list if c.args[0] == "send_email"]
    assert len(emails) == sends
    if sends:
        assert emails[0].args[1] == {"to": "ana@exemplo.com", "body": CONTENT}
    assert o._pending_action is None and not Path(o._PENDING_ACTION_FILE).exists()


@pytest.mark.parametrize("plan,tools", [
    (planner.Plan("enviar", [planner.Step("p1", "send_email", {"body": "literal"})]),
     ["send_email"]),
    (planner.Plan("escrever", [_plan().passos[0], planner.Step(
        "p2", "write_document", {"content": "{p1}"}, ["p1"])]),
     ["web_search", "write_document"]),
])
def test_literal_s3_and_referenced_s2_still_run(rig, plan, tools):
    o, ran, _, stage = rig
    stage(plan)
    assert _turn(o, "sim")[1] == "AGENT_ACTION_DONE"
    assert [c.args[0] for c in ran.call_args_list] == tools
    assert ran.call_args.args[1] == ({"body": "literal"} if len(tools) == 1 else
                                     {"content": CONTENT})


@pytest.mark.parametrize("pause", [False, True])
def test_planner_pause_contract(pause):
    plan = _plan([planner.Step("p3", "create_reminder", {"title": "literal"})])
    run, on_step = Mock(return_value=CONTENT), Mock()
    result = planner.execute(plan, run, on_step=on_step, **(
        {"pause_before": lambda s: s.tool == "send_email"} if pause else {}))
    assert result.paused_at == ("p2" if pause else None)
    assert result.success is (not pause) and not result.cancelled
    assert run.call_count == (1 if pause else 3)
    assert on_step.call_count == 3
    assert result.results[0].success
    if pause:
        assert all(r.skipped and not r.success and r.error == "awaiting confirmation"
                   for r in result.results[1:])


@pytest.mark.parametrize("ref,rounds", [("{p1}", 2), ("{p2}", 3)])
def test_remaining_dependencies_and_repeated_pause(rig, ref, rounds):
    o, ran, _, stage = rig
    stage(_plan([planner.Step("p3", "create_reminder", {"title": ref},
                              ["p1", "p2"])]))
    assert _turn(o, "sim")[1] == "AGENT_AWAITING_CONFIRMATION"
    email, reminder = o._pending_action["args"]["passos"]
    assert email["args"]["body"] == CONTENT and email["depende_de"] == []
    assert reminder["args"]["title"] == (CONTENT if ref == "{p1}" else ref)
    assert reminder["depende_de"] == ["p2"]
    assert _turn(o, "sim")[1] == ("AGENT_ACTION_DONE" if rounds == 2 else
                                      "AGENT_AWAITING_CONFIRMATION")
    if rounds == 3:
        assert [c.args[0] for c in ran.call_args_list] == ["web_search", "send_email"]
        assert o._pending_action["args"]["passos"][0]["args"] == {"title": CONTENT}
        assert _turn(o, "sim")[1] == "AGENT_ACTION_DONE"
    assert [c.args[0] for c in ran.call_args_list] == [
        "web_search", "send_email", "create_reminder"]


def test_remainder_resolves_structured_fields_but_keeps_future_refs(rig):
    o, ran, _, stage = rig
    output = {"text": CONTENT, "fields": {"title": "Projeto X"}}
    ran.return_value = output
    stage(_plan([planner.Step("p3", "create_reminder",
                              {"title": "{p1.title} / {p2}"}, ["p1", "p2"])],
                body="{p1.title}: {p1.text}"))
    reply, mode = _turn(o, "sim")
    assert mode == "AGENT_AWAITING_CONFIRMATION" and ran.call_count == 1
    email, reminder = o._pending_action["args"]["passos"]
    resolved = planner._resolve_refs({"body": "{p1.title}: {p1.text}"}, {
        "p1": planner.StepResult("p1", True, output)})
    assert email["args"]["body"] == resolved["body"] == "Projeto X: " + CONTENT
    assert reminder["args"]["title"] == "Projeto X / {p2}"
    assert reminder["depende_de"] == ["p2"] and resolved["body"] in reply


@pytest.mark.parametrize("failure", ["write", "exception", "conflict", "cancel"])
def test_no_remainder_is_offered_when_persistence_refuses(rig, monkeypatch, failure):
    o, ran, summary, stage = rig
    stage()
    persist = o._persist_pending_action
    delivery = Delivery()
    o.set_turn_delivery(delivery)

    def refuse():
        if failure == "exception":
            raise OSError("cannot lock pending file")
        elif failure == "cancel":
            delivery.cancelled.set()
        elif failure == "conflict":
            Path(o._PENDING_ACTION_FILE).write_text(json.dumps({
                "type": "send_email", "args": {}, "id": "other",
                "origin": {"channel": "ui", "conversation": "B"}}))
        persist()
    monkeypatch.setattr(o, "_persist_pending_action", refuse)
    if failure == "write":
        monkeypatch.setattr("src.core.staging_mixin.os.replace",
                            Mock(side_effect=OSError("disk full")))
    reply, mode = _turn(o, "sim")
    assert mode == {"write": "AGENT_STAGE_ERROR", "exception": "AGENT_STAGE_ERROR",
                    "conflict": "AGENT_STAGE_CONFLICT",
                    "cancel": "AGENT_ACTION_CANCELLED"}[failure]
    assert "Antes de enviar" not in reply and ran.call_count == 1
    summary.assert_not_called()
    if failure == "conflict":
        assert o._pending_action["id"] == "other"
    else:
        assert o._pending_action is None and not Path(o._PENDING_ACTION_FILE).exists()
