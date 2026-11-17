"""The offline replay harness for the semantic router (s83).

The router sat in shadow mode for eight weeks and logged 56 events, the
last on 24/08 — waiting for live traffic to calibrate it had been tried
twice. The corpus was already on disk: every sentence Pedro has said to
Cortex is in the audit log, it had just never been scored.

What is tested here is the harvesting, because that is where a replay can
quietly lie: score the wrong sentences and the thresholds that come out are
confident and wrong.
"""
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "router_replay",
    Path(__file__).resolve().parents[1] / "scripts" / "router_replay.py",
)
router_replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(router_replay)


def _audit(tmp_path, rows) -> Path:
    path = tmp_path / "audit.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                    encoding="utf-8")
    return path


def test_harvests_what_the_user_actually_said(tmp_path):
    path = _audit(tmp_path, [
        {"stage": "orchestrator", "event": "request",
         "payload": {"input": "tem vaga nova?"}},
        {"stage": "reflex", "event": "miss", "payload": {"input": "que horas são"}},
        {"stage": "voice", "event": "transcribed", "payload": {"text": "toca matuê"}},
    ])
    assert router_replay.harvest(path) == [
        "tem vaga nova?", "que horas são", "toca matuê"]


def test_ignores_everything_that_is_not_the_user_speaking(tmp_path):
    path = _audit(tmp_path, [
        {"stage": "engine", "event": "token_usage", "payload": {"input": "x"}},
        {"stage": "tool", "event": "dispatch", "payload": {"name": "send_email"}},
        {"stage": "web_search", "event": "snippets", "payload": {"query": "clima"}},
        {"stage": "orchestrator", "event": "response", "payload": {"mode": "OFFLINE"}},
    ])
    assert router_replay.harvest(path) == []


def test_de_duplicates_but_keeps_first_appearance_order(tmp_path):
    path = _audit(tmp_path, [
        {"stage": "orchestrator", "event": "request", "payload": {"input": "oi cortex"}},
        {"stage": "orchestrator", "event": "request", "payload": {"input": "tem vaga?"}},
        {"stage": "orchestrator", "event": "request", "payload": {"input": "oi cortex"}},
    ])
    assert router_replay.harvest(path) == ["oi cortex", "tem vaga?"]


@pytest.mark.parametrize("junk", ["\x1b\x1b", "\x1bc", "ok", "?", "   ", "12"])
def test_drops_terminal_noise_and_fragments_with_no_intent(tmp_path, junk):
    path = _audit(tmp_path, [
        {"stage": "orchestrator", "event": "request", "payload": {"input": junk}},
    ])
    assert router_replay.harvest(path) == []


def test_a_missing_log_is_not_a_crash(tmp_path):
    assert router_replay.harvest(tmp_path / "nope.jsonl") == []


def test_a_corrupt_line_does_not_stop_the_harvest(tmp_path):
    path = tmp_path / "audit.jsonl"
    path.write_text(
        "{not json at all\n"
        + json.dumps({"stage": "orchestrator", "event": "request",
                      "payload": {"input": "tem vaga nova?"}}) + "\n",
        encoding="utf-8")
    assert router_replay.harvest(path) == ["tem vaga nova?"]


def test_a_payload_that_is_not_a_dict_is_skipped(tmp_path):
    path = _audit(tmp_path, [
        {"stage": "orchestrator", "event": "request", "payload": "uma string"},
        {"stage": "orchestrator", "event": "request", "payload": {"input": "ok vamos"}},
    ])
    assert router_replay.harvest(path) == ["ok vamos"]
