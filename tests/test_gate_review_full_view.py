"""The approval view shows complete, terminal-safe values without changing execution."""
import pytest

from src import gate_review
from src.core.gate_queue import GateQueue


@pytest.fixture
def queue(tmp_path):
    return GateQueue(str(tmp_path / "gate_queue.jsonl"))


def test_full_body_and_one_argument_per_line(queue):
    body = "Resumo da tarefa. " * 28 + "FIM-DO-CORPO-XYZ"
    queue.enqueue("send_email", {"to": "a@b", "body": body})
    out = gate_review.render_list(queue)
    assert f"       args:\n         to=a@b\n         body={body}" in out
    assert "…" not in out


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_multiline_body_is_indented(queue, newline):
    queue.enqueue("send_email", {"body": newline.join(["Primeira", "Segunda", "Terceira"])})
    out = gate_review.render_list(queue)
    assert "         body=Primeira\n              Segunda\n              Terceira" in out
    assert not {"Primeira", "Segunda", "Terceira"}.intersection(out.split("\n"))
    assert "\r" not in out


def test_controls_are_visible_and_approval_preserves_stored_args(queue):
    body = "antes\x1b[2Kmeio\rfim\u202eDEPOIS\n\túltima"
    args = {"to": "a@b", "body": body, "extra": [1, {"texto": body}]}
    gid = queue.enqueue("send_email", args)
    stored = queue.get(gid)["args"]
    out = gate_review.render_list(queue)
    assert "antes\\x1b[2Kmeio\\x0dfim\\u202eDEPOIS" in out
    assert not any(char in out for char in ("\x1b", "\r", "\u202e"))
    assert "\n              \túltima" in out
    assert queue.get(gid)["args"] == stored == args
    calls = []

    def runner(action_type, action_args):
        calls.append((action_type, action_args))
        return "ok", "AGENT_ACTION_DONE"

    gate_review.approve(gid, queue=queue, runner=runner)
    assert calls == [("send_email", stored)]


def test_all_required_control_ranges_are_escaped(queue):
    codes = [*range(32), *range(127, 160), *range(0x202a, 0x202f), *range(0x2066, 0x206a)]
    codes = [code for code in codes if code not in (9, 10)]
    queue.enqueue("send_email", {"body": "|".join(chr(code) for code in codes)})
    out = gate_review.render_list(queue)
    for code in codes:
        assert chr(code) not in out
        escaped = f"\\x{code:02x}" if code <= 255 else f"\\u{code:04x}"
        assert escaped in out


@pytest.mark.parametrize("field", ["type", "origin", "reason"])
def test_metadata_is_complete_safe_and_indented(queue, field):
    value = "Contexto " * 60 + "\x1b[2K\r\u202eFIM-DO-CONTEXTO\r\ncontinuação"
    entry = {"type": "send_email", "origin": "studier", "reason": "revisar"}
    entry[field] = value
    queue.enqueue(entry.pop("type"), {}, **entry)
    out = gate_review.render_list(queue)
    assert "Contexto " * 60 + "\\x1b[2K\\x0d\\u202eFIM-DO-CONTEXTO" in out
    assert not any(char in out for char in ("\x1b", "\r", "\u202e"))
    assert "continuação" in out
    assert "continuação" not in out.split("\n")


@pytest.mark.parametrize("value", [1234567890, ["item" * 100, "\u202e"], {"texto": "valor " * 100}])
def test_non_string_values_are_complete_and_safe(queue, value):
    queue.enqueue("send_email", {"value": value})
    out = gate_review.render_list(queue)
    assert "         value=" + str(value).replace("\u202e", "\\u202e") in out
    assert "\u202e" not in out
    assert "…" not in out
