"""Turn language and the claimed plan survive another turn on the same instance."""
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.core import planner
from src.core.orchestrator import CortexOrchestrator
from tests.test_dispatch_parity import _build_real_orch
from tests.test_pending_identity import EMAIL


@pytest.fixture
def orch(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    o = _build_real_orch(tmp_path)
    o.background, o._offer = False, None
    return o


ASK = "envia email para joao@exemplo.com; assunto: Atraso; corpo: Vou atrasar."
ASK_PT = ASK.replace("Vou atrasar.", "Chego logo.")


def turn(o, text, stream, lang):
    if stream:
        frames = list(o.process_stream(text, language=lang))
        return "".join(f[0] for f in frames), frames[-1][2], frames
    reply, mode = o.process(text, language=lang)
    return reply, mode, []


@pytest.mark.parametrize("stream,pieces", [(False, False), (True, False), (True, True)])
def test_concurrent_email_previews(orch, monkeypatch, stream, pieces):
    staged, languages, release = (threading.Barrier(2) for _ in range(3))
    preview = orch._human_confirmation_prompt
    observed = {}

    def paused_preview(action):
        lang = "pt" if action["args"]["body"] == "Chego logo" else "en"
        if lang == "en":
            staged.wait(timeout=10)  # EN is staged; PT can now begin its turn.
            languages.wait(timeout=10)
        observed[lang] = orch._reply_lang
        reply = preview(action)
        if pieces and orch._token_emitter() is not None:
            orch._token_emitter()(reply)
        return reply

    def pause_pt(text):
        if text == ASK_PT:
            languages.wait(timeout=10)  # PT has set its language in _process_core.
            release.wait(timeout=10)

    monkeypatch.setattr(orch, "_human_confirmation_prompt", paused_preview)
    monkeypatch.setattr(orch, "_shadow_classify", pause_pt)

    def english():
        # The stream request thread must not use its own language for worker frames.
        orch._reply_lang = "pt"
        return turn(orch, ASK, stream, "en")

    with ThreadPoolExecutor(max_workers=2) as pool:
        en = pool.submit(english)
        staged.wait(timeout=10)
        pt = pool.submit(turn, orch, ASK_PT, stream, "pt")
        try:
            en_result = en.result(timeout=10)
        finally:
            release.wait(timeout=10)
        pt_result = pt.result(timeout=10)

    assert en_result[1] == pt_result[1] == "AGENT_AWAITING_CONFIRMATION"
    assert "Confirm sending the email?" in en_result[0]
    assert "Confirmar envio do email?" in pt_result[0]
    assert observed == {"en": "en", "pt": "pt"}
    assert all(lang == "en" for _, _, _, lang in en_result[2])
    assert all(lang == "pt" for _, _, _, lang in pt_result[2])
    if pieces:
        assert any(not final for _, final, _, _ in en_result[2])
    assert orch._reply_lang == "pt"  # Last set, readable outside either turn.


@pytest.mark.parametrize("stream", [False, True])
def test_pooled_thread_uses_fallback_after_turn(orch, monkeypatch, stream):
    if stream:
        monkeypatch.setattr(orch, "_is_pure_chat", lambda _: True)
        orch.engine_analytical.chat_stream.return_value = iter(["Hello"])
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(turn, orch, "explain gravity", stream, "en").result(timeout=10)
        assert orch._reply_lang == "en"
        # A different thread updates the shared fallback while this worker is idle.
        orch._reply_lang = "pt"
        monkeypatch.setattr(orch, "_process_core", lambda *a, **kw: (
            orch._reply_lang, "PROBE"))
        assert pool.submit(orch.process, "probe").result(timeout=10) == ("pt", "PROBE")


def test_thread_language_isolated_with_shared_fallback(orch):
    barrier = threading.Barrier(2)

    def english():
        orch._reply_lang = "en"
        barrier.wait(timeout=10)
        barrier.wait(timeout=10)
        return orch._reply_lang

    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(english)
        barrier.wait(timeout=10)
        orch._reply_lang = "pt"
        barrier.wait(timeout=10)
        assert result.result(timeout=10) == "en"
    assert orch._reply_lang == "pt"


@pytest.mark.parametrize("stream", [False, True])
def test_language_cleanup_on_error(orch, monkeypatch, stream):
    def fail(*a, **kw):
        orch._reply_lang = "en"
        raise RuntimeError("turn failed")

    monkeypatch.setattr(orch, "_process_core", fail)
    with pytest.raises(RuntimeError, match="turn failed"):
        turn(orch, ASK, stream, "en")
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(setattr, orch, "_reply_lang", "pt").result(timeout=10)
    assert orch._reply_lang == "pt"


def test_language_cleanup_on_stream_close(orch, monkeypatch):
    monkeypatch.setattr(orch, "_is_pure_chat", lambda _: True)
    orch.engine_analytical.chat_stream.return_value = iter(["Hello", " there"])
    stream = orch.process_stream("explain gravity", language="en")
    assert next(stream) == ("Hello", False, None, "en")
    stream.close()
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(setattr, orch, "_reply_lang", "pt").result(timeout=10)
    assert orch._reply_lang == "pt"


@pytest.mark.parametrize("stream", [False, True])
def test_claimed_plan_survives_email_staging(orch, monkeypatch, stream):
    orch._pending_action = {"type": "execute_plan", "args": {
        "objetivo": "Read the approved document", "lang": "pt", "passos": [
            {"id": "p1", "tool": "read_document", "args": {"filename": "approved.txt"}}
        ]}}
    orch._persist_pending_action()
    claimed, staged = threading.Barrier(2), threading.Barrier(2)
    execute = orch._execute_staged_plan
    ran = []

    def pause_plan(action=None):
        claimed.wait(timeout=10)
        staged.wait(timeout=10)
        # Keep the pre-fix call compatible, so it fails on the wrong executed plan.
        return execute() if action is None else execute(action)

    monkeypatch.setattr(orch, "_execute_staged_plan", pause_plan)
    monkeypatch.setattr(orch, "_execute_tool_unstaged", lambda name, args, **kw: (
        ran.append((name, args)), "document contents")[1])
    monkeypatch.setattr(planner, "summarise", lambda *a, **kw: "Done")
    with ThreadPoolExecutor(max_workers=1) as pool:
        approval = pool.submit(turn, orch, "sim", stream, "pt")
        claimed.wait(timeout=10)
        try:
            assert turn(orch, ASK, False, "pt")[1] == "AGENT_AWAITING_CONFIRMATION"
            email = orch._read_pending_raw()
        finally:
            staged.wait(timeout=10)
        assert approval.result(timeout=10)[1] == "AGENT_ACTION_DONE"
    assert ran == [("read_document", {"filename": "approved.txt"})]
    assert email["type"] == EMAIL["type"]
    assert orch._read_pending_raw() == email == orch._pending_action


def test_new_instance_language(monkeypatch):
    o = CortexOrchestrator.__new__(CortexOrchestrator)
    assert getattr(o, "_reply_lang", "pt") == "pt"
    o._reply_lang = "en"
    assert o._reply_lang == "en"
    with monkeypatch.context() as patch:
        patch.setattr(o, "_reply_lang", "pt")
        assert o._reply_lang == "pt"
    assert o._reply_lang == "en"
