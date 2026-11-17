"""141b: durable channel-local usage and metrics across stream pool pulls."""
import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

import src.api as api
import src.main as cli
from src.core import turn_metrics
from src.core.engine import OllamaEngine
from src.core.usage_answer import previous_usage_answer
from src.ui import chat_store as cs
from tests.test_taught_facts import orch  # noqa: F401
from tests.test_usage_answer import EN, PT, USAGE, assert_no_model


QUESTION = "quantos tokens você gastou?"
EMPTY = "Ainda não há uma resposta anterior nesta conversa."
UNKNOWN = "Não tenho a contagem da última resposta."
ZERO = "A última resposta foi dada pelo código, sem usar o modelo (0 tokens)."
CHANNELS = [("/ask", "127.0.0.1", "terminal", cs.TERMINAL_LABEL),
            ("/ask", "100.64.0.42", "network", cs.NETWORK_LABEL),
            ("/ask_voice", "127.0.0.1", "voice", cs.VOICE_LABEL),
            ("/ask_stream", "127.0.0.1", "voice", cs.VOICE_LABEL),
            ("native", None, "terminal", cs.TERMINAL_LABEL)]


@pytest.fixture
def channel_orch(orch, monkeypatch):
    import ollama

    monkeypatch.setattr(api, "orchestrator", orch)
    monkeypatch.setattr(api, "SECRET_API_KEY", "test-fake")
    monkeypatch.setattr(api, "_ALLOWLIST", [])
    monkeypatch.setattr(cli, "CortexOrchestrator", lambda: orch)
    for name in ("chat", "generate"):
        monkeypatch.setattr(ollama, name, MagicMock(side_effect=AssertionError("no Ollama")))

    def counted():
        # Exercise the real engine hook with mocked model responses.
        engine = SimpleNamespace(model_name="cortex-lite")
        for incoming, outgoing in [(2000, 100), (913, 48)]:
            OllamaEngine._log_token_usage(engine, "chat", {
                "prompt_eval_count": incoming, "eval_count": outgoing}, {})

    def chat(*args, **kwargs):
        counted()
        return "Model answer."

    def stream(*args, **kwargs):
        yield "Model "
        counted()  # After the first chunk, on a later HTTP iterator pull.
        yield "answer."

    for engine in (orch.engine_analytical, orch.engine_creative):
        engine.max_history = 10
        engine.chat.side_effect = chat
        engine.chat_with_tools.side_effect = lambda *a, **kw: (chat(), None)
        engine.chat_stream.side_effect = stream
    api.limiter.reset()
    yield orch
    orch.set_turn_origin(None)
    turn_metrics.collect()
    api.limiter.reset()


def send(monkeypatch, capsys, path, host, text):
    if path == "native":
        typed = iter([text, "sair"])
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(typed))
        with pytest.raises(SystemExit):
            cli.run()
        return capsys.readouterr().out
    with TestClient(api.app, client=(host, 50000)) as client:
        response = client.post(path, headers={"X-Cortex-Key": "test-fake"},
                               json={"user_input": text})
    assert response.status_code == 200
    if path == "/ask":
        return response.json()["response"]
    if path == "/ask_stream":
        frames = [json.loads(line[6:]) for line in response.text.splitlines()
                  if line.startswith("data: ")]
        return "".join(frame.get("chunk", "") for frame in frames)
    return response.text


def latest(label):
    cid = cs.store.find_titled(cs.daily_title(label))
    return cs.store.get_turns(cid)[-1]


@pytest.mark.parametrize("path,host,channel,label", CHANNELS)
@pytest.mark.parametrize("lang", ["pt", "en"])
def test_channel_round_trip(channel_orch, monkeypatch, capsys, path, host, channel, label, lang):
    # Other channels and UI have different counts and must remain untouched.
    for other, other_label in [("terminal", cs.TERMINAL_LABEL),
                               ("network", cs.NETWORK_LABEL), ("voice", cs.VOICE_LABEL)]:
        if other != channel:
            cs.append_daily(other, other_label, "other", "other")
    ui = cs.store.create_conversation("UI")["id"]
    cs.store.append_turn(ui, "ui", "ui", usage={**USAGE, "eval_count": 999})
    untouched = {c["id"]: cs.store.get_turns(c["id"]) for c in cs.store.list_conversations()}
    assert "Model answer." in send(monkeypatch, capsys, path, host, "explain gravity")
    assert latest(label)["usage"] == USAGE
    assert latest(label)["source"] == channel
    for engine in (channel_orch.engine_analytical, channel_orch.engine_creative):
        engine.reset_mock()
    channel_orch._needs_realtime.reset_mock()
    question = QUESTION if lang == "pt" else "how many tokens did you use?"
    expected = PT if lang == "pt" else EN
    spoken = expected.replace("\n", " ") if path == "/ask_voice" else expected
    assert spoken in send(monkeypatch, capsys, path, host, question)
    assert latest(label)["reply"] == expected
    assert latest(label)["status_mode"] == "USAGE_PREVIOUS"
    assert latest(label)["usage"] is None
    assert_no_model(channel_orch)
    for cid, turns in untouched.items():
        assert cs.store.get_turns(cid) == turns
    assert (ZERO if lang == "pt" else
            "The last answer was given by code, without using the model (0 tokens).") in send(
                monkeypatch, capsys, path, host, question)
    assert_no_model(channel_orch)


@pytest.mark.parametrize("path,host,channel,label", CHANNELS)
@pytest.mark.parametrize("previous", ["missing", "empty", "yesterday", "code", "legacy", "error"])
def test_previous_boundaries(channel_orch, monkeypatch, capsys, path, host, channel, label, previous):
    if previous == "yesterday":
        cid = cs.store.create_conversation(cs.daily_title(label, datetime.now() - timedelta(days=1)))["id"]
        cs.store.append_turn(cid, "old", "old", usage=USAGE)
    elif previous != "missing":
        cid = cs.store.conversation_titled(cs.daily_title(label))
        if previous in {"code", "legacy", "error"}:
            cs.store.append_turn(cid, "old", "old", status_mode="SELF_MODEL" if previous == "code" else "OFFLINE")
    if previous == "error":
        monkeypatch.setattr(cs.store, "_connect", lambda: None)
    expected = ZERO if previous == "code" else UNKNOWN if previous in {"legacy", "error"} else EMPTY
    assert expected in send(monkeypatch, capsys, path, host, QUESTION)
    assert_no_model(channel_orch)


@pytest.mark.parametrize("channel,label", [(c, l) for _, _, c, l in CHANNELS[:3]])
def test_lookup_is_read_only(channel_orch, channel, label):
    before = cs.store.list_conversations()
    assert previous_usage_answer({"channel": channel, "conversation": None}, "pt") == EMPTY
    assert cs.store.list_conversations() == before


@pytest.mark.parametrize("read", ["find_titled", "get_turns"])
@pytest.mark.parametrize("channel,label", [("terminal", cs.TERMINAL_LABEL),
                                          ("network", cs.NETWORK_LABEL), ("voice", cs.VOICE_LABEL)])
def test_lookup_storage_errors(channel_orch, monkeypatch, read, channel, label):
    cs.append_daily(channel, label, "old", "old")
    monkeypatch.setattr(cs.store, read, MagicMock(side_effect=OSError("unavailable")))
    assert previous_usage_answer({"channel": channel, "conversation": None}, "pt") == UNKNOWN


def test_two_measured_channels_keep_their_counts(channel_orch, monkeypatch, capsys):
    send(monkeypatch, capsys, "/ask", "127.0.0.1", "explain gravity")
    def different(*args, **kwargs):
        turn_metrics.add(1, 2, "other-model")
        return "different", None
    channel_orch.engine_analytical.chat.side_effect = lambda *a, **kw: different()[0]
    channel_orch.engine_analytical.chat_with_tools.side_effect = different
    send(monkeypatch, capsys, "/ask_voice", "127.0.0.1", "explain gravity")
    assert PT == send(monkeypatch, capsys, "/ask", "127.0.0.1", QUESTION)
    assert "1 token de entrada e 2 de saída" in send(monkeypatch, capsys, "/ask_voice", "127.0.0.1", QUESTION)


def test_stream_later_pulls_on_different_threads(channel_orch, monkeypatch):
    # Capture the sync iterator that Starlette normally passes to
    # iterate_in_threadpool; force alternating workers instead of relying on
    # AnyIO's pool to happen to migrate this request.
    monkeypatch.setattr(api, "StreamingResponse", lambda content, **kw: content)
    counted_pulls = []
    pull_index = None

    def stream(*args, **kwargs):
        # Count directly on two separate HTTP pulls. A deferred worker can
        # inherit the first pull's bucket and mask missing later attaches.
        yield "", False, None, "en"  # First HTTP pull: language prelude only.
        engine = SimpleNamespace(model_name="cortex-lite")
        for incoming, outgoing, chunk, final in [
            (2000, 100, "Model ", False), (913, 48, "answer.", True),
        ]:
            OllamaEngine._log_token_usage(engine, "chat", {
                "prompt_eval_count": incoming, "eval_count": outgoing}, {})
            counted_pulls.append((pull_index, threading.get_ident()))
            yield chunk, final, "OFFLINE", "en"

    monkeypatch.setattr(channel_orch, "process_stream", stream)
    request = Request({"type": "http", "method": "POST", "path": "/ask_stream",
                       "headers": [], "client": ("127.0.0.1", 50000), "app": api.app})
    events = asyncio.run(api.process_query_stream.__wrapped__(
        request, api.CortexRequest(user_input="explain gravity")))
    done = object()
    ids = set()
    pull_threads = []
    pool_calls = []
    def pull():
        turn_metrics.reset()
        prior = turn_metrics.current()
        try:
            frame = next(events, done)
            assert turn_metrics.current() is prior
            pool_calls.append(prior["calls"])
            return threading.get_ident(), frame
        finally:
            turn_metrics.collect()
    with ThreadPoolExecutor(max_workers=1) as first, ThreadPoolExecutor(max_workers=1) as second:
        for i in range(20):
            pull_index = i
            tid, frame = (first if i % 2 == 0 else second).submit(pull).result(timeout=3)
            ids.add(tid)
            pull_threads.append(tid)
            if frame is done:
                break
        else:
            pytest.fail("stream did not finish")
    assert len(ids) == 2
    assert [index for index, _ in counted_pulls] == [1, 2]
    assert counted_pulls[0][1] != pull_threads[0]
    assert counted_pulls[0][1] != counted_pulls[1][1]
    assert latest(cs.VOICE_LABEL)["usage"] == USAGE
    assert all(calls == 0 for calls in pool_calls)  # No pool contamination.


def test_stream_deferred_worker_inherits_usage(channel_orch, monkeypatch, capsys):
    threads = []
    original = channel_orch.engine_analytical.chat_with_tools.side_effect

    def tracked(*args, **kwargs):
        threads.append(threading.current_thread().name)
        return original(*args, **kwargs)

    channel_orch.engine_analytical.chat_with_tools.side_effect = tracked
    send(monkeypatch, capsys, "/ask_stream", "127.0.0.1", "responder sobre gravidade")
    assert threads == ["cortex-stream-turn"]
    assert latest(cs.VOICE_LABEL)["usage"] == USAGE


@pytest.mark.parametrize("path,host,channel,label", CHANNELS)
def test_origin_shape_and_bucket_cleanup(channel_orch, monkeypatch, capsys, path, host, channel, label):
    seen = []

    def fail(*args, **kwargs):
        seen.append(channel_orch._turn_origin())
        turn_metrics.add(11, 7, "failed-model")
        raise RuntimeError("turn failed")

    if path == "/ask_stream":
        def fail_stream(*args, **kwargs):
            fail()
            yield
        monkeypatch.setattr(channel_orch, "process_stream", fail_stream)
    else:
        monkeypatch.setattr(channel_orch, "process", fail)
    if path == "native":
        send(monkeypatch, capsys, path, host, "current")
        assert turn_metrics.current() is None
    else:
        with TestClient(api.app, client=(host, 50000)) as client:
            async def one_worker():
                import anyio
                anyio.to_thread.current_default_thread_limiter().total_tokens = 1
            async def probe():
                import anyio
                return await anyio.to_thread.run_sync(
                    lambda: (turn_metrics.current(), channel_orch._turn_origin()))
            client.portal.call(one_worker)
            response = client.post(path, headers={"X-Cortex-Key": "test-fake"},
                                   json={"user_input": "current"})
            assert response.status_code == (200 if path == "/ask_stream" else 500)
            assert client.portal.call(probe) == (None, None)
    assert seen == [{"channel": channel, "conversation": None}]


@pytest.mark.parametrize("origin", [None, {"channel": "unknown"}, {"channel": "ui", "conversation": None}])
def test_unknown_channel_fallback(origin):
    assert previous_usage_answer(origin, "pt") == "Não consigo identificar a conversa deste turno."
    assert previous_usage_answer(origin, "en") == "I can't identify this turn's conversation."
