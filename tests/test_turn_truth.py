"""Per-reply model/network truth; no real network or LLM is needed."""
import ast
from pathlib import Path
import threading
from unittest.mock import Mock

import pytest

from src.core import turn_metrics
from tests.test_ui_routes import _MockOrchestrator, _parse_sse, ui_app, client


MARKS_ONLINE = {
    "tools/web_search.py",  # Tavily and DuckDuckGo search attempts.
    "tools/weather.py",  # Open-Meteo geocoding and forecast.
    "integrations/gmail_client.py",  # Shared service entry, including refresh.
    "integrations/calendar_client.py",  # Shared service entry, including refresh.
    "integrations/ats.py",  # Public job feeds, including FAA.
}
NOT_ONLINE = {
    "memory/embeddings.py": "Localhost Ollama embeddings.",
    "tools/vision_processor.py": "Localhost Ollama vision.",
    "voice/listener.py": "Local API client; not the UI turn's outbound path.",
    "doctor.py": "Diagnostics are not on a turn's path.",
    "ui/aggregator.py": "Dashboard net probe is not on a turn's path.",
    "integrations/google_oauth.py": "Refresh is marked by Gmail/Calendar before load_credentials; interactive setup is outside turns.",
}
NETWORK = ("requests", "httpx", "urllib.request", "googleapiclient",
           "duckduckgo_search", "ddgs", "google.auth.transport.requests",
           "socket.create_connection", "urlopen")


def scan_network(root):
    """Resolve import aliases and calls, ignoring comments and docstrings."""
    found = {}
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        aliases, names = {}, set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    name = (f"{node.module}.{alias.name}"
                            if isinstance(node, ast.ImportFrom) else alias.name)
                    aliases[alias.asname or (alias.name if isinstance(node, ast.ImportFrom)
                                             else alias.name.split(".")[0])] = (
                        name if alias.asname or isinstance(node, ast.ImportFrom)
                        else name.split(".")[0])
                    names.add(name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                head, dot, tail = name.partition(".")
                names.add(aliases.get(head, head) + (dot + tail if dot else ""))
        if any(name == net or name.startswith(net + ".")
               for name in names for net in NETWORK):
            found[path.relative_to(root).as_posix()] = tree
    return found


def assert_network_guard(root):
    assert not MARKS_ONLINE.intersection(NOT_ONLINE)
    for name, tree in scan_network(root).items():
        assert name in MARKS_ONLINE or name in NOT_ONLINE, f"Unclassified network module: {name}"
        if name in MARKS_ONLINE:
            assert any(isinstance(n, ast.Call)
                       and ast.unparse(n.func) == "turn_metrics.mark_online"
                       for n in ast.walk(tree)), f"Missing online mark: {name}"
        else:
            assert NOT_ONLINE[name]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Unexpected real network/LLM call")
    monkeypatch.setattr("requests.sessions.Session.request", blocked)
    monkeypatch.setattr("httpx.HTTPTransport.handle_request", blocked)
    monkeypatch.setattr("urllib.request.urlopen", blocked)
    monkeypatch.setattr("socket.create_connection", blocked)
    monkeypatch.setattr("googleapiclient.http.HttpRequest.execute", blocked)
    monkeypatch.setattr("src.tools.web_search.DDGS", blocked)
    monkeypatch.setattr("src.core.engine.ollama.chat", blocked)
    turn_metrics.attach(None)
    yield
    turn_metrics.attach(None)


@pytest.fixture
def engines(monkeypatch):
    from src.core.engine import CortexMemory, OllamaEngine
    memory = Mock(retrieve_relevant_context=Mock(return_value=""),
                  get_user_profile=Mock(return_value=""))
    monkeypatch.setattr("src.core.engine.CortexMemory",
                        Mock(return_value=memory, _is_refusal=CortexMemory._is_refusal))
    monkeypatch.setattr("src.core.engine.ollama.chat", lambda **kw: {
        "message": {"content": "Fictitious reply."},
        "prompt_eval_count": 12, "eval_count": 3,
    })
    return [OllamaEngine(model_name=name) for name in ("fake-analytical", "fake-creative")]


def test_models_are_distinct_sorted_and_local(engines):
    turn_metrics.reset()
    engines[1].chat("hello", response_language="en")
    assert turn_metrics.current()["models"] == ["fake-creative"]
    for engine in (engines[0], engines[1]):
        engine.chat("hello", response_language="en")
    assert turn_metrics.collect() == {
        "prompt_eval_count": 36, "eval_count": 9, "calls": 3,
        "models": ["fake-analytical", "fake-creative"], "online": [],
    }


@pytest.mark.parametrize("path,source", [
    ("tavily", "web_search"), ("ddg", "web_search"), ("weather", "weather"),
    ("gmail_read", "gmail"), ("gmail_send", "gmail"), ("calendar", "calendar"),
    ("jobs", "jobs"), ("faa", "jobs"),
    ("gmail_refresh", "gmail"), ("calendar_refresh", "calendar"),
])
def test_mark_precedes_failed_request(path, source, monkeypatch):
    import requests
    from src.tools.web_search import WebSearcher
    from src.tools import weather
    from src.integrations import ats, gmail_client as gmail, calendar_client as calendar

    attempted = []

    def fail(*args, **kwargs):
        attempted.append(True)
        assert turn_metrics.current()["online"] == [source]
        raise requests.ConnectionError("fictitious failure after possible transmission")

    monkeypatch.setattr("requests.post", fail)
    monkeypatch.setattr("requests.get", fail)
    monkeypatch.setattr("src.tools.web_search.DDGS", fail)
    for module, resource, refresh in ((gmail, "messages", "gmail_refresh"),
                                      (calendar, "events", "calendar_refresh")):
        service = Mock()
        target = service.users().messages() if resource == "messages" else service.events()
        for operation in ("list", "get", "send", "insert", "delete"):
            getattr(target, operation).return_value.execute.side_effect = fail
        monkeypatch.setattr(module, "build", Mock(return_value=service))
        monkeypatch.setattr(module, "load_credentials",
                            fail if path == refresh else lambda: object())

    searcher = WebSearcher(cost_tracker=Mock())
    searcher.tavily_key = "fake-key"
    actions = {
        "tavily": lambda: searcher._search_tavily("fictitious query", 1),
        "ddg": lambda: searcher._search_ddg("fictitious query", 1),
        "weather": lambda: weather.fetch_weather("Fictitious City"),
        "gmail_read": gmail.list_recent,
        "gmail_send": lambda: gmail.send_email("fake@example.invalid", "fake", "fake"),
        "calendar": lambda: calendar.create_event("fake", "2026-01-01T10:00:00", "2026-01-01T11:00:00"),
        "jobs": lambda: ats.fetch_greenhouse("fictitious-board"),
        "faa": lambda: ats.fetch_findapprenticeship(api_key="fake-key"),
        "gmail_refresh": gmail.list_recent,
        "calendar_refresh": calendar.list_events,
    }
    turn_metrics.reset()
    try:
        actions[path]()
    except (requests.ConnectionError, weather.WeatherUnavailable):
        pass
    assert attempted
    assert turn_metrics.collect() == {
        "prompt_eval_count": 0, "eval_count": 0, "calls": 0,
        "models": [], "online": [source],
    }


def test_reflex_empty_and_worker_buckets():
    turn_metrics.reset()
    assert turn_metrics.collect() is None
    turn_metrics.reset()
    bucket = turn_metrics.current()
    errors = []

    def worker():
        try:
            turn_metrics.mark_online("ignored")  # no bucket in this thread
            assert turn_metrics.collect() is None
            turn_metrics.attach(bucket)
            for source in ("weather", "web_search", "weather"):
                turn_metrics.mark_online(source)
            turn_metrics.attach(None)
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert not errors
    assert turn_metrics.collect() == {
        "prompt_eval_count": 0, "eval_count": 0, "calls": 0,
        "models": [], "online": ["weather", "web_search"],
    }


@pytest.mark.parametrize("online", [False, True])
def test_route_persists_truth(client, ui_app, engines, online):
    from src.ui.routes import chat_store

    class Orchestrator(_MockOrchestrator):
        def process(self, text, image_b64=None, language=None):
            if online:
                turn_metrics.mark_online("web_search")
            return engines[0].chat(text, response_language="en"), "OK"

    ui_app.state.orchestrator = Orchestrator()
    conversation = chat_store.create_conversation(title="Fictitious conversation")
    response = client.post("/ui/chat/stream", headers={"X-Cortex-Key": "test-key-xyz"},
                           json={"text": "hello", "conversation_id": conversation["id"]})
    assert response.status_code == 200
    final = _parse_sse(response.text)[-1]
    assert final["type"] == "final"
    usage = {"prompt_eval_count": 12, "eval_count": 3, "calls": 1,
             "models": ["fake-analytical"], "online": ["web_search"] if online else []}
    assert final["turn"]["usage"] == usage
    assert chat_store.get_turns(conversation["id"])[-1]["usage"] == usage


def test_network_guard():
    assert_network_guard(Path(__file__).resolve().parents[1] / "src")


@pytest.mark.parametrize("code", ["import requests", "import httpx as h",
    "from urllib.request import urlopen as open_url", "import googleapiclient.discovery",
    "from duckduckgo_search import DDGS", "from ddgs import DDGS",
    "import socket as s\ns.create_connection(('example.invalid', 443))"])
def test_network_guard_rejects_new_module(tmp_path, code):
    (tmp_path / "unmarked.py").write_text(code)
    with pytest.raises(AssertionError, match="Unclassified network module: unmarked.py"):
        assert_network_guard(tmp_path)
