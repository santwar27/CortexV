"""
Tests for `src/doctor.py` — the local-stack health check.

The doctor's job is to talk to live subsystems (Ollama, the API server),
so the parts worth unit-testing are the OFFLINE ones: the model-name
matching, the env-var policy, Palace integrity against a throwaway DB, the
voice-model file probe, and the exit-code aggregation. The network checks
are exercised by monkeypatching, so this whole file stays offline — it can
run inside `make check` / the pre-commit hook without Ollama up.
"""
import sqlite3

import pytest

from src import doctor
from src.doctor import OK, WARN, FAIL, Result


# --- _has_model: tag-agnostic matching ------------------------------------

@pytest.mark.parametrize("configured,pulled,expected", [
    ("cortex-lite", ["cortex-lite:latest"], True),
    ("cortex-lite:latest", ["cortex-lite:latest"], True),
    ("llava", ["llava:latest", "llama3:latest"], True),
    ("nomic-embed-text", ["nomic-embed-text:latest"], True),
    ("cortex-lite", ["cortex-criativo:latest"], False),
    ("missing", [], False),
])
def test_has_model(configured, pulled, expected):
    assert doctor._has_model(pulled, configured) is expected


# --- check_env: names only, required vs recommended -----------------------

def test_env_fail_without_api_key(monkeypatch):
    monkeypatch.delenv("CORTEX_API_KEY", raising=False)
    assert doctor.check_env().status == FAIL


def test_env_warn_missing_optional(monkeypatch):
    monkeypatch.setenv("CORTEX_API_KEY", "x")
    for v in ("TAVILY_API_KEY", "SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET"):
        monkeypatch.delenv(v, raising=False)
    assert doctor.check_env().status == WARN


def test_env_ok_all_set(monkeypatch):
    for v in ("CORTEX_API_KEY", "TAVILY_API_KEY",
              "SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET"):
        monkeypatch.setenv(v, "x")
    assert doctor.check_env().status == OK


def test_env_never_leaks_value(monkeypatch):
    monkeypatch.setenv("CORTEX_API_KEY", "super-secret-value")
    for v in ("TAVILY_API_KEY", "SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET"):
        monkeypatch.delenv(v, raising=False)
    assert "super-secret-value" not in doctor.check_env().detail


# --- check_palace: integrity against a real temp DB -----------------------

def _make_palace(path, *, facts=0, docs=0, events=0):
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE triplas (sujeito TEXT, predicado TEXT, valid_until TEXT);"
        "CREATE TABLE documentos (texto TEXT);"
        "CREATE TABLE eventos (payload TEXT);"
    )
    conn.executemany("INSERT INTO triplas VALUES ('s','p',NULL)", [()] * facts)
    conn.executemany("INSERT INTO documentos VALUES ('d')", [()] * docs)
    conn.executemany("INSERT INTO eventos VALUES ('e')", [()] * events)
    conn.commit()
    conn.close()


def test_palace_ok(monkeypatch, tmp_path):
    (tmp_path / "data").mkdir()
    _make_palace(str(tmp_path / "data" / "palace.db"), facts=3, docs=5, events=2)
    monkeypatch.setattr(doctor, "REPO_ROOT", tmp_path)
    r = doctor.check_palace()
    assert r.status == OK
    assert "3 facts" in r.detail and "5 docs" in r.detail and "2 events" in r.detail


def test_palace_fail_missing(monkeypatch, tmp_path):
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(doctor, "REPO_ROOT", tmp_path)
    assert doctor.check_palace().status == FAIL


def test_palace_counts_only_open_facts(monkeypatch, tmp_path):
    # Bitemporal: a closed (superseded) tripla has valid_until set and must
    # not be counted as a live fact.
    (tmp_path / "data").mkdir()
    db = str(tmp_path / "data" / "palace.db")
    _make_palace(db, facts=2, docs=0, events=0)
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO triplas VALUES ('s','p','2026-01-01')")  # closed
    conn.commit()
    conn.close()
    monkeypatch.setattr(doctor, "REPO_ROOT", tmp_path)
    assert "2 facts" in doctor.check_palace().detail


# --- check_gate_queue: background-safe action queue -----------------------

def test_gate_queue_ok_when_empty(monkeypatch, tmp_path):
    from src.core import gate_queue as gq_mod
    fake = gq_mod.GateQueue(path=str(tmp_path / "gq.jsonl"))
    monkeypatch.setattr(gq_mod, "gate_queue", fake)
    assert doctor.check_gate_queue().status == OK


def test_gate_queue_warns_on_pending(monkeypatch, tmp_path):
    from src.core import gate_queue as gq_mod
    fake = gq_mod.GateQueue(path=str(tmp_path / "gq.jsonl"))
    fake.enqueue("send_email", {"to": "x@y.z"}, reason="test")
    monkeypatch.setattr(gq_mod, "gate_queue", fake)
    r = doctor.check_gate_queue()
    assert r.status == WARN
    assert "1 action" in r.detail


def test_gate_queue_ok_when_file_absent(monkeypatch, tmp_path):
    # Never-queued is healthy: no file on disk, no pending, OK (not FAIL).
    from src.core import gate_queue as gq_mod
    fake = gq_mod.GateQueue(path=str(tmp_path / "never_created.jsonl"))
    monkeypatch.setattr(gq_mod, "gate_queue", fake)
    assert doctor.check_gate_queue().status == OK


def test_gate_queue_fail_on_read_error(monkeypatch):
    from src.core import gate_queue as gq_mod

    class _Boom:
        def pending(self):
            raise RuntimeError("corrupt log")
    monkeypatch.setattr(gq_mod, "gate_queue", _Boom())
    assert doctor.check_gate_queue().status == FAIL


# --- check_voice_model: file probe ----------------------------------------

def test_voice_model_ok(monkeypatch, tmp_path):
    models = tmp_path / "src" / "voice" / "models"
    models.mkdir(parents=True)
    (models / "hey_cortex.onnx").write_bytes(b"x" * 2048)
    monkeypatch.setattr(doctor, "REPO_ROOT", tmp_path)
    assert doctor.check_voice_model().status == OK


def test_voice_model_warn_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor, "REPO_ROOT", tmp_path)
    assert doctor.check_voice_model().status == WARN


# --- check_api: 401 means the server is up and enforcing auth -------------

def test_api_401_is_ok(monkeypatch):
    class _Resp:
        status_code = 401
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp())
    assert doctor.check_api().status == OK


def test_api_down_is_warn(monkeypatch):
    import requests
    def _boom(*a, **k):
        raise requests.ConnectionError("refused")
    monkeypatch.setattr(requests, "get", _boom)
    assert doctor.check_api().status == WARN


def test_api_500_is_warn(monkeypatch):
    class _Resp:
        status_code = 502
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp())
    assert doctor.check_api().status == WARN


# --- exit-code aggregation ------------------------------------------------

def test_main_returns_0_on_all_ok(monkeypatch):
    monkeypatch.setattr(doctor, "CHECKS",
                        [lambda: Result("a", OK, ""), lambda: Result("b", OK, "")])
    assert doctor.main() == 0


def test_main_returns_0_on_warn(monkeypatch):
    monkeypatch.setattr(doctor, "CHECKS",
                        [lambda: Result("a", OK, ""), lambda: Result("b", WARN, "")])
    assert doctor.main() == 0


def test_main_returns_1_on_fail(monkeypatch):
    monkeypatch.setattr(doctor, "CHECKS",
                        [lambda: Result("a", OK, ""), lambda: Result("b", FAIL, "")])
    assert doctor.main() == 1


def test_run_checks_never_aborts_on_crash(monkeypatch):
    def _crash():
        raise RuntimeError("boom")
    monkeypatch.setattr(doctor, "CHECKS",
                        [_crash, lambda: Result("ok", OK, "")])
    results = doctor.run_checks()
    assert len(results) == 2
    assert results[0].status == FAIL
    assert results[1].status == OK
