"""/fitness/run endpoint — Apple Watch auto-sync sink (s76 cont.35).

TestClient over the real app (auth + IP allowlist), with the fitness log
pointed at a temp DB via the module singleton.
"""
import pytest
from fastapi.testclient import TestClient

import src.api as api_mod
from src.integrations import fitness as fitness_mod
from src.integrations.fitness import FitnessLog

_AUTH = {"X-Cortex-Key": "test-key-xyz"}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(api_mod, "SECRET_API_KEY", "test-key-xyz")
    monkeypatch.setattr(api_mod, "_ALLOWLIST", [])
    monkeypatch.setattr(fitness_mod, "_log_singleton",
                        FitnessLog(db_path=str(tmp_path / "f.db")))
    return TestClient(api_mod.app, raise_server_exceptions=False)


def test_run_logs_and_reports_pace(client):
    r = client.post("/fitness/run", headers=_AUTH,
                    json={"distance_km": 5.0, "duration_min": 25.0,
                          "start_date": "2026-05-01T07:30:00"})
    assert r.status_code == 200
    assert "5km" in r.text and "5:00/km" in r.text


def test_run_is_idempotent_by_start_date(client):
    body = {"distance_km": 5.0, "duration_min": 25.0,
            "start_date": "2026-05-01T07:30:00"}
    assert client.post("/fitness/run", headers=_AUTH, json=body).status_code == 200
    r2 = client.post("/fitness/run", headers=_AUTH, json=body)  # same workout again
    assert r2.status_code == 200
    assert "já estava registrada" in r2.text
    # and the store really has just one run
    assert len(fitness_mod.get_log().run_history()) == 1


def test_run_accepts_units_in_strings(client):
    # the Shortcut can dump the raw Workout properties (with units)
    r = client.post("/fitness/run", headers=_AUTH,
                    json={"distance_km": "5 km", "duration_min": "25 min",
                          "start_date": "2026-05-02T07:00:00"})
    assert r.status_code == 200
    assert "5km" in r.text and "5:00/km" in r.text


def test_run_rejects_nonpositive(client):
    r = client.post("/fitness/run", headers=_AUTH,
                    json={"distance_km": 0, "duration_min": 25})
    assert r.status_code == 400


def test_run_requires_auth(client):
    r = client.post("/fitness/run",
                    json={"distance_km": 5, "duration_min": 25})
    assert r.status_code in (401, 403)


# ---------- /fitness/health-export (Health Auto Export sink) ----------

_HAE_BODY = {"data": {"workouts": [
    {"name": "Running", "start": "2026-06-01 07:30:00 -0300",
     "distance": {"qty": 5.0, "units": "km"},
     "duration": {"qty": 25, "units": "min"}},
    {"name": "Cycling", "start": "2026-06-02 08:00:00 -0300",
     "distance": {"qty": 15, "units": "km"},
     "duration": {"qty": 40, "units": "min"}},
]}}


def test_health_export_imports_runs_and_dedups(client):
    r = client.post("/fitness/health-export", headers=_AUTH, json=_HAE_BODY)
    assert r.status_code == 200
    assert "1 novas" in r.text                     # only the run, not cycling
    assert len(fitness_mod.get_log().run_history()) == 1
    r2 = client.post("/fitness/health-export", headers=_AUTH, json=_HAE_BODY)
    assert "0 novas" in r2.text and "1 já existiam" in r2.text


def test_health_export_requires_auth(client):
    r = client.post("/fitness/health-export", json={"data": {"workouts": []}})
    assert r.status_code in (401, 403)
