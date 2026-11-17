"""A failed fitness write must never sound like success, a duplicate or empty."""
from contextlib import contextmanager
import sqlite3
import sys

import pytest
from fastapi.testclient import TestClient

import src.api as api_mod
from scripts import import_apple_health as cli
from src.integrations import fitness as fitness_mod
from src.integrations.apple_health import import_export
from src.integrations.fitness import FitnessLog


RUN_API_ERROR = "Não consegui gravar a corrida agora — nada foi gravado. Tenta de novo daqui a pouco."
RUN_ERRORS = (
    "Não consegui registrar essa corrida agora — nada foi gravado. Tenta de novo daqui a pouco.",
    "I couldn't log that run just now — nothing was saved. Try again in a moment.",
)
LIFT_ERRORS = (
    "Não consegui registrar esse treino agora — nada foi gravado. Tenta de novo daqui a pouco.",
    "I couldn't log that workout just now — nothing was saved. Try again in a moment.",
)
RUN_BODY = {"distance_km": 5, "duration_min": 25,
            "start_date": "2026-05-01T07:30:00Z"}
BATCH = {"data": {"workouts": [
    {"name": "Running", "start": f"2026-05-0{day} 07:30:00 +0000",
     "distance": {"qty": 5, "units": "km"},
     "duration": {"qty": 25, "units": "min"}}
    for day in (1, 2)
]}}
XML = """<HealthData>
<Workout workoutActivityType="HKWorkoutActivityTypeRunning" duration="25"
 durationUnit="min" totalDistance="5" totalDistanceUnit="km"
 startDate="2026-05-01 07:30:00 +0000"/>
</HealthData>"""


@pytest.fixture
def store(tmp_path, monkeypatch):
    log = FitnessLog(db_path=tmp_path / "fitness.db")
    conn = log._connect()
    conn.execute("PRAGMA busy_timeout = 10")
    monkeypatch.setattr(fitness_mod, "_log_singleton", log)
    yield log
    conn.close()


@pytest.fixture
def client(store, tmp_path, monkeypatch):
    monkeypatch.setattr(api_mod, "SECRET_API_KEY", "test-fake")
    monkeypatch.setattr(api_mod, "_ALLOWLIST", [])
    monkeypatch.setattr(api_mod, "_LOG_DIR", tmp_path)
    with TestClient(api_mod.app, raise_server_exceptions=False) as client:
        client.headers["X-Cortex-Key"] = "test-fake"
        yield client


@contextmanager
def locked(store):
    """A separate SQLite connection stands in for another process."""
    conn = sqlite3.connect(store.db_path, timeout=0.01)
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield
    finally:
        conn.rollback()
        conn.close()


def deny_commits(conn, *, after=0):
    """Make SQLite itself reject COMMIT while still permitting ROLLBACK."""
    commits = []

    def authorize(action, arg, *_):
        if action == sqlite3.SQLITE_TRANSACTION and arg == "COMMIT":
            commits.append(arg)
            if len(commits) > after:
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    conn.set_authorizer(authorize)
    return commits


@pytest.fixture
def orch(store, tmp_path, monkeypatch):
    from tests.test_dispatch_parity import _build_real_orch
    o = _build_real_orch(tmp_path)
    o.background = False
    monkeypatch.setattr("src.integrations.contacts.contact_book.list_all", lambda: [])
    for engine in (o.engine_analytical, o.engine_creative):
        for method in ("chat", "chat_with_tools", "chat_stream"):
            getattr(engine, method).side_effect = AssertionError("fitness called an engine")
    return o


@pytest.fixture
def events(monkeypatch):
    events = []
    monkeypatch.setattr("src.core.fitness_mixin.audit.log",
                        lambda *args, **kw: events.append((args[1], args[2], kw["payload"])))
    return events


def test_run_api_failure_retry_then_duplicate(store, client):
    with locked(store):
        response = client.post("/fitness/run", json=RUN_BODY)
    assert response.status_code == 503
    assert response.text == RUN_API_ERROR
    assert response.headers["content-type"].startswith("text/plain")
    assert store.run_history() == []
    assert not store._connect().in_transaction
    response = client.post("/fitness/run", json=RUN_BODY)
    assert response.status_code == 200
    assert response.text == "Corrida registrada: 5km em 25min (5:00/km)."
    response = client.post("/fitness/run", json=RUN_BODY)
    assert response.status_code == 200
    assert response.text == "Corrida já estava registrada (sem mudança)."
    assert len(store.run_history()) == 1


def test_run_api_connection_unavailable(store, client, monkeypatch):
    monkeypatch.setattr(store, "_connect", lambda: None)
    response = client.post("/fitness/run", json=RUN_BODY)
    assert (response.status_code, response.text) == (503, RUN_API_ERROR)


def test_health_export_partial_failure_and_safe_retry(store, client):
    conn = store._connect()
    commits = deny_commits(conn, after=1)
    try:
        response = client.post("/fitness/health-export", json=BATCH)
    finally:
        conn.set_authorizer(None)
    assert len(commits) == 2
    assert response.status_code == 503
    assert response.text == (
        "Corridas: 1 novas, 0 já existiam, 1 não consegui gravar (de 2 no payload).")
    assert not conn.in_transaction
    assert len(store.run_history()) == 1
    response = client.post("/fitness/health-export", json=BATCH)
    assert (response.status_code, response.text) == (
        200, "Corridas: 1 novas, 1 já existiam (de 2 no payload).")
    assert len(store.run_history()) == 2


def test_health_export_no_failure_text(client):
    response = client.post("/fitness/health-export", json=BATCH)
    assert (response.status_code, response.text) == (
        200, "Corridas: 2 novas, 0 já existiam (de 2 no payload).")


def test_xml_import_and_cli_report_failure(store, tmp_path, monkeypatch, capsys):
    path = tmp_path / "export.xml"
    path.write_text(XML)
    monkeypatch.setattr(cli, "get_log", lambda: store)
    monkeypatch.setattr(sys, "argv", ["import_apple_health.py", str(path)])
    with locked(store):
        stats = import_export(str(path), store)
        exit_code = cli.main()
    assert stats == {"imported": 0, "skipped": 0, "failed": 1, "total": 1}
    assert exit_code == 1
    assert capsys.readouterr().out == (
        "Corridas: 0 importadas, 0 já existiam, 1 não consegui gravar "
        "(de 1 encontradas no export).\n")
    assert store.run_history() == []


def test_cli_success_text_is_unchanged(store, tmp_path, monkeypatch, capsys):
    path = tmp_path / "export.xml"
    path.write_text(XML)
    monkeypatch.setattr(cli, "get_log", lambda: store)
    monkeypatch.setattr(sys, "argv", ["import_apple_health.py", str(path)])
    assert cli.main() == 0
    assert capsys.readouterr().out == (
        "Corridas: 1 importadas, 0 já existiam (de 1 encontradas no export).\n")


@pytest.mark.parametrize("entry", ["core", "stream"])
@pytest.mark.parametrize("text,reply,mode,event", [
    ("corri 5km em 25min", RUN_ERRORS[0], "FITNESS_RUN_ERROR", "run_log_failed"),
    ("ran 5km in 25min", RUN_ERRORS[1], "FITNESS_RUN_ERROR", "run_log_failed"),
    ("registra supino 3x10 80kg", LIFT_ERRORS[0], "FITNESS_LIFT_ERROR", "lift_log_failed"),
    ("log bench press 3x10 80kg", LIFT_ERRORS[0], "FITNESS_LIFT_ERROR", "lift_log_failed"),
])
def test_manual_failure_is_fixed_through_both_entrypoints(
        store, orch, events, entry, text, reply, mode, event):
    with locked(store):
        if entry == "core":
            result = orch._process_core(text)
        else:
            chunks = list(orch.process_stream(text))
            assert chunks[-1][1] is True
            result = chunks[-1][0], chunks[-1][2]
    assert result == (reply, mode)
    assert ("fitness", event, {}) in events
    assert not any(name in ("run_logged", "lift_logged") for _, name, _ in events)
    for engine in (orch.engine_analytical, orch.engine_creative):
        engine.chat.assert_not_called()
        engine.chat_with_tools.assert_not_called()
        engine.chat_stream.assert_not_called()
    assert store.run_history() == []
    assert store.exercise_history("bench press") == []


@pytest.mark.parametrize("entry,reply_lang,text,lift,reply", [
    ("handler", "en", "agachamento 4x8 100kg", ("agachamento", "4", "8", "100"),
     LIFT_ERRORS[0]),
    ("handler", "pt", "log bench press 3x10 80kg", ("bench press", "3", "10", "80"),
     LIFT_ERRORS[0]),
    ("core", "en", "agachamento 4x8 100kg", ("agachamento", "4", "8", "100"),
     LIFT_ERRORS[0]),
])
def test_lift_failure_reply_uses_command_language(
        store, orch, events, entry, reply_lang, text, lift, reply):
    orch._reply_lang = reply_lang
    with locked(store):
        if entry == "handler":
            result = orch._handle_fitness_log_lift(*lift, raw_text=text)
        else:
            result = orch._process_core(text)
    assert result == (reply, "FITNESS_LIFT_ERROR")
    assert events == [("fitness", "lift_log_failed", {})]
    assert store.exercise_history(lift[0]) == []
    for engine in (orch.engine_analytical, orch.engine_creative):
        engine.chat.assert_not_called()
        engine.chat_with_tools.assert_not_called()
        engine.chat_stream.assert_not_called()


def test_lift_success_and_failure_use_same_command_language(store, orch):
    # "log " is not an English language trigger; both replies stay Portuguese.
    raw_text = "log bench press 3x10 80kg"
    success = orch._handle_fitness_log_lift("bench press", "3", "10", "80", raw_text=raw_text)
    with locked(store):
        error = orch._handle_fitness_log_lift("bench press", "3", "10", "80", raw_text=raw_text)
    assert (success, error) == (
        ("Registrei: bench press 3x10 80kg (volume 2400kg).", "FITNESS_LIFT_LOGGED"),
        (LIFT_ERRORS[0], "FITNESS_LIFT_ERROR"),
    )


DELETES = [
    ("_handle_fitness_undo_meal", "desfaz a última refeição", "undo last meal",
     "FITNESS_UNDO", "undo_last_meal",
     "Não consegui desfazer a refeição agora — deu erro no registro, nada foi apagado. Tenta de novo daqui a pouco.",
     "I couldn't undo the meal just now — the log failed, nothing was deleted. Try again in a moment.",
     "Não há refeição pra desfazer.", "No meal to undo."),
    ("_handle_fitness_undo_workout", "desfaz o último treino", "undo last workout",
     "FITNESS_WORKOUT_UNDO", "undo_last_workout",
     "Não consegui desfazer o treino agora — deu erro no registro, nada foi apagado. Tenta de novo daqui a pouco.",
     "I couldn't undo the workout just now — the log failed, nothing was deleted. Try again in a moment.",
     "Não há treino pra desfazer.", "No workout to undo."),
    ("_handle_fitness_clear_today", "limpa as refeições de hoje", "clear today's meals",
     "FITNESS_CLEAR", "clear_today",
     "Não consegui apagar as refeições de hoje — deu erro no registro, nada foi apagado.",
     "I couldn't clear today's meals — the log failed, nothing was deleted.",
     "Não havia nada registrado hoje.", "Nothing was logged today."),
    ("_handle_fitness_clear_workouts", "limpa os treinos de hoje", "clear today's workouts",
     "FITNESS_WORKOUT_CLEAR", "clear_workouts_today",
     "Não consegui apagar os treinos de hoje — deu erro no registro, nada foi apagado.",
     "I couldn't clear today's workouts — the log failed, nothing was deleted.",
     "Não havia treino registrado hoje.", "No workouts logged today."),
]


@pytest.mark.parametrize("case", DELETES, ids=lambda case: case[3])
@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("failure", ["locked", "unavailable"])
def test_delete_failure_is_not_empty(store, orch, events, monkeypatch, case, lang, failure):
    method, pt, en, mode, event, err_pt, err_en, *_ = case
    assert store.log_meal(store.parse_meal("100g de arroz")[0])
    assert store.log_run(5, 25)
    conn = store._connect()
    if failure == "unavailable":
        monkeypatch.setattr(store, "_connect", lambda: None)
    with locked(store):
        result = getattr(orch, method)(raw_text=pt if lang == "pt" else en)
    assert result == (err_pt if lang == "pt" else err_en, mode + "_ERROR")
    payload = {"removed": 0, "error": True}
    if event == "undo_last_workout":
        payload["kind"] = None
    assert events == [("fitness", event, payload)]
    assert conn.execute("SELECT COUNT(*) FROM meals").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM workouts").fetchone()[0] == 1
    assert not conn.in_transaction


@pytest.mark.parametrize("case", DELETES, ids=lambda case: case[3])
@pytest.mark.parametrize("lang", ["pt", "en"])
def test_delete_empty_still_says_empty(orch, events, case, lang):
    method, pt, en, mode, event, _, _, empty_pt, empty_en = case
    assert getattr(orch, method)(raw_text=pt if lang == "pt" else en) == (
        empty_pt if lang == "pt" else empty_en, mode + "_EMPTY")
    assert "error" not in events[0][2]


@pytest.mark.parametrize("method,table", [
    ("undo_last_meal", "meals"), ("clear_today", "meals"),
    ("undo_last_workout", "workouts"), ("clear_workouts_today", "workouts"),
])
def test_failed_delete_commit_rolls_back_before_next_write(store, method, table):
    assert store.log_meal(store.parse_meal("100g de arroz")[0])
    assert store.log_run(5, 25)
    conn = store._connect()
    commits = deny_commits(conn)
    try:
        result = getattr(store, method + "_result")()
    finally:
        conn.set_authorizer(None)
    assert commits == ["COMMIT"]
    assert result.error is True and result.removed == 0
    assert not conn.in_transaction
    assert store.log_lift("fictitious lift", 2, 4, 10)
    expected = 1 if table == "meals" else 2
    assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == expected
    with sqlite3.connect(store.db_path) as observer:
        assert observer.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == expected


def test_shared_connection_commit_failure_rolls_back(store):
    conn = store._connect()
    conn.execute("INSERT INTO workouts (ts, kind, name) VALUES ('test', 'run', 'corrida')")
    commits = deny_commits(conn)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            conn.commit()
    finally:
        conn.set_authorizer(None)
    assert commits == ["COMMIT"]
    assert not conn.in_transaction
    assert store.log_run(5, 25)
    assert len(store.run_history()) == 1


@pytest.mark.parametrize("method,args", [
    ("log_run", (5, 25)), ("log_lift", ("fictitious lift", 2, 4, 10)),
    ("import_run", (5, 25, "2026-05-01 07:30:00")),
])
@pytest.mark.parametrize("failure", ["locked", "unavailable", "commit"])
def test_write_results_report_errors_and_bool_wrappers_stay_false(
        store, monkeypatch, method, args, failure):
    conn = store._connect()
    if failure == "unavailable":
        monkeypatch.setattr(store, "_connect", lambda: None)
    if failure == "commit":
        deny_commits(conn)
    try:
        with locked(store) if failure == "locked" else context_without_lock():
            assert getattr(store, method + "_result")(*args) == "error"
            assert getattr(store, method)(*args) is False
    finally:
        conn.set_authorizer(None)
    assert not conn.in_transaction
    assert conn.execute("SELECT COUNT(*) FROM workouts").fetchone()[0] == 0


@contextmanager
def context_without_lock():
    yield


@pytest.mark.parametrize("method,args", [
    ("_handle_fitness_log_run", (0, 25)),
    ("_handle_fitness_log_run", ("bad", 25)),
    ("_handle_fitness_log_run", (float("inf"), 25)),
    ("_handle_fitness_log_lift", ("fictitious lift", 0, 4, 10)),
    ("_handle_fitness_log_lift", ("", 2, 4, 10)),
    ("_handle_fitness_log_lift", ("fictitious lift", "bad", 4, 10)),
])
def test_invalid_manual_values_still_fall_through(orch, method, args):
    assert getattr(orch, method)(*args) is None
