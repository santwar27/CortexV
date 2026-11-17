"""Apple Health import parser tests (Fitness Engine slice 3, s76)."""
import zipfile

import pytest

from src.integrations.apple_health import iter_runs, import_export, _to_km, _to_min, _norm_ts
from src.integrations.fitness import FitnessLog


# --- old format: distance/duration as Workout attributes ---
_OLD = """<?xml version="1.0" encoding="UTF-8"?>
<HealthData locale="pt_BR">
 <ExportDate value="2026-06-16 12:00:00 -0300"/>
 <Workout workoutActivityType="HKWorkoutActivityTypeRunning" duration="25.5"
   durationUnit="min" totalDistance="5.0" totalDistanceUnit="km"
   startDate="2026-05-01 07:30:00 -0300" endDate="2026-05-01 07:55:30 -0300"/>
 <Workout workoutActivityType="HKWorkoutActivityTypeCycling" duration="40"
   durationUnit="min" totalDistance="15" totalDistanceUnit="km"
   startDate="2026-05-02 08:00:00 -0300"/>
 <Record type="HKQuantityTypeIdentifierStepCount" value="1000"
   startDate="2026-05-01 07:30:00 -0300"/>
</HealthData>
"""

# --- new format: distance inside nested WorkoutStatistics ---
_NEW = """<?xml version="1.0" encoding="UTF-8"?>
<HealthData>
 <Workout workoutActivityType="HKWorkoutActivityTypeRunning" duration="24"
   durationUnit="min" startDate="2026-05-03 06:00:00 -0300">
   <WorkoutStatistics type="HKQuantityTypeIdentifierDistanceWalkingRunning" sum="6.0" unit="km"/>
   <WorkoutStatistics type="HKQuantityTypeIdentifierActiveEnergyBurned" sum="350" unit="kcal"/>
 </Workout>
 <Workout workoutActivityType="HKWorkoutActivityTypeRunning" duration="30"
   durationUnit="min" totalDistance="3.1" totalDistanceUnit="mi"
   startDate="2026-05-04 06:00:00 -0300"/>
</HealthData>
"""


def _write(tmp_path, text, name="export.xml"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


# ---------- unit conversions ----------

def test_to_km_and_min():
    assert _to_km("5", "km") == 5.0
    assert round(_to_km("3.1", "mi"), 3) == 4.989
    assert _to_min("25.5", "min") == 25.5
    assert _to_min("1530", "s") == 25.5
    assert _to_min("1", "h") == 60.0


def test_norm_ts_drops_timezone():
    assert _norm_ts("2026-05-01 07:30:00 -0300") == "2026-05-01 07:30:00"


def test_parse_distance_km_lenient():
    from src.integrations.apple_health import parse_distance_km
    assert parse_distance_km(5) == 5.0
    assert parse_distance_km("5") == 5.0
    assert parse_distance_km("5,02 km") == 5.02
    assert round(parse_distance_km("3.1 mi"), 3) == 4.989
    assert parse_distance_km("5000 m") == 5.0


def test_parse_duration_min_lenient():
    from src.integrations.apple_health import parse_duration_min
    assert parse_duration_min("25 min") == 25.0
    assert parse_duration_min("1530 s") == 25.5
    assert parse_duration_min("0:25:30") == 25.5
    assert parse_duration_min("25:30") == 25.5
    assert parse_duration_min(1500) == 25.0    # bare > 180 → seconds
    assert parse_duration_min("90") == 90.0    # bare ≤ 180 → minutes


# ---------- iter_runs ----------

def test_iter_runs_old_format_only_running(tmp_path):
    runs = list(iter_runs(_write(tmp_path, _OLD)))
    assert len(runs) == 1                       # cycling + record excluded
    assert runs[0]["distance_km"] == 5.0
    assert runs[0]["duration_min"] == 25.5
    assert runs[0]["ts"] == "2026-05-01 07:30:00"


def test_iter_runs_new_format_and_miles(tmp_path):
    runs = list(iter_runs(_write(tmp_path, _NEW)))
    assert len(runs) == 2
    assert runs[0]["distance_km"] == 6.0        # nested WorkoutStatistics
    assert runs[1]["distance_km"] == 4.989      # 3.1 mi → km


# ---------- import_export + dedup ----------

def test_import_export_and_idempotent(tmp_path):
    fl = FitnessLog(db_path=str(tmp_path / "f.db"))
    path = _write(tmp_path, _OLD)
    stats = import_export(path, fl)
    assert stats == {"imported": 1, "skipped": 0, "failed": 0, "total": 1}
    # the run is queryable with the right pace, at its real date
    runs = fl.run_history()
    assert len(runs) == 1
    assert runs[0]["pace_min_km"] == 5.1        # 25.5 / 5.0
    assert runs[0]["date"] == "2026-05-01"
    # re-import → everything skipped (idempotent)
    again = import_export(path, fl)
    assert again == {"imported": 0, "skipped": 1, "failed": 0, "total": 1}
    assert len(fl.run_history()) == 1


def test_import_from_zip(tmp_path):
    zip_path = tmp_path / "export.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("apple_health_export/export.xml", _OLD)
        zf.writestr("apple_health_export/export_cda.xml", "<ignored/>")
    fl = FitnessLog(db_path=str(tmp_path / "f.db"))
    stats = import_export(str(zip_path), fl)
    assert stats["imported"] == 1


def test_import_runs_mix_into_manual_history(tmp_path):
    fl = FitnessLog(db_path=str(tmp_path / "f.db"))
    fl.log_run(5, 30)                            # a manual run "now"
    import_export(_write(tmp_path, _NEW), fl)    # + 2 historical
    assert len(fl.run_history(20)) == 3


# ---------- Health Auto Export (REST) payload parsing ----------

_HAE = {  # qty/units shape, nested under data.workouts
    "data": {"workouts": [
        {"name": "Running", "start": "2026-06-01 07:30:00 -0300",
         "distance": {"qty": 5.0, "units": "km"},
         "duration": {"qty": 25, "units": "min"}},
        {"name": "Outdoor Cycle", "start": "2026-06-02 08:00:00 -0300",
         "distance": {"qty": 15, "units": "km"},
         "duration": {"qty": 40, "units": "min"}},
    ]}
}

_HAE_FLAT = {  # flat numbers, duration in seconds, top-level workouts
    "workouts": [
        {"type": "Running", "startDate": "2026-06-03 06:00:00",
         "distance": 6.0, "duration": 1530},
    ]
}


def test_iter_export_runs_qty_units_filters_cycling():
    from src.integrations.apple_health import iter_export_runs
    runs = list(iter_export_runs(_HAE))
    assert len(runs) == 1                          # cycling excluded
    assert runs[0]["distance_km"] == 5.0
    assert runs[0]["duration_min"] == 25.0
    assert runs[0]["ts"] == "2026-06-01 07:30:00"


def test_iter_export_runs_flat_and_seconds():
    from src.integrations.apple_health import iter_export_runs
    runs = list(iter_export_runs(_HAE_FLAT))
    assert len(runs) == 1
    assert runs[0]["distance_km"] == 6.0
    assert runs[0]["duration_min"] == 25.5         # 1530 s


def test_iter_export_runs_garbage_is_empty():
    from src.integrations.apple_health import iter_export_runs
    assert list(iter_export_runs({})) == []
    assert list(iter_export_runs({"data": {}})) == []
    assert list(iter_export_runs("nope")) == []


def test_the_same_run_through_both_endpoints_is_one_run(tmp_path):
    """The export kept the "T", /fitness/run made it a space: one run sent
    through both was stored twice (s88, Codex lote 11)."""
    import src.api as api
    from src.integrations import apple_health as ah
    from src.integrations.fitness import FitnessLog
    log = FitnessLog(db_path=str(tmp_path / "f.db"))
    assert log.import_run(5, 25, ah._norm_ts("2026-10-18T07:30:00+01:00"))
    assert not log.import_run(5, 25, api._run_ts("2026-10-18T07:30:00+01:00"))
    assert len(log.run_history(50)) == 1


@pytest.mark.parametrize("raw,wall", [
    ("2026-10-18T07:30:00+01:00", "2026-10-18 07:30:00"),
    ("2026-10-18 07:30:00 +0100", "2026-10-18 07:30:00"),
    ("2026-10-18T07:30:00Z", "2026-10-18 07:30:00"),
    ("2026-10-18T07:30", "2026-10-18 07:30:00"),
    ("2026-10-18 07:30:00.123 -0300", "2026-10-18 07:30:00"),
    ("ontem de manhã", None),
])
def test_run_start_writes_every_format_one_way(raw, wall):
    from src.integrations.apple_health import run_start
    assert run_start(raw) == wall


def test_two_runs_an_hour_apart_in_the_repeated_hour_are_two_runs(tmp_path):
    """25/10/2026: 01:30 BST and 01:30 GMT are an hour apart; by wall clock
    the second was dropped as a duplicate (s88, Codex lote 11)."""
    from src.integrations import apple_health as ah
    from src.integrations.fitness import FitnessLog
    log = FitnessLog(db_path=str(tmp_path / "f.db"))
    for raw, km in (("2026-10-25 01:30:00 +0100", 5), ("2026-10-25 01:30:00 +0000", 6)):
        assert log.import_run(km, 25, ah._norm_ts(raw), start_utc=ah.run_instant(raw))
    assert len(log.run_history(50)) == 2
    # and re-importing the same export still adds nothing
    raw = "2026-10-25 01:30:00 +0000"
    assert not log.import_run(6, 25, ah._norm_ts(raw), start_utc=ah.run_instant(raw))


@pytest.mark.parametrize("raw,instant", [
    ("2026-10-25 01:30:00 +0100", "2026-10-25T00:30:00Z"),
    ("2026-10-25T01:30:00+00:00", "2026-10-25T01:30:00Z"),
    ("2026-10-18T07:30:00Z", "2026-10-18T07:30:00Z"),
    ("2026-05-01 07:30:45 -0300", "2026-05-01T10:30:45Z"),
])
def test_run_instant(raw, instant):
    from src.integrations.apple_health import run_instant
    assert run_instant(raw) == instant


def test_a_run_stored_before_the_instant_is_still_a_duplicate(tmp_path):
    """Rows imported before start_utc existed carry none: matched by ts."""
    from src.integrations.fitness import FitnessLog
    log = FitnessLog(db_path=str(tmp_path / "f.db"))
    assert log.import_run(5, 25, "2026-06-01 07:30:00")          # legacy, no instant
    assert not log.import_run(5, 25, "2026-06-01 07:30:00",
                              start_utc="2026-06-01T06:30:00Z")


class TestAMeasureIsReadWhole:
    """"5 yd" became 5 km and "1 h 30 min" 60 minutes: an unknown unit fell
    to the default and only the first part was read (s88, Codex lote 11)."""

    @pytest.mark.parametrize("raw", ["5 yd", "5 furlongs", "5 km 200"])
    def test_an_unknown_distance_is_refused(self, raw):
        from src.integrations.apple_health import parse_distance_km
        with pytest.raises(ValueError):
            parse_distance_km(raw)

    @pytest.mark.parametrize("raw,minutes", [
        ("1 h 30 min", 90.0), ("1h30", 90.0), ("1 hora e 30 minutos", 90.0),
        ("1h 2m 10s", 62.17), ("25 min", 25.0), ("1530 s", 25.5), ("25", 25.0),
    ])
    def test_a_compound_duration_adds_up(self, raw, minutes):
        from src.integrations.apple_health import parse_duration_min
        assert round(parse_duration_min(raw), 2) == minutes

    @pytest.mark.parametrize("raw", ["30 min 5 parsecs", "meia hora", "25 x"])
    def test_an_unknown_duration_is_refused(self, raw):
        from src.integrations.apple_health import parse_duration_min
        with pytest.raises(ValueError):
            parse_duration_min(raw)


class TestAnInfiniteNumberIsNotARun:
    """"duration": 1e309 became infinity, passed "> 0", was imported, and
    "meu pace" crashed on int(inf) (s88, Codex lote 11)."""

    def test_the_export_skips_it(self):
        import json
        from src.integrations.apple_health import iter_export_runs
        payload = json.loads('{"workouts": [{"name": "Outdoor Run", "distance": 5, '
                             '"duration": 1e309, "start": "2026-10-18 07:30:00 +0100"}]}')
        assert list(iter_export_runs(payload)) == []

    @pytest.mark.parametrize("value", [float("inf"), float("nan")])
    def test_the_parsers_refuse_it(self, value):
        from src.integrations.apple_health import parse_distance_km, parse_duration_min
        for parse in (parse_distance_km, parse_duration_min):
            with pytest.raises(ValueError):
                parse(value)

    def test_the_store_refuses_it_too(self, tmp_path):
        from src.integrations.fitness import FitnessLog
        log = FitnessLog(db_path=str(tmp_path / "f.db"))
        assert not log.import_run(5, float("inf"), "2026-10-18 07:30:00")
        assert not log.log_run(float("nan"), 25)
        assert log.run_history(10) == []
