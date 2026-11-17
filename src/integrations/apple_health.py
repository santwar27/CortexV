"""
Apple Health export parser → Fitness Engine (slice 3, s76).

Imports RUNNING workouts from an Apple Health export into the fitness store.
Apple Health exports a zip (Health.app → profile → "Export All Health Data")
whose `export.xml` holds every record — often hundreds of MB — so we STREAM it
with iterparse and keep only running workouts (distance + duration → our `run`
rows). Each run is stored at its REAL date so the trends are chronological,
and de-duplicated by timestamp so re-importing the same export adds nothing.

Two export shapes are handled:
  - older: <Workout duration="25.5" durationUnit="min"
            totalDistance="5.0" totalDistanceUnit="km" startDate="…"/>
  - newer (iOS 14+): distance moved into a nested
            <WorkoutStatistics type="…DistanceWalkingRunning" sum="5.0" unit="km"/>

Soberania: 100% local — reads a file the user exported, writes the local DB,
nothing leaves the machine. Strength workouts aren't imported (Apple Health
has no per-exercise sets/reps), only distance cardio runs.
"""
from __future__ import annotations

import logging
import math
import re
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterator, Optional, Union

log = logging.getLogger(__name__)

_RUN_TYPES = {"HKWorkoutActivityTypeRunning"}
_MILE_KM = 1.60934


# Every unit understood, and nothing else: "5 yd" became 5 km and "5
# furlongs" too, because an unknown unit fell through to the km default (s88,
# Codex lote 11 — invariant 1). An unknown unit is now an error.
_KM_UNITS = {"", "km", "k", "kms", "kilometer", "kilometers", "kilometre",
             "kilometres", "quilometro", "quilometros", "quilômetro", "quilômetros"}
_MI_UNITS = {"mi", "mile", "miles", "milha", "milhas"}
_M_UNITS = {"m", "meter", "meters", "metre", "metres", "metro", "metros"}
_MIN_UNITS = {"", "m", "min", "mins", "minute", "minutes", "minuto", "minutos"}
_SEC_UNITS = {"s", "sec", "secs", "second", "seconds", "segundo", "segundos"}
_HOUR_UNITS = {"h", "hr", "hrs", "hour", "hours", "hora", "horas"}


def _finite(v: float) -> float:
    """A duration of 1e309 in the JSON became infinity, passed "> 0", was
    stored, and "meu pace" crashed on int(inf) (s88, Codex lote 11)."""
    if not math.isfinite(v):
        raise ValueError(f"número inválido: {v!r}")
    return v


def _to_km(value: str, unit: Optional[str]) -> float:
    v = _finite(float(str(value).replace(",", ".")))
    u = (unit or "").strip().lower()
    if u in _MI_UNITS:
        return v * _MILE_KM
    if u in _M_UNITS:
        return v / 1000.0
    if u in _KM_UNITS:
        return v
    raise ValueError(f"unidade de distância desconhecida: {unit!r}")


def _to_min(value: str, unit: Optional[str]) -> float:
    v = _finite(float(str(value).replace(",", ".")))
    u = (unit or "min").strip().lower()
    if u in _SEC_UNITS:
        return v / 60.0
    if u in _HOUR_UNITS:
        return v * 60.0
    if u in _MIN_UNITS:
        return v
    raise ValueError(f"unidade de duração desconhecida: {unit!r}")


_ONE_MEASURE_RE = re.compile(r"^\s*(-?\d+(?:[.,]\d+)?)\s*([A-Za-zÀ-ÿ]*)\s*$")
# "1 h 30 min", "1h30", "1h 2m 10s", "1 hora e 30 minutos" — every part read.
_DURATION_PART_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*([A-Za-zÀ-ÿ]*)")
_DURATION_GLUE_RE = re.compile(r"^(?:\s|,|\be\b|\band\b)*$", re.IGNORECASE)


def parse_distance_km(raw: Union[str, float, int]) -> float:
    """Lenient distance → km, for the /fitness/run endpoint (a Shortcut may
    send "5", 5, "5,02 km", "3.1 mi", "5000 m"). The whole string is one
    measure in a known unit, or it is refused."""
    if isinstance(raw, (int, float)):
        return _finite(float(raw))
    m = _ONE_MEASURE_RE.match(str(raw))
    if not m:
        raise ValueError(f"distância inválida: {raw!r}")
    return _to_km(m.group(1), m.group(2))


def parse_duration_min(raw: Union[str, float, int]) -> float:
    """Lenient duration → minutes. Handles "25", "25 min", "1530 s",
    "0:25:30"/"25:30" (H:MM:SS / MM:SS) and compounds — "1 h 30 min" was
    read as 60, dropping the 30 (s88, Codex lote 11). A BARE number ≤ 180 is
    read as minutes, > 180 as seconds (a real run is minutes; Apple's raw
    duration is seconds, so a big bare number is almost certainly seconds)."""
    if isinstance(raw, (int, float)):
        v = _finite(float(raw))
        return v / 60.0 if v > 180 else v
    s = str(raw).strip()
    if ":" in s:
        try:
            parts = [float(p) for p in s.split(":")]
        except ValueError:
            raise ValueError(f"duração inválida: {raw!r}")
        if len(parts) == 3:
            return parts[0] * 60 + parts[1] + parts[2] / 60.0
        if len(parts) == 2:
            return parts[0] + parts[1] / 60.0
        raise ValueError(f"duração inválida: {raw!r}")
    parts = list(_DURATION_PART_RE.finditer(s))
    if not parts or not _DURATION_GLUE_RE.match(_DURATION_PART_RE.sub(" ", s)):
        raise ValueError(f"duração inválida: {raw!r}")
    if len(parts) == 1 and not parts[0].group(2):  # bare number: minutes if small, else seconds
        v = float(parts[0].group(1).replace(",", "."))
        return v / 60.0 if v > 180 else v
    total, prev = 0.0, None
    for p in parts:
        num, unit = p.group(1), p.group(2)
        if not unit:
            # "1h30": a bare number after hours is minutes, after minutes seconds
            unit = {"h": "min", "min": "s"}.get(prev, None)
            if unit is None:
                raise ValueError(f"duração inválida: {raw!r}")
        total += _to_min(num, unit)
        u = unit.lower()
        prev = "h" if u in _HOUR_UNITS else "min" if u in _MIN_UNITS else "s"
    return total


_START_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2})(?::(\d{2}))?(?:\.\d+)?"
    r"\s*(Z|[+-]\d{2}:?\d{2})?$",
    re.IGNORECASE,
)


def run_start(raw: Optional[str]) -> Optional[str]:
    """A run's start, as every importer writes it: the wall clock the run was
    logged at, "YYYY-MM-DD HH:MM:SS". The health export and /fitness/run each
    cut the string their own way — one kept the "T", the other made it a
    space — so the same run sent through both got two identities and was
    stored twice (s88, Codex lote 11, reproduced). None when unparseable."""
    m = _START_RE.match((raw or "").strip())
    if not m:
        return None
    return f"{m.group(1)} {m.group(2)}:{m.group(3) or '00'}"


def run_instant(raw: Optional[str]) -> Optional[str]:
    """The run's start as an INSTANT, "YYYY-MM-DDTHH:MM:SSZ" — the identity
    the store deduplicates on. The wall clock alone made 01:30 BST and 01:30
    GMT on 25/10/2026, an hour apart, one run, and the second was dropped as
    a duplicate (s88, Codex lote 11, reproduced). An offset given is used; a
    start with none is read as this machine's local time."""
    m = _START_RE.match((raw or "").strip())
    if not m:
        return None
    try:
        naive = datetime.strptime(run_start(raw), "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return None
    off = m.group(4)
    if off:
        if off.upper() == "Z":
            tz = timezone.utc
        else:
            sign = 1 if off[0] == "+" else -1
            digits = off[1:].replace(":", "")
            tz = timezone(sign * timedelta(hours=int(digits[:2]), minutes=int(digits[2:])))
        instant = naive.replace(tzinfo=tz)
    else:
        instant = naive.astimezone()
    return instant.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _norm_ts(raw: Optional[str]) -> str:
    """Apple's "2026-05-01 07:30:45 -0300" → "2026-05-01 07:30:45" (the
    wall clock the run was logged at — run_start)."""
    raw = (raw or "").strip()
    return run_start(raw) or (raw[:19] if len(raw) >= 19 else raw)


def _workout_to_run(elem: ET.Element) -> Optional[Dict]:
    """A running <Workout> element → {distance_km, duration_min, ts} or None
    when it lacks the distance/duration we need."""
    ts = _norm_ts(elem.get("startDate"))
    dur = elem.get("duration")
    dur_unit = elem.get("durationUnit")
    dist = elem.get("totalDistance")
    dist_unit = elem.get("totalDistanceUnit")
    if dist is None:  # newer format — look in the nested statistics
        for st in elem.iter("WorkoutStatistics"):
            if "Distance" in (st.get("type") or ""):
                dist = st.get("sum") or st.get("value")
                dist_unit = st.get("unit")
                break
    if not (ts and dur and dist):
        return None
    try:
        km = round(_to_km(dist, dist_unit), 3)
        mn = round(_to_min(dur, dur_unit), 2)
    except (ValueError, TypeError):
        return None
    if km <= 0 or mn <= 0:
        return None
    return {"distance_km": km, "duration_min": mn, "ts": ts,
            "start_utc": run_instant(elem.get("startDate"))}


def iter_runs(xml_source) -> Iterator[Dict]:
    """Stream {distance_km, duration_min, ts} for each running Workout.

    Memory stays flat: root.clear() runs after each TOP-LEVEL <Workout>/<Record>
    (the millions of records), but NOT on nested elements — so a <Workout>'s
    own <WorkoutStatistics> children are still readable at its end event."""
    context = ET.iterparse(xml_source, events=("start", "end"))
    _, root = next(context)  # first 'start' is the <HealthData> root
    for event, elem in context:
        if event != "end":
            continue
        if elem.tag == "Workout":
            if elem.get("workoutActivityType") in _RUN_TYPES:
                rec = _workout_to_run(elem)
                if rec:
                    yield rec
            root.clear()
        elif elem.tag == "Record":
            root.clear()  # the bulk of the file — drop as we go


def _open_xml(path: str):
    """File-like for the export's XML, whether `path` is the .zip or the .xml.
    Inside the zip it's `apple_health_export/export.xml` (NOT export_cda.xml)."""
    if path.lower().endswith(".zip"):
        zf = zipfile.ZipFile(path)
        name = next(
            (n for n in zf.namelist()
             if n.endswith("export.xml") and "cda" not in n.lower()),
            None,
        )
        if name is None:
            raise ValueError("export.xml não encontrado no zip do Apple Health.")
        return zf.open(name)
    return open(path, "rb")


def _qty_unit(v):
    """A Health Auto Export value is either a bare number or a
    {"qty": N, "units": "km"} object. Returns (value, unit-or-None)."""
    if isinstance(v, dict):
        return v.get("qty", v.get("value")), v.get("units", v.get("unit"))
    return v, None


def iter_export_runs(payload) -> Iterator[Dict]:
    """Yield {distance_km, duration_min, ts} for RUNNING workouts in a Health
    Auto Export REST payload. Best-effort across app-version shapes:
    `{"data":{"workouts":[…]}}`, `{"workouts":[…]}`, or a bare list. Each
    workout: name/type contains "run"; distance/duration are bare numbers OR
    {qty,units}; start under start/startDate/date."""
    workouts = None
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, dict):
            workouts = data.get("workouts")
        if workouts is None:
            workouts = payload.get("workouts")
    elif isinstance(payload, list):
        workouts = payload
    if not isinstance(workouts, list):
        return
    for w in workouts:
        if not isinstance(w, dict):
            continue
        name = str(w.get("name") or w.get("type")
                   or w.get("workoutActivityType") or "").lower()
        if "run" not in name and "corrida" not in name:
            continue
        dist_raw = w.get("distance")
        if dist_raw is None:
            dist_raw = w.get("distanceWalkingRunning") or w.get("totalDistance")
        dur_raw = w.get("duration")
        start = w.get("start") or w.get("startDate") or w.get("date")
        if dist_raw is None or dur_raw is None or not start:
            continue
        dval, dunit = _qty_unit(dist_raw)
        tval, tunit = _qty_unit(dur_raw)
        try:
            km = _to_km(dval, dunit) if dunit else parse_distance_km(dval)
            mn = _to_min(tval, tunit) if tunit else parse_duration_min(tval)
        except (ValueError, TypeError):
            continue
        if km <= 0 or mn <= 0:
            continue
        yield {"distance_km": round(km, 3), "duration_min": round(mn, 2),
               "ts": _norm_ts(str(start)), "start_utc": run_instant(str(start))}


def import_export(path: str, fitness_log) -> Dict[str, int]:
    """Import running workouts from an export.zip/.xml into the fitness log.
    Returns {imported, skipped, failed, total} (skipped = already present)."""
    stats = {"imported": 0, "skipped": 0, "failed": 0, "total": 0}
    src = _open_xml(path)
    try:
        for run in iter_runs(src):
            stats["total"] += 1
            result = fitness_log.import_run_result(
                run["distance_km"], run["duration_min"], run["ts"],
                start_utc=run.get("start_utc"))
            if result == "added":
                stats["imported"] += 1
            elif result == "duplicate":
                stats["skipped"] += 1
            else:
                stats["failed"] += 1
    finally:
        try:
            src.close()
        except Exception:
            pass
    return stats
