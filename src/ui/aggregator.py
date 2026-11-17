"""Data aggregator for the /ui dashboard.

Collects a snapshot of Cortex state from file-based sources — no LLM,
and no network EXCEPT net_speed_snapshot (idea #29), which makes one
small, audited, lease-gated probe per TTL window. Each helper degrades
to a safe empty value when its source is missing so the dashboard
renders even on a fresh checkout.

Sources:
  - LogicGate state file (gate_state.json) — internet lease
  - AuditLogger JSONL (~/.cortex/audit.jsonl) — recent events
  - StudyQueue JSONL (data/study_queue.jsonl) — pending topics
  - data/knowledge/{pending,approved}/ — studies awaiting review
  - launchctl list — service health for the 4 com.cortex.* agents
  - platform / time — host meta

Keep this module pure-Python and dependency-free beyond stdlib +
modules already in src/core/. The dashboard route imports
`build_snapshot()` and shoves the dict into the Jinja2 context.
"""

import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------- Lease ----------

def lease_snapshot(state_file: str = "data/gate_state.json") -> Dict[str, Any]:
    """Read gate_state.json directly. Avoids instantiating LogicGate
    (which has regex compilation cost) for a one-shot read."""
    info: Dict[str, Any] = {
        "active": False,
        "expires_at": None,
        "remaining_minutes": None,
    }
    try:
        with open(state_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw = data.get("expires_at")
        if not raw:
            return info
        exp = datetime.fromisoformat(raw)
        now = datetime.now(timezone.utc)
        if exp <= now:
            return info
        info["active"] = True
        info["expires_at"] = exp.isoformat()
        info["remaining_minutes"] = int((exp - now).total_seconds() // 60)
    except (FileNotFoundError, json.JSONDecodeError, ValueError, KeyError):
        pass
    return info


# ---------- Audit ----------

def _audit_path() -> str:
    return os.environ.get(
        "CORTEX_AUDIT_LOG",
        os.path.expanduser("~/.cortex/audit.jsonl"),
    )


def _local_hms(ts: str) -> str:
    """ISO-8601 UTC timestamp → local HH:MM:SS. The audit log stores UTC;
    the dashboard should read in the user's wall-clock time. Falls back to
    a naive slice if the string doesn't parse."""
    if not ts:
        return ""
    try:
        return datetime.fromisoformat(ts).astimezone().strftime("%H:%M:%S")
    except (ValueError, TypeError):
        return ts[11:19] if len(ts) >= 19 else ts


# The dashboard polls every few seconds and the audit JSONL is append-only
# with no retention yet (M3-A5) — reading the WHOLE file twice per tick
# (audit + voice panels) grows linearly with history. A bounded tail window
# read by seeking from the end keeps each tick O(window), not O(file).
# ~256KB ≈ a thousand records — far more than either panel shows.
_TAIL_READ_BYTES = int(os.getenv("CORTEX_UI_AUDIT_TAIL_BYTES", str(256 * 1024)))


def _tail_lines(path: str, max_bytes: int = 0) -> Optional[List[str]]:
    """Last lines of `path`, reading at most `max_bytes` from the end
    (default `_TAIL_READ_BYTES`). The first line of a truncated window is
    almost certainly partial and is dropped. Returns None on OSError so
    callers keep their existing degrade path."""
    max_bytes = max_bytes or _TAIL_READ_BYTES
    try:
        with open(path, "rb") as f:
            # Size from the OPEN handle, not the path: the audit rotates
            # at >10MB by atomic rename + fresh file (s72), and a stat
            # taken before open() can describe the pre-rotation file —
            # the stale size would seek past the fresh file's EOF and
            # the panel would read empty for a tick. fstat pins size and
            # content to the same inode.
            size = os.fstat(f.fileno()).st_size
            if size > max_bytes:
                f.seek(size - max_bytes)
                data = f.read()
                nl = data.find(b"\n")
                data = data[nl + 1:] if nl >= 0 else b""
            else:
                data = f.read()
    except OSError:
        return None
    return data.decode("utf-8", errors="replace").splitlines()


def _shorten(value: Any, limit: int = 80) -> str:
    """Stringify and truncate so payloads don't blow up the page."""
    if value is None:
        return ""
    s = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def audit_snapshot(tail: int = 30) -> Dict[str, Any]:
    """Last `tail` audit records, plus a today-count and the source
    path so the dashboard can show "no events yet" honestly when the
    file doesn't exist.

    Reads only the bounded tail window (M10-A7), so `count_today` counts
    events within that window — exact on a normal day (~1000 records fit),
    a lower bound on an extremely busy one. The panel is a pulse check,
    not an accounting ledger; the full file stays on disk for `make`
    tooling that needs exactness."""
    path = _audit_path()
    info: Dict[str, Any] = {
        "available": False,
        "path": path,
        "recent": [],
        "count_today": 0,
    }
    if not os.path.exists(path):
        return info
    lines = _tail_lines(path)
    if lines is None:
        return info
    info["available"] = True
    today_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    parsed: List[Dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Valid JSON is not a record: a "null" or "[]" line raised on .get()
        # and took /ui/data down with a 500 (s88, Codex lote 7).
        if not isinstance(rec, dict):
            continue
        parsed.append(rec)
    info["count_today"] = sum(
        1 for r in parsed if isinstance(r.get("ts"), str) and r["ts"].startswith(today_prefix)
    )
    recent = parsed[-tail:]
    info["recent"] = [
        {
            "ts": r.get("ts", ""),
            "trace_id": r.get("trace_id", ""),
            "stage": r.get("stage", ""),
            "event": r.get("event", ""),
            "latency_ms": r.get("latency_ms"),
            "payload_preview": _shorten(r.get("payload")),
        }
        for r in reversed(recent)  # newest first for UI scanning
    ]
    return info


# ---------- Studier ----------

def studier_snapshot(
    queue_path: str = "data/study_queue.jsonl",
    knowledge_root: str = "data/knowledge",
) -> Dict[str, Any]:
    pending: List[Dict[str, str]] = []
    knowledge_pending: List[Dict[str, Any]] = []
    knowledge_approved_count = 0

    # Queue (pending topics waiting for Studier nightly run)
    if os.path.exists(queue_path):
        try:
            with open(queue_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("status") == "pending":
                        pending.append({
                            "topic": rec.get("topic", "(no topic)"),
                            "added_at": rec.get("added_at", ""),
                        })
        except OSError:
            pass

    # Knowledge directory: pending = ready to review, approved = ingested
    pending_dir = Path(knowledge_root) / "pending"
    approved_dir = Path(knowledge_root) / "approved"
    if pending_dir.is_dir():
        try:
            for entry in sorted(pending_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if entry.is_file() and not entry.name.startswith("."):
                    knowledge_pending.append({
                        "name": entry.name,
                        "mtime": datetime.fromtimestamp(
                            entry.stat().st_mtime, tz=timezone.utc
                        ).isoformat(),
                    })
        except OSError:
            pass
    if approved_dir.is_dir():
        try:
            knowledge_approved_count = sum(
                1 for p in approved_dir.iterdir()
                if p.is_file() and not p.name.startswith(".")
            )
        except OSError:
            pass

    return {
        "queue_pending": pending,
        "knowledge_pending": knowledge_pending[:20],  # cap for UI
        "knowledge_approved_count": knowledge_approved_count,
    }


# ---------- Services (launchd) ----------

# Agents the dashboard REQUIRES loaded. Listed statically so a clear
# "missing" row shows even when launchctl never heard of the label (vs
# silently omitting it). For scheduled jobs (nightly/backup/weekly/
# heartbeat) "loaded, no PID" is the healthy idle state — only uvicorn
# is expected to hold a PID. heartbeat + weekly added in the Fable 5
# review (M10-A11c/M7-A1): both were loaded/loadable but absent here,
# so the chip claimed full health while the watchdog was dark.
_EXPECTED_SERVICES = [
    "com.cortex.uvicorn",
    "com.cortex.nightly",
    "com.cortex.backup",
    "com.cortex.weekly",
    "com.cortex.heartbeat",
]

# Agents whose plists exist in launchd/ but are OFF BY CHOICE (the user,
# 2026-06-10): `daily` (the 08:00 spoken briefing) and `dream` are
# deliberately not loaded; `voice` runs as a manual foreground daemon
# (mic TCC grant needs an interactive Terminal), not under launchd.
# Shown in the modal as "off by choice" so a future contributor doesn't
# read them as failures — and NEVER counted against chip health. If one
# is loaded anyway, its real status is shown.
_DEFERRED_SERVICES = [
    "com.cortex.voice",
    "com.cortex.daily",
    "com.cortex.dream",
    "com.cortex.nudge",  # proactive nudges — opt-in like daily (Pedro loads it)
]


# Short-TTL cache for the DEFAULT services_snapshot invocation (M10-A7):
# the dashboard polls every few seconds per open tab, and each uncached call
# is a `launchctl list` subprocess — service state changes on the scale of
# minutes, not seconds. Explicit-args calls (tests, custom probes) bypass.
_SERVICES_TTL_S = float(os.getenv("CORTEX_UI_SERVICES_TTL", "5"))
_services_cache: Dict[str, Any] = {"at": 0.0, "rows": None}


def services_snapshot(
    expected: Optional[List[str]] = None,
    runner: Optional[Any] = None,
    deferred: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Parse `launchctl list` output into structured rows for the
    expected com.cortex.* agents. Output rows: label, running (bool),
    pid (int|None), exit_code (int|None), missing (bool), deferred
    (bool — off by choice, excluded from chip health).

    `runner` is a hook for tests — pass a callable that returns the
    raw launchctl stdout. Default invokes the binary (cached for
    `_SERVICES_TTL_S` seconds — see above)."""
    use_cache = expected is None and runner is None and deferred is None
    if use_cache:
        cached = _services_cache["rows"]
        if cached is not None and (time.monotonic() - _services_cache["at"]) < _SERVICES_TTL_S:
            return cached
    if expected is None:
        labels = _EXPECTED_SERVICES + (
            _DEFERRED_SERVICES if deferred is None else deferred
        )
        deferred_set = set(
            _DEFERRED_SERVICES if deferred is None else deferred
        )
    else:
        labels = expected + (deferred or [])
        deferred_set = set(deferred or [])
    raw = ""
    if runner is not None:
        try:
            raw = runner() or ""
        except Exception:
            raw = ""
    else:
        try:
            result = subprocess.run(
                ["launchctl", "list"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            raw = result.stdout or ""
        except (FileNotFoundError, subprocess.SubprocessError):
            raw = ""

    # launchctl list output: "PID\tStatus\tLabel" rows, header included.
    found: Dict[str, Dict[str, Any]] = {}
    for line in raw.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        pid_s, status_s, label = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if label.startswith("PID") or label == "Label":
            continue  # header row
        pid: Optional[int]
        if pid_s == "-" or not pid_s:
            pid = None
        else:
            try:
                pid = int(pid_s)
            except ValueError:
                pid = None
        try:
            exit_code = int(status_s) if status_s not in ("-", "") else None
        except ValueError:
            exit_code = None
        found[label] = {
            "label": label,
            "running": pid is not None,
            "pid": pid,
            "exit_code": exit_code,
        }

    rows: List[Dict[str, Any]] = []
    for label in labels:
        row = found.get(label)
        if row is None:
            row = {
                "label": label,
                "running": False,
                "pid": None,
                "exit_code": None,
                "missing": True,
            }
        else:
            row["missing"] = False
        row["deferred"] = label in deferred_set
        rows.append(row)
    if use_cache:
        _services_cache["rows"] = rows
        _services_cache["at"] = time.monotonic()
    return rows


# ---------- Memory (Palace) ----------

def memory_snapshot(db_path: str = "data/palace.db") -> Dict[str, Any]:
    """Palace row counts + file size (M3-A5) — the store is append-only
    with no retention policy yet, so the dashboard is where growth gets
    SEEN before it becomes a latency problem. Opens read_only (a
    dashboard must never mutate the patient — M3-A7) and degrades to
    available=False on any failure."""
    try:
        if not os.path.exists(db_path):
            return {"available": False, "counts": {}, "db_bytes": 0}
        from src.memory.palace import Palace

        palace = Palace(db_path=db_path, read_only=True)
        try:
            counts = palace.counts()
        finally:
            palace.close()
        return {
            "available": True,
            "counts": counts,
            "db_bytes": os.path.getsize(db_path),
        }
    except Exception:
        return {"available": False, "counts": {}, "db_bytes": 0}


# ---------- Host ----------

def host_snapshot() -> Dict[str, Any]:
    return {
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "monotonic": time.monotonic(),
    }


# ---------- Cost ledger ----------

def cost_snapshot() -> Dict[str, Any]:
    """Cost-tracker summary for the dashboard chip + modal. Returns
    month-to-date + all-time aggregates plus the recent rows. Reads
    the same SQLite ledger that `/ui/costs` serves as JSON — this
    snapshot is the human-readable mirror.

    Fail-soft by transitive design: `cost_tracker.summary()` and
    `.recent()` already swallow their own errors and return empty
    structures, so nothing here can break the dashboard render.

    The import is INSIDE the function on purpose (M10-A11a, considered
    and kept): tests rebind the `cost_tracker` singleton on its module
    (monkeypatch.setattr(ct_mod, "cost_tracker", fresh)) — a module-top
    import here would freeze the original instance at import time and
    the patch (and the tmp-DB isolation it provides) would stop working."""
    from src.core.cost_tracker import cost_tracker
    now = datetime.now(timezone.utc)
    mtd_since = now.replace(
        day=1, hour=0, minute=0, second=0, microsecond=0,
    ).isoformat()
    mtd = cost_tracker.summary(since=mtd_since)
    all_time = cost_tracker.summary()
    return {
        # `available` mirrors the audit/studier convention: True iff
        # the ledger has any history (otherwise the chip dot reads
        # "off" to communicate cold-start, not failure).
        "available": all_time.get("total_events", 0) > 0,
        "mtd": mtd,
        "all_time": all_time,
        "recent": cost_tracker.recent(limit=15),
    }


# ---------- Voice telemetry ----------

# Mirrors src/voice/telemetry.STAGE — kept as a local literal so this
# pure aggregator doesn't import the voice package (which pulls in
# audio/whisper deps). One string; drift is caught by the snapshot tests.
STAGE_VOICE = "voice"

# Phase events emitted by src/voice/telemetry.py, in timeline order. The
# inter-phase latency lives on the *arriving* event (e.g. the time STT
# took is the latency on "transcribed").
_VOICE_PHASE_EVENTS = ("listened", "transcribed", "thinking", "replied")


def voice_snapshot(tail_sessions: int = 15) -> Dict[str, Any]:
    """Recent voice-pipeline sessions for the dashboard "Voz" panel.

    Reads the same audit JSONL, keeps only `stage="voice"` records, and
    regroups them by trace_id into one row per spoken interaction:
    wake → listened → transcribed → thinking → replied → done, with the
    per-phase latency and the transcribed text. Also computes average
    latency per phase over the shown sessions so "is it healthy / where's
    the time going" is answerable at a glance.

    Fail-soft: a missing file or a malformed line degrades to empty."""
    path = _audit_path()
    info: Dict[str, Any] = {
        "available": False,
        "path": path,
        "recent": [],
        "count_today": 0,
        "averages": {},
    }
    if not os.path.exists(path):
        return info
    # Bounded tail window, same rationale as audit_snapshot (M10-A7).
    lines = _tail_lines(path)
    if lines is None:
        return info

    # Group voice events by trace, preserving first-seen order.
    sessions: "Dict[str, Dict[str, Any]]" = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict) or rec.get("stage") != STAGE_VOICE:
            continue
        trace, event = str(rec.get("trace_id") or "?"), rec.get("event", "")
        if not isinstance(event, str):        # a key: a list there raised too
            continue
        s = sessions.setdefault(trace, {"trace_id": trace, "events": {}})
        # Last write wins per event type (follow-up turns re-emit replied);
        # the timeline only needs one of each.
        s["events"][event] = rec

    if not sessions:
        return info
    info["available"] = True
    today_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    rows: List[Dict[str, Any]] = []
    for s in sessions.values():
        ev = s["events"]
        wake = ev.get("wake", {})
        transcribed = ev.get("transcribed", {})
        replied = ev.get("replied", {})
        done = ev.get("done", {})
        listened = ev.get("listened", {})
        t_payload, w_payload, l_payload = (
            (e.get("payload") if isinstance(e.get("payload"), dict) else {})
            for e in (transcribed, wake, listened))
        phases = {
            e: (ev[e].get("latency_ms") if e in ev else None)
            for e in _VOICE_PHASE_EVENTS
        }
        rows.append({
            "trace_id": s["trace_id"],
            "ts": wake.get("ts", ""),
            # ts is stored UTC; show it in the machine's local time so the
            # panel matches the wall clock (and the spoken time reflex).
            "ts_local": _local_hms(wake.get("ts", "")),
            "text": t_payload.get("text", ""),
            "lang": t_payload.get("lang", ""),
            "score": w_payload.get("score"),
            "listened_s": l_payload.get("duration_s"),
            "status_mode": (replied.get("payload") or {}).get("status_mode", ""),
            "phases": phases,
            "total_ms": done.get("latency_ms"),
            "complete": "done" in ev,
        })

    rows.sort(key=lambda r: r["ts"], reverse=True)  # newest first
    info["count_today"] = sum(
        1 for r in rows if isinstance(r["ts"], str) and r["ts"].startswith(today_prefix)
    )
    shown = rows[:tail_sessions]
    info["recent"] = shown

    # Average latency per phase + total over the shown sessions. ALWAYS
    # emit every key (None when no session reached that phase) so the
    # template can read averages.thinking etc. unconditionally — a
    # partial session (e.g. a reflex reply that never spoke, or a cut-off
    # turn) must not 500 the dashboard. Means skip None so one missing
    # phase doesn't poison the average of the others.
    averages: Dict[str, Optional[float]] = {}
    for key in (*_VOICE_PHASE_EVENTS, "total"):
        vals = [
            (r["total_ms"] if key == "total" else r["phases"].get(key))
            for r in shown
        ]
        vals = [v for v in vals if isinstance(v, (int, float))]
        averages[key] = round(sum(vals) / len(vals), 1) if vals else None
    info["averages"] = averages
    return info


# ---------- Top-level ----------

# ---------- Net speed (idea #29, s74) ----------

# One small download (~100KB default) per TTL window, NOT a speedtest:
# an active speedtest is seconds of saturated traffic and the dashboard
# ticks every 5s per open tab. This is proactive network traffic, so it
# follows the house rules: only with an active lease (no lease → the
# chip says "offline" and NOTHING is fetched), audited as
# background_read (M7-A4), and only the fixed URL leaves the machine —
# nothing from the Palace (SOBERANIA regra 1).
_NETSPEED_URL = os.getenv(
    "CORTEX_UI_NETSPEED_URL",
    "https://speed.cloudflare.com/__down?bytes=100000",
)
_NETSPEED_TTL_S = float(os.getenv("CORTEX_UI_NETSPEED_TTL", "60"))
_NETSPEED_TIMEOUT_S = 5.0
_netspeed_cache: Dict[str, Any] = {"at": 0.0, "data": None}


def _netspeed_fetch(url: str, timeout: float):
    """Open the probe URL — split out so tests fake the network.

    macOS python.org builds don't see the system CA store, so bare
    urllib fails every https handshake with CERTIFICATE_VERIFY_FAILED
    (caught live by Pedro, s74 — the chip showed 'erro' with a healthy
    connection). The rest of Cortex never hit this because it talks
    through `requests`, which bundles certifi; do the same here."""
    import ssl
    import urllib.request
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:  # certifi gone → system default, better than nothing
        ctx = None
    req = urllib.request.Request(url, headers={"User-Agent": "cortex-ui"})
    return urllib.request.urlopen(  # noqa: S310 — fixed https URL
        req, timeout=timeout, context=ctx)


def net_speed_snapshot(
    lease: Optional[Dict[str, Any]] = None,
    fetcher: Optional[Any] = None,
) -> Dict[str, Any]:
    """Latency + throughput estimate for the dashboard chip.

    Returns {status: "ok"|"offline"|"error", latency_ms, mbps,
    measured_at}. latency is time-to-first-byte of the probe request;
    mbps is the body read throughput — a floor estimate, honest enough
    for "is my internet fine", not a speedtest.

    `lease`/`fetcher` are test hooks (explicit args bypass the cache,
    mirroring services_snapshot). The OFFLINE branch is checked on
    every call and never cached: a lease granted between ticks must
    measure on the very next one — and a lease REVOKED between ticks
    must stop showing a number sourced from traffic we could no longer
    have generated."""
    info: Dict[str, Any] = {
        "status": "offline",
        "latency_ms": None,
        "mbps": None,
        "measured_at": None,
    }
    use_cache = lease is None and fetcher is None
    if lease is None:
        lease = lease_snapshot()
    if not lease.get("active"):
        return info

    if use_cache:
        cached = _netspeed_cache["data"]
        if cached is not None and (
                time.monotonic() - _netspeed_cache["at"]) < _NETSPEED_TTL_S:
            return cached

    # The probe is about to leave the machine — breadcrumb first, so a
    # failed fetch is still visible in the audit (the weather lesson).
    try:
        from src.core.audit import audit
        audit.log("–", "background_read", "net_speed",
                  payload={"url": _NETSPEED_URL})
    except Exception:  # noqa: BLE001 — audit must never break the dashboard
        pass

    fetch = fetcher or _netspeed_fetch
    try:
        t0 = time.perf_counter()
        with fetch(_NETSPEED_URL, _NETSPEED_TIMEOUT_S) as resp:
            t1 = time.perf_counter()  # headers in → TTFB
            body = resp.read()
            t2 = time.perf_counter()
        info["status"] = "ok"
        info["latency_ms"] = int((t1 - t0) * 1000)
        read_s = t2 - t1
        if body and read_s > 0:
            info["mbps"] = round(len(body) * 8 / read_s / 1_000_000, 1)
        info["measured_at"] = datetime.now().strftime("%H:%M:%S")
    except Exception:  # noqa: BLE001 — degrade honest, never a number
        info["status"] = "error"
        info["measured_at"] = datetime.now().strftime("%H:%M:%S")

    # Errors are cached too: a dead probe host must not be re-hit on
    # every 5s tick for the whole TTL window.
    if use_cache:
        _netspeed_cache["data"] = info
        _netspeed_cache["at"] = time.monotonic()
    return info


def build_snapshot() -> Dict[str, Any]:
    """Single entry point — assembles everything for the template."""
    return {
        "host": host_snapshot(),
        "lease": lease_snapshot(),
        "net": net_speed_snapshot(),
        "audit": audit_snapshot(),
        "studier": studier_snapshot(),
        "services": services_snapshot(),
        "cost": cost_snapshot(),
        "voice": voice_snapshot(),
        "memory": memory_snapshot(),
    }
