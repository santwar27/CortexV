"""
Cortex doctor — one-shot health check of the local stack.

Run:  venv/bin/python -m src.doctor   (or `make doctor`)

Prints one line per subsystem with a status of OK / WARN / FAIL, then a
summary. Exit code is 0 unless something is FAIL — a WARN is a
degraded-but-usable state (e.g. an optional integration not configured, or
the voice daemon intentionally off), not a reason to fail a script.

This is the design-reference ch. 16 "doctor command" idea, adapted to Cortex's ACTUAL
stack rather than the book's generic list:

  - Ollama (the local model server) reachable + the configured models pulled
  - Palace (structured + semantic memory on one SQLite file) integrity
  - Gate queue (background-safe S3 action queue) readable + pending count
  - Embeddings (nomic-embed-text) answering a live round-trip
  - Voice (custom wake model shipped + listener running, supervised)
  - API server (FastAPI/uvicorn) responding

There is deliberately NO ChromaDB check: Palace replaced it. The point of a
doctor is to mirror what the system depends on today, so it stays honest as
the architecture moves.

Security: this NEVER prints secret values — only whether a var is set.
"""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

from dotenv import load_dotenv


# Status constants. Ordered by severity so we can pick the worst one for the
# process exit code.
OK = "OK"
WARN = "WARN"
FAIL = "FAIL"
_SEVERITY = {OK: 0, WARN: 1, FAIL: 2}

# ANSI colours, disabled when stdout isn't a TTY (e.g. piped into a log) so
# the output stays grep-friendly.
_USE_COLOR = sys.stdout.isatty()
_COLORS = {OK: "\033[32m", WARN: "\033[33m", FAIL: "\033[31m"}
_RESET = "\033[0m"


def _paint(status: str) -> str:
    if not _USE_COLOR:
        return status
    return f"{_COLORS.get(status, '')}{status}{_RESET}"


@dataclass
class Result:
    name: str
    status: str
    detail: str


REPO_ROOT = Path(__file__).resolve().parent.parent


# --- individual checks -----------------------------------------------------
# Each check is a no-argument function returning a Result. Checks must never
# raise: a broken subsystem is the thing we're here to report, so they catch
# their own failures and turn them into FAIL.


def check_env() -> Result:
    """Required vs recommended env vars. Names only — never values."""
    required = ["CORTEX_API_KEY"]
    recommended = ["TAVILY_API_KEY", "SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET"]

    missing_req = [v for v in required if not os.getenv(v)]
    missing_rec = [v for v in recommended if not os.getenv(v)]

    if missing_req:
        return Result("env", FAIL, f"missing required: {', '.join(missing_req)}")
    if missing_rec:
        return Result("env", WARN, f"missing optional: {', '.join(missing_rec)}")
    return Result("env", OK, "required + optional vars set")


def _ollama_models() -> List[str]:
    """Tag-bearing model names Ollama has pulled, e.g. 'cortex-lite:latest'."""
    import ollama

    resp = ollama.list()
    out: List[str] = []
    for m in resp.get("models", []):
        name = m.get("model") or m.get("name")
        if name:
            out.append(name)
    return out


def _full_name(name: str) -> str:
    """'cortex-lite' → 'cortex-lite:latest'; a tag given is kept. Ollama's
    own default — the only normalisation: two tags of one base are two
    models (s88, Codex lote 10)."""
    return name if ":" in name else f"{name}:latest"


def _pin_key(name: str) -> str:
    """The baseline's key: ':latest' stays implicit, as it always was in
    model_digests.json, and any other tag is spelled out."""
    full = _full_name(name)
    return full[: -len(":latest")] if full.endswith(":latest") else full


def _has_model(pulled: List[str], configured: str) -> bool:
    """'cortex-lite' matches 'cortex-lite:latest'. Matching on the base alone
    reported 'cortex-lite:seguro' present when only ':latest' was pulled."""
    want = _full_name(configured)
    return any(_full_name(p) == want for p in pulled)


def check_ollama() -> Result:
    """Is the model server up and reachable?"""
    try:
        models = _ollama_models()
    except Exception as e:  # noqa: BLE001 — any failure means "not reachable"
        return Result(
            "ollama", FAIL,
            f"unreachable (is `ollama serve` running?): {type(e).__name__}",
        )
    return Result("ollama", OK, f"reachable · {len(models)} models pulled")


def check_models() -> Result:
    """Are the models Cortex is configured to use actually pulled?

    Analytical is core (no analytical model → Cortex can't think). Creative,
    vision and the embedder are degraded-mode WARNs."""
    try:
        pulled = _ollama_models()
    except Exception:  # noqa: BLE001 — ollama check already reported the why
        return Result("models", FAIL, "can't list models (Ollama unreachable)")

    analytical = os.getenv("CORTEX_MODEL_ANALYTICAL", "cortex-lite")
    optional = {
        "creative": os.getenv("CORTEX_MODEL_CREATIVE", "cortex-criativo"),
        "vision": os.getenv("CORTEX_MODEL_VISION", "llava"),
        "embed": os.getenv("CORTEX_EMBED_MODEL", "nomic-embed-text"),
    }

    if not _has_model(pulled, analytical):
        return Result("models", FAIL, f"analytical model '{analytical}' not pulled")

    missing_opt = [f"{role}:{name}" for role, name in optional.items()
                   if not _has_model(pulled, name)]
    if missing_opt:
        return Result("models", WARN, f"not pulled — {', '.join(missing_opt)}")
    return Result("models", OK, f"analytical '{analytical}' + creative/vision/embed present")


def _ollama_model_digests() -> dict:
    """name → content digest for every pulled model (from ollama.list)."""
    import ollama

    out: dict = {}
    for m in ollama.list().get("models", []):
        name = m.get("model") or m.get("name")
        digest = m.get("digest")
        if name and digest:
            out[name] = digest
    return out


def _digest_baseline_path() -> Path:
    return Path(
        os.getenv("CORTEX_MODEL_DIGEST_FILE",
                  str(Path.home() / ".cortex" / "model_digests.json"))
    )


def check_model_digests() -> Result:
    """Trust-on-first-use digest pin for the models Cortex uses
    (supply-chain hygiene — [[security-scan-local-ai]] item 4). Ollama models
    are GGUF (no pickle → no arbitrary-code-execution on load), so this is a
    LIGHT guard against a silently swapped or re-pulled model — the scan's
    'pin by hash, don't trust the name alone' — not a critical control.

    First run records the current digests as the baseline; later runs WARN if
    a pinned model's digest changed or the model went missing. Fail-soft: any
    error degrades to WARN, never crashes the doctor."""
    import json

    configured = {
        os.getenv("CORTEX_MODEL_ANALYTICAL", "cortex-lite"),
        os.getenv("CORTEX_MODEL_CREATIVE", "cortex-criativo"),
        os.getenv("CORTEX_MODEL_VISION", "llava"),
        os.getenv("CORTEX_EMBED_MODEL", "nomic-embed-text"),
    }
    try:
        current_all = _ollama_model_digests()
    except Exception:  # noqa: BLE001 — ollama check already reported the why
        return Result("model_digests", WARN, "can't read digests (Ollama unreachable)")

    # Resolve each configured name to ITS pulled digest — the full name, with
    # no tag meaning :latest. By base alone the first tag listed won: with
    # cortex-lite:seguro configured and :latest listed first, :seguro could be
    # swapped with no warning (s88, Codex lote 10, reproduced). Not pulled →
    # skipped; check_models flags those.
    pulled_digests = {_full_name(n): d for n, d in current_all.items()}
    pinned: dict = {}
    for name in configured:
        full = _full_name(name)
        if full in pulled_digests:
            pinned[_pin_key(full)] = pulled_digests[full]

    path = _digest_baseline_path()
    try:
        baseline = json.loads(path.read_text()) if path.exists() else None
    except Exception:  # noqa: BLE001 — corrupt baseline shouldn't crash doctor
        baseline = None
    if isinstance(baseline, dict):
        baseline = {_pin_key(k): v for k, v in baseline.items()}

    if baseline is None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(pinned, indent=2, sort_keys=True))
        except Exception:  # noqa: BLE001
            return Result("model_digests", WARN, "couldn't write digest baseline")
        return Result("model_digests", OK,
                      f"baseline recorded for {len(pinned)} model(s) (first run)")

    changed = [b for b, d in pinned.items()
               if b in baseline and baseline[b] != d]
    missing = [b for b in baseline if b not in pinned]
    if changed or missing:
        parts = []
        if changed:
            parts.append("digest changed: " + ", ".join(sorted(changed)))
        if missing:
            parts.append("no longer pulled: " + ", ".join(sorted(missing)))
        return Result("model_digests", WARN,
                      "; ".join(parts) + " — verify you re-pulled it on purpose")
    return Result("model_digests", OK, f"{len(pinned)} model digest(s) match baseline")


def check_palace() -> Result:
    """Palace DB exists, passes SQLite integrity_check, and reports counts."""
    db = REPO_ROOT / "data" / "palace.db"
    if not db.exists():
        return Result("palace", FAIL, f"{db.relative_to(REPO_ROOT)} not found")
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                return Result("palace", FAIL, f"integrity_check: {integrity}")
            facts = conn.execute(
                "SELECT COUNT(*) FROM triplas WHERE valid_until IS NULL"
            ).fetchone()[0]
            docs = conn.execute("SELECT COUNT(*) FROM documentos").fetchone()[0]
            events = conn.execute("SELECT COUNT(*) FROM eventos").fetchone()[0]
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001
        return Result("palace", FAIL, f"{type(e).__name__}: {e}")
    return Result("palace", OK, f"integrity ok · {facts} facts, {docs} docs, {events} events")


def check_gate_queue() -> Result:
    """Background-safe action queue. The JSONL
    file may not exist (nothing was ever queued) — that's healthy. Pending
    actions are a WARN, not an error: they're parked waiting for your review
    (`make gate`). A read/parse failure of the log is a FAIL."""
    try:
        from src.core.gate_queue import gate_queue

        pending = gate_queue.pending()
    except Exception as e:  # noqa: BLE001
        return Result("gate_queue", FAIL, f"{type(e).__name__}: {e}")
    n = len(pending)
    if n == 0:
        return Result("gate_queue", OK, "no actions awaiting approval")
    return Result("gate_queue", WARN, f"{n} action(s) awaiting approval — run `make gate`")


def check_embeddings() -> Result:
    """Live round-trip through the embedder — catches a missing or corrupt
    embed model that a simple 'is it pulled' check would miss."""
    try:
        from src.memory.embeddings import OllamaEmbedder

        vec = OllamaEmbedder().embed("cortex doctor ping")
        dims = len(vec)
        if dims <= 0:
            return Result("embeddings", FAIL, "embedder returned an empty vector")
    except Exception as e:  # noqa: BLE001
        return Result("embeddings", FAIL, f"{type(e).__name__}: {e}")
    return Result("embeddings", OK, f"live round-trip · {dims} dims")


def check_voice_model() -> Result:
    """Is the trained 'Hey Cortex' wake model shipped? Without it the
    listener falls back to the builtin 'Hey Jarvis'."""
    onnx = REPO_ROOT / "src" / "voice" / "models" / "hey_cortex.onnx"
    if onnx.exists():
        kb = onnx.stat().st_size / 1024
        return Result("voice.model", OK, f"hey_cortex.onnx present ({kb:.0f} KB)")
    return Result("voice.model", WARN, "hey_cortex.onnx missing — falls back to hey_jarvis")


# The argv patterns scripts/cortex-voice-start.sh guards on (s88). The old
# bare "src.voice.listener" also matched any command naming
# src/voice/listener.py (an editor, a `git add`) and counted it as a second
# listener. tests/test_voice_supervisor.py pins these to the script's copy.
LISTENER_PAT = r" -m src\.voice\.listener$"
SUPERVISOR_PAT = r"^/bin/sh .*/cortex-voice-supervise\.sh$"


def _pgrep(pattern: str) -> List[str]:
    """PIDs whose full argv matches `pattern` (pgrep -f; none → [])."""
    proc = subprocess.run(["pgrep", "-f", pattern],
                          capture_output=True, text=True, timeout=5)
    return proc.stdout.split()


def check_voice_daemon() -> Result:
    """Is the voice listener running, and supervised? Optional — Pedro may
    run Cortex text-only, so a stopped daemon is a WARN, not a FAIL. A
    listener with no scripts/cortex-voice-supervise.sh above it is a WARN
    too: it hears, but a crash leaves Cortex deaf (s88)."""
    try:
        listeners = _pgrep(LISTENER_PAT)
        supervisors = _pgrep(SUPERVISOR_PAT)
    except Exception as e:  # noqa: BLE001
        return Result("voice.daemon", WARN, f"could not check ({type(e).__name__})")
    if listeners:
        extra = f", {len(listeners)} procs" if len(listeners) > 1 else ""
        if supervisors:
            return Result("voice.daemon", OK,
                          f"listener running (pid {listeners[0]}{extra}), supervised")
        return Result("voice.daemon", WARN,
                      f"listener running (pid {listeners[0]}{extra}) with no supervisor "
                      "— a crash leaves it deaf; restart it supervised: "
                      "pkill -f 'src.voice.listener'; sleep 1; "
                      "scripts/cortex-voice-start.sh")
    if supervisors:
        return Result("voice.daemon", WARN,
                      f"listener down, supervisor (pid {supervisors[0]}) restarting it "
                      "— see logs/voice.err.log")
    return Result("voice.daemon", WARN, "listener not running (text-only mode)")


def check_api() -> Result:
    """Is the FastAPI/uvicorn server responding? Voice + iPhone need it; the
    CLI doesn't — so a down server is a WARN."""
    import requests

    base = os.getenv("CORTEX_API_URL", "http://127.0.0.1:8000").rstrip("/")
    url = f"{base}/ui"
    try:
        resp = requests.get(url, timeout=3)
    except Exception as e:  # noqa: BLE001
        return Result("api", WARN, f"server not responding at {base} ({type(e).__name__})")
    # Any sub-500 response means uvicorn is alive and serving. A 401/403 is
    # the HEALTHY state: the server is up and the auth gate is enforcing.
    # Only a 5xx (server-side error) is worth flagging.
    if resp.status_code < 400:
        return Result("api", OK, f"responding at {base}")
    if resp.status_code < 500:
        return Result("api", OK, f"up at {base} (auth-protected, HTTP {resp.status_code})")
    return Result("api", WARN, f"{base} returned HTTP {resp.status_code}")


CHECKS: List[Callable[[], Result]] = [
    check_env,
    check_ollama,
    check_models,
    check_model_digests,
    check_palace,
    check_gate_queue,
    check_embeddings,
    check_voice_model,
    check_voice_daemon,
    check_api,
]


def run_checks() -> List[Result]:
    """Run every check in order. Each is self-contained, so one failure
    never aborts the rest — a doctor that stops at the first problem isn't
    much of a doctor."""
    results: List[Result] = []
    for check in CHECKS:
        try:
            results.append(check())
        except Exception as e:  # noqa: BLE001 — belt-and-suspenders
            results.append(Result(check.__name__, FAIL, f"check crashed: {e}"))
    return results


def main() -> int:
    load_dotenv()
    print("\nCortex doctor — local stack health\n")
    results = run_checks()
    width = max(len(r.name) for r in results)
    worst = OK
    for r in results:
        print(f"  [{_paint(r.status):>4}]  {r.name:<{width}}  {r.detail}")
        if _SEVERITY[r.status] > _SEVERITY[worst]:
            worst = r.status

    fails = sum(1 for r in results if r.status == FAIL)
    warns = sum(1 for r in results if r.status == WARN)
    print()
    if worst == FAIL:
        print(f"  {fails} FAIL, {warns} WARN — Cortex won't run correctly until fixed.")
        return 1
    if worst == WARN:
        print(f"  all core systems OK · {warns} WARN (degraded but usable).")
        return 0
    print("  all green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
