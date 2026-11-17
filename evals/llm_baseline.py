"""Baseline storage for LLM behaviour evals (fase 2, sessão B).

A baseline is a snapshot of what each LLM case replied at a known-good
moment, keyed by the exact `(model, prompt_version)` that produced it.
Later runs compare against it (cosine similarity, see `similarity.py`)
to surface drift — the answer changed shape since we last looked.

Key decisions (pre-approved, see HANDOFF "Decisões fase 2"):
  - ONE file per (model, prompt_version):
        evals/baselines/<model>__v<N>.json
    so swapping model or bumping the prompt yields a NEW file and the
    old one survives for A/B comparison (Llama vs Gemma, v1 vs v2).
  - Regenerated ONLY by explicit `make eval-llm-rebaseline`. Never
    written as a side effect of a normal eval run — a baseline is a
    deliberate "this is correct now" act, not an accident.
  - The files ARE committed: `git diff` on a rebaseline is the human-
    readable record of how answers shifted between versions.

Model-agnostic: nothing here knows or assumes Llama. The model name is
whatever the engine reported (from CORTEX_MODEL).
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union

# evals/baselines/ — sibling of this module.
BASELINE_DIR = Path(__file__).resolve().parent / "baselines"

# Filenames embed model + version; sanitise so an exotic model name
# (e.g. a registry path with slashes) can't escape the directory.
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_model(model: str) -> str:
    cleaned = _SAFE.sub("-", model.strip()).strip("-")
    return cleaned or "unknown"


def _safe_version(prompt_version: Union[int, str]) -> str:
    """Filename-safe version segment. Ints render plainly (1 -> 'v1'); a
    dormant named prompt (e.g. 'frontier', M2-B2) is sanitised like the
    model name so the swap's opt-in prompt can be baselined too."""
    cleaned = _SAFE.sub("-", str(prompt_version).strip()).strip("-")
    return cleaned or "unknown"


def baseline_path(model: str, prompt_version: Union[int, str], base_dir: Optional[Path] = None) -> Path:
    """Resolve the baseline file for a (model, prompt_version) pair.

    `base_dir` override exists for tests (tmp_path) — production passes
    nothing and gets the package's baselines/ directory.
    """
    root = base_dir or BASELINE_DIR
    return root / f"{_safe_model(model)}__v{_safe_version(prompt_version)}.json"


def save_baseline(
    model: str,
    prompt_version: Union[int, str],
    replies: Dict[str, str],
    base_dir: Optional[Path] = None,
) -> Path:
    """Write `replies` (case name -> reply text) as the baseline for the
    given pair. Overwrites the file for THIS pair (that's what an explicit
    rebaseline means); a different pair is a different file, untouched.

    Cases are sorted by name so the JSON diff is stable across runs.
    """
    path = baseline_path(model, prompt_version, base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "model": model,
        "prompt_version": prompt_version,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cases": {name: replies[name] for name in sorted(replies)},
    }
    path.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def load_baseline(
    model: str,
    prompt_version: Union[int, str],
    base_dir: Optional[Path] = None,
) -> Optional[Dict[str, str]]:
    """Return the case->reply mapping for the pair, or None if no baseline
    has been recorded yet (first run on a new model/prompt). Callers treat
    None as "no comparison available", not an error."""
    path = baseline_path(model, prompt_version, base_dir)
    if not path.exists():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    cases = doc.get("cases", {})
    return {str(k): str(v) for k, v in cases.items()}


def replies_from_results(results: List) -> Dict[str, str]:
    """Pull case-name -> reply text out of EvalResult payloads. Only
    LLM_BEHAVIOR results carry a 'reply'; others are skipped."""
    out: Dict[str, str] = {}
    for r in results:
        reply = r.payload.get("reply")
        if reply is not None:
            out[r.case.name] = reply
    return out
