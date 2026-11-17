#!/usr/bin/env python3
"""Replay Pedro's real sentences through the semantic router — offline (s83).

Why this exists. The router has been in shadow mode since July and has
logged 56 events, the last on 24/08: waiting for live traffic to calibrate
it has been tried twice and produced no corpus in eight weeks. But the
corpus already exists — ~/.cortex/audit.jsonl holds every sentence Pedro
has actually said to Cortex, roughly 600 distinct ones. They were never
scored because the shadow only runs on the paths it is wired into.

This reads them, scores each one with `SemanticRouter.explain()`, and
reports where the bands fall. It EXECUTES NOTHING: no tool runs, no action
is staged, no state is touched. It only embeds text.

The labels are weak but real: the deterministic dispatch layer already
knows what it does with each sentence. So each utterance lands in one of
two buckets.

  ALREADY ROUTED — a regex handles it today. Here the router can be
  checked: agreeing is evidence it understands the intent, disagreeing is
  a confusion we can name.

  CHAT TODAY — no regex fires, so Cortex talks. Anything the router would
  route here is either a rescue (a phrasing the regexes miss — the whole
  point of the feature) or a false positive. Only Pedro can tell those
  apart, so they are printed for review, highest score first.

Usage:
    venv/bin/python scripts/router_replay.py [--limit N] [--top N]
    venv/bin/python scripts/router_replay.py --sweep        # thresholds
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

# The .env decides which embedder runs (CORTEX_EMBED_MODEL=bge-m3). Without
# it this script silently calibrated against nomic-embed-text while
# production used bge-m3 — different vectors, different thresholds, a useless
# answer delivered confidently.
load_dotenv(PROJECT_ROOT / ".env")

AUDIT = Path(os.getenv("CORTEX_AUDIT_LOG",
                       str(Path.home() / ".cortex" / "audit.jsonl")))


def harvest(path: Path) -> List[str]:
    """Every sentence the user actually said, de-duplicated, order kept."""
    seen: Dict[str, None] = {}
    if not path.exists():
        return []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            payload = row.get("payload")
            if not isinstance(payload, dict):
                continue
            stage, event = row.get("stage"), row.get("event")
            if stage == "orchestrator" and event == "request":
                text = payload.get("input")
            elif stage == "reflex":
                text = payload.get("input")
            elif stage == "voice" and event in ("transcribed", "transcribed_extra"):
                text = payload.get("text")
            else:
                continue
            if not isinstance(text, str):
                continue
            text = text.strip()
            # Control characters are terminal noise, not speech; one- and
            # two-character fragments carry no intent to route.
            if len(text) < 4 or not any(ch.isalpha() for ch in text):
                continue
            seen.setdefault(text, None)
    return list(seen)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="score only the first N")
    ap.add_argument("--top", type=int, default=20, help="how many to print per list")
    ap.add_argument("--sweep", action="store_true", help="threshold sweep table")
    args = ap.parse_args()

    from evals.dispatch import simulate_dispatch
    from src.core.reflex import reflex_layer
    from src.core.semantic_router import SemanticRouter
    from src.memory.embeddings import OllamaEmbedder

    utterances = harvest(AUDIT)
    if args.limit:
        utterances = utterances[:args.limit]
    if not utterances:
        print(f"nada pra replayar em {AUDIT}")
        return 1

    embedder = OllamaEmbedder()
    print(f"corpus: {len(utterances)} falas distintas de {AUDIT}")
    print(f"embedder: {embedder.model}\n", flush=True)

    router = SemanticRouter(embedder.embed)

    rows: List[Tuple[str, str, object]] = []   # (text, dispatch_mode, decision)
    for i, text in enumerate(utterances, 1):
        if i % 50 == 0:
            print(f"  … {i}/{len(utterances)}", flush=True)
        try:
            # The reflex answers BEFORE the router is ever consulted, so a
            # sentence it owns can never be a router false positive however
            # badly the router scores it. Counting those as errors is how a
            # calibration ends up solving a problem that does not exist.
            if reflex_layer.try_respond(text) is not None:
                mode = "REFLEX"
            else:
                mode = simulate_dispatch(text).mode
        except Exception:
            mode = "?"
        rows.append((text, mode, router.explain(text)))

    reflex = [r for r in rows if r[1] == "REFLEX"]
    already = [r for r in rows if r[1] not in ("OFFLINE", "REFLEX", "?")]
    chat = [r for r in rows if r[1] == "OFFLINE"]

    print(f"\n{'='*72}\nONDE AS FALAS CAEM HOJE")
    print(f"  respondidas pelo reflexo (nunca chegam ao router): {len(reflex)}")
    print(f"  já roteadas por regex : {len(already)}")
    print(f"  conversa (sem regex)  : {len(chat)}  ← só estas alcançam o router")

    agree = [r for r in already if r[2].mode == r[1]]
    disagree = [r for r in already if r[2].mode != r[1]]
    print(f"\nNAS JÁ ROTEADAS, o router concorda com a regex em "
          f"{len(agree)}/{len(already)}")
    if disagree:
        print("  discordâncias mais confiantes (regex → router):")
        for text, mode, d in sorted(disagree, key=lambda r: -r[2].score)[:args.top]:
            print(f"    {d.score:.3f}  {mode:16} → {str(d.mode):16} {text[:46]!r}")

    would_route = [r for r in chat if r[2].accepted]
    print(f"\nNA CONVERSA, o router ROTEARIA {len(would_route)}/{len(chat)} "
          f"com os limiares de hoje "
          f"(score≥{router.min_score}, margem≥{router.min_margin})")
    for text, _m, d in sorted(would_route, key=lambda r: -r[2].score)[:args.top]:
        print(f"    {d.score:.3f} m={d.margin:.3f}  {str(d.mode):16} {text[:46]!r}")

    print("\n  as mais próximas de passar (recusadas, maiores scores):")
    near = sorted((r for r in chat if not r[2].accepted),
                  key=lambda r: -r[2].score)[:args.top]
    for text, _m, d in near:
        print(f"    {d.score:.3f} m={d.margin:.3f}  {str(d.mode):16} "
              f"[{d.reason}] {text[:40]!r}")

    by_reason = Counter(r[2].reason or "accepted" for r in rows)
    print("\nMOTIVOS:", dict(by_reason))

    if args.sweep:
        print(f"\n{'='*72}\nVARREDURA DE LIMIAR  (margem fixa em {router.min_margin})")
        print("  score   concorda/já-roteadas   rotearia-conversa")
        for step in range(50, 86, 2):
            floor = step / 100.0
            ok = sum(1 for r in already
                     if r[2].score >= floor and r[2].margin >= router.min_margin
                     and r[2].mode == r[1])
            fp = sum(1 for r in chat
                     if r[2].score >= floor and r[2].margin >= router.min_margin)
            print(f"  {floor:.2f}    {ok:>5}/{len(already):<10}        {fp}")

    modes = Counter(str(r[2].mode) for r in chat if r[2].accepted)
    if modes:
        print("\nÍMÃS DE FALSO POSITIVO (modos que mais puxam conversa):",
              dict(modes.most_common(8)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
