"""
Smoke test for the implicit preference engine (Tier 2.3) against a live
Ollama instance.

Pedro runs this manually with Ollama up. It exists because every unit
test mocks the LLM, so we haven't actually seen what Llama 3.1 8B Q4
produces against real pt-BR conversation chunks. This script seeds a
throwaway Palace, drives the Studier's preference extractor, and prints
the full output for visual calibration.

Run from repo root:
  venv/bin/python scripts/smoke_preferences.py

Look for:
  - Are the extracted predicado slugs reasonable (lowercase, prefixed
    `prefere_`, semantic)?
  - Is the confidence distribution sensible — i.e. NOT all clustered at
    0.95+ regardless of input strength? If everything is ≥0.95, the
    0.55 floor in preferences.py is effectively inert and we should
    raise it.
  - Does the USER PROFILE block render cleanly in the system prompt?

Nothing here is persistent — the Palace lives at /tmp/cortex_smoke_palace.db
and gets removed on each run. Your real Palace at data/palace.db is
untouched.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

# Make `src.*` importable when running from repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ollama  # noqa: E402

from src.core.memory import CortexMemory  # noqa: E402
from src.memory.palace import Palace  # noqa: E402
from src.memory.preferences import (  # noqa: E402
    extract_preferences,
    persist_preferences,
    recent_conversations,
)

SMOKE_DB = "/tmp/cortex_smoke_palace.db"
MODEL = os.getenv("CORTEX_MODEL_ANALYTICAL", "cortex-lite")

# Sample conversation turns the user might have had over a day. Mix of
# explicit preferences, a one-off request that should be SKIPPED, and a
# factual question that should produce zero preferences.
SEEDS = [
    "User: prefiro respostas curtas e diretas, sem rodeio\n"
    "Cortex: Entendido.",

    "User: sempre trabalho de madrugada, costumo dormir às 6 da manhã\n"
    "Cortex: OK, vou levar isso em conta nos horários sugeridos.",

    "User: pode falar em português comigo, prefiro\n"
    "Cortex: Combinado, vou responder em pt-br.",

    "User: toca uma música do Matuê agora\n"  # one-off, should be skipped
    "Cortex: Tocando BACKSTAGE de Matuê.",

    "User: o que é GGUF?\n"  # factual question, no preference
    "Cortex: GGUF é um formato de arquivo para LLMs quantizados...",
]


def main() -> int:
    # Fresh start each run.
    for ext in ("", "-wal", "-shm"):
        path = SMOKE_DB + ext
        if os.path.exists(path):
            os.remove(path)

    print(f"=== Cortex preference smoke ({datetime.now().isoformat(timespec='seconds')}) ===")
    print(f"Model: {MODEL}")
    print(f"Palace: {SMOKE_DB}")
    print()

    p = Palace(db_path=SMOKE_DB)
    now_iso = datetime.now(timezone.utc).isoformat()
    for i, text in enumerate(SEEDS):
        p.add_document("conversation", text, created_at=now_iso)
        print(f"  seed[{i}]: {text.splitlines()[0][:80]}")

    print()
    print("--- Building corpus for extraction ---")
    corpus = recent_conversations(p, hours=24, max_chars=8000)
    print(f"corpus length: {len(corpus)} chars")
    print()

    print("--- Calling extractor (Ollama, may take a few seconds) ---")
    triples = extract_preferences(corpus, llm_chat_fn=ollama.chat, model=MODEL)
    print(f"got {len(triples)} candidate triple(s):")
    for t in triples:
        print(f"  {t.predicado:35s} → {t.objeto:25s} "
              f"(conf={t.confidence:.2f}) evidence={t.evidence!r}")
    print()

    print("--- Persisting (showing ALL, no floor) ---")
    # No floor so we can SEE confidence distribution. The real path
    # applies the production floor.
    written = persist_preferences(p, triples, min_confidence=0.0)
    print(f"persisted: {written} row(s)")
    print()

    print("--- All preferences in Palace (min_confidence=0) ---")
    for row in p.get_preferences(min_confidence=0.0):
        print(f"  {row['predicado']:35s} → {row['objeto']:25s} "
              f"conf={row['confidence']:.2f}  source={row['source']}")
    print()

    print("--- USER PROFILE block as it would land in the system prompt ---")
    mem = CortexMemory(palace=p)
    profile = mem.get_user_profile()
    if profile:
        print(profile.rstrip())
    else:
        print("(empty — no preference cleared the production confidence floor)")
    print()

    print("=== Calibration questions ===")
    print(
        "1. Are the slugs sensible (lowercase, prefixed `prefere_`)?\n"
        "2. Are confidences spread, or all >= 0.95? (If all >= 0.95,\n"
        "   raise the floor in preferences.py.)\n"
        "3. Is the one-off Matuê request correctly excluded?\n"
        "4. Is the factual GGUF question correctly excluded?\n"
        "5. Does the USER PROFILE block render with the expected\n"
        "   3 stable preferences?"
    )

    p.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
