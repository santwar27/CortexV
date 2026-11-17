"""Single source of truth for Ollama's `keep_alive` and `think` values.

Ollama's `keep_alive` is last-call-wins: any `ollama.generate` /
`ollama.chat` WITHOUT the kwarg resets the model's unload timer to the
default 5 min. The interactive engine sets 30m (`2aa9b11`), but every
other caller — the realtime classifier, the planner's decompose/summarise,
the nightly Studier, the deep/dream scaffolds — used the bare API and
silently downgraded that timer, so the warm-model win was intermittent.

The fix (review's second option, kept for test-surface reasons: dozens of
tests patch `<module>.ollama.<fn>`) is to pass `keep_alive=KEEP_ALIVE` at
each call site rather than wrap the call. This module holds the shared values
those sites — and engine.py — import, so each knob lives in exactly one
place. THINK follows the same per-call-site pattern; the one-word realtime
classifier always disables thinking. `tests/test_ollama_client.py` pins both
keywords at every site.
"""
from __future__ import annotations

import os

# "30m" balances snappiness vs RAM on a 16 GB Mac; "-1" pins forever, "5m"
# restores Ollama's default. Read once at import (env knob CORTEX_KEEP_ALIVE).
KEEP_ALIVE = os.getenv("CORTEX_KEEP_ALIVE", "30m")

# Thinking mode. Gemma 4+ ships a native chain-of-thought "thinking" channel
# that Ollama turns ON by default for capable models. Cortex keeps it OFF: the
# orchestrator routes by deterministic reflex/JSON, not CoT, and a thinking
# block would burn num_ctx + num_predict and leak raw reasoning into the reply
# (and the TTS). `think=False` is a safe no-op on models without the capability
# (Llama 3.1 ignores it). This single flag is the seam the future "deep mode"
# toggle flips to True per query; env-tunable so experiments need no code edit.
THINK = os.getenv("CORTEX_THINK", "0").strip().lower() in ("1", "true", "yes", "on")
