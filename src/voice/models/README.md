# Custom wake-word models

Holds custom-trained openWakeWord `.onnx` files. Today the only one is
`hey_cortex.onnx` — the project's custom wake word, **trained 2026-05-31**
on a rented NVIDIA GPU via the [`briankelley/atlas-voice-training`](https://github.com/briankelley/atlas-voice-training)
pipeline (the David Scripka Colab notebook is dead in 2026 — see
`docs/wake_word_training_plan.md` for the full story and the runbook).

The model is **English-only** ("Hey Cortex"): the synthetic-positive
generator ships no PT-BR voice, so "Hei Córtex" can't be trained this
way. Pedro speaks EN-UK daily, so EN-only is the practical v1. PT-BR is
deferred (training plan §6).

## Why here (and not `data/voice/`)

`data/voice/` is gitignored and reserved for **third-party** models
that are downloaded on demand and large (Whisper, Piper). The custom
`hey_cortex.onnx` is small (~200 KB), identity-tied to this project,
and only reproducible with a paid GPU training run. So it lives **in
the repo, committed**, so a fresh clone of Cortex works out of the box.

## How to activate (no code change)

`scripts/cortex-voice-start.sh` already wires this up: if the `.onnx`
is present it exports `CORTEX_WAKE_MODEL_PATH` (and a tuned
`CORTEX_WAKE_THRESHOLD`) before launching the listener. So a normal
start uses the custom model automatically.

Manual / one-off activation:

```bash
export CORTEX_WAKE_MODEL_PATH=src/voice/models/hey_cortex.onnx
export CORTEX_WAKE_THRESHOLD=0.35   # see below
./scripts/cortex-voice-start.sh     # or restart the listener
```

Unset (or empty) `CORTEX_WAKE_MODEL_PATH` → the listener falls back to
the built-in `hey_jarvis` model bundled with openwakeword. That's the
rollback path if the custom model ever misbehaves.

## Detection threshold

The model trained **conservative**: accuracy 0.77, **recall ~0.40**,
**0.0 false positives/hour**. In plain terms — it almost never fires by
mistake, but at the default 0.5 threshold it can miss a "Hey Cortex".
Because the false-positive rate is ~0, there's headroom to lower the
threshold and recover recall. `CORTEX_WAKE_THRESHOLD` (resolved in
`wake_config.py`, default 0.5) controls this; the launcher defaults it
to **0.35**. Tune up toward 0.5 if you ever get false fires, down toward
0.3 if it's missing you. Out-of-range / non-numeric values fall back to
0.5.

## Lookup-key contract

openWakeWord uses the file basename (without `.onnx`) as the score-dict
key in `model.predict(frame)`. So `hey_cortex.onnx` ⇒ key `hey_cortex`.
The resolver derives this automatically — no manual config.

## Smoke-test the model

```bash
CORTEX_WAKE_MODEL_PATH=src/voice/models/hey_cortex.onnx \
CORTEX_WAKE_THRESHOLD=0.35 \
    ./venv/bin/python -m src.voice.wake_test
```

Prints rolling confidence + a banner on each detection. Test "Hey
Cortex" near and far from the mic, plus music / background noise as a
false-positive check.
