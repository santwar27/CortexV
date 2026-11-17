"""
Wake-word config resolver.

One place that decides which openWakeWord model the listener (and any
smoke script) should load. Reads `CORTEX_WAKE_MODEL_PATH` from env:

  - **set** → use the custom `.onnx` at that path (e.g. the
    `hey_cortex.onnx` trained via Colab and dropped in
    `src/voice/models/`). openWakeWord uses the file basename (without
    `.onnx`) as the score-dict key, so the lookup key is derived from
    the path.
  - **unset / empty** → fall back to the built-in pre-trained
    `hey_jarvis` model bundled in the openwakeword package. This is the
    Etapa 9 placeholder ("Hey Jarvis" voice trigger) that ships until
    the custom Cortex wake word is trained.

The resolver is a pure function so it can be unit-tested without
loading the heavyweight openwakeword runtime.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class WakeConfig:
    """Resolved wake-word setup the listener consumes.

    - `model_spec`: the value passed to `WakeModel(wakeword_models=[...])`.
      A path for custom models, a builtin name otherwise.
    - `lookup_key`: the key to use when reading the score out of
      `wake.predict(frame)` — openWakeWord keys its output dict by the
      model basename without extension.
    - `phrase_hint`: human-readable phrase shown to the user at startup
      ("Say 'Hey Cortex' ..." vs "Say 'Hey Jarvis' ...").
    - `source`: 'env' when a custom path is in use, 'builtin' otherwise.
      Useful for logging and for tests to assert behaviour.
    """
    model_spec: str
    lookup_key: str
    phrase_hint: str
    source: str


# Built-in pre-trained model name (openwakeword ships the .onnx inside
# its package). Kept as a constant so the listener doesn't need to know.
_BUILTIN_NAME = "hey_jarvis"
_BUILTIN_PHRASE = "Hey Jarvis"

# Phrase shown when a custom model is loaded. Hardcoded to "Hey Cortex"
# because that's the only custom model the project trains today; if
# multiple customs ever ship, derive this from a manifest beside the
# .onnx instead of guessing from the filename.
_CUSTOM_PHRASE = "Hey Cortex"

# Default detection threshold for the wake score (openWakeWord scores are
# in [0, 1]). The custom hey_cortex model trains conservative — high
# precision (~0 false positives/hour) but modest recall — so in practice
# it's run at a lower threshold (~0.35) to recover recall, which the ~0
# FP rate leaves headroom for. Override via CORTEX_WAKE_THRESHOLD.
_DEFAULT_THRESHOLD = 0.5


def resolve_wake_threshold(env_value: str | None = None) -> float:
    """Decide the wake detection threshold.

    Reads `CORTEX_WAKE_THRESHOLD` from env (or an explicit override for
    tests). Empty / non-numeric / out-of-range values fall back to the
    default so a stray `export CORTEX_WAKE_THRESHOLD=` or a typo can't
    silently set a threshold of 0.0 (fires on everything) or break the
    listener at startup."""
    raw = env_value if env_value is not None else os.environ.get("CORTEX_WAKE_THRESHOLD", "")
    raw = (raw or "").strip()
    if not raw:
        return _DEFAULT_THRESHOLD
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_THRESHOLD
    # openWakeWord scores live in (0, 1]; reject anything outside so a
    # bad value degrades to the safe default rather than a dead listener.
    if value <= 0.0 or value > 1.0:
        return _DEFAULT_THRESHOLD
    return value


def resolve_wake_config(env_path: str | None = None) -> WakeConfig:
    """Decide which wake model to load.

    `env_path` is an explicit override mostly for tests; production
    callers pass nothing and the resolver reads `CORTEX_WAKE_MODEL_PATH`
    from `os.environ`. An empty / whitespace-only value is treated as
    unset so a stray `export CORTEX_WAKE_MODEL_PATH=` in a shell
    doesn't silently break the fallback."""
    raw = env_path if env_path is not None else os.environ.get("CORTEX_WAKE_MODEL_PATH", "")
    path = (raw or "").strip()
    if not path:
        return WakeConfig(
            model_spec=_BUILTIN_NAME,
            lookup_key=_BUILTIN_NAME,
            phrase_hint=_BUILTIN_PHRASE,
            source="builtin",
        )
    base = os.path.basename(path)
    key = os.path.splitext(base)[0] or base
    return WakeConfig(
        model_spec=path,
        lookup_key=key,
        phrase_hint=_CUSTOM_PHRASE,
        source="env",
    )
