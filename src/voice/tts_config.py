"""
TTS engine config resolver.

One place that decides which text-to-speech backend the listener uses to
voice a reply, and which voice per language. Reads from env:

  - ``CORTEX_TTS`` = ``"piper"`` | ``"say"`` — pick the engine.
    **unset / empty** → auto: use Piper if both configured voice models
    exist on disk, else fall back to the macOS built-in ``say``. So just
    dropping the ``.onnx`` voices into ``data/voice/piper/`` is enough to
    switch over; deleting them (or ``CORTEX_TTS=say``) rolls back.
  - ``CORTEX_PIPER_VOICE_PT`` / ``CORTEX_PIPER_VOICE_EN`` — paths to the
    Piper ``.onnx`` voice models (PT-BR + EN). Defaults point at the
    faber / alan medium voices fetched into ``data/voice/piper/``.
  - ``CORTEX_VOICE_PT`` / ``CORTEX_VOICE_EN`` — macOS ``say`` voice names
    (Luciana / Daniel), used by the ``say`` engine and as the fallback.

Piper is local/offline (fits model-sovereignty — no cloud TTS) and gives
a far less robotic voice than ``say``. The resolver is a pure function
(filesystem probe injectable) so it can be unit-tested without loading
onnxruntime / the Piper runtime.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Mapping, Optional

# Default Piper voice model paths — the medium-tier faber (PT-BR) and
# alba (EN-GB) voices fetched from rhasspy/piper-voices. Live in
# data/voice/piper/ (gitignored, like the generic Whisper models — large
# and not identity-tied, unlike the small hey_cortex.onnx in models/).
# The user picked these by ear (session 28): faber for PT, alba for EN after
# A/B-ing the en_GB catalogue (alan, northern, vctk southern males, cori,
# alba, jenny, aru). alba is medium-tier; no high tier exists for it.
_DEFAULT_PIPER_PT = "data/voice/piper/pt_BR-faber-medium.onnx"
_DEFAULT_PIPER_EN = "data/voice/piper/en_GB-alba-medium.onnx"

# macOS `say` voice names — the previous engine, kept as the fallback.
_DEFAULT_SAY_PT = "Luciana"
_DEFAULT_SAY_EN = "Daniel"


@dataclass(frozen=True)
class TTSConfig:
    """Resolved TTS setup the listener consumes.

    - ``engine``: ``"piper"`` (neural, local) or ``"say"`` (macOS builtin).
    - ``piper_model_pt`` / ``piper_model_en``: voice model paths used when
      ``engine == "piper"``.
    - ``say_voice_pt`` / ``say_voice_en``: ``say -v`` voice names used when
      ``engine == "say"`` (and for per-sentence fallback if a Piper synth
      ever fails at runtime).
    - ``source``: how the engine was chosen — ``"piper-env"`` (CORTEX_TTS=
      piper), ``"piper-auto"`` (models present, no explicit env),
      ``"say-env"`` (CORTEX_TTS=say), ``"say-fallback"`` (Piper wanted but
      a model file is missing). Useful for startup logging + tests.
    """
    engine: str
    piper_model_pt: str
    piper_model_en: str
    say_voice_pt: str
    say_voice_en: str
    source: str

    def piper_model_for(self, lang: str) -> str:
        return self.piper_model_pt if lang == "pt" else self.piper_model_en

    def say_voice_for(self, lang: str) -> str:
        return self.say_voice_pt if lang == "pt" else self.say_voice_en


def resolve_tts_config(
    env: Optional[Mapping[str, str]] = None,
    exists: Callable[[str], bool] = os.path.exists,
) -> TTSConfig:
    """Decide the TTS engine + per-language voices.

    ``env`` defaults to ``os.environ``; ``exists`` is the filesystem probe
    (injectable so tests don't need real model files on disk). Selection:

      - ``CORTEX_TTS=say`` → ``say`` unconditionally (``say-env``).
      - ``CORTEX_TTS=piper`` → Piper if **both** voice models exist, else
        degrade to ``say`` (``say-fallback``) rather than crash the reply.
      - unset / empty / anything else → auto: Piper iff both models exist
        (``piper-auto``), else ``say`` (``say-fallback``).

    Both models are required for Piper so an EN reply can't hit a missing
    EN voice mid-conversation; a half-installed voice set degrades cleanly
    to ``say`` instead.
    """
    env = env if env is not None else os.environ

    piper_pt = (env.get("CORTEX_PIPER_VOICE_PT") or "").strip() or _DEFAULT_PIPER_PT
    piper_en = (env.get("CORTEX_PIPER_VOICE_EN") or "").strip() or _DEFAULT_PIPER_EN
    say_pt = (env.get("CORTEX_VOICE_PT") or "").strip() or _DEFAULT_SAY_PT
    say_en = (env.get("CORTEX_VOICE_EN") or "").strip() or _DEFAULT_SAY_EN

    requested = (env.get("CORTEX_TTS") or "").strip().lower()
    models_present = exists(piper_pt) and exists(piper_en)

    def _say(source: str) -> TTSConfig:
        return TTSConfig("say", piper_pt, piper_en, say_pt, say_en, source)

    def _piper(source: str) -> TTSConfig:
        return TTSConfig("piper", piper_pt, piper_en, say_pt, say_en, source)

    if requested == "say":
        return _say("say-env")
    if requested == "piper":
        return _piper("piper-env") if models_present else _say("say-fallback")
    # unset / empty / unrecognised → auto-detect from disk
    return _piper("piper-auto") if models_present else _say("say-fallback")
