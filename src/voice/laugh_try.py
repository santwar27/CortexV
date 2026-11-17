"""Audition local laughs: python -m src.voice.laugh_try [--lang pt|en].

Uses exported TTS configuration only; never loads dotenv, the listener,
microphones, wake/STT models or the API. No downloads or network calls.
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
import wave
from pathlib import Path

from src.voice.tts_config import resolve_tts_config


CANDIDATES = {
    "pt": ("Há, há!", "Ha ha.", "Rá, rá!", "He, he.", "Hahaha!"),
    "en": ("Ha, ha!", "Ha ha.", "Ha!", "Heh heh.", "Hahaha!"),
}
_VOICES: dict = {}


def _voice(model_path):
    if model_path not in _VOICES:
        from piper import PiperVoice

        _VOICES[model_path] = PiperVoice.load(model_path)
    return _VOICES[model_path]


def synth_piper(model_path, text, out_path, *, voice_loader=_voice):
    """Listener and audition share local synthesis and say fallback semantics."""
    try:
        voice = voice_loader(model_path)
        with wave.open(out_path, "wb") as wf:
            voice.synthesize_wav(text, wf)
        return True
    except Exception as exc:
        print(f"  [tts: piper synth failed ({exc}) → say fallback]")
        return False


def prepare_tts(text, lang, config, wav_path, *, synth=synth_piper):
    """Prepare the same TTS command for streamed sentences and auditions.

    Playback timing stays with the caller (the listener double-buffers).
    """
    if config.engine == "piper":
        if synth(config.piper_model_for(lang), text, wav_path):
            return ["afplay", wav_path]
    return ["say", "-v", config.say_voice_for(lang), text]


def main(argv=None, *, speak=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=("pt", "en"), default="pt")
    args = parser.parse_args(argv)

    def audition(speaker):
        for candidate in CANDIDATES[args.lang]:
            print(candidate, flush=True)
            speaker(candidate, args.lang)

    if speak is not None:  # Offline tests: no configuration or TTS reads.
        audition(speak)
        return
    config = resolve_tts_config()
    with tempfile.TemporaryDirectory(prefix="cortex-laugh-") as folder:
        wav_path = str(Path(folder) / "laugh.wav")

        def local_speak(text, lang):
            subprocess.run(prepare_tts(text, lang, config, wav_path), check=True)

        audition(local_speak)


if __name__ == "__main__":
    main()
