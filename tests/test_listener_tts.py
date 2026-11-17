"""
Piper double-buffer + voice pre-load (M5-A8).

The synth gap at sentence boundaries came from strict serialisation:
_speak waited for the previous afplay BEFORE synthesising the next WAV
into a single shared path. The fix alternates two per-process WAVs so
synth can overlap playback. These tests pin the pieces that make that
safe — the alternation invariant, the voice cache, and the fail-soft
pre-load — with fakes only (no piper import, no audio hardware), same
posture as test_listener_recording.
"""
import os

import pytest

import src.voice.listener as l


class FakeVoice:
    def __init__(self):
        self.synths = []

    def synthesize_wav(self, text, wf):
        # Set the WAV params like the real PiperVoice does — wave.open
        # raises on close otherwise and _synth_piper would report False.
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        wf.writeframes(b"\x00\x00")
        self.synths.append(text)


class TestDoubleBufferPaths:
    def test_two_distinct_buffers(self):
        a, b = l._TTS_WAV_PATHS
        assert a != b

    def test_alternation_never_repeats_consecutively(self):
        # The invariant the double-buffer rests on: consecutive synths
        # never target the same file, so the WAV being written is never
        # the WAV afplay is playing (playback is serial via _wait_say).
        seq = [l._next_tts_wav() for _ in range(6)]
        for prev, cur in zip(seq, seq[1:]):
            assert prev != cur
        assert set(seq) == set(l._TTS_WAV_PATHS)


class TestPiperVoiceCache:
    @pytest.fixture(autouse=True)
    def clean_cache(self, monkeypatch):
        monkeypatch.setattr(l, "_PIPER_VOICES", {})

    def test_get_piper_voice_caches_by_path(self, monkeypatch):
        loads = []

        def fake_load(path):
            loads.append(path)
            return FakeVoice()

        import sys
        import types
        fake_mod = types.SimpleNamespace(PiperVoice=types.SimpleNamespace(load=fake_load))
        monkeypatch.setitem(sys.modules, "piper", fake_mod)

        v1 = l._get_piper_voice("model-pt.onnx")
        v2 = l._get_piper_voice("model-pt.onnx")
        assert v1 is v2
        assert loads == ["model-pt.onnx"]  # ~0.7s load paid exactly once

    def test_synth_piper_writes_via_cached_voice(self, tmp_path, monkeypatch):
        voice = FakeVoice()
        monkeypatch.setattr(l, "_PIPER_VOICES", {"m.onnx": voice})
        out = str(tmp_path / "out.wav")
        assert l._synth_piper("m.onnx", "olá", out) is True
        assert voice.synths == ["olá"]

    def test_synth_failure_returns_false(self, monkeypatch):
        class Broken:
            def synthesize_wav(self, text, wf):
                raise RuntimeError("boom")

        monkeypatch.setattr(l, "_PIPER_VOICES", {"m.onnx": Broken()})
        assert l._synth_piper("m.onnx", "olá", os.devnull) is False


class TestPreload:
    @pytest.fixture(autouse=True)
    def clean_cache(self, monkeypatch):
        monkeypatch.setattr(l, "_PIPER_VOICES", {})

    @staticmethod
    def _fake_config(engine: str):
        # TTS_CONFIG is a frozen dataclass — swap the whole object.
        import types
        return types.SimpleNamespace(
            engine=engine,
            piper_model_for=lambda lang: f"model-{lang}.onnx",
        )

    def test_preload_warms_pt_voice(self, monkeypatch):
        warmed = []
        monkeypatch.setattr(l, "TTS_CONFIG", self._fake_config("piper"))
        monkeypatch.setattr(l, "_get_piper_voice", lambda p: warmed.append(p))
        l.preload_piper_voices()
        assert warmed == ["model-pt.onnx"]  # PT only — EN stays lazy

    def test_preload_noop_when_engine_is_say(self, monkeypatch):
        monkeypatch.setattr(l, "TTS_CONFIG", self._fake_config("say"))
        monkeypatch.setattr(
            l, "_get_piper_voice",
            lambda p: (_ for _ in ()).throw(AssertionError("must not load")))
        l.preload_piper_voices()

    def test_preload_failure_is_soft(self, monkeypatch):
        # A missing model must not block boot: first turn just pays the
        # lazy load like before.
        monkeypatch.setattr(l, "TTS_CONFIG", self._fake_config("piper"))

        def boom(path):
            raise FileNotFoundError(path)

        monkeypatch.setattr(l, "_get_piper_voice", boom)
        l.preload_piper_voices()  # must not raise
