"""
tts_config.resolve_tts_config — env-driven TTS engine/voice resolver.

Pure function over an env mapping + a filesystem-probe callable, so it
runs without Piper / onnxruntime and without real model files on disk
(both injected). This is the contract the listener's _speak() relies on
to pick Piper vs the macOS `say` fallback.
"""
from src.voice.tts_config import (
    _DEFAULT_PIPER_EN,
    _DEFAULT_PIPER_PT,
    _DEFAULT_SAY_EN,
    _DEFAULT_SAY_PT,
    resolve_tts_config,
)

# Filesystem probes for injection.
_ALL_PRESENT = lambda _p: True       # noqa: E731
_NONE_PRESENT = lambda _p: False     # noqa: E731


def _present(*paths):
    paths = set(paths)
    return lambda p: p in paths


class TestAutoDetect:
    def test_unset_env_with_models_uses_piper(self):
        cfg = resolve_tts_config(env={}, exists=_ALL_PRESENT)
        assert cfg.engine == "piper"
        assert cfg.source == "piper-auto"

    def test_unset_env_without_models_falls_back_to_say(self):
        cfg = resolve_tts_config(env={}, exists=_NONE_PRESENT)
        assert cfg.engine == "say"
        assert cfg.source == "say-fallback"

    def test_unrecognised_value_treated_as_auto(self):
        cfg = resolve_tts_config(env={"CORTEX_TTS": "espeak"}, exists=_ALL_PRESENT)
        assert cfg.engine == "piper"
        assert cfg.source == "piper-auto"


class TestExplicitEngine:
    def test_say_forced_even_with_models(self):
        cfg = resolve_tts_config(env={"CORTEX_TTS": "say"}, exists=_ALL_PRESENT)
        assert cfg.engine == "say"
        assert cfg.source == "say-env"

    def test_say_is_case_insensitive(self):
        cfg = resolve_tts_config(env={"CORTEX_TTS": "  SAY "}, exists=_ALL_PRESENT)
        assert cfg.engine == "say"

    def test_piper_forced_with_models(self):
        cfg = resolve_tts_config(env={"CORTEX_TTS": "piper"}, exists=_ALL_PRESENT)
        assert cfg.engine == "piper"
        assert cfg.source == "piper-env"

    def test_piper_requested_but_models_missing_degrades(self):
        cfg = resolve_tts_config(env={"CORTEX_TTS": "piper"}, exists=_NONE_PRESENT)
        assert cfg.engine == "say"
        assert cfg.source == "say-fallback"


class TestBothModelsRequired:
    def test_only_pt_present_degrades_to_say(self):
        # An EN reply can't hit a missing EN voice mid-conversation — a
        # half-installed voice set degrades cleanly to `say`.
        cfg = resolve_tts_config(env={}, exists=_present(_DEFAULT_PIPER_PT))
        assert cfg.engine == "say"
        assert cfg.source == "say-fallback"

    def test_only_en_present_degrades_to_say(self):
        cfg = resolve_tts_config(env={}, exists=_present(_DEFAULT_PIPER_EN))
        assert cfg.engine == "say"

    def test_both_present_uses_piper(self):
        cfg = resolve_tts_config(
            env={}, exists=_present(_DEFAULT_PIPER_PT, _DEFAULT_PIPER_EN)
        )
        assert cfg.engine == "piper"


class TestVoiceOverrides:
    def test_custom_piper_paths_are_probed_and_carried(self):
        env = {
            "CORTEX_PIPER_VOICE_PT": "/custom/pt.onnx",
            "CORTEX_PIPER_VOICE_EN": "/custom/en.onnx",
        }
        cfg = resolve_tts_config(env=env, exists=_present("/custom/pt.onnx", "/custom/en.onnx"))
        assert cfg.engine == "piper"
        assert cfg.piper_model_pt == "/custom/pt.onnx"
        assert cfg.piper_model_en == "/custom/en.onnx"

    def test_default_paths_when_unset(self):
        cfg = resolve_tts_config(env={}, exists=_ALL_PRESENT)
        assert cfg.piper_model_pt == _DEFAULT_PIPER_PT
        assert cfg.piper_model_en == _DEFAULT_PIPER_EN

    def test_custom_say_voices(self):
        env = {"CORTEX_TTS": "say", "CORTEX_VOICE_PT": "Joana", "CORTEX_VOICE_EN": "Alex"}
        cfg = resolve_tts_config(env=env, exists=_NONE_PRESENT)
        assert cfg.say_voice_pt == "Joana"
        assert cfg.say_voice_en == "Alex"

    def test_default_say_voices(self):
        cfg = resolve_tts_config(env={"CORTEX_TTS": "say"}, exists=_NONE_PRESENT)
        assert cfg.say_voice_pt == _DEFAULT_SAY_PT
        assert cfg.say_voice_en == _DEFAULT_SAY_EN

    def test_empty_string_overrides_fall_back_to_defaults(self):
        env = {"CORTEX_PIPER_VOICE_PT": "  ", "CORTEX_VOICE_EN": ""}
        cfg = resolve_tts_config(env=env, exists=_ALL_PRESENT)
        assert cfg.piper_model_pt == _DEFAULT_PIPER_PT
        assert cfg.say_voice_en == _DEFAULT_SAY_EN


class TestModelSelectionHelpers:
    def test_piper_model_for_language(self):
        cfg = resolve_tts_config(env={}, exists=_ALL_PRESENT)
        assert cfg.piper_model_for("pt") == _DEFAULT_PIPER_PT
        assert cfg.piper_model_for("en") == _DEFAULT_PIPER_EN

    def test_say_voice_for_language(self):
        cfg = resolve_tts_config(env={"CORTEX_TTS": "say"}, exists=_NONE_PRESENT)
        assert cfg.say_voice_for("pt") == _DEFAULT_SAY_PT
        assert cfg.say_voice_for("en") == _DEFAULT_SAY_EN

    def test_non_pt_language_treated_as_en(self):
        # _speak only ever passes "pt" or "en"; any non-"pt" maps to EN.
        cfg = resolve_tts_config(env={}, exists=_ALL_PRESENT)
        assert cfg.piper_model_for("fr") == _DEFAULT_PIPER_EN
