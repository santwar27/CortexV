"""
wake_config.resolve_wake_config — env-driven model resolver.

Lives without openwakeword installed at test time: the resolver is a
pure function over a string. The contract here is what the listener and
the smoke script rely on tomorrow when the custom hey_cortex.onnx drops.
"""
import pytest

from src.voice.wake_config import resolve_wake_config, resolve_wake_threshold


class TestBuiltinFallback:
    def test_unset_env_uses_hey_jarvis(self, monkeypatch):
        monkeypatch.delenv("CORTEX_WAKE_MODEL_PATH", raising=False)
        cfg = resolve_wake_config()
        assert cfg.model_spec == "hey_jarvis"
        assert cfg.lookup_key == "hey_jarvis"
        assert cfg.phrase_hint == "Hey Jarvis"
        assert cfg.source == "builtin"

    def test_empty_env_treated_as_unset(self, monkeypatch):
        # `export CORTEX_WAKE_MODEL_PATH=` in a shell sets an empty value
        # — must NOT silently break the fallback.
        monkeypatch.setenv("CORTEX_WAKE_MODEL_PATH", "")
        cfg = resolve_wake_config()
        assert cfg.source == "builtin"
        assert cfg.model_spec == "hey_jarvis"

    def test_whitespace_only_env_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv("CORTEX_WAKE_MODEL_PATH", "   ")
        assert resolve_wake_config().source == "builtin"

    def test_explicit_none_uses_env(self, monkeypatch):
        # Passing None to env_path falls through to os.environ, not to
        # the builtin shortcut.
        monkeypatch.setenv("CORTEX_WAKE_MODEL_PATH", "src/voice/models/hey_cortex.onnx")
        cfg = resolve_wake_config(env_path=None)
        assert cfg.source == "env"


class TestCustomModel:
    def test_path_drives_spec_and_key(self):
        cfg = resolve_wake_config(env_path="src/voice/models/hey_cortex.onnx")
        assert cfg.model_spec == "src/voice/models/hey_cortex.onnx"
        # openwakeword keys its predict() output by the basename without
        # extension — this is the exact lookup the listener will do.
        assert cfg.lookup_key == "hey_cortex"
        assert cfg.phrase_hint == "Hey Cortex"
        assert cfg.source == "env"

    def test_basename_only_path(self):
        # A bare filename (no dir) still resolves cleanly.
        cfg = resolve_wake_config(env_path="hey_cortex.onnx")
        assert cfg.model_spec == "hey_cortex.onnx"
        assert cfg.lookup_key == "hey_cortex"

    def test_no_extension_falls_back_to_basename(self):
        # Defensive: a path with no extension shouldn't yield an empty
        # lookup_key (which would silently always read 0.0 from predict).
        cfg = resolve_wake_config(env_path="models/some_wake_model")
        assert cfg.lookup_key == "some_wake_model"

    @pytest.mark.parametrize("path", [
        "/abs/path/hey_cortex.onnx",
        "./rel/hey_cortex.onnx",
        "src/voice/models/hey_cortex.onnx",
    ])
    def test_various_path_shapes_resolve(self, path):
        cfg = resolve_wake_config(env_path=path)
        assert cfg.lookup_key == "hey_cortex"
        assert cfg.source == "env"

    def test_explicit_path_wins_over_env(self, monkeypatch):
        # An explicit env_path arg ignores whatever is in os.environ —
        # this is what the smoke script uses when run with `--model`.
        monkeypatch.setenv("CORTEX_WAKE_MODEL_PATH", "src/voice/models/other.onnx")
        cfg = resolve_wake_config(env_path="src/voice/models/hey_cortex.onnx")
        assert cfg.lookup_key == "hey_cortex"


class TestThreshold:
    def test_unset_env_is_default(self, monkeypatch):
        monkeypatch.delenv("CORTEX_WAKE_THRESHOLD", raising=False)
        assert resolve_wake_threshold() == 0.5

    def test_empty_env_is_default(self, monkeypatch):
        # `export CORTEX_WAKE_THRESHOLD=` must not become 0.0 (fires on
        # everything) — fall back to the safe default.
        monkeypatch.setenv("CORTEX_WAKE_THRESHOLD", "")
        assert resolve_wake_threshold() == 0.5

    def test_valid_value_parsed(self):
        # The custom hey_cortex model is run lower than default to
        # recover its conservative recall.
        assert resolve_wake_threshold(env_value="0.35") == 0.35

    def test_whitespace_trimmed(self):
        assert resolve_wake_threshold(env_value="  0.4  ") == 0.4

    def test_non_numeric_is_default(self):
        assert resolve_wake_threshold(env_value="loud") == 0.5

    @pytest.mark.parametrize("bad", ["0", "0.0", "-0.2", "1.5", "2"])
    def test_out_of_range_is_default(self, bad):
        # Scores live in (0, 1]; anything outside degrades to the
        # default rather than a dead or trigger-happy listener.
        assert resolve_wake_threshold(env_value=bad) == 0.5

    def test_upper_bound_one_is_allowed(self):
        assert resolve_wake_threshold(env_value="1.0") == 1.0
