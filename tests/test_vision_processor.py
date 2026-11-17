"""
M4-A5: the VisionProcessor request was the ONE call in the repo without a
timeout — a hung Llava (the likeliest Ollama hang on the M1: big model,
memory-pressure swap) pinned the turn's thread forever. These tests pin
the timeout ceiling, the speakable timeout message, and the env-driven
host (hardcoded localhost ignored OLLAMA_HOST).
"""
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.tools.vision_processor import VisionProcessor


class TestTimeout:
    def test_post_carries_timeout(self):
        vp = VisionProcessor()
        with patch("src.tools.vision_processor.requests.post") as post:
            post.return_value = MagicMock(
                json=lambda: {"response": "a cat"},
                raise_for_status=lambda: None,
            )
            assert vp.analyze_image("what is it?", "b64data") == "a cat"
        timeout = post.call_args.kwargs.get("timeout")
        assert timeout is not None, "request must have a ceiling (M4-A5)"
        connect, read = timeout
        assert connect <= 10
        assert read <= 600

    def test_timeout_is_speakable_not_raised(self):
        vp = VisionProcessor()
        with patch("src.tools.vision_processor.requests.post",
                   side_effect=requests.exceptions.ReadTimeout("slow llava")):
            reply = vp.analyze_image("describe", "b64data")
        assert isinstance(reply, str)
        assert "didn't answer in time" in reply
        # The [SYSTEM ERROR] marker keeps the reply out of memory
        # (CortexMemory._ERROR_MARKERS) — same contract as the generic path.
        assert reply.startswith("[SYSTEM ERROR]")

    def test_connect_error_still_soft(self):
        vp = VisionProcessor()
        with patch("src.tools.vision_processor.requests.post",
                   side_effect=requests.exceptions.ConnectionError("down")):
            reply = vp.analyze_image("describe", "b64data")
        assert reply.startswith("[SYSTEM ERROR]")


class TestHostFromEnv:
    def test_default_localhost(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        assert VisionProcessor().api_url == "http://localhost:11434/api/generate"

    def test_env_host_with_scheme(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "http://studio.local:11434/")
        assert VisionProcessor().api_url == "http://studio.local:11434/api/generate"

    def test_env_host_bare(self, monkeypatch):
        # Ollama's own convention allows host:port without a scheme.
        monkeypatch.setenv("OLLAMA_HOST", "192.168.1.50:11434")
        assert VisionProcessor().api_url == "http://192.168.1.50:11434/api/generate"
