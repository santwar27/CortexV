"""Browser audio stays local; no real decoder, Whisper, or orchestrator."""
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.ui import routes


def test_pyav_dependency_is_pinned():
    requirements = Path(__file__).resolve().parents[1] / "requirements.txt"
    pins = [line.split("#", 1)[0].strip()
            for line in requirements.read_text(encoding="utf-8").splitlines()]
    assert "av==17.0.1" in pins


@pytest.fixture
def voice(monkeypatch, tmp_path):
    monkeypatch.setenv("CORTEX_API_KEY", "test-fake")
    cache = tmp_path / "model"
    cache.mkdir()
    (cache / "tokenizer.json").write_text("{}")
    state = SimpleNamespace(audio=np.zeros(16000), lang="pt", paths=[], cache=cache)
    state.decode = Mock(side_effect=lambda path, **kw: decode(path, **kw))

    def decode(path, sampling_rate):
        assert sys.modules["src.voice.stt"]._lock.locked()
        assert sampling_rate == 16000
        assert Path(path).read_bytes() == state.body
        return state.audio

    def transcribe(audio, **kwargs):
        assert audio is state.audio

        def segments():
            assert sys.modules["src.voice.stt"]._lock.locked()
            yield SimpleNamespace(text=" private transcript ")

        return segments(), SimpleNamespace(language=kwargs["language"] or state.lang)

    state.model = Mock(transcribe=Mock(side_effect=transcribe))
    state.load = Mock(return_value=state.model)
    state.resolve = Mock(return_value=str(cache))
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(
        WhisperModel=state.load, decode_audio=state.decode))
    monkeypatch.setitem(sys.modules, "faster_whisper.utils", SimpleNamespace(
        download_model=state.resolve))
    stt = sys.modules.get("src.voice.stt")
    if stt is not None:
        monkeypatch.setattr(stt, "_model", None)
    original_temp = tempfile.NamedTemporaryFile

    def temporary(*args, **kwargs):
        file = original_temp(*args, dir=tmp_path, **kwargs)
        state.paths.append(Path(file.name))
        return file

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", temporary)
    state.audit = Mock()
    monkeypatch.setattr(routes.audit, "log", state.audit)

    class ForbiddenOrchestrator:
        def __getattribute__(self, name):
            raise AssertionError("Transcription must not touch the orchestrator")

    app = FastAPI()
    app.state.orchestrator = ForbiddenOrchestrator()
    app.include_router(routes.router)
    with TestClient(app) as state.client:
        yield state
    assert "private transcript" not in str(state.audit.call_args_list)


def post(voice, body=b"blob", content_type="audio/webm;codecs=opus", auth=True):
    voice.body = body
    headers = {"Content-Type": content_type}
    if auth:
        headers["X-Cortex-Key"] = "test-fake"
    response = voice.client.post("/ui/voice/transcribe", content=body, headers=headers)
    assert all(not path.exists() for path in voice.paths)
    return response


def test_requires_auth(voice):
    assert post(voice, auth=False).status_code == 401
    assert not voice.paths
    voice.decode.assert_not_called()
    voice.load.assert_not_called()


@pytest.mark.parametrize("content_type", ["application/json", "video/webm", ""])
def test_requires_audio_type(voice, content_type):
    assert post(voice, content_type=content_type).status_code == 415
    assert not voice.paths
    voice.decode.assert_not_called()


def test_body_limit(voice):
    assert post(voice, body=b"x" * (10 * 1024 * 1024 + 1)).status_code == 413
    voice.decode.assert_not_called()
    voice.load.assert_not_called()


def test_decoded_duration_limit(voice):
    voice.audio = np.zeros(60 * 16000 + 1)
    response = post(voice)
    assert response.status_code == 413
    assert response.json() == {"error": "too_long"}
    voice.load.assert_not_called()
    assert voice.paths


def test_bad_audio(voice):
    voice.decode.side_effect = ValueError("invalid media")
    response = post(voice)
    assert response.status_code == 400
    assert response.json() == {"error": "bad_audio"}
    voice.load.assert_not_called()
    assert voice.paths


@pytest.mark.parametrize("failure", ["load", "inference", "tokenizer"])
def test_unavailable(voice, failure):
    if failure == "tokenizer":
        (voice.cache / "tokenizer.json").unlink()
    else:
        target = voice.load if failure == "load" else voice.model.transcribe
        target.side_effect = RuntimeError("unavailable private transcript")
    response = post(voice)
    assert response.status_code == 503
    assert response.json() == {"error": "stt_unavailable"}
    assert voice.paths
    if failure == "tokenizer":
        voice.load.assert_not_called()


@pytest.mark.parametrize("lang,passes", [("pt", [None]), ("en", [None]),
                                        ("de", [None, "pt"])])
def test_success_language_settings_and_reuse(voice, lang, passes):
    voice.lang = lang
    voice.audio = np.zeros(60 * 16000)  # exactly 60 seconds is accepted
    for _ in range(2):
        response = post(voice, content_type="audio/mp4")
        assert response.status_code == 200
        expected_lang = lang if lang in {"pt", "en"} else "pt"
        assert response.json() == {"text": "private transcript", "lang": expected_lang}
    voice.load.assert_called_once_with(
        "small", device="cpu", compute_type="int8", local_files_only=True)
    voice.resolve.assert_called_once_with("small", local_files_only=True)
    calls = voice.model.transcribe.call_args_list
    assert [call.kwargs["language"] for call in calls] == passes * 2
    for call in calls:
        assert call.kwargs["beam_size"] == int(os.getenv("CORTEX_STT_BEAM", "5"))
        assert call.kwargs["vad_filter"] is True
    assert voice.audit.call_count == 2
    for call in voice.audit.call_args_list:
        assert set(call.kwargs["payload"]) == {"duration_s", "lang"}
        assert call.kwargs["payload"]["lang"] == expected_lang
        assert call.kwargs["payload"]["duration_s"] >= 0
    assert voice.paths


def test_empty_text_is_allowed(voice):
    voice.model.transcribe.side_effect = lambda *a, **kw: (
        iter([]), SimpleNamespace(language="en"))
    response = post(voice)
    assert response.status_code == 200
    assert response.json() == {"text": "", "lang": "en"}
