"""Local-only file transcription, independent of the microphone listener."""
import os
import threading

SAMPLE_RATE = 16000
WHISPER_MODEL_SIZE = "small"
STT_BEAM_SIZE = int(os.getenv("CORTEX_STT_BEAM", "5"))
_model = None
_lock = threading.Lock()


class BadAudio(Exception):
    pass


class TooLong(Exception):
    pass


class STTUnavailable(Exception):
    pass


def transcribe_file(path: str) -> tuple[str, str]:
    global _model
    # Covers loading AND consuming Whisper's lazy segments: one job at a time.
    with _lock:
        try:
            from faster_whisper import WhisperModel, decode_audio
            from faster_whisper.utils import download_model
        except Exception as exc:
            raise STTUnavailable from exc
        try:
            audio = decode_audio(path, sampling_rate=SAMPLE_RATE)
            if not len(audio):
                raise ValueError("No decoded samples")
        except Exception as exc:
            raise BadAudio from exc
        if len(audio) > 60 * SAMPLE_RATE:
            raise TooLong
        try:
            if _model is None:
                # Cache lookup only. This version of Whisper otherwise fetches
                # a fallback tokenizer even with local_files_only=True.
                cached = download_model(WHISPER_MODEL_SIZE, local_files_only=True)
                if not os.path.isfile(os.path.join(cached, "tokenizer.json")):
                    raise STTUnavailable
                _model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu",
                                      compute_type="int8", local_files_only=True)

            # Mirrors listener.transcribe's PT/EN rule, without recording.
            def run(language):
                segments, info = _model.transcribe(
                    audio, beam_size=STT_BEAM_SIZE, vad_filter=True, language=language)
                return " ".join(s.text for s in segments).strip(), info.language

            text, lang = run(None)
            return run("pt") if lang not in {"pt", "en"} else (text, lang)
        except Exception as exc:
            raise STTUnavailable from exc
