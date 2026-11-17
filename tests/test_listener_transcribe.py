"""
listener.transcribe — language guard.

Whisper auto-detect sometimes mislabels short PT audio (Brazilian proper
nouns) as German/Polish; Cortex is PT/EN only, so a non-pt/en guess
triggers a forced-PT re-transcription. Whisper is mocked — no model, no
audio hardware. (The listener imports fine in this env; the rest of it
stays live-only.)
"""
import numpy as np

from src.voice import listener


class _Seg:
    def __init__(self, text):
        self.text = text


class _Info:
    def __init__(self, language, prob):
        self.language = language
        self.language_probability = prob


class _FakeWhisper:
    """transcribe() yields a scripted (text, lang, prob) per call."""
    def __init__(self, *results):
        self._results = list(results)
        self.calls = []  # records the `language` kwarg of each call

    def transcribe(self, audio, beam_size=1, vad_filter=True, language=None):
        self.calls.append(language)
        text, lang, prob = self._results[len(self.calls) - 1]
        return ([_Seg(text)], _Info(lang, prob))


def _audio():
    return np.zeros(1600, dtype=np.int16)


def test_pt_accepted_no_retry():
    w = _FakeWhisper(("toca matuê", "pt", 0.9))
    text, lang, prob = listener.transcribe(_audio(), w)
    assert (text, lang) == ("toca matuê", "pt")
    assert w.calls == [None]  # auto-detect only, no retry


def test_en_accepted_no_retry():
    w = _FakeWhisper(("play queen", "en", 0.95))
    text, lang, _ = listener.transcribe(_audio(), w)
    assert (text, lang) == ("play queen", "en")
    assert len(w.calls) == 1


def test_exotic_language_retries_as_pt():
    # Auto-detect → Polish gibberish; forced-PT retry → clean PT.
    w = _FakeWhisper(
        ("matuje niespoczywanie", "pl", 0.4),
        ("toca matuê no spotify", "pt", 0.8),
    )
    text, lang, _ = listener.transcribe(_audio(), w)
    assert text == "toca matuê no spotify"
    assert lang == "pt"
    assert w.calls == [None, "pt"]  # auto first, then forced pt


def test_german_misdetect_retries_as_pt():
    w = _FakeWhisper(
        ("Matui und Spotify", "de", 0.5),
        ("matuê no spotify", "pt", 0.7),
    )
    text, lang, _ = listener.transcribe(_audio(), w)
    assert lang == "pt"
    assert w.calls[1] == "pt"


def test_retry_result_returned_even_if_still_not_pt_en():
    # Defensive: we don't loop forever — the forced-PT result is returned
    # as-is even if Whisper somehow still reports an odd language.
    w = _FakeWhisper(
        ("???", "pl", 0.3),
        ("ainda estranho", "pl", 0.3),
    )
    text, lang, _ = listener.transcribe(_audio(), w)
    assert text == "ainda estranho"
    assert w.calls == [None, "pt"]  # exactly one retry, no infinite loop
