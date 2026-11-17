"""Speech newline boundaries preserve punctuation and existing list pauses."""
import pytest

from src.core.text_utils import strip_markdown_for_speech


@pytest.mark.parametrize("text,expected", [
    (
        "A última resposta usou 2.913 tokens de entrada e 148 de saída, "
        "em 2 chamadas ao modelo cortex-lite.\n"
        "A contagem cobre apenas as chamadas ao modelo registradas pelo código.",
        "A última resposta usou 2.913 tokens de entrada e 148 de saída, "
        "em 2 chamadas ao modelo cortex-lite. "
        "A contagem cobre apenas as chamadas ao modelo registradas pelo código.",
    ),
    ("Não sei essa parte.\nHoje eu rodo…", "Não sei essa parte. Hoje eu rodo…"),
    ("primeira\nsegunda", "primeira, segunda"),
    ("- primeira\n- segunda\n- terceira", "primeira, segunda, terceira"),
    ("* primeira\n* segunda", "primeira, segunda"),
    ("+ primeira\n+ segunda", "primeira, segunda"),
    ("1. primeira\n2. segunda", "1. primeira, 2. segunda"),
    ("primeira\n\nsegunda", "primeira. segunda"),
    ("primeira.\n\nsegunda", "primeira.. segunda"),
    ("# Título\n**forte** e _ênfase_\n[link](https://example.test)",
     "Título, forte e ênfase, link"),
])
def test_speech_newline_boundaries(text, expected):
    assert strip_markdown_for_speech(text) == expected


@pytest.mark.parametrize("punctuation", [".", "!", "?", "…", ":", ";"])
@pytest.mark.parametrize("trailing", ["", " \t"])
def test_sentence_punctuation_needs_no_comma(punctuation, trailing):
    assert strip_markdown_for_speech(f"Primeira{punctuation}{trailing}\nSegunda") == (
        f"Primeira{punctuation} Segunda")
