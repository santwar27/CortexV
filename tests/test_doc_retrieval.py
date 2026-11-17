"""Choosing WHICH part of a long document the model sees (s83).

Clipping to the first 6.000 characters is honest and dumb: on Pedro's
punctuation worksheet it lost every question past page two, so "qual a
resposta da questão 19?" could not be answered from a document that
contains it. These tests pin the three outcomes and, above all, that the
failure mode is the old clip rather than an exception in the middle of a
turn.
"""
import numpy as np
import pytest

from src.core.doc_retrieval import chunk_text, select_relevant
from src.core.orchestrator import _partial_read_note


def _fake_embed(text: str):
    """A stand-in for bge-m3: one dimension per marker word."""
    low = (text or "").lower()
    return np.asarray([low.count("alpha"), low.count("beta"),
                       low.count("gama")], dtype=np.float32)


def _doc(paragraphs=30, marker_at=25, marker="beta"):
    parts = []
    for i in range(paragraphs):
        word = marker if i == marker_at else "alpha"
        parts.append(f"Paragraph {i}. " + f"{word} " * 40)
    return "\n\n".join(parts)


class TestChunking:

    def test_splits_a_long_document(self):
        chunks = chunk_text(_doc())
        assert len(chunks) > 1
        assert all(c.strip() for c in chunks)

    def test_respects_the_size_roughly(self):
        for chunk in chunk_text(_doc(), size=500, overlap=50):
            assert len(chunk) <= 500 + 200          # overlap + a paragraph tail

    def test_a_wall_of_text_with_no_paragraphs_still_splits(self):
        wall = "Uma frase qualquer. " * 400          # no blank lines at all
        chunks = chunk_text(wall, size=600, overlap=60)
        assert len(chunks) > 3

    def test_short_text_is_one_chunk(self):
        assert chunk_text("uma linha só") == ["uma linha só"]


class TestSelection:

    def test_a_short_document_is_passed_whole(self):
        text, how = select_relevant("curto", "pergunta", budget=6000,
                                    embed=_fake_embed)
        assert (text, how) == ("curto", "whole")

    def test_the_relevant_passage_is_found_deep_in_the_document(self):
        """The bug this exists for: the answer sits past the clip."""
        doc = _doc(paragraphs=30, marker_at=25)
        clipped = doc[:2000]
        assert "beta" not in clipped                 # the old behaviour missed it
        text, how = select_relevant(doc, "beta?", budget=2000, embed=_fake_embed)
        assert how == "retrieved"
        assert "beta" in text
        assert len(text) <= 2000

    def test_passages_come_back_in_reading_order(self):
        doc = "\n\n".join(f"Paragraph {i}. " + ("beta " if i in (2, 20) else "alpha ") * 40
                          for i in range(30))
        text, how = select_relevant(doc, "beta?", budget=3000, embed=_fake_embed)
        assert how == "retrieved"
        assert text.index("Paragraph 2.") < text.index("Paragraph 20.")

    def test_a_gap_is_marked_so_two_passages_do_not_read_as_one(self):
        doc = "\n\n".join(f"Paragraph {i}. " + ("beta " if i in (1, 25) else "alpha ") * 40
                          for i in range(30))
        text, _ = select_relevant(doc, "beta?", budget=3000, embed=_fake_embed)
        assert "[…]" in text

    def test_without_a_question_it_clips(self):
        doc = _doc()
        text, how = select_relevant(doc, None, budget=1500, embed=_fake_embed)
        assert how == "clipped" and text == doc[:1500]

    def test_an_embedding_failure_falls_back_to_the_clip(self):
        """Ollama's embedding endpoint answered 500 twice this week; a turn
        must not die because of it."""
        def boom(_text):
            raise RuntimeError("embeddings down")
        doc = _doc()
        text, how = select_relevant(doc, "beta?", budget=1500, embed=boom)
        assert how == "clipped" and text == doc[:1500]

    def test_it_can_be_switched_off(self, monkeypatch):
        monkeypatch.setenv("CORTEX_DOC_RETRIEVAL", "0")
        doc = _doc()
        text, how = select_relevant(doc, "beta?", budget=1500, embed=_fake_embed)
        assert how == "clipped"

    def test_a_document_is_not_re_embedded_per_question(self):
        """Pedro asks four things about the same PDF."""
        calls = []

        def counting(text):
            calls.append(text)
            return _fake_embed(text)
        doc = _doc(paragraphs=40, marker_at=30)
        select_relevant(doc, "beta?", budget=2000, embed=counting)
        first = len(calls)
        select_relevant(doc, "beta again?", budget=2000, embed=counting)
        # second pass embeds the QUESTION only
        assert len(calls) == first + 1


class TestTheUserIsTold:

    def test_whole_says_nothing(self):
        assert _partial_read_note("whole") == ""

    def test_retrieved_says_it_searched(self):
        assert "busquei" in _partial_read_note("retrieved")
        assert "searched" in _partial_read_note("retrieved", "en")

    def test_clipped_says_it_read_only_the_start(self):
        assert "primeira parte" in _partial_read_note("clipped")
        assert "only the first part" in _partial_read_note("clipped", "en")
