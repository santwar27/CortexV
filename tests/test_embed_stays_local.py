"""What is embedded never leaves the machine (SOBERANIA 1 — s88, Codex lote 11).

The embedder carries the Palace's memory and, in the job ranker, Pedro's whole
résumé. CORTEX_EMBED_URL pointed at another host sent it there (reproduced
with the transport stubbed — no network). It is refused unless another
machine is chosen on purpose with CORTEX_ALLOW_REMOTE_EMBED=1.
"""
import pytest
import requests

from src.memory.embeddings import OllamaEmbedder


class _Ok:
    def raise_for_status(self):
        pass

    def json(self):
        return {"embedding": [0.1, 0.2]}


@pytest.fixture
def sent(monkeypatch):
    calls = []
    monkeypatch.setattr(requests, "post",
                        lambda url, json=None, timeout=None: calls.append(url) or _Ok())
    monkeypatch.delenv("CORTEX_ALLOW_REMOTE_EMBED", raising=False)
    return calls


@pytest.mark.parametrize("url", ["https://servidor-remoto.example/api/embeddings",
                                 "http://192.168.1.20:11434/api/embeddings",
                                 "http://localhost.evil.com/api/embeddings"])
def test_a_remote_url_is_refused_before_anything_is_sent(sent, url):
    with pytest.raises(RuntimeError, match="non-local"):
        OllamaEmbedder(url=url).embed("CURRICULO: Pedro …")
    assert sent == []


@pytest.mark.parametrize("url", ["http://localhost:11434/api/embeddings",
                                 "http://127.0.0.1:11434/api/embeddings",
                                 "http://[::1]:11434/api/embeddings"])
def test_the_local_ollama_still_works(sent, url):
    assert OllamaEmbedder(url=url).embed("texto").tolist() == pytest.approx([0.1, 0.2])
    assert sent == [url]


def test_another_machine_only_on_purpose(sent, monkeypatch):
    monkeypatch.setenv("CORTEX_ALLOW_REMOTE_EMBED", "1")
    OllamaEmbedder(url="http://192.168.1.20:11434/api/embeddings").embed("texto")
    assert sent == ["http://192.168.1.20:11434/api/embeddings"]
