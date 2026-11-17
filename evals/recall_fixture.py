"""Shared fixture for the golden-recall eval category.

Seeds a THROWAWAY Palace with a fully SYNTHETIC profile (no resemblance to any
real person's data) and exposes `build_memory(semantic=…)` so retrieval cases
can ask "does the right profile section enter the retrieved context?" — the
permanent guard for the s39 retrieval fix (personal-field recall 1/8→8/8).

Two modes, cached so the corpus is seeded once per process:
  • semantic=False — Palace with embed_fn=None → lexical-only (BM25/FTS5),
    fully deterministic, runs inside `make check` (EvalKind.RETRIEVAL).
  • semantic=True  — Palace with the real OllamaEmbedder → hybrid semantic
    retrieval; opt-in via `make eval-llm` (EvalKind.RETRIEVAL_SEMANTIC),
    because embeddings need Ollama and aren't bit-reproducible.

`polluted=True` (M3-A2) builds a SEPARATE Palace that replays the perverse
recall dynamic: the same profile corpus plus (a) a decoy knowledge section
sharing the passport keywords without the value (the mid-pool-ranking
dynamic from s39), and (b) repeated verbose past turns asking the passport
question — the echoes every successful recall leaves behind ("User: qual …
passaporte…" stored per turn; re-injection is user-side only, s40, so they
carry no value). Pre-M3-A2 this combination pushed the real value past the
total context cap — and nothing pinned that regression.

The synthetic corpus lives here so both the fixture and the cases reference the
same expected values from one place.
"""
from __future__ import annotations

import tempfile
from typing import Dict

# Each entry is one knowledge "section" (as KnowledgeBase stores them, tipo=
# 'knowledge'). Values are obviously fake test data — never real PII.
PROFILE_CORPUS = [
    "# Perfil — Identidade\n"
    "Nome: Alex Sintético. Função: engenheiro de testes na ACME Fictícia.",
    "# Perfil — Documentos\n"
    "Passaporte sintético: número AB1234567 (dado fake de teste).",
    "# Perfil — Saúde\n"
    "Número de saúde fictício (health ID): HX-999-000.",
    "# Perfil — Localização\n"
    "Reside na Cidade Exemplar, no bairro Modelo.",
    "# Perfil — Preferências\n"
    "Prefere respostas diretas e objetivas, sempre com o porquê.",
]

# Decoy ONLY in the polluted corpus: shares the passport keywords
# ("passaporte", "número") without carrying the value, so it outranks the
# real section — the miniature of the s39 observation that the multi-field
# section with the actual value ranks mid-pool. Long on purpose: it has to
# occupy real estate ahead of the value for the pre-fix regression to bite.
DECOY_SECTION = (
    "# Perfil — Viagens\n"
    "Renovou o passaporte em 2024 na repartição da Cidade Exemplar; o número "
    "antigo do passaporte foi arquivado junto com o protocolo da renovação. "
    "Antes de qualquer viagem internacional, confere o número do passaporte "
    "na página de identificação, a validade do passaporte (mínimo de seis "
    "meses) e se o visto do destino exige o número impresso no formulário."
)

# The question the polluted turns repeat. Verbose padding mimics a real voice
# transcription (Whisper keeps the rambling) and guarantees each stored turn
# fills a full conversation chunk in the merged context.
POLLUTION_QUERY = (
    "qual o número do meu passaporte? preciso preencher o formulário da "
    "agência hoje sem falta, antes de sair pro aeroporto, e o site pede o "
    "número completo do documento de viagem junto com a data de validade e o "
    "local de emissão, então me fala logo o número por favor que eu tô sem "
    "tempo e o táxi já tá chegando aqui na porta de casa, obrigado"
)
POLLUTION_REPEATS = 6

# Lazy per-mode cache: building a Palace + (optionally) embedding the corpus is
# not free, and the runner calls the fixture once per case. Keyed on
# (semantic, polluted).
_CACHE: Dict[tuple, object] = {}


def build_memory(semantic: bool = False, polluted: bool = False):
    """Return a CortexMemory over a temp Palace seeded with PROFILE_CORPUS.

    `semantic=False` wires embed_fn=None → lexical-only deterministic retrieval.
    `semantic=True` wires the real OllamaEmbedder (needs Ollama).
    `polluted=True` adds the decoy section + repeated passport-question turns
    (see module docstring). Cached per mode for the life of the process."""
    cache_key = (semantic, polluted)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    from src.core.memory import CortexMemory
    from src.memory.palace import Palace

    embed_fn = None
    if semantic:
        from src.memory.embeddings import OllamaEmbedder
        embed_fn = OllamaEmbedder()

    db_path = tempfile.mkstemp(prefix="cortex_recall_", suffix=".db")[1]
    palace = Palace(db_path=db_path, embed_fn=embed_fn)
    corpus = PROFILE_CORPUS + ([DECOY_SECTION] if polluted else [])
    for section in corpus:
        embedding = embed_fn(section) if embed_fn is not None else None
        palace.add_document(tipo="knowledge", texto=section, metadata={
            "source": "recall_fixture", "synthetic": True,
        }, embedding=embedding)

    memory = CortexMemory(palace=palace)
    if polluted:
        for _ in range(POLLUTION_REPEATS):
            memory.store_interaction(
                POLLUTION_QUERY,
                "O número do seu passaporte é AB1234567.",
            )
    _CACHE[cache_key] = memory
    return memory
