"""
Tests for the Palace-backed CortexMemory and KnowledgeBase wrappers.

Both classes are exercised against an injected in-memory Palace with a
deterministic embed_fn, so no Ollama, no filesystem persistence, no
network. The contract under test is the *public API* preserved across
the swap from ChromaDB to Palace.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from src.core.knowledge_base import KnowledgeBase
from src.core.memory import CortexMemory
from src.memory.palace import Palace


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------


def _toy_embed(text: str) -> np.ndarray:
    """Deterministic 4-dim embedding: bag-of-words over a tiny vocabulary."""
    vocab = ["cats", "dogs", "pizza", "knowledge"]
    v = np.zeros(4, dtype=np.float32)
    low = text.lower()
    for i, w in enumerate(vocab):
        if w in low:
            v[i] = 1.0
    norm = float(np.linalg.norm(v))
    if norm < 1e-9:
        v[0] = 1.0
    else:
        v = v / norm
    return v.astype(np.float32)


@pytest.fixture
def palace():
    p = Palace(db_path=":memory:", embed_fn=_toy_embed)
    yield p
    p.close()


@pytest.fixture
def kb(palace, tmp_path):
    return KnowledgeBase(
        approved_dir=str(tmp_path / "approved"),
        palace=palace,
    )


@pytest.fixture
def mem(palace):
    return CortexMemory(palace=palace)


# ----------------------------------------------------------------------
# CortexMemory
# ----------------------------------------------------------------------


class TestCortexMemory:
    def test_store_interaction_persists_one_document(self, mem, palace):
        mem.store_interaction("what's the weather?", "sunny and 22°C")
        docs = palace.list_documents(tipo="conversation")
        assert len(docs) == 1
        assert "User: what's the weather?" in docs[0]["texto"]
        assert "Cortex: sunny and 22°C" in docs[0]["texto"]

    def test_store_interaction_refuses_error_markers(self, mem, palace):
        mem.store_interaction("hi", "[OFFLINE] no model loaded")
        mem.store_interaction("ok", "Insufficient data due to engine failure")
        mem.store_interaction("test", "[SYSTEM ERROR] crash")
        assert palace.list_documents(tipo="conversation") == []

    def test_store_interaction_refuses_refusals(self, mem, palace):
        # Refusals must NOT be persisted — storing them poisons future
        # retrieval (the model sees its own past refusal as context and
        # refuses again). Regression for the "quem descobriu o Brasil?"
        # escalating-refusal loop (session 28).
        mem.store_interaction("quem descobriu o Brasil?",
                              "Desculpe, mas não posso ajudar com essa pergunta.")
        mem.store_interaction("conta uma história",
                              "Olá! Sinto muito, mas não posso atender a esse pedido.")
        mem.store_interaction("agenda algo",
                              "Peço desculpas, mas não posso cumprir esse pedido.")
        mem.store_interaction("tell me a joke",
                              "I'm sorry, but I can't help with that.")
        assert palace.list_documents(tipo="conversation") == []

    def test_store_interaction_keeps_legit_answer_with_sorry(self, mem, palace):
        # "Desculpe" that is NOT a refusal (e.g. apologising for delay) must
        # still be stored — the filter is anchored to refusal phrasing.
        mem.store_interaction("qual a resposta?",
                              "Desculpe pela demora! A resposta é 42.")
        assert len(palace.list_documents(tipo="conversation")) == 1

    def test_retrieve_relevant_context_merges_convo_and_knowledge(
        self, mem, palace, tmp_path
    ):
        # one convo turn about cats
        mem.store_interaction("tell me about cats", "cats are wonderful pets")
        # one knowledge briefing about cats — index directly via the
        # KnowledgeBase mounted on the same Palace
        path = tmp_path / "briefing.md"
        path.write_text("# Pedro — feline biology\n\ncats are mammals.")
        mem.knowledge.index_file(str(path))

        ctx = mem.retrieve_relevant_context("cats", n_results=2)
        assert ctx, "context should be non-empty"
        # s39: retrieval surfaces only the USER side of a past turn.
        assert "tell me about cats" in ctx            # user side kept
        assert "cats are wonderful pets" not in ctx   # assistant side dropped
        assert "[KNOWLEDGE]" in ctx
        assert "feline biology" in ctx

    def test_study_comes_back_fenced_as_study_never_as_knowledge(
        self, mem, tmp_path
    ):
        # s88 (Codex lote 14 — invariant 3): an approved web study came back
        # as [KNOWLEDGE], which the prompt calls the user's own facts. Live,
        # "qual minha profissão?" got the studied person's job, 3 in 3.
        from src.core.prompt_safety import OPEN, fence_left_open
        study = tmp_path / "pizza_20261018_031244.md"
        study.write_text("# Topic: pizza napoletana\n\n## Quem fez\n"
                         "A profissão dela é arquiteta. pizza")
        profile = tmp_path / "perfil.md"
        profile.write_text("# Pedro — Perfil\n\n## Comida\nPedro gosta de pizza.")
        mem.knowledge.index_file(str(study))
        mem.knowledge.index_file(str(profile))

        ctx = mem.retrieve_relevant_context("pizza", n_results=3)
        at_study = ctx.index("arquiteta")
        assert fence_left_open(ctx[:at_study])            # inside a fence
        assert ctx[:ctx.rfind(OPEN, 0, at_study)].endswith("[STUDY]\n")
        assert "[KNOWLEDGE]\n# Topic" not in ctx
        at_own = ctx.index("Pedro gosta de pizza")
        assert not fence_left_open(ctx[:at_own])          # his data stays his
        assert ctx.rfind("[KNOWLEDGE]", 0, at_own) > ctx.rfind("[STUDY]", 0, at_own)

    def test_budget_cut_through_a_study_closes_its_fence(
        self, mem, tmp_path, monkeypatch
    ):
        # The knowledge budget cuts the tail chunk; with studies fenced, that
        # cut must not leave the fence open over the conversation after it.
        from src.core.prompt_safety import OPEN, fence_left_open
        monkeypatch.setattr(CortexMemory, "_KNOWLEDGE_BUDGET", 150)
        mem.store_interaction("tell me about pizza dough", "flour and water")
        study = tmp_path / "pizza_20261018_031244.md"
        study.write_text("# Topic: pizza history\n\n" + "pizza napoletana " * 40)
        mem.knowledge.index_file(str(study))

        ctx = mem.retrieve_relevant_context("pizza", n_results=3)
        assert OPEN in ctx and not fence_left_open(ctx)
        assert not fence_left_open(ctx[:ctx.index("tell me about pizza dough")])

    def test_fencing_a_study_does_not_push_out_pedros_chunk(
        self, mem, monkeypatch
    ):
        # Measured live (s88): two fenced studies ahead of the user's birthplace
        # chunk spent the budget on fence markers, cut the chunk from 140 to
        # 45 chars, and "onde eu nasci?" went from right 3/3 to refused 3/3.
        # The plain layout of these two chunks is 199 chars: it fits in 200.
        monkeypatch.setattr(CortexMemory, "_KNOWLEDGE_BUDGET", 200)
        study = "# Topic: Marina Kowalczyk\n## Nascimento\n" + "n" * 40
        own = "# Pedro — Perfil\n## Identidade\nNasceu em CIDADE-DO-PEDRO." + "p" * 34
        monkeypatch.setattr(mem.knowledge, "retrieve_relevant",
                            lambda q, n_results=3: [study, own])
        monkeypatch.setattr(mem._palace, "search", lambda *a, **kw: [])

        ctx = mem.retrieve_relevant_context("onde eu nasci?")
        assert own in ctx

    def test_retrieve_relevant_context_empty_db(self, mem):
        assert mem.retrieve_relevant_context("anything") == ""

    def test_retrieve_drops_assistant_reply(self, mem):
        # s39: a confused/hallucinated assistant reply must NOT re-enter PAST
        # CONTEXT (it primes the 8B to echo the confusion). Only the user's
        # side of a stored turn is surfaced on retrieval.
        mem.store_interaction(
            "fale do meu perfil", "Pedro é você, conforme o meu perfil pessoal")
        ctx = mem.retrieve_relevant_context("perfil", n_results=2)
        assert "fale do meu perfil" in ctx          # user side kept
        assert "Pedro é você" not in ctx            # confused reply dropped
        assert "Cortex:" not in ctx

    def test_user_side_helper(self):
        from src.core.memory import CortexMemory
        assert CortexMemory._user_side("User: oi\nCortex: olá") == "User: oi"
        # no separator → whole text (legacy/malformed row)
        assert CortexMemory._user_side("sem separador") == "sem separador"
        assert CortexMemory._user_side("") == ""

    def test_floor_drops_low_similarity_convo_hit(self, mem, palace, monkeypatch):
        # Bug B-2: a generic query otherwise drags in the least-irrelevant turn
        # (RRF returns top-k by rank). With the floor on, a convo hit whose
        # semantic similarity is below it is dropped. toy_embed: "cats" and
        # "dogs" are orthogonal → cosine 0.0 for the off-topic turn.
        from src.core.memory import CortexMemory
        monkeypatch.setattr(CortexMemory, "_CONVO_SIM_FLOOR", 0.55)
        mem.store_interaction("tell me about cats", "cats are nice")
        mem.store_interaction("dogs are loud", "indeed they bark")
        ctx = mem.retrieve_relevant_context("cats", n_results=2)
        assert "tell me about cats" in ctx       # sim 1.0 ≥ floor → kept
        assert "dogs are loud" not in ctx        # sim 0.0 < floor → dropped

    def test_floor_kill_switch_keeps_all(self, mem, palace, monkeypatch):
        # Floor = 0 → no filtering (the kill-switch restores prior behaviour).
        from src.core.memory import CortexMemory
        monkeypatch.setattr(CortexMemory, "_CONVO_SIM_FLOOR", 0.0)
        mem.store_interaction("tell me about cats", "cats are nice")
        mem.store_interaction("dogs are loud at night", "indeed they bark")
        ctx = mem.retrieve_relevant_context("cats", n_results=2)
        assert "tell me about cats" in ctx
        assert "dogs are loud at night" in ctx    # not dropped when floor off

    def test_floor_keeps_hit_with_no_semantic_score(self, monkeypatch):
        # An FTS/frequency-only hit has _sem_sim=None (no semantic signal to
        # judge) → never floored out, so keyword recall can't regress. Build a
        # lexical-only Palace (embed_fn=None → semantic channel off).
        from src.core.memory import CortexMemory
        from src.memory.palace import Palace
        monkeypatch.setattr(CortexMemory, "_CONVO_SIM_FLOOR", 0.55)
        p = Palace(db_path=":memory:", embed_fn=None)
        m = CortexMemory(palace=p)
        m.store_interaction("the unique token zebra appeared", "noted")
        ctx = m.retrieve_relevant_context("zebra", n_results=2)
        p.close()
        assert "zebra" in ctx                     # lexical hit survives the floor

    def test_retrieve_relevant_context_can_exclude_knowledge(
        self, mem, palace, tmp_path
    ):
        # Same setup as the merge test, but include_knowledge=False must
        # drop the [KNOWLEDGE] briefing while keeping conversation memory.
        # This is the ACT-path behaviour that fixes bug 5: the tool-calling
        # turn shouldn't get the user's profile briefings injected.
        mem.store_interaction("tell me about cats", "cats are wonderful pets")
        path = tmp_path / "briefing.md"
        path.write_text("# Pedro — feline biology\n\ncats are mammals.")
        mem.knowledge.index_file(str(path))

        ctx = mem.retrieve_relevant_context(
            "cats", n_results=2, include_knowledge=False
        )
        assert "tell me about cats" in ctx       # conversation (user side) kept
        assert "[KNOWLEDGE]" not in ctx          # knowledge suppressed
        assert "feline biology" not in ctx

    def test_retrieve_context_truncates_long_chunks(self, mem, palace, monkeypatch):
        # Latency fix (s29): each conversation chunk is capped so the
        # injected context doesn't balloon prompt-eval (an 11k-char context
        # cost ~19.6s TTFT on the M1 8B). A long reply must be truncated.
        from src.core.memory import CortexMemory
        monkeypatch.setattr(CortexMemory, "_CHUNK_CHAR_CAP", 100)
        monkeypatch.setattr(CortexMemory, "_CONTEXT_CHAR_CAP", 100000)
        # The USER side is what gets retrieved now (s39) — make IT the long part.
        mem.store_interaction("cats " * 500, "short reply")  # ~2500-char user side
        ctx = mem.retrieve_relevant_context("cats", n_results=1)
        assert ctx, "context should be non-empty"
        assert len(ctx) <= 102, f"chunk not truncated: {len(ctx)} chars"
        assert ctx.endswith("…")

    def test_retrieve_context_caps_total_size(self, mem, palace, monkeypatch):
        # Even with many hits, the merged context is capped overall so TTFT
        # stays bounded.
        from src.core.memory import CortexMemory
        monkeypatch.setattr(CortexMemory, "_CHUNK_CHAR_CAP", 10000)
        monkeypatch.setattr(CortexMemory, "_CONTEXT_CHAR_CAP", 300)
        for i in range(5):
            mem.store_interaction(f"q{i} about dogs", "dog " * 200)
        ctx = mem.retrieve_relevant_context("dogs", n_results=5)
        assert len(ctx) <= 301, f"total not capped: {len(ctx)} chars"

    # ---- M3-A2: knowledge-first budget + echo dedup -------------------

    def test_knowledge_survives_fat_conversation(
        self, mem, palace, tmp_path, monkeypatch
    ):
        # The headline of M3-A2: knowledge (authoritative — it's where the
        # value lives) must survive the total cap even when the retrieved
        # conversation is fat. Pre-fix, convo came first and the head-keeping
        # truncate cut the knowledge — the first thing lost was the answer.
        from src.core.memory import CortexMemory
        monkeypatch.setattr(CortexMemory, "_CHUNK_CHAR_CAP", 200)
        monkeypatch.setattr(CortexMemory, "_CONTEXT_CHAR_CAP", 300)
        monkeypatch.setattr(CortexMemory, "_KNOWLEDGE_BUDGET", 200)
        for i in range(3):
            mem.store_interaction(f"q{i} long dogs question " + "blah " * 100,
                                  "short reply")
        path = tmp_path / "briefing.md"
        path.write_text("# Pedro — dog registry\n\ndogs licence VALUE-XYZ.")
        mem.knowledge.index_file(str(path))

        ctx = mem.retrieve_relevant_context("dogs", n_results=3)
        assert "[KNOWLEDGE]" in ctx
        assert "VALUE-XYZ" in ctx
        assert len(ctx) <= 301

    def test_knowledge_comes_before_conversation(self, mem, palace, tmp_path):
        # Order pin: [KNOWLEDGE] leads the merge so the head-keeping total
        # truncate always favours it over conversation snippets.
        mem.store_interaction("tell me about cats", "cats are wonderful pets")
        path = tmp_path / "briefing.md"
        path.write_text("# Pedro — feline biology\n\ncats are mammals.")
        mem.knowledge.index_file(str(path))

        ctx = mem.retrieve_relevant_context("cats", n_results=2)
        assert "[KNOWLEDGE]" in ctx and "tell me about cats" in ctx
        assert ctx.index("[KNOWLEDGE]") < ctx.index("tell me about cats")

    def test_conversation_inherits_knowledge_slack(
        self, mem, palace, tmp_path, monkeypatch
    ):
        # The knowledge budget is a CEILING, not a carve-out: when knowledge
        # is short, conversation gets the actual slack — not just
        # (total − budget). Budget 200 but knowledge ~60 chars → convo may
        # use ~240 of the 300 total.
        from src.core.memory import CortexMemory
        monkeypatch.setattr(CortexMemory, "_CHUNK_CHAR_CAP", 10000)
        monkeypatch.setattr(CortexMemory, "_CONTEXT_CHAR_CAP", 300)
        monkeypatch.setattr(CortexMemory, "_KNOWLEDGE_BUDGET", 200)
        mem.store_interaction("a dogs question " + "word " * 100, "reply")
        path = tmp_path / "briefing.md"
        path.write_text("# Pedro — dogs\n\ntiny.")
        mem.knowledge.index_file(str(path))

        ctx = mem.retrieve_relevant_context("dogs", n_results=1)
        assert "[KNOWLEDGE]" in ctx
        convo_part = ctx.split("a dogs question", 1)
        assert len(convo_part) == 2, "conversation must still be present"
        # total used must exceed (total − budget) + knowledge: the convo
        # spilled into the budget the knowledge didn't use.
        assert len(ctx) > 200, f"convo did not inherit the slack: {len(ctx)}"

    def test_echo_of_current_query_is_dropped(self, mem, palace):
        # The perverse dynamic of M3-A2: a stored past copy of the CURRENT
        # question (answer side already dropped on retrieval, s39) carries
        # zero information — it must not occupy the cap.
        mem.store_interaction("qual o meu CPF?", "É 123.456.789-00")
        ctx = mem.retrieve_relevant_context("qual o meu cpf", n_results=3)
        assert "qual o meu CPF?" not in ctx
        assert ctx == ""  # nothing else stored → nothing to inject

    def test_echoes_do_not_spend_the_slots_of_the_answer(self, mem, palace):
        # s84, seen live on the real 8B: every ask stores the question, and on
        # the next ask its echo took a slot in the top-k and was then dropped,
        # pushing the answer out — the second "qual o meu prato favorito?" got
        # an invented dish. Cuts must skip a candidate, not spend a slot.
        mem.store_interaction("tell me about cats", "I don't know yet")
        mem.store_interaction("Tell me about cats!", "Still not sure")
        mem.store_interaction("my cats are called Tom and Jerry", "lovely names")
        ctx = mem.retrieve_relevant_context("tell me about cats", n_results=1)
        assert "my cats are called Tom and Jerry" in ctx

    def test_one_word_user_side_is_not_injected(self, mem, palace):
        # s83 probe: "sim", "c", "Ah!", "play" were about a quarter of all
        # injections. With the reply dropped on re-injection they carry
        # nothing, and bge-m3 scores them 0.50-0.65 against almost anything,
        # so the floor doesn't catch them. Measured on the echo key.
        mem.store_interaction("cats?", "Yes, cats are great")
        mem.store_interaction("tell me about cats", "cats are nice")
        ctx = mem.retrieve_relevant_context("cats", n_results=3)
        assert "tell me about cats" in ctx
        assert "cats?" not in ctx

    def test_short_turn_kill_switch_keeps_it(self, mem, palace, monkeypatch):
        from src.core.memory import CortexMemory
        monkeypatch.setattr(CortexMemory, "_MIN_USER_CHARS", 0)
        mem.store_interaction("cats!", "Yes, cats are great")
        ctx = mem.retrieve_relevant_context("about cats", n_results=3)
        assert "cats!" in ctx

    def test_past_question_is_not_injected(self, mem, palace):
        # s84: only the user side comes back, so an old question never carries
        # its answer — and the 8B reads it as context anyway (an old "por que
        # azul?" turned into "your favourite colour is blue").
        mem.store_interaction("do you know what my cats are called?", "No idea")
        mem.store_interaction("my cats are called Tom and Jerry", "lovely names")
        ctx = mem.retrieve_relevant_context("cats names", n_results=3)
        assert "my cats are called Tom and Jerry" in ctx
        assert "do you know what my cats are called?" not in ctx

    def test_question_rule_uses_only_the_question_mark(self, mem, palace):
        # Interrogative openers are NOT the signal: "como eu disse…" is a
        # statement and must survive.
        mem.store_interaction("como eu disse, my cats are called Tom", "noted")
        ctx = mem.retrieve_relevant_context("cats names", n_results=3)
        assert "como eu disse, my cats are called Tom" in ctx

    def test_question_rule_kill_switch(self, mem, palace, monkeypatch):
        monkeypatch.setenv("CORTEX_CTX_SKIP_QUESTIONS", "0")
        mem.store_interaction("do you know what my cats are called?", "No idea")
        ctx = mem.retrieve_relevant_context("cats names", n_results=3)
        assert "do you know what my cats are called?" in ctx

    def test_near_identical_echoes_are_deduped(self, mem, palace):
        # Two past turns that normalise to the same question must inject
        # only one copy — repeated asking must not crowd the cap.
        mem.store_interaction("Qual é o número do meu CPF?", "É 123")
        mem.store_interaction("qual, é o número do meu... CPF!!", "É 123")
        ctx = mem.retrieve_relevant_context("documento cpf número", n_results=3)
        assert ctx.lower().count("cpf") == 1, f"echo not deduped: {ctx!r}"

    def test_echo_key_normalisation(self):
        # casefold + 'User:' label stripped + punctuation collapsed; accents
        # are word chars and survive (PT-BR text must not be mangled).
        key = CortexMemory._echo_key
        assert key("User: Qual meu CPF?") == "qual meu cpf"
        assert key("  user:   QUAL, meu — CPF!! ") == "qual meu cpf"
        assert key("número de saúde") == "número de saúde"
        assert key("") == ""

    # ---- M3-A4: access_count bumps only on INJECTION ------------------

    def _count_by_text(self, palace, fragment: str) -> int:
        for d in palace.list_documents(tipo="conversation"):
            if fragment in d["texto"]:
                return d["access_count"]
        raise AssertionError(f"doc with {fragment!r} not found")

    def test_injected_hit_gets_access_bump(self, mem, palace):
        mem.store_interaction("tell me about cats", "cats are nice")
        ctx = mem.retrieve_relevant_context("cats", n_results=2)
        assert "tell me about cats" in ctx
        assert self._count_by_text(palace, "tell me about cats") == 1

    def test_floor_dropped_hit_gets_no_bump(self, mem, palace, monkeypatch):
        # The headline of M3-A4: a hit the relevance floor discards was NOT
        # injected — it must not get richer. Before the fix, search() bumped
        # it anyway (the bump lived below the filter).
        monkeypatch.setattr(CortexMemory, "_CONVO_SIM_FLOOR", 0.55)
        mem.store_interaction("tell me about cats", "cats are nice")
        mem.store_interaction("dogs are loud", "indeed they bark")
        ctx = mem.retrieve_relevant_context("cats", n_results=2)
        assert "dogs are loud" not in ctx            # floored out
        assert self._count_by_text(palace, "dogs are loud") == 0
        assert self._count_by_text(palace, "tell me about cats") == 1

    def test_echo_deduped_hit_gets_no_bump(self, mem, palace):
        # Echo of the current query is dropped by dedup → not injected →
        # no bump. (This was the live pathology: the passport echo sat at
        # access_count 37 in the real palace.db, fed by bumps on hits the
        # dedup discards.)
        mem.store_interaction("qual o meu CPF?", "É 123.456.789-00")
        ctx = mem.retrieve_relevant_context("qual o meu cpf", n_results=3)
        assert ctx == ""
        assert self._count_by_text(palace, "qual o meu CPF?") == 0


# ----------------------------------------------------------------------
# KnowledgeBase
# ----------------------------------------------------------------------


class TestKnowledgeBase:
    def test_index_file_writes_one_knowledge_doc(self, kb, palace, tmp_path):
        path = tmp_path / "topic.md"
        path.write_text("# Topic: pizza\n\npizza recipe details.")
        assert kb.index_file(str(path)) is True
        docs = palace.list_documents(tipo="knowledge")
        assert len(docs) == 1
        assert docs[0]["metadata"]["topic"] == "pizza"
        assert docs[0]["metadata"]["filename"] == "topic.md"

    def test_index_file_is_idempotent_on_filename(self, kb, palace, tmp_path):
        path = tmp_path / "topic.md"
        path.write_text("# Topic: pizza\n\nv1")
        kb.index_file(str(path))
        path.write_text("# Topic: pizza\n\nv2 revised")
        kb.index_file(str(path))
        docs = palace.list_documents(tipo="knowledge")
        assert len(docs) == 1
        assert "v2 revised" in docs[0]["texto"]

    def test_remove_file_drops_the_row(self, kb, palace, tmp_path):
        path = tmp_path / "topic.md"
        path.write_text("# Topic: x\n\nbody")
        kb.index_file(str(path))
        assert kb.count() == 1
        assert kb.remove_file("topic.md") is True
        assert kb.count() == 0

    def test_remove_file_missing_returns_false(self, kb):
        assert kb.remove_file("nope.md") is False

    def test_reindex_all_walks_directory(self, palace, tmp_path):
        approved = tmp_path / "approved"
        approved.mkdir()
        (approved / "a.md").write_text("# Topic: a\n\nbody A")
        (approved / "b.md").write_text("# Topic: b\n\nbody B")
        (approved / "ignored.txt").write_text("not a briefing")

        kb = KnowledgeBase(approved_dir=str(approved), palace=palace)
        assert kb.reindex_all() == 2
        assert kb.count() == 2

    def test_retrieve_relevant_bumps_access_count(self, kb, palace, tmp_path):
        # M3-A4: knowledge has no floor/dedup downstream — every returned
        # hit is injected, so retrieve_relevant marks them all as used.
        path = tmp_path / "topic.md"
        path.write_text("# Topic: knowledge base\n\nknowledge is power.")
        kb.index_file(str(path))
        hits = kb.retrieve_relevant("knowledge", n_results=2)
        assert hits
        docs = palace.list_documents(tipo=kb.DOC_TIPO)
        assert any(d["access_count"] == 1 for d in docs)

    def test_retrieve_relevant_returns_strings(self, kb, tmp_path):
        path = tmp_path / "k.md"
        path.write_text("# Topic: pizza\n\npizza recipe")
        kb.index_file(str(path))
        hits = kb.retrieve_relevant("pizza", n_results=3)
        assert hits and isinstance(hits[0], str)
        assert "pizza recipe" in hits[0]

    def test_missing_file_index_returns_false(self, kb):
        assert kb.index_file("/nonexistent/path/x.md") is False

    def test_index_handles_briefing_without_topic_line(self, kb, palace, tmp_path):
        path = tmp_path / "untitled.md"
        path.write_text("just a body, no topic header")
        assert kb.index_file(str(path)) is True
        doc = palace.list_documents(tipo="knowledge")[0]
        # falls back to basename
        assert doc["metadata"]["topic"] == "untitled.md"

    # ----- section chunking (s37 passport-retrieval fix) -----

    PROFILE_MD = (
        "# Pedro — Perfil Brasileiro\n\n"
        "Dados da cidadania brasileira.\n\n"
        "## Identificação\n- Nome: Pedro\n\n"
        "## Passaporte brasileiro\n- Número: XX000000\n- Validade: 2030\n\n"
        "## Endereço no Brasil\n- CEP: 00000-000\n"
    )

    def test_index_file_chunks_by_h2_section(self, kb, palace, tmp_path):
        path = tmp_path / "perfil_brasileiro.md"
        path.write_text(self.PROFILE_MD)
        assert kb.index_file(str(path)) is True
        docs = palace.list_documents(tipo="knowledge")
        # preamble + 3 sections = 4 chunks, all from the one file
        assert len(docs) == 4
        assert {d["metadata"]["filename"] for d in docs} == {"perfil_brasileiro.md"}
        assert all(d["metadata"]["chunk_count"] == 4 for d in docs)
        # the passport section is its own retrievable chunk…
        passport = [d for d in docs if "## Passaporte brasileiro" in d["texto"]]
        assert len(passport) == 1
        # …and it carries the H1 so 'passaporte brasileiro' disambiguates
        assert passport[0]["texto"].startswith("# Pedro — Perfil Brasileiro")
        assert "Número: XX000000" in passport[0]["texto"]

    def test_count_counts_files_not_chunks(self, kb, palace, tmp_path):
        path = tmp_path / "perfil_brasileiro.md"
        path.write_text(self.PROFILE_MD)
        kb.index_file(str(path))
        assert len(palace.list_documents(tipo="knowledge")) == 4  # chunks
        assert kb.count() == 1                                    # files

    def test_reindex_replaces_all_chunks_no_dup(self, kb, palace, tmp_path):
        path = tmp_path / "perfil_brasileiro.md"
        path.write_text(self.PROFILE_MD)
        kb.index_file(str(path))
        kb.index_file(str(path))  # re-index same basename
        docs = palace.list_documents(tipo="knowledge")
        assert len(docs) == 4      # not 8 — prior chunks were deleted first
        assert kb.count() == 1

    @staticmethod
    def _ranked_kb(ranked, calls):
        """A KnowledgeBase over a stub Palace that returns the top k of a
        fixed ranking and records every search's k and what gets used."""
        class _P:
            def search(self, q, k=3, tipo=None, **kw):
                calls.append(k)
                return ranked[:k]

            def mark_used(self, ids):
                calls.append(("used", list(ids)))
        kb = KnowledgeBase.__new__(KnowledgeBase)
        kb._palace = _P()
        return kb

    @staticmethod
    def _einstein(i):
        return {"id": i, "_sem_sim": 0.5,
                "texto": f"# Topic: Albert Einstein: biografia\n## S{i}\nNasceu em Ulm."}

    @staticmethod
    def _own(i, section):
        return {"id": i, "_sem_sim": 0.45, "texto": f"# Pedro — Perfil\n## {section}"}

    def test_a_study_the_question_does_not_name_gives_its_slot_away(self):
        # s88, measured: with an Einstein briefing approved, its chunks held 5
        # of the 6 slots for "onde eu nasci?" and the user's birthplace chunk
        # ranked 8th; the 8B answered "Você nasceu em Ulm".
        ranked = [self._einstein(1), self._einstein(2), self._own(3, "Vida no UK"),
                  self._einstein(4), self._own(5, "Identificação")]
        calls = []
        out = self._ranked_kb(ranked, calls).retrieve_relevant("onde eu nasci?", 3)
        assert out == [ranked[2]["texto"], ranked[4]["texto"]]
        assert calls[-1] == ("used", [3, 5])

    def test_a_study_the_question_names_still_enters(self):
        ranked = [self._einstein(1), self._own(2, "Vida no UK")]
        calls = []
        out = self._ranked_kb(ranked, calls).retrieve_relevant(
            "onde o Einstein nasceu?", 2)
        assert out == [ranked[0]["texto"], ranked[1]["texto"]]
        assert calls == [2, ("used", [1, 2])]   # one search, as before

    def test_without_a_study_turned_away_it_is_one_search(self):
        ranked = [self._own(1, "Identificação"), self._own(2, "Vida no UK")]
        calls = []
        self._ranked_kb(ranked, calls).retrieve_relevant("onde eu nasci?", 2)
        assert calls == [2, ("used", [1, 2])]

    def test_every_chunk_of_a_studier_briefing_reads_as_a_study(self, tmp_path):
        # Pins the coupling the [STUDY] tag rests on: the file the Studier
        # writes, chunked the way it is indexed, is recognised chunk by chunk.
        from src.core.studier import NightlyStudier
        studier = NightlyStudier.__new__(NightlyStudier)
        studier.output_dir = str(tmp_path)
        path = studier._save(
            "Marina Kowalczyk",
            "## Resumo executivo\nNasceu em Gdańsk.\n\n## Profissão\nArquiteta.",
            ["Marina Kowalczyk biografia"],
        )
        with open(path, encoding="utf-8") as f:
            chunks = KnowledgeBase._chunk_markdown(f.read())
        assert len(chunks) >= 2
        assert all(KnowledgeBase.is_study_chunk(c) for c in chunks)

    def test_profile_chunks_are_not_studies(self):
        chunks = KnowledgeBase._chunk_markdown(
            "# Pedro — Perfil Brasileiro\n\nIntro.\n\n"
            "## Identificação\nNasceu em X.\n\n## Endereço\nRua Y.")
        assert chunks
        assert not any(KnowledgeBase.is_study_chunk(c) for c in chunks)

    def test_chunk_markdown_no_h2_is_single_chunk(self):
        # Short studier briefing (H1 + body, no '##') stays one chunk.
        out = KnowledgeBase._chunk_markdown("# Topic: pizza\n\npizza recipe")
        assert out == ["# Topic: pizza\n\npizza recipe"]

    def test_chunk_markdown_empty_returns_nothing(self):
        assert KnowledgeBase._chunk_markdown("   ") == []

    def test_chunk_markdown_ignores_headings_inside_code_fences(self):
        # A '## step' / '# comment' inside a ``` fence must NOT split or be
        # read as the H1 — only the real '##' sections do.
        md = (
            "# Topic: deploy\n\n"
            "## Setup\nrun it.\n\n"
            "```bash\n# comment, not a heading\n## step two, also not\n```\n\n"
            "## Teardown\nstop it.\n"
        )
        out = KnowledgeBase._chunk_markdown(md)
        # preamble + Setup + Teardown = 3 chunks (the fence stays whole)
        assert len(out) == 3
        setup = [c for c in out if "## Setup" in c][0]
        assert "# comment, not a heading" in setup   # fence rode with Setup
        assert "## step two" in setup                # not split out
        # H1 prefix is the real title, not the fenced '# comment'
        assert all(c.startswith("# Topic: deploy") for c in out)
