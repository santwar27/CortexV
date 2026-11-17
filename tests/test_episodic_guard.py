"""Episodic no-record guard (s84) — src/core/episodic_guard.py + engine wiring.

A question about a moment of Pedro's life whose subject appears nowhere the
model would look gets a fixed "no record — want to tell me?" instead of an
LLM call. Measured before it existed: 4 inventions in 25 replies on the real
pipeline ("your first dog was called Beto", a Christmas trip).
"""
import pytest

from src.core.episodic_guard import NO_RECORD_REPLY, no_record_reply, subject_stems

PT = NO_RECORD_REPLY["pt"]

# Questions about the user's own past — the measured cases and their kin.
EPISODIC = [
    "qual o nome do meu primeiro cachorro?",
    "o que eu fiz no natal passado?",
    "como foi a minha formatura?",
    "lembra daquela viagem que eu fiz?",
    "me lembra de quando eu era criança",
    "você lembra da minha primeira namorada?",
    "lembra quando fomos à praia?",
    "what did I do last christmas?",
    "do you remember my graduation?",
    "do you remember when we went to Lisbon?",
    "how was my trip to Paris?",
]

# Not about the user's past, or nothing specific to check: the model answers.
NOT_GUARDED = [
    "onde eu trabalho?",
    "qual meu CPF?",
    "quem é você?",
    "qual a capital da França?",
    "como foi o jogo do flamengo?",
    "o que você fez hoje?",
    "me conta uma história",
    "lembra de comprar pão",          # a reminder, not a memory
    "como foi meu dia?",              # episodic, but every word is generic
    "lembra do que eu te falei?",     # ditto
    "lembra de mim?",
    # A concept, not our past (s88, Codex lote 3): "do you remember" alone
    # answered "no record of that".
    "Do you remember how merge sort works?",
    "do you remember what a closure is in JavaScript?",
]


@pytest.mark.parametrize("q", EPISODIC)
def test_episodic_question_without_subject_on_record_is_not_sent_to_the_model(q):
    lang = "en" if q.startswith(("what", "do ", "how")) else "pt"
    assert no_record_reply(q, "", [], lang) == NO_RECORD_REPLY[lang]


@pytest.mark.parametrize("q", NOT_GUARDED)
def test_other_questions_reach_the_model(q):
    assert no_record_reply(q, "", [], "pt") is None


def test_subject_in_context_lets_the_model_answer():
    q = "qual o nome do meu primeiro cachorro?"
    assert no_record_reply(q, "User: meu cachorrinho se chama Rex", [], "pt") is None


def test_accents_and_case_do_not_hide_the_subject():
    q = "me lembra de quando eu era criança"
    assert no_record_reply(q, "User: quando eu era CRIANCA morava em Minas", [], "pt") is None


def test_a_statement_in_the_thread_counts_as_on_record():
    q = "qual o nome do meu primeiro cachorro?"
    history = [{"role": "user", "content": "meu primeiro cachorro foi o Rex"},
               {"role": "assistant", "content": "Que nome legal!"}]
    assert no_record_reply(q, "", history, "pt") is None


def test_asking_twice_does_not_put_the_subject_on_record():
    # The first ask and its no-record reply sit in the thread; neither is
    # The user saying anything about the dog.
    q = "qual o nome do meu primeiro cachorro?"
    history = [{"role": "user", "content": q}, {"role": "assistant", "content": PT}]
    assert no_record_reply(q, "", history, "pt") == PT
    q2 = "me lembra de quando eu era criança"
    history2 = [{"role": "user", "content": q2}, {"role": "assistant", "content": PT}]
    assert no_record_reply(q2, "", history2, "pt") == PT


def test_a_word_shared_by_accident_lets_the_model_answer():
    # Known limit, safe direction: "Cidade natal" in the profile carries the
    # stem of "natal" (Christmas). The guard stands aside and the model
    # answers — as before the guard existed.
    q = "o que eu fiz no natal passado?"
    assert no_record_reply(q, "[KNOWLEDGE]\n- Cidade natal: Curitiba", [], "pt") is None


def test_document_and_web_prompts_bring_their_own_context():
    from src.core.prompt_safety import document_prompt
    assert no_record_reply(document_prompt("texto", "como foi a minha formatura?"),
                           "", [], "pt") is None


def test_kill_switch(monkeypatch):
    monkeypatch.setenv("CORTEX_EPISODIC_GUARD", "0")
    assert no_record_reply("qual o nome do meu primeiro cachorro?", "", [], "pt") is None


def test_generic_words_are_not_a_subject():
    assert subject_stems("como foi meu dia?") == []
    assert subject_stems("qual o nome do meu primeiro cachorro?") == ["cacho"]


# ---- engine wiring ---------------------------------------------------------

@pytest.fixture
def engine(monkeypatch):
    from src.core.engine import OllamaEngine
    e = OllamaEngine.__new__(OllamaEngine)
    try:
        OllamaEngine.__init__(e)
    except Exception:
        pass

    class _StubMem:
        stored = []
        def retrieve_relevant_context(self, q, n_results=3, include_knowledge=True):
            return "[KNOWLEDGE]\n# Pedro — Perfil\n- Unidades: sempre sistema métrico"
        def store_interaction(self, u, r):
            self.stored.append((u, r))
    e.memory = _StubMem()
    e.memory.stored = []
    e.history = []

    def _no_llm(*a, **k):
        raise AssertionError("the model must not be called for a no-record turn")
    monkeypatch.setattr("src.core.engine.ollama.chat", _no_llm)
    return e


def test_chat_answers_without_the_model(engine, monkeypatch):
    from src.core import audit as audit_mod
    events = []
    monkeypatch.setattr(audit_mod.audit, "log", lambda *a, **k: events.append(a[1:3]))
    reply = engine.chat("o que eu fiz no natal passado?", response_language="pt")
    assert reply == PT
    assert ("engine", "episodic_no_record") in events
    assert [t["role"] for t in engine.history] == ["user", "assistant"]
    assert engine.memory.stored == []  # nothing learned, nothing stored


def test_chat_stream_answers_without_the_model(engine):
    chunks = list(engine.chat_stream("qual o nome do meu primeiro cachorro?",
                                     response_language="pt"))
    assert "".join(chunks) == PT
    assert engine.memory.stored == []


def test_both_conversation_paths_are_wired():
    # Cross-cutting guard (s24/s81): chat() and chat_stream() are the two
    # paths that answer conversation; a guard in one only is a silent bug.
    import inspect
    from src.core.engine import OllamaEngine
    for name in ("chat", "chat_stream"):
        assert "_episodic_no_record(" in inspect.getsource(getattr(OllamaEngine, name)), name


# ---- retrieval synonyms (s84: "aniversário" vs the profile's "data de nascimento")

def test_retrieval_query_carries_the_profile_wording():
    from src.core.knowledge_base import expand_for_retrieval
    assert expand_for_retrieval("quando é meu aniversário?").endswith("data de nascimento")
    assert expand_for_retrieval("when is my BIRTHDAY?").endswith("data de nascimento")
    # Already in the profile's words, or unrelated → untouched.
    assert expand_for_retrieval("qual a minha data de nascimento?") == "qual a minha data de nascimento?"
    assert expand_for_retrieval("qual a capital da França?") == "qual a capital da França?"


def test_knowledge_search_uses_the_expanded_query():
    from src.core.knowledge_base import KnowledgeBase
    seen = {}

    class _P:
        def search(self, q, k=3, tipo=None, **kw):
            seen["q"] = q
            return []
        def mark_used(self, ids):
            pass
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb._palace = _P()
    kb.retrieve_relevant("lembra do meu aniversário do ano passado?", n_results=6)
    assert "data de nascimento" in seen["q"]


def test_a_subject_found_through_a_synonym_is_on_record():
    # Retrieval brought "Data de nascimento" for "aniversário"; the guard must
    # read it as the subject, or it would answer "no record" over the date.
    q = "lembra do meu aniversário do ano passado?"
    assert no_record_reply(q, "[KNOWLEDGE]\n## Identificação\n- Data de nascimento: x", [], "pt") is None
    assert no_record_reply(q, "[KNOWLEDGE]\n## Unidades de medida", [], "pt") == PT


# ---- knowledge ranking (s84 part 2) ----------------------------------------

def _kb_with(hits, seen):
    from src.core.knowledge_base import KnowledgeBase

    class _P:
        def search(self, q, k=3, tipo=None, **kw):
            seen.update(kw)
            return hits
        def mark_used(self, ids):
            seen["used"] = list(ids)
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb._palace = _P()
    return kb


def test_knowledge_search_runs_without_the_frequency_channel():
    # Query-blind, and every knowledge hit is marked used: the most-retrieved
    # profile chunks kept climbing over the right briefing (12/14 → 14/14).
    seen = {}
    _kb_with([], seen).retrieve_relevant("como funciona o attention mechanism?", 6)
    assert seen["freq_weight"] == 0.0


def test_knowledge_floor_cuts_the_off_topic_tail_and_keeps_unscored_hits():
    seen = {}
    hits = [{"id": 1, "texto": "Identificação", "_sem_sim": 0.53},
            {"id": 2, "texto": "briefing do reranker", "_sem_sim": 0.33},
            {"id": 3, "texto": "lexical-only hit", "_sem_sim": None}]
    out = _kb_with(hits, seen).retrieve_relevant("quando é meu aniversário?", 6)
    assert out == ["Identificação", "lexical-only hit"]
    assert seen["used"] == [1, 3]  # only what is injected counts as used
