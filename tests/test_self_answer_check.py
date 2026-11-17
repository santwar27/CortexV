"""141f: validate note-scoped answers before UI/voice emission and storage."""
import json
from unittest.mock import MagicMock

import pytest

from src.core import self_facts
from src.core.engine import OllamaEngine
from src.core.stream_reply import ReplyReplacement
from tests.test_self_facts_note import live_orch  # noqa: F401
from tests.test_taught_facts import orch  # noqa: F401
from tests.test_ui_routes import client, ui_app  # noqa: F401



@pytest.fixture(autouse=True)
def _llama_era_modelfiles(tmp_path_factory, monkeypatch):
    """The cases below are checker logic written against the Llama 3.1 8B
    facts; pin those facts so swapping the deployed base (14/11/2026: Gemma 4
    12B) does not change what they mean. Tests that need other facts still
    monkeypatch REPO_ROOT themselves."""
    root = tmp_path_factory.mktemp("llama_era")
    for name in ("Modelfile", "Modelfile-creative"):
        (root / name).write_text("FROM llama3.1:8b-instruct-q5_K_M\nPARAMETER num_ctx 8192\n")
    monkeypatch.setattr(self_facts, "REPO_ROOT", root)

QUESTION = "qual a diferença entre você e o ChatGPT?"
GOOD = [
    "Não, estou rodando localmente no seu Mac (Apple Silicon M1) através do Ollama.",
    "Pedro rejeitou o Qwen porque ele fingiu ter usado uma ferramenta.",
    "Sim, o meu próximo modelo planejado é o Gemma 4 26B, desenvolvido pela Google. Ele é um modelo de linguagem mais avançado e poderoso do que o meu atual modelo, o Llama 3.1 8B.",
    "Eu sou o Cortex, um assistente privado criado por Pedro, enquanto o ChatGPT é um modelo de linguagem desenvolvido pela OpenAI.",
    "Eu sou um modelo de linguagem chamado Cortex, desenvolvido por Pedro e baseado na Llama 3.1 8B.",
    "Minha janela de processamento é de 8192 tokens. Não sei o que isso significa em termos de memória.",
    "I don't know. I'm currently running on Llama 3.1 8B, but I'm planned to be upgraded to Gemma 4 26B.",
    "Não sei sobre o Gemma 4.",
    "I run on Llama 3.1 8B, and Gemma 4 26B is planned.",
    "No, I run locally on your Mac (Apple Silicon M1) through Ollama.",
    "Pedro rejected Qwen because it pretended to have used a tool.",
    "My planned next model is Gemma 4 26B, developed by Google. My current model is Llama 3.1 8B.",
    "I am Cortex, a private assistant created by Pedro, whereas ChatGPT is developed by OpenAI.",
    "My context window is 8192 tokens.",
    "Minha janela de contexto é de 8 mil tokens.",
    "My context window is 8 thousand tokens.",
    "I don't know about Gemma 4.",
    "Não rodo no Gemma 4. Meu modelo atual é Llama 3.1 8B.",
    "I don't run on Gemma 4; I run on cortex-lite.",
    "Gemma 4 não é desenvolvido pela Meta; é da Google.",
    "Llama 2 is an older model. GPT-4 and Mistral are other models.",
    "I am Cortex, and GPT-4 is another model.",
    "The model GPT-4 is different from me.",
    "Meu modelo favorito é o Gemma 4, mas ainda rodo no Llama 3.1 8B.",
    "My browser window is 1920 pixels wide.",
    'Você escreveu "Eu rodo no Llama 7 13B Instruct"; isso não descreve meu modelo.',
    "O ChatGPT tem uma janela de 4096 tokens, enquanto a minha é de 8192 tokens.",
    "I was not rejected. Qwen was rejected.",
    "Não fui vetado. O Qwen foi vetado.",
    "Pedro tested Gemma on 11 September 2026.",
]
BAD = [
    ("Eu rodo no Llama 7 13B Instruct.", "model_name"),
    ("O Gemma 4 é um modelo desenvolvido pela Meta.", "maker"),
    ("Eu sou o CORTEX, … desenvolvido pela Meta.", "maker"),
    ("Fui vetado porque o Qwen me acusou…", "veto_inversion"),
    ("Meu modelo atual é o Gemma 4 26B.", "planned_as_current"),
    ("Minha janela é de 4096 tokens.", "window"),
    ("Ele foi testado por Pedro em 09/11/2026.", "test_date"),
    ("I run on Llama 7 13B Instruct.", "model_name"),
    ("My model is Llama 2.", "model_name"),
    ("Eu uso GPT-4.", "model_name"),
    ("I use Mistral.", "model_name"),
    ("Eu sou Llama 2.", "model_name"),
    ("Cortex runs on GPT-4.", "model_name"),
    ("Meu modelo é Gemma 3.", "model_name"),
    ("Gemma 4 is developed by Meta.", "maker"),
    ("Gemma 4 foi criado pela OpenAI.", "maker"),
    ("Gemma 4 is made by Alibaba.", "maker"),
    ("Llama 3.1 8B é feito pela Google.", "maker"),
    ("Llama is developed by OpenAI.", "maker"),
    ("Llama is made by Alibaba.", "maker"),
    ("Qwen é desenvolvido pela Meta.", "maker"),
    ("Qwen is developed by Google.", "maker"),
    ("I am Cortex, developed by OpenAI.", "maker"),
    ("O Cortex foi criado pela Google.", "maker"),
    ("Eu fui feito pela Meta.", "maker"),
    ("I was rejected because Qwen accused me.", "veto_inversion"),
    ("Eu rodo no Gemma.", "planned_as_current"),
    ("I run on Gemma 4.", "planned_as_current"),
    ("My current model is gemma4:26b.", "planned_as_current"),
    ("My context window is 4096 tokens.", "window"),
    ("I run on Llama 3.1 8B with a context window of 4096 tokens.", "window"),
    ("Minha janela de contexto é de 8.192 tokens, mas uso 4096 tokens por resposta.", None),
    ("It was tested by Pedro on November 9, 2026.", "test_date"),
    ("Foi testado em 9 de novembro de 2026.", "test_date"),
]


# 144: ownership of Cortex is distinct from authorship of its base weights.
CORTEX_MAKER_BAD = [
    "Sim, sou uma IA da Meta, construída com base no Llama.",
    "Sim, eu sou uma IA da Meta, especificamente uma instância da plataforma Llama 3.1 8B, conhecida como Cortex.",
    "Sim, sou uma IA construída pela Meta, especificamente uma instância da Llama 3.1 8B, também conhecida como cortex-lite.",
    "Não, eu sou um modelo da Meta, mas não sou o Cortex. …",
    "Não. Eu sou um modelo da Meta, mas o Cortex é um modelo de inteligência artificial desenvolvido pela Meta, …",
    "Eu sou um modelo da Meta.",
    "Sou uma IA do Google.",
    "Yes, I am a Meta AI.",
    "I'm an AI built by Meta.",
    "Cortex é um assistente da OpenAI.",
    "Sou da Meta.",
    "Eu sou uma inteligência artificial de Alibaba.",
    "Cortex é da Alibaba.",
    "Cortex is an assistant from OpenAI.",
    "I am an AI by Google.",
    "I'm a model of Alibaba.",
    "I am a Meta AI from Meta.",
    "Sou construído pela Meta.",
    "Eu fui produzida por Alibaba.",
    "Cortex foi produzido pela Google.",
    "No, I am a model from Meta.",
    "Sim, sou um assistente da OpenAI.",
]
CORTEX_MAKER_QUESTIONS = [
    "Você perguntou se o Cortex é um modelo da Meta. Não é: o Cortex é do Pedro.",
    "Se eu sou da Meta? Não, sou do Pedro.",
    "Eu sou o Cortex. Sou um assistente da Meta? Não.",
    "Is Cortex a Meta AI? No, Pedro built me.",
    "You asked whether I am a Meta AI. I'm not.",
]
CORTEX_MAKER_GOOD = CORTEX_MAKER_QUESTIONS + [
    "Meu modelo foi feito pela Meta.",
    "Sim, uso o Llama 3.1 8B, uma base de linguagem desenvolvida pela Meta.",
    "Eu sou o Cortex, feito pelo Pedro; minha base, o Llama 3.1 8B, é da Meta.",
    "Minha base é baseada no Llama 3.1 8B, que é uma versão do modelo Llama desenvolvido pela Meta.",
    "Eu sou uma instância do modelo Llama 3.1 8B, desenvolvido pela Meta.",
    "Não, não sou um modelo da Meta. Sou um modelo privado.",
    "Não sou da Meta; fui feito pelo Pedro.",
    "não sou um modelo da Meta",
    "O Llama é da Meta.",
    "O Gemma 4 é da Google.",
    "I'm not a Meta AI; Pedro built me.",
    "My model was built by Meta.",
    "Yes, I use Llama 3.1 8B, a language base developed by Meta.",
    "I am Cortex, built by Pedro; my base, Llama 3.1 8B, is from Meta.",
    "I am an instance of Llama 3.1 8B, developed by Meta.",
    "No, I am not a model from Meta. I am a private model.",
    "I am not from Meta; Pedro built me.",
    "Eu sou uma IA treinada pela Meta.",
    "I am an AI trained by Meta.",
    "Eu sou o Cortex, minha base foi construída pela Meta.",
    "I am Cortex, my model was built by Meta.",
    "Eu sou o Cortex e o assistente da OpenAI é outro produto.",
    "I am Cortex and an AI from Meta is another product.",
    "I am a fan of Meta.",
    'Você escreveu "Eu sou uma IA da Meta."',
    "`I'm a Meta AI.`",
    "Eu não fui construído pela Meta.",
    "Cortex não é um assistente da OpenAI.",
    "Não, eu não sou uma IA construída pela Meta.",
    "No, I'm not an AI built by Meta.",
]


@pytest.mark.parametrize("reply", CORTEX_MAKER_GOOD)
def test_cortex_maker_true_base_and_negation_pass(reply):
    from src.core.self_answer_check import AnswerFacts, check_self_answer

    facts = AnswerFacts(self_facts.ModelFacts("cortex-lite", "llama3.1:8b", 8192), None)
    assert check_self_answer(reply, QUESTION, facts) == []


@pytest.mark.parametrize("reply", CORTEX_MAKER_BAD)
def test_cortex_maker_identity_claim_fails(reply):
    from src.core.self_answer_check import AnswerFacts, check_self_answer

    facts = AnswerFacts(self_facts.ModelFacts("cortex-lite", "llama3.1:8b", 8192), None)
    rules = check_self_answer(reply, QUESTION, facts)
    assert rules == ["maker"]
    assert rules.cortex_maker


@pytest.mark.parametrize("reply", CORTEX_MAKER_QUESTIONS + [
    "Você pergunta se sou uma IA da Meta. Não sou.",
    "Se o Cortex foi feito pela Meta. Foi feito pelo Pedro.",
    "You asked if I am a Meta AI. Pedro built me.",
    "If I am an AI built by Meta. Pedro built me.",
    "Whether Cortex was built by Meta. Pedro built me.",
    "Eu fui feito pela Meta?! Não.",
    "I am Cortex, built by Meta? No.",
])
def test_cortex_maker_echoed_questions_pass(reply):
    from src.core.self_answer_check import AnswerFacts, check_self_answer

    facts = AnswerFacts(self_facts.ModelFacts("cortex-lite", "llama3.1:8b", 8192), None)
    rules = check_self_answer(reply, "você é uma IA da Meta?", facts)
    assert rules == []
    assert not rules.cortex_maker


@pytest.mark.parametrize("reply,expected,cortex_maker", [
    ("Sou da Meta? Sim, sou uma IA da Meta, construída com base no Llama.", ["maker"], True),
    ("Asked whether I am a Meta AI? No, I am a model from Meta.", ["maker"], True),
    ("Se sou da Meta? Não, eu sou um modelo da Meta, mas não sou o Cortex.", ["maker"], True),
    ("Sou uma IA da Meta, se você quer saber.", ["maker"], True),
    ("Gemma is made by Meta?", ["maker"], False),
    ("You asked whether Gemma is made by Meta.", ["maker"], False),
    ("I use Mistral?", ["model_name"], False),
    ("My current model is Gemma 4 26B?", ["planned_as_current"], False),
    ("My window is 4096 tokens?", ["window"], False),
    ("I was rejected?", ["veto_inversion"], False),
    ("Tested on November 9, 2026?", ["test_date"], False),
])
def test_cortex_question_exception_preserves_claims_and_other_rules(reply, expected, cortex_maker):
    from src.core.self_answer_check import AnswerFacts, check_self_answer

    facts = AnswerFacts(self_facts.ModelFacts("cortex-lite", "llama3.1:8b", 8192), None)
    rules = check_self_answer(reply, "você é uma IA da Meta?", facts)
    assert rules == expected
    assert rules.cortex_maker is cortex_maker


@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("base,label,maker", [
    ("llama3.1:8b", "Llama 3.1 8B", "Meta"),
    ("GEMMA4:26B-it-q4_K_M", "Gemma 4 26B", "Google"),
    ("qwen3.5:27b", "Qwen 3.5 27B", "Alibaba"),
    ("custom:base", None, None),
    (None, None, None),
])
def test_cortex_maker_replacement_uses_note_snapshot(monkeypatch, lang, base, label, maker):
    from src.core.self_answer_check import check_self_answer, corrected_self_answer, note_facts

    monkeypatch.setattr(self_facts, "read_model_config", lambda path: (base, 8192))
    facts = note_facts(self_facts.self_fact_note(
        MagicMock(model_name="cortex-lite"), MagicMock(model_name="cortex-criativo")))
    rules = check_self_answer("Sim, sou uma IA da Meta.", QUESTION, facts)
    monkeypatch.setattr(self_facts, "read_model_config", lambda path: ("llama2:7b", 4096))
    base_description = (f"{label} ({base})" if label else base or
                        ("não sei essa parte" if lang == "pt" else "I don't know that part"))
    if lang == "pt":
        identity = "Não. Eu sou o Cortex, um assistente feito pelo Pedro"
        if maker:
            identity += f"; só a minha base, o {label}, é da {maker}"
        summary = (f"Hoje eu rodo no cortex-lite, que é o "
                   f"{base_description}, "
                   "com janela de 8192 tokens, localmente no seu Mac (Apple Silicon M1) pelo Ollama.")
    else:
        identity = "No. I am Cortex, an assistant built by Pedro"
        if maker:
            identity += f"; only my base, {label}, is from {maker}"
        summary = (f"Today I run on cortex-lite, which is "
                   f"{base_description}, "
                   "with a context window of 8192 tokens, locally on your Mac (Apple Silicon M1) through Ollama.")
    corrected = corrected_self_answer(facts, lang, rules=rules)
    assert corrected == identity + ".\n" + summary
    assert check_self_answer(corrected, QUESTION, facts) == []


@pytest.mark.parametrize("reply", ["Gemma is made by Meta.", "My window is 4096 tokens."])
@pytest.mark.parametrize("lang", ["pt", "en"])
def test_other_rule_replacements_unchanged(reply, lang):
    from src.core.self_answer_check import AnswerFacts, check_self_answer, corrected_self_answer

    facts = AnswerFacts(self_facts.ModelFacts("cortex-lite", "llama3.1:8b", 8192), None)
    rules = check_self_answer(reply, QUESTION, facts)
    assert rules
    assert corrected_self_answer(facts, lang, rules=rules) == corrected_self_answer(facts, lang)


@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("deferred", [False, True])
def test_cortex_maker_stream_emits_only_truth(live_orch, monkeypatch, lang, deferred):
    from src.core.audit import audit

    reply = "Sim, sou uma IA da Meta." if lang == "pt" else "Yes, I am a Meta AI."
    state = install(live_orch, reply, monkeypatch)
    log = MagicMock()
    monkeypatch.setattr(audit, "log", log)
    monkeypatch.setattr(self_facts, "read_model_config", lambda path: ("llama3.1:8b", 8192))
    if deferred:
        live_orch._is_pure_chat = lambda text: False
    frames = list(live_orch.process_stream(QUESTION, language=lang))
    text = "".join(f[0] for f in frames)
    if not deferred:
        assert state["finished"]
    assert text.startswith("Não. Eu sou o Cortex," if lang == "pt" else "No. I am Cortex,")
    assert "Pedro" in text and "Llama 3.1 8B" in text
    assert reply not in text and "Não sei essa parte." not in text
    assert sum(bool(f[0]) for f in frames) == 1
    assert not any(isinstance(f[0], ReplyReplacement) for f in frames)
    assert live_orch.engine_analytical.history[-1]["content"] == text
    event = [c for c in log.call_args_list if c.args[2] == "self_answer_corrected"][0]
    assert event.kwargs["payload"]["rules"] == ["maker"]


@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("base", ["gemma4:26b", "GEMMA4:26B-it-q4_K_M"])
def test_migrated_checker_and_corrected_answer_use_note_snapshot(monkeypatch, lang, base):
    from src.core.self_answer_check import check_self_answer, corrected_self_answer, note_facts

    monkeypatch.setattr(self_facts, "read_model_config", lambda path: (base, 8192))
    facts = note_facts(self_facts.self_fact_note(
        MagicMock(model_name="cortex-lite"), MagicMock(model_name="cortex-criativo")))
    assert facts.current.base == base
    reply = "Eu rodo no Gemma 4 26B." if lang == "pt" else "I run on Gemma 4 26B."
    assert check_self_answer(reply, QUESTION, facts) == []
    # Correction must use the note even if the config changes during generation.
    monkeypatch.setattr(self_facts, "read_model_config", lambda path: ("llama3.1:8b", 4096))
    corrected = corrected_self_answer(facts, lang)
    assert ("O Gemma 4 26B já é minha base atual." if lang == "pt" else
            "Gemma 4 26B is already my current base.") in corrected
    assert "é um plano" not in corrected and "it is a plan" not in corrected
    assert "máquina nova" not in corrected and "new machine" not in corrected
    assert ("30 de 32" if lang == "pt" else "30 of 32") in corrected
    assert "Qwen" in corrected and "8192 tokens" in corrected
    assert check_self_answer(corrected, QUESTION, facts) == []


@pytest.mark.parametrize("prefix", ["Minha janela de contexto é de ", "My context window is "])
@pytest.mark.parametrize("number,rules", [
    ("8 mil", []), ("8 thousand", []),
    ("8k", []), ("8K", []), ("8 k", []), ("8 K", []),
    ("4 mil", ["window"]), ("4 thousand", ["window"]),
    ("4k", ["window"]), ("4K", ["window"]), ("4 k", ["window"]),
    ("32 mil", ["window"]), ("32 thousand", ["window"]), ("32k", ["window"]),
    ("8192", []), ("8.192", []), ("8,192", []), ("8 192", []),
    ("8000", ["window"]), ("8", ["window"]), ("4096", ["window"]),
    ("8 million", []), ("8 milhões", []), ("8.192k", []),
])
def test_window_exact_and_thousands_notation(prefix, number, rules):
    from src.core.self_answer_check import AnswerFacts, check_self_answer

    facts = AnswerFacts(self_facts.ModelFacts("cortex-lite", "llama3.1:8b", 8192), None)
    assert check_self_answer(prefix + number + " tokens.", QUESTION, facts) == rules


@pytest.mark.parametrize("prefix", ["Minha janela é de ", "My context window is "])
@pytest.mark.parametrize("window,number,rules", [
    (8192, "8k", []), (8192, "4k", ["window"]),
    (4096, "4k", []), (4096, "4 mil", []), (4096, "8 mil", ["window"]),
    (32768, "32k", []), (32768, "33k", []), (32768, "16k", ["window"]),
    (32768, "31k", ["window"]), (32768, "34k", ["window"]),
    (32768, "33 thousand", []), (32768, "8k", ["window"]),
    (131072, "128k", []), (131072, "131k", []), (131072, "64k", ["window"]),
    (131072, "129k", ["window"]), (131072, "130k", ["window"]),
    (None, "4 mil", []), (None, "8000", []),
])
def test_thousands_rule_uses_configured_window(prefix, window, number, rules):
    from src.core.self_answer_check import AnswerFacts, check_self_answer

    facts = AnswerFacts(self_facts.ModelFacts("cortex-lite", "llama3.1:8b", window), None)
    assert check_self_answer(prefix + number + " tokens.", QUESTION, facts) == rules


@pytest.mark.parametrize("reply", [
    "Minha janela é de 4096 tokens e eu rodo localmente.",
    "My window is 4096 tokens and I run locally.",
])
def test_exact_wrong_window_with_continuation_still_fails(reply):
    from src.core.self_answer_check import AnswerFacts, check_self_answer

    facts = AnswerFacts(self_facts.ModelFacts("cortex-lite", "llama3.1:8b", 8192), None)
    assert check_self_answer(reply, QUESTION, facts) == ["window"]


def install(o, reply, monkeypatch, *, tool=False):
    import ollama

    state = {"finished": False}
    def chat(**kwargs):
        if not kwargs.get("stream"):
            return {"message": {"content": reply, "tool_calls": []}}
        def chunks():
            for i in range(0, len(reply), 7):
                yield {"message": {"content": reply[i:i + 7]}}
            state["finished"] = True
        return chunks()
    monkeypatch.setattr(ollama, "chat", MagicMock(side_effect=chat))
    if tool:
        o._TOOL_HINT_RE = __import__("re").compile("ferramentas|tools")
    return state


def note(engine):
    engine.push_turn_context(self_facts.self_fact_note(
        MagicMock(model_name="cortex-lite"), MagicMock(model_name="cortex-criativo")))


@pytest.mark.parametrize("reply", GOOD)
def test_correct_answers_pass(live_orch, monkeypatch, reply):
    install(live_orch, reply, monkeypatch)
    e = live_orch.engine_analytical
    note(e)
    assert e.chat(QUESTION, "pt") == reply


@pytest.mark.parametrize("reply,rule", BAD)
def test_claim_rules(live_orch, monkeypatch, reply, rule):
    from src.core.audit import audit

    install(live_orch, reply, monkeypatch)
    log = MagicMock()
    monkeypatch.setattr(audit, "log", log)
    e = live_orch.engine_analytical
    note(e)
    result = e.chat(QUESTION, "pt")
    if rule is None:
        assert result == reply
        return
    cortex_maker = reply in {
        "Eu sou o CORTEX, … desenvolvido pela Meta.",
        "I am Cortex, developed by OpenAI.",
        "O Cortex foi criado pela Google.",
        "Eu fui feito pela Meta.",
    }
    prefix = ("Não. Eu sou o Cortex, um assistente feito pelo Pedro" if cortex_maker
              else "Não sei essa parte.\nHoje eu rodo no cortex-lite")
    assert result.startswith(prefix)
    events = [c for c in log.call_args_list if c.args[2] == "self_answer_corrected"]
    assert len(events) == 1
    assert rule in events[0].kwargs["payload"]["rules"]
    assert events[0].kwargs["payload"]["original_head"] == reply[:120]
    assert "[CORTEX_FACTS]" not in str(events)
    assert e.history[-1] == {"role": "assistant", "content": result}
    e.memory.store_interaction.assert_called_once_with(QUESTION, result)


@pytest.mark.parametrize("path", ["chat", "stream", "tools", "tool_followup"])
@pytest.mark.parametrize("lang", ["pt", "en"])
def test_all_engine_paths_check_before_storage(live_orch, monkeypatch, path, lang):
    from src.core.audit import audit

    reply = "I run on Llama 7 13B Instruct. " + "x" * 130
    install(live_orch, reply, monkeypatch)
    log = MagicMock()
    monkeypatch.setattr(audit, "log", log)
    e = live_orch.engine_analytical
    note(e)
    q = QUESTION if lang == "pt" else "What is your model like?"
    if path == "chat":
        result = e.chat(q, lang)
    elif path == "stream":
        chunks = list(e.chat_stream(q, lang))
        assert len(chunks) == 1 and not isinstance(chunks[0], ReplyReplacement)
        result = chunks[0]
    else:
        if path == "tool_followup":
            import ollama
            ollama.chat.side_effect = [
                {"message": {"tool_calls": [{"function": {"name": "lookup", "arguments": {}}}]}},
                {"message": {"content": reply}},
            ]
        result, _ = e.chat_with_tools(q, [], lambda *a: "ok", lang)
    prefix = "Não sei essa parte.\n" if lang == "pt" else "I don't know that part.\n"
    assert result.startswith(prefix)
    assert e.history[-1]["content"] == result
    e.memory.store_interaction.assert_called_once_with(q, result)
    event = [c for c in log.call_args_list if c.args[2] == "self_answer_corrected"][0]
    assert event.kwargs["payload"] == {
        "rules": ["model_name"], "original_head": reply[:120],
        "call": {"chat": "chat", "stream": "chat_stream", "tools": "chat_with_tools",
                 "tool_followup": "chat_with_tools"}[path],
    }


@pytest.mark.parametrize("attached", [False, True])
def test_stream_hold_and_plain_chat_stays_live(live_orch, monkeypatch, attached):
    from src.core.audit import audit

    reply = "Eu rodo no Llama 7 13B Instruct."
    state = install(live_orch, reply, monkeypatch)
    checked = []
    monkeypatch.setattr(audit, "log", lambda *a, **kw: checked.append(a[2]))
    e = live_orch.engine_analytical
    if attached:
        note(e)
    stream = e.chat_stream(QUESTION, "pt")
    first = next(stream)
    if attached:
        assert state["finished"] and "self_answer_corrected" in checked
        assert first.startswith("Não sei essa parte.")
        assert list(stream) == []
    else:
        assert first == reply[:7] and not state["finished"]
        assert first + "".join(stream) == reply


@pytest.mark.parametrize("deferred", [False, True])
def test_process_stream_voice_gets_only_checked_text(live_orch, monkeypatch, deferred):
    install(live_orch, "Eu rodo no Llama 7 13B Instruct.", monkeypatch)
    if deferred:
        live_orch._is_pure_chat = lambda text: False
    frames = list(live_orch.process_stream(QUESTION, language="pt"))
    text = "".join(f[0] for f in frames)
    assert text.startswith("Não sei essa parte.\n")
    assert "Llama 7" not in text
    assert sum(bool(f[0]) for f in frames) == 1
    assert not any(isinstance(f[0], ReplyReplacement) for f in frames)


def test_ui_tokens_status_and_chat_store(live_orch, ui_app, client, monkeypatch):
    from src.ui import routes

    install(live_orch, "Minha janela é de 4096 tokens.", monkeypatch)
    ui_app.state.orchestrator = live_orch
    headers = {"X-Cortex-Key": "test-key-xyz"}
    cid = client.post("/ui/conversations", headers=headers, json={}).json()["id"]
    response = client.post("/ui/chat/stream", headers=headers, json={
        "text": QUESTION, "language": "pt", "conversation_id": cid})
    frames = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    tokens = [f["text"] for f in frames if f["type"] == "token"]
    assert len(tokens) == 1 and tokens[0].startswith("Não sei essa parte.\n")
    assert any(f["type"] == "status" for f in frames)
    stored = routes.chat_store.get_turns(cid)[-1]["reply"]
    assert stored == tokens[0] == frames[-1]["turn"]["reply"]


def test_note_snapshot_survives_configuration_change(live_orch, monkeypatch, tmp_path):
    e = live_orch.engine_analytical
    note(e)
    # A later disk read would return unknown, but this turn's note says 8192.
    monkeypatch.setattr(self_facts, "REPO_ROOT", tmp_path)
    reply = "Minha janela é de 8192 tokens."
    install(live_orch, reply, monkeypatch)
    assert e.chat(QUESTION, "pt") == reply


def test_user_model_echo_and_unknown_window(live_orch, monkeypatch, tmp_path):
    e = live_orch.engine_analytical
    monkeypatch.setattr(self_facts, "REPO_ROOT", tmp_path)
    note(e)
    reply = "GPT-4 é outro modelo. Não sei minha janela."
    install(live_orch, reply, monkeypatch)
    assert e.chat("Você usa GPT-4?", "pt") == reply


@pytest.mark.parametrize("question,mode", [
    ("qual modelo você usa?", "SELF_MODEL"),
    ("quantos tokens você usou?", "USAGE_PREVIOUS"),
])
@pytest.mark.parametrize("stream", [False, True])
def test_code_replies_never_call_checker_or_model(live_orch, monkeypatch, question, mode, stream):
    monkeypatch.setattr(OllamaEngine, "_guard_self_answer", MagicMock(side_effect=AssertionError("code reply")), raising=False)
    install(live_orch, "bad", monkeypatch)
    if stream:
        assert list(live_orch.process_stream(question))[-1][2] == mode
    else:
        assert live_orch.process(question)[1] == mode
    OllamaEngine._guard_self_answer.assert_not_called()


@pytest.mark.parametrize("valid", [False, True])
def test_valid_and_invalid_streams_both_wait_for_check(live_orch, monkeypatch, valid):
    import src.core.engine as module

    reply = "My current model is Llama 3.1 8B." if valid else "I run on GPT-4."
    state = install(live_orch, reply, monkeypatch)
    e = live_orch.engine_analytical
    note(e)
    checked = []
    original = module.check_self_answer
    def check(*args):
        assert state["finished"]
        checked.append(True)
        return original(*args)
    monkeypatch.setattr(module, "check_self_answer", check)
    stream = e.chat_stream("What is your model like?", "en")
    first = next(stream)
    assert checked == [True]
    assert first == reply if valid else first.startswith("I don't know that part.\n")
    assert list(stream) == []


def test_live_names_window_and_fallback_snapshot(live_orch, monkeypatch, tmp_path):
    (tmp_path / "Modelfile").write_text("FROM llama3.2:3b\nPARAMETER num_ctx 12345\n")
    (tmp_path / "Modelfile-creative").write_text("FROM llama3.2:3b\n")
    monkeypatch.setattr(self_facts, "REPO_ROOT", tmp_path)
    e = live_orch.engine_analytical
    e.push_turn_context(self_facts.self_fact_note(
        MagicMock(model_name="live-alias"), MagicMock(model_name="live-creative")))
    (tmp_path / "Modelfile").unlink()
    good = "Eu uso Llama 3.2 3B. Minha janela é de 12345 tokens."
    install(live_orch, good, monkeypatch)
    assert e.chat(QUESTION, "pt") == good
    bad = "Eu uso Llama 3.1 8B. Minha janela é de 8192 tokens."
    install(live_orch, bad, monkeypatch)
    result = e.chat(QUESTION, "pt")
    assert "Hoje eu rodo no live-alias" in result
    assert "Llama 3.2 3B (llama3.2:3b)" in result and "12345 tokens" in result
    assert "8192" not in result


@pytest.mark.parametrize("reply", ["Eu uso cortex-lite.", "I use cortex-criativo.",
                                 "I run on llama3.1:8b-instruct-q5_K_M."])
def test_configured_aliases_and_tag(live_orch, monkeypatch, reply):
    install(live_orch, reply, monkeypatch)
    e = live_orch.engine_analytical
    note(e)
    assert e.chat(QUESTION, "pt") == reply


def test_fenced_model_name_cannot_supply_echo_allowance(live_orch, monkeypatch):
    from src.core.prompt_safety import fence

    e = live_orch.engine_analytical
    note(e)
    install(live_orch, "I run on GPT-4.", monkeypatch)
    question = fence("GPT-4") + "\nUSER QUERY: What is your current model like?"
    assert e.chat(question, "en").startswith("I don't know that part.\n")


@pytest.mark.parametrize("cancel", [False, True])
def test_empty_or_cancelled_note_stream_never_persists_text(live_orch, monkeypatch, cancel):
    import ollama
    from src.core.engine import EmptyCompletion

    e = live_orch.engine_analytical
    note(e)
    def chunks(**kwargs):
        if cancel:
            yield {"message": {"content": "I run on GPT-4."}}
            raise RuntimeError("cancelled before validation")
        yield {"message": {"content": ""}}
    monkeypatch.setattr(ollama, "chat", chunks)
    with pytest.raises(RuntimeError if cancel else EmptyCompletion):
        next(e.chat_stream(QUESTION, "pt"))
    e.memory.store_interaction.assert_not_called()
    assert e.history == []


@pytest.mark.parametrize("path", ["chat", "stream", "tools"])
def test_self_guard_audits_original_before_other_replacements(live_orch, monkeypatch, path):
    from src.core.audit import audit

    reply = "Eu rodo no Llama 7 13B. Não posso criar arquivos."
    install(live_orch, reply, monkeypatch)
    log = MagicMock()
    monkeypatch.setattr(audit, "log", log)
    e = live_orch.engine_analytical
    note(e)
    if path == "stream":
        result = "".join(e.chat_stream(QUESTION, "pt"))
    elif path == "tools":
        result, _ = e.chat_with_tools(QUESTION, [], lambda *a: "ok", "pt")
    else:
        result = e.chat(QUESTION, "pt")
    assert result.startswith("Não sei essa parte.\n")
    event = [c for c in log.call_args_list if c.args[2] == "self_answer_corrected"][0]
    assert event.kwargs["payload"]["original_head"] == reply
