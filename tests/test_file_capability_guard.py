"""Task 121: fake model replies, no tools or external services needed."""
import pytest

from src.core.memory import CortexMemory
from src.core.stream_reply import ReplyReplacement
from tests.test_engine_mac_hallucination import rig as mac_rig, TOOLS  # noqa: F401
from tests.test_taught_facts import orch  # noqa: F401


REPLACEMENT = {
    "pt": 'Eu consigo salvar arquivos. Para gravar o código da minha última resposta, '
          'diga "salva esse código". Para guardar um texto, diga "escreve no <nome> <texto>".',
    "en": 'I can save files. To save the code from my last reply, say "save this code". '
          'To keep a text, say "store <name> <text>".',
}
DENIALS = [
    ("pt", "Não posso criar um documento com o código, pois isso iria contra os princípios de privacidade."),
    ("pt", "Não tenho permissão para criar arquivos."),
    ("pt", "O Cortex é um assistente de texto e não tem a capacidade de criar arquivos ou salvar conteúdo em disco."),
    ("pt", "Não consigo salvar um arquivo no seu Mac."),
    ("pt", "Eu não tenho a capacidade de gravar documentos."),
    ("pt", "Não tenho como escrever um doc."),
    ("pt", "Não posso criar arquivos diretamente."),
    ("pt", "O Cortex não pode salvar documentos."),
    ("pt", "Aqui está a explicação.\nPorém, não consigo criar um arquivo."),
    ("en", "No, I cannot create files directly."),
    ("en", "I can't save a file on your Mac."),
    ("en", "I am unable to write documents."),
    ("en", "I don't have the ability to create files."),
    ("en", "Cortex does not have the ability to save files."),
    ("en", "I can explain the code. However, I can’t create a document."),
]
UNTOUCHED = [
    "Não posso acessar a internet agora",
    "não consigo abrir esse PDF — ele está protegido",
    "I can't read that file",
    "não salvei nada na memória",
    "Sim, posso criar arquivos",
    'A ferramenta respondeu: "Não posso criar arquivos: acesso negado."',
    'The tool returned: “I cannot create files: permission denied.”',
    'Tool error:\n> I cannot create files in this directory.',
    '[SYSTEM ERROR] I cannot create files in this directory.',
    'Tool returned: I cannot create files in this directory.',
    '```js\n// cannot create file\n// I cannot create files\n```',
    'Exemplo:\n~~~text\nNão posso criar arquivos\n~~~',
    'O erro é `I cannot create files`.',
    'Meu editor não tem a capacidade de criar arquivos.',
    "I can't create a directory.",
]


@pytest.fixture
def rig(mac_rig, monkeypatch):
    # The task-110 rig replaces the class with a factory. Plain chat also
    # needs the real class's pure refusal classifier, after construction.
    monkeypatch.setattr("src.core.engine.CortexMemory", CortexMemory)
    return mac_rig


def final_reply(chunks):
    parts = []
    for chunk in chunks:
        if isinstance(chunk, ReplyReplacement):
            parts.clear()
        parts.append(chunk)
    return "".join(parts)


def answer(rig, path, reply, language="pt", user_input="can you create files?"):
    engine, model, executor, _audit = rig
    if path == "stream":
        # Split inside denial words and sentences; final text must supersede
        # the live tokens even when the denial is only apparent at the end.
        model.return_value = iter({"message": {"content": reply[i:i + 3]}}
                                  for i in range(0, len(reply), 3))
        return final_reply(engine.chat_stream(user_input, language))
    model.return_value = {"message": {"content": reply, "tool_calls": []}}
    if path == "tools":
        result, tool = engine.chat_with_tools(user_input, TOOLS, executor, language)
        assert tool is None
        return result
    return engine.chat(user_input, language)


@pytest.mark.parametrize("path", ["chat", "stream", "tools"])
@pytest.mark.parametrize("language,reply", DENIALS)
def test_denials_replaced_and_only_replacement_persisted(rig, path, language, reply):
    engine, model, executor, audit = rig
    expected = REPLACEMENT[language]
    assert answer(rig, path, reply, language) == expected
    executor.assert_not_called()
    model.assert_called_once()
    audit.assert_called_once_with("–", "engine", "file_capability_denial_blocked", payload={
        "original_head": reply[:120], "user_input": "can you create files?",
        "mode": "OFFLINE", "call": {"chat": "chat", "stream": "chat_stream",
                                     "tools": "chat_with_tools"}[path],
    })
    assert engine.history == [
        {"role": "user", "content": "can you create files?"},
        {"role": "assistant", "content": expected},
    ]
    engine.memory.store_interaction.assert_called_once_with("can you create files?", expected)


@pytest.mark.parametrize("path", ["chat", "stream", "tools"])
@pytest.mark.parametrize("reply", UNTOUCHED)
def test_unrelated_denials_quotes_errors_and_code_untouched(rig, path, reply):
    assert answer(rig, path, reply) == reply
    rig[3].assert_not_called()
    rig[0].memory.store_interaction.assert_called_once_with("can you create files?", reply)


@pytest.mark.parametrize("language,reply", DENIALS)
def test_tool_backed_reply_untouched(rig, language, reply):
    engine, model, executor, audit = rig
    model.side_effect = [
        {"message": {"content": "", "tool_calls": [{"function": {
            "name": "write_document", "arguments": {"filename": "note", "content": "text"},
        }}]}},
        {"message": {"content": reply}},
    ]
    assert engine.chat_with_tools("save my text", TOOLS, executor, language) == (reply, "write_document")
    executor.assert_called_once()
    audit.assert_not_called()
    assert engine.history[-1]["content"] == reply
    engine.memory.store_interaction.assert_called_once_with("save my text", reply)


@pytest.mark.parametrize("path", ["sync", "stream", "deferred"])
@pytest.mark.parametrize("language,reply", DENIALS)
def test_both_orchestrator_entry_points_use_guard(orch, rig, path, language, reply):
    engine, model, _executor, audit = rig
    orch.engine_analytical = orch.engine_creative = engine
    model.side_effect = lambda **kw: (iter([{"message": {"content": reply}}])
                                     if kw.get("stream") else {"message": {"content": reply}})
    if path != "sync":
        if path == "deferred":
            orch._is_pure_chat = lambda _text: False
        frames = list(orch.process_stream("can you create files?", language=language))
        result, mode = final_reply(f[0] for f in frames), frames[-1][2]
    else:
        result, mode = orch.process("can you create files?", language=language)
    assert (result, mode) == (REPLACEMENT[language], "OFFLINE")
    assert engine.history[-1]["content"] == REPLACEMENT[language]
    assert sum(c.args[2] == "file_capability_denial_blocked" for c in audit.call_args_list) == 1


@pytest.mark.parametrize("command,filename,content", [
    ("escreve no notas texto de exemplo", "notas", "texto de exemplo"),
    ("store notes example text", "notes", "example text"),
])
def test_replacement_document_command_matches_real_route(command, filename, content):
    from src.core.dispatch.triggers import WRITE_RE

    match = WRITE_RE.match(command)
    assert match is not None
    assert match.groups() == (filename, content)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("mode", ["LOCAL_RAG", "ONLINE_RAG"])
def test_rag_synthesis_does_not_correct_tool_backed_denial(orch, rig, stream, mode):
    engine, model, _executor, audit = rig
    reply = DENIALS[0][1]
    model.return_value = (iter([{"message": {"content": reply}}]) if stream else
                          {"message": {"content": reply}})
    emitted = []
    if stream:
        orch.set_token_emitter(emitted.append)
    assert orch._synthesise(engine, "summarise this document", "pt", mode) == (reply, mode)
    if stream:
        assert "".join(emitted) == reply
    audit.assert_not_called()
    engine.memory.store_interaction.assert_called_once_with("summarise this document", reply)


def test_unexecuted_tool_with_invalid_arguments_still_guarded(rig):
    engine, model, executor, audit = rig
    reply = DENIALS[0][1]
    model.side_effect = [
        {"message": {"tool_calls": [{"function": {
            "name": "write_document", "arguments": "invalid json",
        }}]}},
        {"message": {"content": reply}},
    ]
    assert engine.chat_with_tools("save my text", TOOLS, executor, "pt") == (REPLACEMENT["pt"], None)
    executor.assert_not_called()
    assert engine.history[-1]["content"] == REPLACEMENT["pt"]
    assert sum(c.args[2] == "file_capability_denial_blocked" for c in audit.call_args_list) == 1


@pytest.mark.parametrize("path", ["chat", "stream", "tools"])
def test_capability_is_a_fact_under_other_profiles(rig, monkeypatch, path):
    monkeypatch.setenv("CORTEX_MODEL_PROFILE", "gemma4")
    assert answer(rig, path, DENIALS[0][1]) == REPLACEMENT["pt"]
