"""Task 127: model claims cannot stand in for a queue mutation."""
import pytest

from src.core.stream_reply import ReplyReplacement
from src.core.hallucination import detect_hallucinated_action, detect_hallucinated_study_action
from tests.test_engine_mac_hallucination import rig as mac_rig  # noqa: F401
from tests.test_file_capability_guard import rig, answer, final_reply  # noqa: F401
from tests.test_taught_facts import orch  # noqa: F401


REPLY = {
    'pt': 'Não coloquei nada na lista de estudos. Para colocar, diga: adiciona na lista de estudos: <tópico>',
    'en': "I didn't add anything to the study list. To add one, say: add to the study list: <topic>",
}
CLAIMS = [
    ('pt', '\n\nAdicionei na sua lista de estudos: redes neurais.'),
    ('pt', 'Coloquei grafos na fila de estudos.'),
    ('pt', 'Incluí na lista de estudo.'),
    ('pt', 'Inclui na lista de estudos.'),
    ('pt', 'Anotei na lista de estudos.'),
    ('pt', 'Acrescentei isso à fila de estudos.'),
    ('en', 'Added vector databases to your study list.'),
    ('en', 'I put compilers on my study queue.'),
    ('en', 'I have queued it in your studies.'),
]


@pytest.mark.parametrize('path', ['chat', 'tools', 'stream'])
@pytest.mark.parametrize('language,claim', CLAIMS)
def test_unsupported_claim_replaced_and_persisted(rig, path, language, claim):
    engine, model, executor, audit = rig
    assert answer(rig, path, claim, language, user_input='tell me more') == REPLY[language]
    executor.assert_not_called()
    assert engine.history[-1]['content'] == REPLY[language]
    engine.memory.store_interaction.assert_called_once_with('tell me more', REPLY[language])


@pytest.mark.parametrize('path', ['chat', 'tools', 'stream'])
@pytest.mark.parametrize('reply', [
    'Você pode adicionar na lista de estudos dizendo “estuda grafos”.',
    'You can add topics to your study list.',
    'Aqui está um exemplo.\nAdicionei na lista de estudos.',
    'Não adicionei nada à lista de estudos.',
    'Adicionei leite à lista de compras.',
])
def test_mentions_are_untouched(rig, path, reply):
    assert answer(rig, path, reply) == reply


@pytest.mark.parametrize('stream', [False, True])
def test_orchestrator_uses_replacement(orch, rig, stream):
    engine, model, executor, audit = rig
    orch.engine_analytical = orch.engine_creative = engine
    claim = CLAIMS[0][1]
    model.side_effect = lambda **kw: (iter([{'message': {'content': claim}}])
                                     if kw.get('stream') else {'message': {'content': claim}})
    if stream:
        frames = list(orch.process_stream('tell me about compilers', language='en'))
        assert final_reply(f[0] for f in frames) == REPLY['en']
    else:
        assert orch.process('tell me about compilers', language='en')[0] == REPLY['en']


def test_stream_remains_live_and_replaces_at_completion(rig):
    engine, model, executor, audit = rig
    claim = CLAIMS[0][1]
    model.return_value = iter([{'message': {'content': claim}},
                               {'message': {'content': ' More invented detail.'}}])
    chunks = engine.chat_stream('tell me more', 'pt')
    first = next(chunks)
    assert first and not isinstance(first, ReplyReplacement)
    rest = list(chunks)
    assert isinstance(rest[-1], ReplyReplacement)
    assert final_reply([first, *rest]) == REPLY['pt']


@pytest.mark.parametrize('language,claim', CLAIMS)
def test_detector_family_includes_study_claims(language, claim):
    assert detect_hallucinated_study_action(claim)
    assert detect_hallucinated_action(claim)


@pytest.mark.parametrize('path', ['chat', 'tools', 'stream'])
def test_study_claim_guard_is_factual_under_other_profiles(rig, monkeypatch, path):
    monkeypatch.setenv('CORTEX_MODEL_PROFILE', 'gemma4')
    assert answer(rig, path, CLAIMS[0][1]) == REPLY['pt']


def test_executed_tool_reply_untouched(rig):
    from tests.test_engine_mac_hallucination import TOOLS
    engine, model, executor, audit = rig
    claim = CLAIMS[1][1]
    model.side_effect = [
        {'message': {'content': '', 'tool_calls': [{'function': {
            'name': 'add_study', 'arguments': {'topic': 'grafos'},
        }}]}},
        {'message': {'content': claim}},
    ]
    assert engine.chat_with_tools('study grafos', TOOLS, executor, 'pt') == (claim, 'add_study')
    executor.assert_called_once_with('add_study', {'topic': 'grafos'})
    audit.assert_not_called()


def test_unexecuted_tool_followup_is_guarded(rig):
    from tests.test_engine_mac_hallucination import TOOLS
    engine, model, executor, audit = rig
    model.side_effect = [
        {'message': {'content': '', 'tool_calls': [{'function': {
            'name': 'add_study', 'arguments': '{broken',
        }}]}},
        {'message': {'content': CLAIMS[0][1]}},
    ]
    assert engine.chat_with_tools('study grafos', TOOLS, executor, 'pt') == (REPLY['pt'], None)
    executor.assert_not_called()
    engine.memory.store_interaction.assert_called_once_with('study grafos', REPLY['pt'])
