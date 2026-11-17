"""Task 127: deterministic study intake on both public entry points."""
from contextlib import contextmanager
from unittest.mock import Mock

import pytest

from src.core.dispatch.registry import simulate
from tests.test_taught_facts import orch  # noqa: F401


REAL_ONE = ('adicione na sua lista de estudos: Hoje e\x1b[200~Principais '
            'acontecimentos de 2024 a 2026 no Reino Unido e no Brasil.')
REAL_TWO = ('Adicionei na sua lista de estudos:\x1b[200~Novidades do macOS '
            'dos últimos dois anos.\x1b[201~')
PHRASES = [
    ('adiciona na sua lista de estudos: redes neurais', 'redes neurais'),
    ('adicione na minha lista de estudo: Rust ownership', 'Rust ownership'),
    ('adicionar à lista de estudos: história do Brasil', 'história do Brasil'),
    ('coloca na fila de estudos o protocolo TCP', 'o protocolo TCP'),
    ('põe na lista de estudos: SQLite WAL', 'SQLite WAL'),
    ('poe na lista de estudos SQLite WAL', 'SQLite WAL'),
    ('bota na sua lista de estudo: Docker', 'Docker'),
    ('inclui na lista de estudos: OAuth 2', 'OAuth 2'),
    ('inclua na lista de estudos: OAuth 2', 'OAuth 2'),
    ('acrescenta na lista de estudos: MLX', 'MLX'),
    ('coloca Kubernetes na sua lista de estudos', 'Kubernetes'),
    ('adiciona isso na lista de estudos: grafos', 'grafos'),
    ('Adicionei na sua lista de estudos: Novidades do macOS dos últimos dois anos.',
     'Novidades do macOS dos últimos dois anos'),
    *[(s, 'compiladores') for s in (
        'estuda essa noite: compiladores', 'estude hoje à noite compiladores',
        'estuda hoje a noite compiladores', 'estuda esta noite compiladores')],
    ('add to your study list: vector databases', 'vector databases'),
    ('add to the study queue vector databases', 'vector databases'),
    ('add vector databases to my study list', 'vector databases'),
    ('put vector databases on your study list', 'vector databases'),
    ('study tonight: compilers', 'compilers'),
    ('study compilers tonight', 'compilers'),
    (REAL_ONE, 'Hoje ePrincipais acontecimentos de 2024 a 2026 no Reino Unido e no Brasil'),
    (REAL_TWO, 'Novidades do macOS dos últimos dois anos'),
    ('estude : "SQLite WAL".', 'SQLite WAL'),
    ('study - “vector databases” !!', 'vector databases !!'),
    ('adiciona na lista de estudos, Docker.', 'Docker'),
    ('coloca Kubernetes na sua lista de estudos !!', 'Kubernetes !!'),
    ('add vector databases to my study list !!', 'vector databases !!'),
    ('estude grafos..', 'grafos'),
    *[(f'{verb} redes neurais', 'redes neurais') for verb in (
        'estude', 'estuda', 'estudar', 'estudo', 'etude', 'estud', 'etudo',
        'study', 'studar', 'studie')],
]


def turn(o, text, stream, language=None):
    if stream:
        frames = list(o.process_stream(text, language=language))
        return ''.join(f[0] for f in frames), frames[-1][2]
    return o.process(text, language=language)


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('text,topic', PHRASES)
def test_both_entry_points_dispatch_clean_topic(orch, stream, text, topic):
    orch._handle_add_study = Mock(return_value=('Queued.', 'STUDY_QUEUED'))
    orch._handle_planner_request = Mock(side_effect=AssertionError('planner reached'))
    assert turn(orch, text, stream) == ('Queued.', 'STUDY_QUEUED')
    orch._handle_add_study.assert_called_once_with(topic)
    orch._needs_realtime.assert_not_called()
    for engine in (orch.engine_analytical, orch.engine_creative):
        engine.chat.assert_not_called()
        engine.chat_stream.assert_not_called()
        engine.chat_with_tools.assert_not_called()


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('text,handler,mode', [
    ('lista de estudos', '_handle_list_studies', 'STUDY_LIST'),
    ('limpa a fila de estudos', '_handle_cancel_studies', 'STUDY_CANCELLED'),
    ('exclui o estudo 2', '_handle_cancel_studies', 'STUDY_CANCELLED'),
    ('adiciona na nota compras: leite', '_handle_note_append', 'NOTE_APPEND'),
    ('põe uma música', None, 'SPOTIFY_PLAY'),
])
def test_neighbor_routes_keep_ownership(orch, stream, text, handler, mode):
    orch._handle_add_study = Mock(side_effect=AssertionError('study stole a route'))
    if handler:
        setattr(orch, handler, Mock(return_value=('Owned.', mode)))
    else:
        orch.spotify.play_anything.return_value = 'Owned.'
    assert turn(orch, text, stream) == ('Owned.', mode)


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('text', [
    'o que tem na lista de estudos?', 'adiciona leite na lista de compras',
    'como eu adiciono algo na lista de estudos?', 'estudei ontem a matéria',
])
def test_non_commands_never_queue(orch, stream, text):
    orch._handle_add_study = Mock(side_effect=AssertionError('not a study command'))
    turn(orch, text, stream)
    orch._handle_add_study.assert_not_called()


@pytest.mark.parametrize('stream', [False, True])
def test_markers_removed_before_even_project_scope(orch, stream):
    seen = []

    @contextmanager
    def scope(text):
        seen.append(text)
        yield

    orch._project_scope = scope
    orch._process_core = Mock(return_value=('OK', 'OFFLINE'))
    orch._process_stream_impl = Mock(return_value=iter([('OK', True, 'OFFLINE', 'pt')]))
    turn(orch, '\x1b[201~a\x1b[200~b\x1b[200~c\x1b[201~\t\x07', stream)
    assert seen == ['abc\t\x07']


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('language,topic,word', [('pt', 'Hoje e notícias', 'Hoje'),
                                               ('en', 'weather Today', 'weather')])
def test_transient_reply_is_plain_localized(orch, tmp_study_queue, stream, language, topic, word):
    orch.study_queue = tmp_study_queue
    expected = {
        'pt': f'Não coloquei na lista: "{topic}" muda com o tempo (tem "{word}"). '
              'Pergunte direto que eu busco na internet. Para colocar mesmo assim, termine com !!',
        'en': f'I didn\'t add it: "{topic}" changes over time (it has "{word}"). '
              "Ask me directly and I'll look it up online. To add it anyway, end it with !!",
    }[language]
    assert turn(orch, f'study {topic}', stream, language) == (expected, 'STUDY_REJECTED_TRANSIENT')
    assert tmp_study_queue.list_pending() == []


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('language', ['pt', 'en'])
def test_real_queue_duplicate_and_force(orch, tmp_study_queue, stream, language):
    orch.study_queue = tmp_study_queue
    assert turn(orch, REAL_TWO, stream, language)[1] == 'STUDY_QUEUED'
    reply, mode = turn(orch, REAL_TWO, stream, language)
    assert mode == 'STUDY_REJECTED_DUPLICATE'
    assert reply == {
        'pt': 'Já tem um estudo parecido: "Novidades do macOS dos últimos dois anos" '
              '(na fila). Para estudar de novo mesmo assim, termine com !!',
        'en': 'There is already a similar study: "Novidades do macOS dos últimos dois anos" '
              '(in the queue). To study it anyway, end it with !!',
    }[language]
    assert turn(orch, REAL_TWO + ' !!', stream, language)[1] == 'STUDY_QUEUED'
    assert [e['topic'] for e in tmp_study_queue.list_pending()] == [
        'Novidades do macOS dos últimos dois anos'] * 2


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('language', ['pt', 'en'])
@pytest.mark.parametrize('topic', ['RAG vs fine-tuning', 'RAG vs finetuning'])
@pytest.mark.parametrize('status,status_pt,status_en', [
    ('pending', 'na fila', 'in the queue'),
    ('studying', 'em estudo', 'being studied'),
    ('saving', 'sendo salvo', 'being saved'),
    ('consumed', 'já estudado', 'already studied'),
    ('done', 'já estudado', 'already studied'),
    ('failed', 'falhou', 'failed'),
    ('unknown', 'no histórico', 'in the history'),
    (None, 'no histórico', 'in the history'),
])
def test_duplicate_names_existing_study_and_status(
        orch, tmp_study_queue, stream, language, topic, status, status_pt, status_en):
    orch.study_queue = tmp_study_queue
    existing = tmp_study_queue.add('RAG vs fine-tuning')
    if status is None:
        existing.pop('status')
        tmp_study_queue._rewrite_atomic([existing])
    else:
        tmp_study_queue.transition(existing['id'], ('pending',), status)
    before = tmp_study_queue._read_all()
    expected = {
        'pt': f'Já tem um estudo parecido: "RAG vs fine-tuning" ({status_pt}). '
              'Para estudar de novo mesmo assim, termine com !!',
        'en': f'There is already a similar study: "RAG vs fine-tuning" ({status_en}). '
              'To study it anyway, end it with !!',
    }[language]
    assert turn(orch, f'study {topic}', stream, language) == (expected, 'STUDY_REJECTED_DUPLICATE')
    assert tmp_study_queue._read_all() == before
    assert turn(orch, f'study {topic} !!', stream, language)[1] == 'STUDY_QUEUED'
    assert tmp_study_queue._read_all()[-1]['topic'] == topic


@pytest.mark.parametrize('stream', [False, True])
def test_real_first_turn_reaches_transient_rule(orch, tmp_study_queue, stream):
    orch.study_queue = tmp_study_queue
    reply, mode = turn(orch, REAL_ONE, stream, 'pt')
    assert mode == 'STUDY_REJECTED_TRANSIENT'
    assert '(tem "Hoje")' in reply
    assert '\x1b' not in reply
    assert not tmp_study_queue.list_pending()


@pytest.mark.parametrize('stream', [False, True])
def test_force_bypasses_transient_after_cleaning(orch, tmp_study_queue, stream):
    orch.study_queue = tmp_study_queue
    assert turn(orch, REAL_ONE + ' !!', stream, 'pt')[1] == 'STUDY_QUEUED'
    assert [e['topic'] for e in tmp_study_queue.list_pending()] == [
        'Hoje ePrincipais acontecimentos de 2024 a 2026 no Reino Unido e no Brasil']


@pytest.mark.parametrize('text', [p[0] for p in PHRASES if '\x1b' not in p[0]])
def test_registry_simulator_also_sees_add(text):
    match = simulate(text)
    assert match is not None and match.mode == 'STUDIER_ADD'
