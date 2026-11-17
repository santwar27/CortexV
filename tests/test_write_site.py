"""Task 124: tool-created web code, literal receipts and channel-local links."""
import importlib
import re
from unittest.mock import Mock

import pytest

from src.tools.local_writer import LocalFileWriter
from tests.test_loss_note import orch, taught_orch, answer  # noqa: F401
from tests.test_engine_mac_hallucination import rig  # noqa: F401
from tests.test_codigo import client, folder_key, HEADERS  # noqa: F401


HTML = '  <!DOCTYPE html>\r\n<html><p>Olá Queens Head!  </p></html>\r\n'
FAILURE = 'Não consegui salvar o código — nada foi gravado.'
DISPATCH = ['_execute_tool', '_execute_tool_unstaged']
PHRASES = [
    'onde está o arquivo?', 'onde esta o arquivo', 'onde salvou?',
    'onde você salvou', 'onde voce salvou?', 'onde ficou o arquivo?',
    'onde ficou o site', 'qual é o link?', 'qual o link', 'passa o link',
    'manda o link?', 'me dá o link', 'me da o link', 'cadê o arquivo?',
    'cade o link', 'where is the file?', 'where did you save it',
    'send me the link', 'give me the link?', "what's the link?",
]


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    # resolve /var -> /private/var on macOS: publisher refuses symlink walks.
    home = tmp_path.resolve() / 'home'
    home.mkdir()
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.setenv('CORTEX_API_KEY', 'test-fake')
    return home


@pytest.fixture
def site_orch(orch, isolated_home):
    orch.writer = LocalFileWriter(str(isolated_home / 'vault'))
    orch._observe_project = Mock()
    return orch


def write(o, dispatch, filename='the_queens_head.html', content=HTML):
    return getattr(o, dispatch)('write_document', {'filename': filename, 'content': content})


@pytest.mark.parametrize('dispatch', DISPATCH)
@pytest.mark.parametrize('filename', ['the_queens_head.html', 'the_queens_head',
                                    'the_queens_head.html.txt'])
@pytest.mark.parametrize('fenced', [False, True])
def test_real_tool_shapes_publish_exact_bytes_and_number(site_orch, isolated_home,
                                                       dispatch, filename, fenced):
    payload = '```html\r\n' + HTML + '```' if fenced else HTML
    for suffix in ('', ' (2)'):
        reply = write(site_orch, dispatch, filename, payload)
        folder = 'the_queens_head' + suffix
        target = isolated_home / 'Documents/Cortex/Codigo' / folder / 'the_queens_head.html'
        assert target.is_file(), reply
        assert target.read_bytes() == HTML.encode()
        assert reply.startswith(f'Salvei ~/Documents/Cortex/Codigo/{folder}/the_queens_head.html.')
        from urllib.parse import quote
        assert f'[Abrir o site](/ui/codigo/{folder_key(folder)}/{quote(folder)}/the_queens_head.html)' in reply
        assert '```html\n' + HTML + '```' in reply
    assert list((isolated_home / 'vault').iterdir()) == []


@pytest.mark.parametrize('dispatch', DISPATCH)
@pytest.mark.parametrize('filename,content,name,stem,language', [
    ('theme.css', ' body { color: red; }\n', 'theme.css', 'theme', 'css'),
    ('app.js', '  alert("oi");\n', 'app.js', 'app', 'js'),
    ('page.htm', '<HTML>hello</HTML>\n', 'page.html', 'page', 'html'),
    ('../../pub.html', HTML, 'pub.html', 'pub', 'html'),
    ('', HTML, 'index.html', 'site', 'html'),
    ('.css', 'a {}\n', 'styles.css', 'site', 'css'),
    ('.js', 'void 0;\n', 'script.js', 'site', 'js'),
    ('note', '\n <HtMl>hi</HTML>', 'note.html', 'note', 'html'),
    ('note.txt', HTML, 'note.html', 'note', 'html'),
])
def test_names_types_defaults(site_orch, isolated_home, dispatch, filename, content,
                             name, stem, language):
    reply = write(site_orch, dispatch, filename, content)
    target = isolated_home / 'Documents/Cortex/Codigo' / stem / name
    assert target.is_file(), reply
    assert target.read_bytes() == content.encode()
    assert f'```{language}\n{content}' in reply
    assert ('[Abrir o site]' in reply) == (language == 'html')
    assert not list((isolated_home / 'vault').iterdir())


@pytest.mark.parametrize('dispatch', DISPATCH)
def test_plain_notes_still_append(site_orch, isolated_home, dispatch):
    for _ in range(2):
        assert 'appended' in write(site_orch, dispatch, 'note', '  plain note  ')
    assert (isolated_home / 'vault/note.txt').read_bytes() == b'plain note\n\nplain note\n\n'
    assert not (isolated_home / 'Documents').exists()


@pytest.mark.parametrize('dispatch', DISPATCH)
def test_publish_failure_never_falls_back_to_notes(site_orch, isolated_home, monkeypatch, dispatch):
    module = importlib.import_module('src.core.save_code')
    monkeypatch.setattr(module, '_write_files', Mock(side_effect=OSError('disk failure')))
    assert write(site_orch, dispatch) == FAILURE
    assert not list((isolated_home / 'vault').iterdir())
    assert not list((isolated_home / 'Documents/Cortex/Codigo').iterdir())


def use_model(o, rig):
    engine, model, _, _ = rig
    o.engine_analytical = o.engine_creative = engine
    model.return_value = {'message': {'content': '', 'tool_calls': [{'function': {
        'name': 'write_document', 'arguments': {'filename': 'the_queens_head.html', 'content': HTML},
    }}]}}
    return engine, model


@pytest.mark.parametrize('path', ['sync', 'stream'])
def test_engine_keeps_literal_reply_without_followup(site_orch, rig, path):
    engine, model = use_model(site_orch, rig)
    text = 'Crie um site para The Queens Head, num arquivo index.html.'
    result = answer(site_orch, text, path)
    assert result.startswith('Salvei ~/Documents/Cortex/Codigo/the_queens_head/')
    assert '```html\n' + HTML + '```' in result
    model.assert_called_once()
    assert engine.history == [{'role': 'user', 'content': text},
                              {'role': 'assistant', 'content': result}]


@pytest.mark.parametrize('path', ['sync', 'stream'])
@pytest.mark.parametrize('phrase', PHRASES)
def test_where_link_is_literal(site_orch, path, phrase):
    saved = write(site_orch, '_execute_tool_unstaged')
    result = answer(site_orch, phrase, path,
                    language='pt',
                    conversation_history=[{'role': 'assistant', 'content': saved}])
    link = re.search(r'\[Abrir o site\]\([^\n]+\)', saved)[0]
    assert result == 'Está em ~/Documents/Cortex/Codigo/the_queens_head/the_queens_head.html. ' + link
    for engine in (site_orch.engine_analytical, site_orch.engine_creative):
        engine.chat.assert_not_called()
        engine.chat_with_tools.assert_not_called()
        engine.chat_stream.assert_not_called()


@pytest.mark.parametrize('path', ['sync', 'stream'])
def test_no_receipt_keeps_existing_routing_and_other_channel_is_ignored(site_orch, path):
    # Baseline today's route, then an unrelated channel's save must not alter it.
    calls = []
    site_orch._run_find_file = lambda args: calls.append(args) or 'existing file route'
    baseline = answer(site_orch, 'onde está o arquivo?', path, conversation_history=[])
    saved = write(site_orch, '_execute_tool_unstaged')
    site_orch.engine_analytical.history = [{'role': 'assistant', 'content': saved}]
    assert answer(site_orch, 'onde está o arquivo?', path, conversation_history=[]) == baseline
    assert not baseline.startswith('Está em ')


@pytest.mark.parametrize('path', ['sync', 'stream'])
@pytest.mark.parametrize('lang', ['pt', 'en'])
def test_where_last_save_non_html_and_gone(site_orch, isolated_home, path, lang):
    first = write(site_orch, '_execute_tool_unstaged')
    last = write(site_orch, '_execute_tool_unstaged', 'app.js', 'void 0;\n')
    messages = [{'role': 'assistant', 'content': r} for r in (first, last)]
    assert answer(site_orch, 'send me the link', path, language=lang, conversation_history=messages) == (
        'Está em ~/Documents/Cortex/Codigo/app/app.js.' if lang == 'pt' else
        "It's in ~/Documents/Cortex/Codigo/app/app.js.")
    folder = isolated_home / 'Documents/Cortex/Codigo/app'
    (folder / 'app.js').unlink()
    folder.rmdir()
    assert answer(site_orch, 'passa o link', path, language=lang, conversation_history=messages) == (
        'A pasta ~/Documents/Cortex/Codigo/app não existe mais.' if lang == 'pt' else
        'The folder ~/Documents/Cortex/Codigo/app no longer exists.')


def test_finder_after_tool_save(site_orch, isolated_home, monkeypatch):
    runner = Mock(return_value=(True, ''))
    monkeypatch.setattr('src.tools.macos_controller._run', runner)
    saved = write(site_orch, '_execute_tool_unstaged')
    assert answer(site_orch, 'mostra no Finder', 'stream', conversation_history=[
        {'role': 'assistant', 'content': saved}]) == 'Abri a pasta no Finder do Mac.'
    assert runner.call_args.args[0] == ['open', '-R', str(
        isolated_home / 'Documents/Cortex/Codigo/the_queens_head/the_queens_head.html')]


def test_ui_tool_reply_persisted_once(client, site_orch, rig, monkeypatch):
    from src.ui import routes
    engine, model = use_model(site_orch, rig)
    monkeypatch.setattr(routes, '_get_orchestrator', lambda request: site_orch)
    cid = routes.chat_store.create_conversation(title='Queens Head')['id']
    response = client.post('/ui/chat/stream', headers=HEADERS, json={
        'text': 'Crie um site para o pub, num arquivo index.html.', 'conversation_id': cid})
    assert response.status_code == 200
    turns = routes.chat_store.get_turns(cid)
    assert len(turns) == 1
    assert turns[0]['reply'].startswith('Salvei ~/Documents/Cortex/Codigo/')
    assert '```html\n' + HTML + '```' in turns[0]['reply']
    assert turns[0]['status_mode'] == 'SAVE_CODE'
    assert engine.history == []
    model.assert_called_once()


@pytest.mark.parametrize('refined', [False, True])
def test_regex_write_path_cannot_append_web_code(site_orch, isolated_home, refined):
    engine = site_orch.engine_analytical
    engine.chat.side_effect = lambda *a, **kw: HTML
    result, mode = site_orch._handle_write(engine, 'pub.html', 'make a site' if refined else HTML)
    assert mode == 'SAVE_CODE'
    assert result.startswith('Salvei ~/Documents/Cortex/Codigo/pub/pub.html.')
    assert (isolated_home / 'Documents/Cortex/Codigo/pub/pub.html').read_bytes() == HTML.encode()
    assert not list((isolated_home / 'vault').iterdir())


def test_approved_plan_keeps_literal_code_and_history(site_orch, monkeypatch):
    site_orch.engine_analytical.model_name = 'fake'
    summary = Mock(side_effect=AssertionError('code must not be summarised'))
    monkeypatch.setattr('src.core.staging_mixin._planner.summarise', summary)
    reply, mode = site_orch._execute_staged_plan({'args': {'passos': [
        {'id': 'p1', 'tool': 'write_document', 'args': {
            'filename': 'the_queens_head.html', 'content': HTML}},
    ]}})
    assert mode == 'SAVE_CODE'
    assert reply.startswith('Salvei ~/Documents/Cortex/Codigo/')
    assert '```html\n' + HTML + '```' in reply
    assert site_orch.engine_analytical.history[-1] == {'role': 'assistant', 'content': reply}
    summary.assert_not_called()


def test_where_keeps_original_html_link_for_multi_file_receipt(site_orch):
    from src.core.save_code import save_last_code
    saved = save_last_code('save this code', [{'role': 'assistant', 'content':
        '**z.html**\n```html\nz\n```\n**a.html**\n```html\na\n```'}], 'pt')
    link = re.search(r'\[Abrir o site\]\([^\n]+\)', saved)[0]
    result = answer(site_orch, 'passa o link', 'stream', language='pt',
                    conversation_history=[{'role': 'assistant', 'content': saved}])
    assert result == 'Está em ~/Documents/Cortex/Codigo/site/z.html. ' + link


@pytest.mark.parametrize('path', ['sync', 'stream'])
def test_no_receipt_still_reaches_find_file(site_orch, rig, path):
    engine, model = use_model(site_orch, rig)
    model.side_effect = [
        {'message': {'content': '', 'tool_calls': [{'function': {
            'name': 'find_file', 'arguments': {'query': 'arquivo'}}}]}},
        {'message': {'content': 'Which file?'}},
    ]
    site_orch._run_find_file = Mock(return_value='ten hits')
    assert answer(site_orch, 'onde está o arquivo?', path, conversation_history=[]) == 'Which file?'
    site_orch._run_find_file.assert_called_once_with({'query': 'arquivo'})
    assert model.call_count == 2


@pytest.mark.parametrize('dispatch', DISPATCH)
def test_missing_secret_fails_before_publication(site_orch, isolated_home, monkeypatch, dispatch):
    monkeypatch.delenv('CORTEX_API_KEY')
    assert write(site_orch, dispatch) == FAILURE
    assert not (isolated_home / 'Documents').exists()
    assert not list((isolated_home / 'vault').iterdir())


@pytest.mark.parametrize('content', ['```html\n<html>broken', '```html\n<html>ok\n```\nextra'])
def test_incomplete_or_mixed_fence_never_silently_discards_content(site_orch, content):
    assert write(site_orch, '_execute_tool', content=content) == FAILURE
