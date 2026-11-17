"""Task 123: folder-keyed, sandboxed saved code and channel-local Finder."""
import base64
import hashlib
import hmac
import importlib
import os
from urllib.parse import quote, unquote, urljoin
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.core.text_utils import strip_markdown_for_speech
from src.ui.routes import router
from tests.test_loss_note import orch, answer, PATHS, taught_orch  # noqa: F401
from tests.test_recall_last import no_model_or_retrieval


CSP = ("sandbox allow-scripts allow-forms; "
       "default-src 'self' 'unsafe-inline' data:; connect-src 'none'")
HEADERS = {'X-Cortex-Key': 'test-fake'}
HTML = '<link rel="stylesheet" href="styles.css"><script src="script.js"></script>'
CODE = '```html\n' + HTML + '\n```\n```css\nbody {}\n```\n```js\nvoid 0;\n```'


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('CORTEX_API_KEY', 'test-fake')
    return tmp_path


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def folder(home):
    folder = home / 'Documents/Cortex/Codigo/bistrô (2)'
    folder.mkdir(parents=True)
    for name, content in {'index.html': HTML, 'styles.css': 'body {}',
                          'script.js': 'void 0;', 'data.json': '{"x":1}',
                          'readme.md': '<script>alert(1)</script>',
                          'notes.txt': '<h1>text</h1>'}.items():
        (folder / name).write_text(content)
    return folder


def folder_key(folder):
    secret = hmac.digest(b'test-fake', b'cortex-codigo-v1', hashlib.sha256)
    return base64.urlsafe_b64encode(hmac.digest(
        secret, folder.encode('utf-8'), hashlib.sha256)).rstrip(b'=').decode('ascii')


def url(folder, name='index.html'):
    return (f'/ui/codigo/{folder_key(folder.name)}/'
            + quote(folder.name, safe='') + '/' + quote(name, safe=''))


@pytest.mark.parametrize('name,mime', [('index.html', 'text/html'), ('styles.css', 'text/css'),
    ('script.js', 'text/javascript'), ('data.json', 'application/json'),
    ('readme.md', 'text/plain'), ('notes.txt', 'text/plain')])
def test_route_types_bytes_and_isolation_headers(client, folder, name, mime):
    response = client.get(url(folder, name))
    assert not response.request.headers.get("cookie")
    assert not response.request.headers.get("x-cortex-key")
    assert response.status_code == 200
    assert response.content == (folder / name).read_bytes()
    assert response.headers['content-type'].split(';')[0] == mime
    assert response.headers['content-security-policy'] == CSP
    assert 'allow-same-origin' not in response.headers['content-security-policy']
    assert response.headers['x-content-type-options'] == 'nosniff'
    assert response.headers['referrer-policy'] == 'no-referrer'
    assert response.headers['cache-control'] == 'no-store'


@pytest.mark.parametrize('credentials', ['none', 'header', 'cookie'])
def test_wrong_other_folder_and_missing_key_are_404(client, folder, credentials):
    headers = HEADERS if credentials == 'header' else {}
    if credentials == 'cookie':
        assert client.post('/ui/login', data={'key': 'test-fake'},
                           follow_redirects=False).status_code == 303
    root = folder.parent
    for name in ('site', 'site (2)'):
        (root / name).mkdir()
        (root / name / 'index.html').write_text(HTML)
    missing = client.get(url(root / 'missing'), headers=headers)
    assert missing.status_code == 404
    paths = [
        url(folder).replace(folder_key(folder.name), 'wrong'),
        url(folder).replace(folder_key(folder.name), 'é'),
        url(folder).replace(folder_key(folder.name), 'test-fake'),
        url(root / 'site (2)').replace(folder_key('site (2)'), folder_key('site')),
        '/ui/codigo/' + quote(folder.name, safe='') + '/index.html',
        '/ui/codigo//site/index.html',
    ]
    for path in paths:
        response = client.get(path, headers=headers, follow_redirects=False)
        assert response.status_code == 404, path
        assert response.content == missing.content
    assert client.get(url(root / 'site')).status_code == 200
    assert client.get(url(root / 'site (2)')).status_code == 200


def test_relative_assets_without_cookie(client, folder):
    page = client.get(url(folder))
    assert page.status_code == 200
    for name in ('styles.css', 'script.js'):
        assert name in page.text
        asset = client.get(urljoin(url(folder), name))
        assert asset.status_code == 200
        assert not asset.request.headers.get('cookie')
        assert asset.headers['content-security-policy'] == CSP
    assert client.get('/ui/data').status_code == 401


@pytest.mark.parametrize('path', [
    '/ui/codigo/site/../secret.txt', '/ui/codigo/%2e%2e/secret.txt',
    '/ui/codigo/site/%2e%2e', '/ui/codigo/site/%2e%2e%2fsecret.txt',
    '/ui/codigo/site/nested%2fsecret.txt', '/ui/codigo/site/%2fetc%2fpasswd',
    '/ui/codigo/%2fetc/passwd', '/ui/codigo/.hidden/secret.txt',
    '/ui/codigo/site/.secret.txt', '/ui/codigo/site/nested/secret.txt',
    '/ui/codigo/site', '/ui/codigo/site/', '/ui/codigo/site/index.html/extra',
    '/ui/codigo/site/secret.txt', '/ui/codigo/link/secret.txt',
    '/ui/codigo/site/directory.txt', '/ui/codigo/site/pipe.txt',
    '/ui/codigo/site/run.svg', '/ui/codigo/site/..%5csecret.txt',
    '/ui/codigo/site/%00.txt',
])
def test_refused_paths_are_indistinguishable_from_missing(client, home, path):
    root = home / 'Documents/Cortex/Codigo'
    site = root / 'site'
    site.mkdir(parents=True)
    outside = home / 'outside'
    outside.mkdir()
    (outside / 'secret.txt').write_text('secret')
    (site / 'secret.txt').symlink_to(outside / 'secret.txt')
    (root / 'link').symlink_to(outside, target_is_directory=True)
    (site / '.secret.txt').write_text('secret')
    (site / 'directory.txt').mkdir()
    (site / 'run.svg').write_text('<svg/>')
    os.mkfifo(site / 'pipe.txt')
    folder_name = path.split('/')[3]
    path = path.replace('/ui/codigo/', '/ui/codigo/' + folder_key(unquote(folder_name)) + '/', 1)
    response = client.get(path, follow_redirects=False)
    missing = client.get(url(root / 'missing', 'missing.txt'))
    assert response.status_code == missing.status_code == 404
    assert response.content == missing.content


@pytest.mark.parametrize('part', ['Documents', 'Cortex', 'Codigo'])
def test_route_refuses_symlinked_ancestors(client, home, part):
    outside = home / 'outside'
    outside.mkdir()
    parent = home
    for component in ('Documents', 'Cortex', 'Codigo'):
        child = parent / component
        if component == part:
            child.symlink_to(outside, target_is_directory=True)
            break
        child.mkdir()
        parent = child
    assert client.get(url(home / 'site')).status_code == 404


def test_route_resolves_home_each_call(client, folder, home, monkeypatch):
    assert client.get(url(folder), headers=HEADERS).status_code == 200
    other = home / 'other'
    other.mkdir()
    monkeypatch.setenv('HOME', str(other))
    assert client.get(url(folder), headers=HEADERS).status_code == 404


def history(code=CODE):
    return [{'role': 'assistant', 'content': code}]


@pytest.mark.parametrize('lang,label', [('pt', 'Abrir o site'), ('en', 'Open the site')])
@pytest.mark.parametrize('path', PATHS)
def test_reply_encoded_link_and_speech(orch, home, path, lang, label):
    command = 'save this code as bistrô (2).html'
    saved = answer(orch, command, path, language=lang, conversation_history=history())
    link = '/ui/codigo/' + folder_key('bistrô (2)') + '/bistr%C3%B4%20%282%29/bistr%C3%B4%20%282%29.html'
    assert saved.endswith(f' [{label}]({link})')
    assert '~/Documents/Cortex/Codigo/bistrô (2)/bistrô (2).html' in saved or (
        '~/Documents/Cortex/Codigo/bistrô (2):' in saved)
    spoken = strip_markdown_for_speech(saved)
    assert spoken.endswith(label)
    assert '/ui/codigo/' not in spoken
    no_model_or_retrieval(orch)


@pytest.mark.parametrize('code', ['```txt\nhello\n```', '```json\n{}\n```'])
def test_no_html_no_link(home, code):
    mod = importlib.import_module('src.core.save_code')
    saved = mod.save_last_code('save this code', history(code), 'en')
    assert saved.startswith('Saved ')
    assert '/ui/codigo/' not in saved


@pytest.fixture
def runner(monkeypatch):
    runner = Mock(return_value=(True, ''))
    monkeypatch.setattr('src.tools.macos_controller._run', runner)
    return runner


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('phrase', ['mostra no finder', 'abre no finder', 'abre a pasta do site',
    'abre a pasta do código', 'abre a pasta do codigo', 'show in finder',
    'show it in finder', 'open the folder', 'Cortex, can you show it in Finder please?'])
def test_finder_phrases_and_argv(orch, home, runner, path, phrase):
    messages = history()
    saved = answer(orch, 'save this code', path, conversation_history=messages)
    messages += [{'role': 'assistant', 'content': saved}]
    result = answer(orch, phrase, path, language='en', conversation_history=messages)
    assert result == "I opened the folder in the Mac's Finder."
    runner.assert_called_once_with(['open', '-R', str(home / 'Documents/Cortex/Codigo/site/index.html')],
                                   timeout=10, lang='en')
    no_model_or_retrieval(orch)


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('lang', ['pt', 'en'])
def test_finder_no_save_and_other_channel(orch, runner, path, lang):
    answer(orch, 'save this code', path, conversation_history=history())
    result = answer(orch, 'mostra no Finder', path, language=lang, conversation_history=[])
    assert result == ('Ainda não salvei código nesta conversa.' if lang == 'pt' else
                      "I haven't saved code in this conversation yet.")
    runner.assert_not_called()
    no_model_or_retrieval(orch)


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('lang', ['pt', 'en'])
def test_finder_voice_remembers_last_save_and_non_html(orch, home, runner, path, lang):
    orch.engine_analytical.history = history('```txt\nhello\n```')
    answer(orch, 'save this code', path, language=lang)
    result = answer(orch, 'mostra no finder', path, language=lang)
    assert result == ('Abri a pasta no Finder do Mac.' if lang == 'pt' else
                      "I opened the folder in the Mac's Finder.")
    assert runner.call_args.args[0] == ['open', '-R', str(home / 'Documents/Cortex/Codigo/codigo/codigo.txt')]


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('lang', ['pt', 'en'])
def test_finder_folder_gone(orch, home, runner, path, lang):
    saved = answer(orch, 'save this code', path, language=lang,
                   conversation_history=history('```html\nhello\n```'))
    folder = home / 'Documents/Cortex/Codigo/site'
    (folder / 'index.html').unlink()
    folder.rmdir()
    result = answer(orch, 'mostra no finder', path, language=lang,
                    conversation_history=[{'role': 'assistant', 'content': saved}])
    assert result == ('A pasta ~/Documents/Cortex/Codigo/site não existe mais.' if lang == 'pt' else
                      'The folder ~/Documents/Cortex/Codigo/site no longer exists.')
    runner.assert_not_called()


@pytest.mark.parametrize('folder_name', ['../outside', '/etc', '.hidden', 'site/nested', 'link'])
def test_finder_path_confinement(orch, home, runner, folder_name):
    outside = home / 'outside'
    outside.mkdir()
    (outside / 'index.html').write_text('secret')
    root = home / 'Documents/Cortex/Codigo'
    root.mkdir(parents=True)
    (root / 'link').symlink_to(outside, target_is_directory=True)
    saved = f'Saved ~/Documents/Cortex/Codigo/{folder_name}/index.html.'
    answer(orch, 'show it in finder', 'sync', conversation_history=[{'role': 'assistant', 'content': saved}])
    runner.assert_not_called()


def test_finder_last_save_and_runner_failure(orch, home, runner):
    first = answer(orch, 'save this code', 'sync', conversation_history=history())
    second = answer(orch, 'save this code', 'sync', conversation_history=history())
    runner.return_value = (False, 'failure')
    result = answer(orch, 'show in finder', 'sync', language='en', conversation_history=[
        {'role': 'assistant', 'content': first}, {'role': 'assistant', 'content': second}])
    assert runner.call_args.args[0][-1] == str(home / 'Documents/Cortex/Codigo/site (2)/index.html')
    assert result == "I couldn't open the folder in the Mac's Finder."


@pytest.mark.parametrize('part', ['Documents', 'Cortex', 'Codigo', 'site', 'index.html'])
def test_finder_refuses_symlinks_at_every_level(orch, home, runner, part):
    outside = home / 'outside'
    outside.mkdir()
    (outside / 'index.html').write_text('secret')
    parent = home
    for component in ('Documents', 'Cortex', 'Codigo', 'site', 'index.html'):
        child = parent / component
        if component == part:
            child.symlink_to(outside / 'index.html' if part == 'index.html' else outside)
            break
        child.mkdir()
        parent = child
    messages = [{'role': 'assistant', 'content': 'Saved ~/Documents/Cortex/Codigo/site/index.html.'}]
    answer(orch, 'show in finder', 'sync', conversation_history=messages)
    runner.assert_not_called()


@pytest.mark.parametrize('path', PATHS)
def test_finder_ignores_user_receipts_and_does_not_announce(orch, runner, path):
    from src.core.orchestrator import CortexOrchestrator
    for name in ('_maybe_announce_pending', '_maybe_announce_abandoned'):
        setattr(orch, name, getattr(CortexOrchestrator, name).__get__(orch))
    orch.studies_review = Mock()
    orch.studies_review.list_pending.side_effect = AssertionError('unrelated notice')
    result = answer(orch, 'mostra no finder', path, language='pt', conversation_history=[
        {'role': 'user', 'content': 'Salvei ~/Documents/Cortex/Codigo/site/index.html.'}])
    assert result == 'Ainda não salvei código nesta conversa.'
    runner.assert_not_called()


@pytest.mark.parametrize('phrase', ['não mostra no finder', 'do not show it in finder',
    'explain show in finder', 'open the folder /etc', 'abre a pasta do site e apaga tudo'])
def test_finder_negative_phrases_do_not_route(orch, runner, phrase):
    mod = importlib.import_module('src.core.save_code')
    assert not mod.is_finder_request(phrase)
    assert answer(orch, phrase, 'sync')
    runner.assert_not_called()


def test_finder_spaces_and_non_html_multi_file(orch, home, runner):
    saved = answer(orch, 'save this code as meu código (2).txt', 'sync', language='pt',
                   conversation_history=history('```txt\nfirst\n```\n```json\n{}\n```'))
    result = answer(orch, 'abre a pasta do código', 'stream', language='pt',
                    conversation_history=[{'role': 'assistant', 'content': saved}])
    assert result == 'Abri a pasta no Finder do Mac.'
    assert runner.call_args.args[0] == ['open', '-R', str(
        home / 'Documents/Cortex/Codigo/meu código (2)/codigo.json')]


@pytest.mark.parametrize('path', PATHS)
def test_save_global_history_strictly_alternates(orch, home, path):
    messages = [{'role': 'user', 'content': 'Make a site'},
                {'role': 'assistant', 'content': CODE}]
    orch.engine_analytical.history = messages
    for command in ('save this code', 'save this code as another.html'):
        receipt = answer(orch, command, path, language='en')
        assert orch.engine_analytical.history[-2:] == [
            {'role': 'user', 'content': command},
            {'role': 'assistant', 'content': receipt}]
        assert [m['role'] for m in orch.engine_analytical.history] == [
            'user', 'assistant'] * (len(orch.engine_analytical.history) // 2)


def test_ui_save_persists_one_pair_only(client, orch, monkeypatch):
    from src.ui import routes
    monkeypatch.setattr(routes, '_get_orchestrator', lambda request: orch)
    convo = routes.chat_store.create_conversation(title='Saved site')
    cid = convo['id']
    routes.chat_store.append_turn(conversation_id=cid, user='Make a site', reply=CODE)
    global_history = [{'role': 'user', 'content': 'Voice'},
                      {'role': 'assistant', 'content': 'Voice reply'}]
    orch.engine_analytical.history = list(global_history)
    response = client.post('/ui/chat/stream', headers=HEADERS, json={
        'text': 'save this code', 'conversation_id': cid, 'language': 'en'})
    assert response.status_code == 200
    turns = routes.chat_store.get_turns(cid)
    assert len(turns) == 2
    assert turns[-1]['user'] == 'save this code'
    assert turns[-1]['status_mode'] == 'SAVE_CODE'
    assert '/ui/codigo/' + folder_key('site') + '/site/index.html' in turns[-1]['reply']
    assert [m['role'] for m in routes._conversation_history_msgs(cid, 5)] == [
        'user', 'assistant', 'user', 'assistant']
    assert orch.engine_analytical.history == global_history


def test_key_rotation_and_missing_secret_fail_closed(client, folder, monkeypatch):
    assert client.get(url(folder)).status_code == 200
    monkeypatch.setenv('CORTEX_API_KEY', 'rotated')
    assert client.get(url(folder)).status_code == 404
    monkeypatch.delenv('CORTEX_API_KEY')
    assert client.get(url(folder)).status_code == 404
