"""Task 120: real files under fake HOME, fake models, no preview execution."""
import errno
import importlib
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from src.core.orchestrator import CortexOrchestrator
from tests.test_loss_note import orch, answer, PATHS, taught_orch  # noqa: F401
from tests.test_recall_last import no_model_or_retrieval
from tests.test_engine_mac_hallucination import rig  # noqa: F401


HTML = '<!DOCTYPE html>\r\n<html lang="pt">\r\n  <p>Olá!  </p>\r\n</html>\r\n'
CSS = 'body {\n  color: #123;  \n}\n'
SINGLE = 'Aqui está:\n```html\r\n' + HTML + '```\n'
NAMED = '**index.html**\n```html\r\n' + HTML + '```\n\n**styles.css**\n```css\n' + CSS + '```'
MISSING = {
    'pt': 'Não achei código nas últimas respostas — me peça o código de novo.',
    'en': "I couldn't find code in the last replies — ask me for the code again.",
}
FAILURE = {
    'pt': 'Não consegui salvar o código — nada foi gravado.',
    'en': "I couldn't save the code — nothing was written.",
}


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('CORTEX_API_KEY', 'test-fake')
    return tmp_path


def history(reply):
    return [{'role': 'user', 'content': 'Faça um site de restaurante.'},
            {'role': 'assistant', 'content': reply}]


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('reply,files', [(SINGLE, {'index.html': HTML}),
                                      (NAMED, {'index.html': HTML, 'styles.css': CSS})])
@pytest.mark.parametrize('lang', ['pt', 'en'])
def test_pedros_replies_saved_verbatim(orch, home, path, reply, files, lang):
    # The contract's literal response must survive pending announcement hooks.
    for name in ('_maybe_announce_pending', '_maybe_announce_abandoned'):
        setattr(orch, name, getattr(CortexOrchestrator, name).__get__(orch))
    orch.studies_review = Mock()
    orch.studies_review.list_pending.side_effect = AssertionError('unrelated notice')
    command_text = 'crie um doc com esse codigo para que eu possa abrir o preview dele'
    actual = answer(orch, command_text, path, language=lang, conversation_history=history(reply))
    folder = home / 'Documents/Cortex/Codigo/site'
    assert folder.is_dir(), actual
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == {
        name: content.encode('utf-8') for name, content in files.items()}
    if len(files) == 1:
        expected = ('Salvei ~/Documents/Cortex/Codigo/site/index.html. Abra no navegador para ver.'
                    if lang == 'pt' else
                    'Saved ~/Documents/Cortex/Codigo/site/index.html. Open it in a browser to view it.')
    else:
        expected = ('Salvei 2 arquivos em ~/Documents/Cortex/Codigo/site: index.html, styles.css. '
                    'Abra o index.html no navegador para ver.' if lang == 'pt' else
                    'Saved 2 files in ~/Documents/Cortex/Codigo/site: index.html, styles.css. '
                    'Open index.html in a browser to view it.')
    label = 'Abrir o site' if lang == 'pt' else 'Open the site'
    from tests.test_codigo import folder_key
    assert actual == expected + f' [{label}](/ui/codigo/{folder_key("site")}/site/index.html)'
    no_model_or_retrieval(orch)


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('command_text', [
    'salva esse código', 'grave esse codigo', 'grava esse html', 'salve esse css',
    'cria um arquivo com esse código', 'salva um documento com esse site',
    'crie um doc com esse codigo para eu abrir o preview',
    'save this code', 'save that html', 'put that code in a file',
    'Cortex, can you save this code please?', 'salva o código',
    'grava isso em um arquivo de código',
    'pode salvar esse código?', 'você pode criar um doc com esse código?',
    'salva esse código em um arquivo', 'save this code in a file',
])
def test_save_forms_never_reach_model(orch, home, path, command_text):
    answer(orch, command_text, path, conversation_history=history(SINGLE))
    assert (home / 'Documents/Cortex/Codigo/site/index.html').read_bytes() == HTML.encode()
    no_model_or_retrieval(orch)


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('command_text,relative_path', [
    ('me salva esse codigo por favor', 'site/index.html'),
    ('me cria um arquivo com esse código', 'site/index.html'),
    ('me grava esse html', 'site/index.html'),
    ('Cortex, me salva esse código', 'site/index.html'),
    ('por favor, me grave esse código', 'site/index.html'),
    ('Cortex, você pode me criar um doc com esse codigo?', 'site/index.html'),
    ('Cortex, me salva esse código em bistrô.html por favor', 'bistrô/bistrô.html'),
])
def test_leading_me_save_forms(orch, home, path, command_text, relative_path):
    reply = answer(orch, command_text, path, conversation_history=history(SINGLE))
    assert (home / 'Documents/Cortex/Codigo' / relative_path).read_bytes() == HTML.encode()
    assert f'~/Documents/Cortex/Codigo/{relative_path}' in reply
    no_model_or_retrieval(orch)


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('command_text', [
    'me save this code', 'me put that code in a file',
    'não me salva esse código', 'me explica esse código',
    'mesalva esse código', 'me me salva esse código',
])
def test_leading_me_does_not_widen_other_requests(orch, home, path, command_text):
    assert module().parse_request(command_text) is None
    assert answer(orch, command_text, path, conversation_history=history(SINGLE))
    assert not (home / 'Documents/Cortex/Codigo').exists()


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('command_text', [
    'podemos usar CSS para esse site', 'explica esse código', 'explain this code',
    'save my notes about code', 'cria um site de restaurante',
    'não salva esse código', 'do not save this code',
    'save this recipe in a file', 'cria um documento sobre código',
])
def test_non_save_requests_do_not_route(orch, home, path, command_text):
    assert answer(orch, command_text, path, conversation_history=history(SINGLE))
    assert not (home / 'Documents/Cortex/Codigo').exists()


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('lang', ['pt', 'en'])
def test_missing_code_and_other_channel(orch, home, path, lang):
    orch.engine_analytical.history = history(SINGLE)
    orch.engine_creative.history = history(NAMED)
    for messages in ([], history('No code here.'), [{'role': 'user', 'content': SINGLE}]):
        assert answer(orch, 'save this code', path, language=lang,
                      conversation_history=messages) == MISSING[lang]
    assert not (home / 'Documents/Cortex/Codigo').exists()
    no_model_or_retrieval(orch)


@pytest.mark.parametrize('path', PATHS)
def test_latest_code_reply_and_voice_history(orch, home, path):
    orch.engine_analytical.history = history(NAMED) + history(SINGLE) + history('You are welcome.')
    answer(orch, 'salva esse código', path)
    folder = home / 'Documents/Cortex/Codigo/site'
    assert [p.name for p in folder.iterdir()] == ['index.html']
    assert (folder / 'index.html').read_bytes() == HTML.encode()


def module():
    return importlib.import_module('src.core.save_code')


def save(reply, command_text='salva esse código', lang='pt'):
    return module().save_last_code(command_text, history(reply), lang)


def test_names_languages_and_collision_suffixes(home):
    reply = ('index.html:\n```html\na\n```\n`styles.css`\n```css\nb\n```\n'
             '```html\nc\n```\n```css\nd\n```\n```js\ne\n```\n'
             '```javascript\nf\n```\n```json\n{}\n```\n```md\ng\n```\n'
             '```markdown\nh\n```\n```txt\ni\n```\n```\nj\n```\n'
             '**runner.py**\n```python\nprint(1)\n```\n```sh\necho hi\n```')
    save(reply)
    folder = home / 'Documents/Cortex/Codigo/site'
    names = ['index.html', 'styles.css', 'index-2.html', 'styles-2.css', 'script.js',
             'script-2.js', 'codigo.json', 'codigo.md', 'codigo-2.md', 'codigo.txt',
             'codigo-2.txt', 'runner.py.txt', 'codigo-3.txt']
    assert {p.name for p in folder.iterdir()} == set(names)
    assert (folder / 'runner.py.txt').read_bytes() == b'print(1)\n'


@pytest.mark.parametrize('command_text', ['salva esse código em bistrô.html',
                                     'save this code as bistrô.html',
                                     'save that code to bistrô.html'])
@pytest.mark.parametrize('reply', [SINGLE, NAMED])
def test_request_name_overrides_single_or_html(home, command_text, reply):
    save(reply, command_text)
    folder = home / 'Documents/Cortex/Codigo/bistrô'
    assert (folder / 'bistrô.html').read_bytes() == HTML.encode()
    if reply == NAMED:
        assert (folder / 'styles.css').read_bytes() == CSS.encode()


@pytest.mark.parametrize('name,expected', [('../x.html', 'x.html'), ('/etc/x', 'x.txt'),
                                         ('.bashrc', 'bashrc.txt'), ('..\\x.html', 'x.html'),
                                         ('..a\x01b.html', 'ab.html')])
@pytest.mark.parametrize('location', ['command_text', 'label', 'bare'])
def test_malicious_names_are_plain_basenames(home, name, expected, location):
    command_text = 'salva esse código' + (f' em {name}' if location == 'command_text' else '')
    prefix = f'**{name}**\n' if location == 'label' else f'{name}:\n' if location == 'bare' else ''
    reply = prefix + SINGLE.split('Aqui está:\n')[1]
    save(reply, command_text)
    root = home / 'Documents/Cortex/Codigo'
    [folder] = root.iterdir()
    assert [p.name for p in folder.iterdir()] == [expected]
    assert (folder / expected).read_bytes() == HTML.encode()
    assert not folder.name.startswith('.') and '..' not in folder.name


def test_existing_folders_and_exclusive_race(home, monkeypatch):
    mod = module()
    root = home / 'Documents/Cortex/Codigo'
    (root / 'site').mkdir(parents=True)
    (root / 'site/index.html').write_bytes(b'keep')
    original = mod._rename_exclusive
    raced = False
    def occupy(fd, source, name):
        nonlocal raced
        if not raced and not (root / name).exists():
            raced = True
            (root / name).mkdir()
            (root / name / 'sentinel').write_bytes(b'keep race')
        return original(fd, source, name)
    monkeypatch.setattr(mod, '_rename_exclusive', occupy)
    reply = save(SINGLE)
    assert '~/Documents/Cortex/Codigo/site (3)/index.html' in reply
    assert (root / 'site/index.html').read_bytes() == b'keep'
    assert (root / 'site (2)/sentinel').read_bytes() == b'keep race'
    assert (root / 'site (3)/index.html').read_bytes() == HTML.encode()
    assert len(list(root.iterdir())) == 3


@pytest.mark.parametrize('lang', ['pt', 'en'])
def test_no_html_has_no_browser_instruction(home, lang):
    reply = save('```css\n' + CSS + '```', lang=lang)
    assert reply == ('Salvei ~/Documents/Cortex/Codigo/codigo/styles.css.' if lang == 'pt'
                     else 'Saved ~/Documents/Cortex/Codigo/codigo/styles.css.')


@pytest.mark.parametrize('operation', ['write', 'read', 'fsync'])
def test_failure_leaves_no_partial_save(home, monkeypatch, operation):
    mod = module()
    original = getattr(os, operation)
    calls = 0
    def fail(fd, *args):
        nonlocal calls
        if stat.S_ISREG(os.fstat(fd).st_mode):
            calls += 1
            if calls == 2:
                raise OSError('injected failure')
        return original(fd, *args)
    monkeypatch.setattr(mod.os, operation, fail)
    assert save(NAMED) == FAILURE['pt']
    root = home / 'Documents/Cortex/Codigo'
    assert not root.exists() or list(root.iterdir()) == []


def test_exclusive_unavailable_has_no_replacing_fallback(home, monkeypatch):
    mod = module()
    monkeypatch.setattr(mod, '_rename_exclusive', Mock(side_effect=OSError(errno.ENOSYS, 'unsupported')))
    monkeypatch.setattr(mod.os, 'rename', Mock(side_effect=AssertionError('unsafe rename')))
    assert save(SINGLE, lang='en') == FAILURE['en']
    assert list((home / 'Documents/Cortex/Codigo').iterdir()) == []


@pytest.mark.parametrize('part', ['Documents', 'Cortex', 'Codigo'])
def test_output_root_never_follows_symlinks(home, monkeypatch, part):
    outside = home / 'elsewhere'
    outside.mkdir()
    parent = home
    for component in ('Documents', 'Cortex', 'Codigo'):
        path = parent / component
        if component == part:
            path.symlink_to(outside)
            break
        path.mkdir()
        parent = path
    assert save(SINGLE) == FAILURE['pt']
    assert list(outside.iterdir()) == []


def test_concurrent_saves_publish_complete_distinct_folders(home):
    with ThreadPoolExecutor(max_workers=4) as pool:
        replies = list(pool.map(lambda _: save(NAMED), range(4)))
    assert all(r.startswith('Salvei 2 arquivos') for r in replies)
    folders = list((home / 'Documents/Cortex/Codigo').iterdir())
    assert len(folders) == 4
    for folder in folders:
        assert {p.name: p.read_bytes() for p in folder.iterdir()} == {
            'index.html': HTML.encode(), 'styles.css': CSS.encode()}


def test_old_and_unclosed_fences_are_not_guessed(home):
    assert module().save_last_code('save this code', history(SINGLE) + history('No code') * 5, 'en') == MISSING['en']
    assert save('```html\nunfinished') == MISSING['pt']


@pytest.mark.parametrize('entry', ['chat', 'chat_stream'])
@pytest.mark.parametrize('reply', [SINGLE, NAMED])
def test_real_engine_history_keeps_pedros_fences_whole(rig, entry, reply, monkeypatch):
    from src.core.memory import CortexMemory
    engine, model, _, _ = rig
    # rig fakes the constructor; restore the class's refusal classifier.
    monkeypatch.setattr('src.core.engine.CortexMemory', CortexMemory)
    message = {'message': {'content': reply}}
    model.return_value = iter([message]) if entry == 'chat_stream' else message
    result = getattr(engine, entry)('Faça um site de restaurante.')
    if entry == 'chat_stream':
        list(result)
    saved = save(engine.history[-1]['content'])
    assert saved.startswith('Salvei ')
    assert engine.history[-1]['content'] == reply.strip()


@pytest.mark.parametrize('reply,command_text,expected', [
    ('```json\n{}\n```', 'save this code as data.json', {'data.json': b'{}\n'}),
    ('**run.sh**\n```sh\necho hi\n```', 'save this code', {'run.sh.txt': b'echo hi\n'}),
    ('```python\nx = 1\n```', 'save this code as run.html', {'run.html.txt': b'x = 1\n'}),
    ('**data.exe**\n```json\n{}\n```', 'save this code', {'data.exe.txt': b'{}\n'}),
    ('**A.js**\n```js\na\n```\n**a.js**\n```js\nb\n```', 'save this code',
     {'A.js': b'a\n', 'a-2.js': b'b\n'}),
])
def test_other_names_and_extensions(home, reply, command_text, expected):
    assert save(reply, command_text).startswith('Salvei ')
    [folder] = (home / 'Documents/Cortex/Codigo').iterdir()
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == expected


def test_short_writes_and_no_chmod_or_execution(home, monkeypatch):
    mod = module()
    original = os.write
    monkeypatch.setattr(mod.os, 'write', lambda fd, data: original(fd, data[:2]))
    monkeypatch.setattr(mod.os, 'chmod', Mock(side_effect=AssertionError('chmod')))
    monkeypatch.setattr(mod.os, 'fchmod', Mock(side_effect=AssertionError('fchmod')))
    monkeypatch.setattr(mod.os, 'system', Mock(side_effect=AssertionError('execute')))
    assert save(SINGLE).startswith('Salvei ')
    assert (home / 'Documents/Cortex/Codigo/site/index.html').read_bytes() == HTML.encode()


def test_readback_mismatch_leaves_nothing(home, monkeypatch):
    mod = module()
    original = os.read
    monkeypatch.setattr(mod.os, 'read', lambda fd, count: original(fd, count).replace(b'html', b'xxxx'))
    assert save(SINGLE) == FAILURE['pt']
    assert list((home / 'Documents/Cortex/Codigo').iterdir()) == []


def test_staging_open_failure_removes_empty_folder(home, monkeypatch):
    mod = module()
    original = os.open
    def fail(path, *args, **kwargs):
        if isinstance(path, str) and path.startswith('.cortex-code-'):
            raise OSError('staging open failed')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(mod.os, 'open', fail)
    assert save(SINGLE) == FAILURE['pt']
    assert list((home / 'Documents/Cortex/Codigo').iterdir()) == []


def test_post_commit_sync_failure_reports_real_save(home, monkeypatch):
    mod = module()
    root = home / 'Documents/Cortex/Codigo'
    root.mkdir(parents=True)
    original = os.fsync
    inode = root.stat().st_ino
    def fail(fd):
        if os.fstat(fd).st_ino == inode:
            raise OSError('sync failed after commit')
        return original(fd)
    monkeypatch.setattr(mod.os, 'fsync', fail)
    assert save(SINGLE).startswith('Salvei ')
    assert (root / 'site/index.html').read_bytes() == HTML.encode()


def test_all_numbered_folders_occupied_leave_existing_data(home):
    root = home / 'Documents/Cortex/Codigo'
    root.mkdir(parents=True)
    for n in range(1, 100):
        (root / ('site' if n == 1 else f'site ({n})')).write_bytes(b'keep')
    assert save(SINGLE) == FAILURE['pt']
    assert len(list(root.iterdir())) == 99
    assert all(p.read_bytes() == b'keep' for p in root.iterdir())


@pytest.mark.parametrize('path', PATHS)
def test_requested_path_is_sanitised_before_file_lookup(orch, home, path):
    reply = answer(orch, 'save this code as /etc/site.html', path,
                   conversation_history=history(SINGLE))
    assert (home / 'Documents/Cortex/Codigo/site/site.html').read_bytes() == HTML.encode()
    assert '~/Documents/Cortex/Codigo/site/site.html' in reply
    no_model_or_retrieval(orch)
