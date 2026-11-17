"""Task 126: opt-in guidance, shared across all chat entry points."""
from unittest.mock import Mock

import pytest

from tests.test_project_context import setup, store, run, seed, system, EXPECTED  # noqa: F401


# Negatives first: mentioning code/the web is not a request to build it.
@pytest.mark.parametrize('text', [
    'abre o site do BBC', 'qual o site da Apple', 'salva esse código',
    'onde está o arquivo', 'passa o link', 'explica esse código',
    'open the BBC website', 'what is Apple\'s website?', 'save this code',
    'explain this code', 'how do websites work?', 'como criar um site?',
    'abre o PDF', 'aumenta o volume do Mac', 'toca Spotify',
    'faz um resumo da página 3 do PDF',
    'cria uma página no Notes com a lista',
    'faz a tradução dessa página',
    'make a note about the website',
    'faz uma pesquisa sobre o site da BBC',
    'escreve um email com o código de rastreio',
    'faz um backup do código',
    'cria um lembrete pra pagar a API amanhã',
    'cria uma página', 'make a page',
    'change a note about the CSS', 'fix an email with code',
    'faz um muito bonito site', 'build a very simple website',
    'create a new small quick basic simple website',
])
def test_negative_detection(text):
    from src.core.build_guides import detect_guide
    assert detect_guide(text) is None


@pytest.mark.parametrize('text,kind', [
    ('Crie um site de uma página só para o pub', 'web'),
    ('faz uma landing page pra minha loja', 'web'),
    ('cria uma página HTML com menu', 'web'),
    ('muda o CSS do site pra verde', 'web'),
    ('create a one-page website for a pub', 'web'),
    ('build me a landing page', 'web'),
    ('escreve um script em Python que soma', 'code'),
    ('cria uma API em FastAPI', 'code'),
    ('write a function that adds numbers', 'code'),
    ('faz um back end pra loja', 'code'),
    ('write Python code and build a website', 'web'),
    ('please implement a JavaScript function', 'code'),
    ('corrige o front-end da loja', 'web'),
    ('faz pra mim um site simples', 'web'),
    ('crie um novo site para a loja', 'web'),
    ('make for me a small website', 'web'),
    ('cria uma página de internet', 'web'),
    ('build a homepage', 'web'), ('create a home page', 'web'),
    ('change the layout', 'web'), ('altera o estilo do site', 'web'),
    ('edit a script', 'code'), ('corrige a função', 'code'),
    ('muda o código', 'code'), ('fix the API', 'code'),
    ('altera o programa', 'code'), ('change the back end', 'code'),
])
def test_positive_detection(text, kind):
    from src.core.build_guides import detect_guide
    assert detect_guide(text) == kind


@pytest.mark.parametrize('path', ['chat', 'stream', 'worker'])
def test_non_build_object_does_not_select_guide_or_window(setup, path):
    from src.core.build_guides import guide_text
    o, seen, _ = setup
    run(o, path, text='make a note about the website')
    assert guide_text('web') not in system(seen)
    assert guide_text('code') not in system(seen)
    assert 'num_ctx' not in seen[-1][1].get('options', {})


def test_files_cached_and_budgeted():
    from src.core.build_guides import guide_text
    from src.core.engine import _est_tokens
    for kind in ('web', 'code'):
        guide = guide_text(kind)
        assert guide and _est_tokens(guide) <= 450
        assert guide_text(kind) is guide


@pytest.mark.parametrize('channel,path', [
    ('ui', 'stream'), ('ui', 'worker'), ('network', 'chat'),
    ('voice', 'stream'), ('voice', 'worker'), ('terminal', 'chat'),
])
def test_all_entry_paths(setup, store, monkeypatch, channel, path):
    from src.core.build_guides import guide_text
    o, seen, _ = setup
    seed(store)
    if path == 'worker':
        monkeypatch.setattr(o, '_is_pure_chat', lambda *_: False)
    monkeypatch.setattr('src.core.orchestrator.reply_notes', lambda *_: 'RECEIPT NOTE')
    before = list(store._conn.iterdump())
    run(o, path, channel, text='crie um site para meu projeto')
    prompt = system(seen)
    assert guide_text('web') in prompt
    assert guide_text('code') not in prompt
    assert EXPECTED in prompt and 'RECEIPT NOTE' in prompt
    assert 'num_ctx' not in seen[-1][1]['options']
    assert seen[-1][1]['options']['num_predict'] == 3000
    for engine in (o.engine_analytical, o.engine_creative):
        assert guide_text('web') not in engine.system_prompt['content']
        assert guide_text('web') not in str(engine.history)
        assert guide_text('web') not in str(engine.memory.store_interaction.call_args_list)
    assert list(store._conn.iterdump()) == before
    run(o, path, channel, text='qual é a capital da França?')
    assert guide_text('web') not in system(seen)
    assert 'num_ctx' not in seen[-1][1].get('options', {})


def test_exception_cleanup(setup, monkeypatch):
    o, _, _ = setup
    monkeypatch.setattr(o, '_process_core', Mock(side_effect=RuntimeError('broken')))
    with pytest.raises(RuntimeError):
        o.process('build a website')
    assert not getattr(o.engine_analytical._turn_context, 'value', None)
    assert o.engine_analytical._turn_num_predict() == 2048
    assert o._web_request_text() == ''


@pytest.mark.parametrize('cap', ['1800', '3000'])
@pytest.mark.parametrize('text', ['build a website', 'write a Python script'])
def test_context_seam_and_trim(setup, monkeypatch, cap, text):
    from src.core import engine as mod
    o, seen, _ = setup
    monkeypatch.setenv('CORTEX_CODE_NUM_PREDICT', cap)
    original = mod.fit_messages_to_ctx
    fit = Mock(wraps=original)
    monkeypatch.setattr(mod, 'fit_messages_to_ctx', fit)
    run(o, text=text)
    assert 'num_ctx' not in seen[-1][1]['options']
    assert seen[-1][1]['options']['num_predict'] == int(cap)
    assert fit.call_args.kwargs['num_ctx'] == 8192
    assert fit.call_args.kwargs['gen_reserve'] == int(cap)
    system_msg = {'role': 'system', 'content': 's' * 15000}
    question = {'role': 'user', 'content': text * 3000}
    assert original([system_msg, {'role': 'assistant', 'content': 'old'}, question],
                    num_ctx=8192) == [system_msg, question]


def test_project_failure_does_not_hide_guide(setup, monkeypatch):
    from src.core.build_guides import guide_text
    o, seen, _ = setup
    monkeypatch.setattr(o, '_project_context', Mock(side_effect=OSError('store')))
    run(o, text='build a website')
    assert guide_text('web') in system(seen)


@pytest.mark.parametrize('path', ['/ask', '/ask_voice', '/ask_stream'])
def test_real_http_entry_points(setup, monkeypatch, path):
    from fastapi.testclient import TestClient
    monkeypatch.setenv('CORTEX_API_KEY', 'test-fake')
    import src.api as api
    from src.core.build_guides import guide_text
    o, seen, _ = setup
    monkeypatch.setattr(api, 'orchestrator', o)
    monkeypatch.setattr(api, 'SECRET_API_KEY', 'test-fake')
    monkeypatch.setattr(api, '_ALLOWLIST', [])
    response = TestClient(api.app).post(path, headers={'X-Cortex-Key': 'test-fake'},
                                       json={'user_input': 'build a website', 'language': 'en'})
    assert response.status_code == 200
    assert guide_text('web') in system(seen)
    assert 'num_ctx' not in seen[-1][1]['options']
    assert seen[-1][1]['options']['num_predict'] == 3000


def test_real_terminal_entry_point(setup, monkeypatch):
    from src import main
    from src.core.build_guides import guide_text
    o, seen, _ = setup
    monkeypatch.setattr(main, 'CortexOrchestrator', lambda: o)
    inputs = iter(['build a website', 'exit'])
    monkeypatch.setattr('builtins.input', lambda *_: next(inputs))
    with pytest.raises(SystemExit):
        main.run()
    assert guide_text('web') in system(seen)


def test_real_ui_chat_entry_point(setup, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.ui.routes import router
    from src.core.build_guides import guide_text
    o, seen, _ = setup
    monkeypatch.setenv('CORTEX_API_KEY', 'test-fake')
    app = FastAPI()
    app.include_router(router)
    app.state.orchestrator = o
    with TestClient(app) as client:
        cid = client.post('/ui/conversations', headers={'X-Cortex-Key': 'test-fake'}, json={}).json()['id']
        response = client.post('/ui/chat/stream', headers={'X-Cortex-Key': 'test-fake'},
                               json={'text': 'build a website', 'conversation_id': cid})
    assert response.status_code == 200
    assert guide_text('web') in system(seen)
    assert 'num_ctx' not in seen[-1][1]['options']
    assert seen[-1][1]['options']['num_predict'] == 3000
