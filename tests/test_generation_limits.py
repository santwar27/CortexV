"""126c: fixed model window, bounded generation and publication regressions."""
import ast
from pathlib import Path
from unittest.mock import Mock

import pytest

from tests.test_project_context import setup, store  # noqa: F401
from tests.test_write_site import isolated_home  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]


def test_modelfiles_match_engine_default():
    tree = ast.parse((ROOT / 'src/core/engine.py').read_text())
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == '_NUM_CTX' for t in n.targets))
    default = int(assignment.value.args[0].args[1].value)
    assert default == 8192
    for name in ('Modelfile', 'Modelfile-creative'):
        values = [int(line.split()[2]) for line in (ROOT / name).read_text().splitlines()
                  if line.startswith('PARAMETER num_ctx ')]
        assert values == [default]


@pytest.mark.parametrize('kind', [None, 'web', 'code'])
@pytest.mark.parametrize('path', ['chat', 'stream', 'tools', 'retry'])
@pytest.mark.parametrize('configured', [None, '777', 'bad', '0', '-1'])
def test_all_calls_capped_and_audited(setup, monkeypatch, kind, path, configured):
    from src.core import engine as mod
    e = setup[0].engine_analytical
    e.push_build_kind(kind)
    for key in ('CORTEX_NUM_PREDICT', 'CORTEX_CODE_NUM_PREDICT'):
        monkeypatch.delenv(key, raising=False)
        if configured is not None:
            monkeypatch.setenv(key, configured)
    cap = 777 if configured == '777' else 3000 if kind else 2048
    seen = []
    def chat(**kw):
        seen.append(kw)
        response = {'message': {'role': 'assistant', 'content': 'Hello.'}, 'done_reason': 'length'}
        if path == 'tools' and len(seen) == 1:
            response['message']['tool_calls'] = [{'function': {'name': 'lookup', 'arguments': {}}}]
        return iter([{'message': {'content': ''}}, response]) if kw.get('stream') else response
    monkeypatch.setattr(mod.ollama, 'chat', chat)
    log = Mock()
    monkeypatch.setattr(mod.audit, 'log', log)
    if path == 'chat':
        assert e.chat('Hello') == 'Hello.'
    elif path == 'stream':
        assert ''.join(e.chat_stream('Hello')) == 'Hello.'
    elif path == 'tools':
        e.chat_with_tools('lookup', [], lambda *_: 'Found')
    else:
        e._retry_past_refusal('Hello', {}, 'fallback')
    sites = {'chat': ['chat'], 'stream': ['chat_stream'],
             'tools': ['chat_with_tools', 'tool_followup'], 'retry': ['refusal_retry']}[path]
    assert len(seen) == len(sites)
    for call in seen:
        assert call['options']['num_predict'] == cap
        assert 'num_ctx' not in call['options']
    events = [c.kwargs['payload'] for c in log.call_args_list if c.args[2] == 'generation_capped']
    assert [(event['call'], event['cap']) for event in events] == [(s, cap) for s in sites]


@pytest.mark.parametrize('path', ['chat', 'retry'])
def test_explicit_cap_preserved_and_reserved(setup, monkeypatch, path):
    from src.core import engine as mod
    e = setup[0].engine_analytical
    e.push_build_kind('web')
    options = {'num_predict': 1234, 'seed': 42}
    fit = Mock(wraps=mod.fit_messages_to_ctx)
    monkeypatch.setattr(mod, 'fit_messages_to_ctx', fit)
    if path == 'chat':
        e.chat('build a website', options=options)
    else:
        e._retry_past_refusal('build a website', options, 'fallback')
    assert setup[1][-1][1]['options'] == options
    assert options == {'num_predict': 1234, 'seed': 42}
    assert fit.call_args.kwargs['gen_reserve'] == 1234
    assert fit.call_args.kwargs['num_ctx'] == 8192


def test_build_both_tool_rounds_reserve_reply_and_keep_guide(setup, monkeypatch):
    from src.core import engine as mod
    from src.core.build_guides import guide_text
    e = setup[0].engine_analytical
    e.push_build_kind('web')
    e.push_turn_context(guide_text('web'))
    monkeypatch.setenv('CORTEX_CODE_NUM_PREDICT', '3000')
    e.history = [{'role': role, 'content': 'old ' * 2000}
                 for role in ['user', 'assistant'] * 5]
    seen = []
    def chat(**kw):
        seen.append(kw)
        return {'message': {'role': 'assistant', 'content': 'Done', 'tool_calls':
                [{'function': {'name': 'lookup', 'arguments': {}}}] if len(seen) == 1 else []}}
    monkeypatch.setattr(mod.ollama, 'chat', chat)
    fit = Mock(wraps=mod.fit_tool_followup)
    monkeypatch.setattr(mod, 'fit_tool_followup', fit)
    e.chat_with_tools('build a website', [], lambda *_: 'result ' * 10000)
    assert fit.call_args.kwargs['gen_reserve'] == 3000
    for call in seen:
        msgs = call['messages']
        assert guide_text('web') in msgs[0]['content']
        assert any(m['role'] == 'user' and
                   m['content'].split('\n\n[SYSTEM NOTE: ', 1)[0].endswith('build a website')
                   for m in msgs)
        cost = sum(mod._est_tokens(m.get('content', '')) +
                   mod._est_tokens(str(m.get('tool_calls') or '')) for m in msgs)
        assert cost + 3000 <= 8192


@pytest.mark.parametrize('text,expected', [
    ('cria o meu site', 'web'), ('faz o meu site novo', 'web'),
    ('build our landing page', 'web'), ('escreve o meu script em Python', 'code'),
    ('atualiza o meu código de rastreio', None),
])
def test_possessive_detector(text, expected):
    from src.core.build_guides import detect_guide
    assert detect_guide(text) == expected


@pytest.mark.parametrize('lang', ['pt', 'en'])
@pytest.mark.parametrize('ending', ['', '</body>', '</body></HTML>'])
def test_incomplete_page_saved_repaired_and_warned(isolated_home, lang, ending):
    from src.core.save_code import save_web_document
    html = ('<html><head><style>#topo{background:#1f3d2b}</style></head><body>'
            '<header id="topo"><a href="#x" style="">Topo</a></header>'
            '<button>See menu</button><section id="menu"><h2>Menu</h2></section>'
            '<footer>© 2023</footer>' + ending)
    reply = save_web_document('index.html', html, lang)
    saved = next((isolated_home / 'Documents').rglob('index.html')).read_text()
    assert '© 2023' not in saved
    assert '<a href="#menu"' in saved
    from src.core.web_repairs import _computed_colors, contrast
    anchor = next((fg, bg) for n, (fg, bg, _) in _computed_colors(saved).items() if n.words() == 'Topo' and n.tag == 'a')
    assert contrast(*anchor) >= 4.5
    warning = ('O código parece incompleto (sem o fim da página). Peça de novo com menos seções.' if lang == 'pt' else
               'The code looks incomplete (the end of the page is missing). Ask again with fewer sections.')
    assert reply.count(warning) == (0 if '</HTML>' in ending else 1)
    if '</HTML>' not in ending:
        fixes = 'Corrigi no código:' if lang == 'pt' else 'Fixed in the code:'
        lines = reply.splitlines()
        assert lines[lines.index(next(s for s in lines if s.startswith(fixes))) + 1] == warning
