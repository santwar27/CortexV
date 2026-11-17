"""Byte-preserving repairs at publication, never at save-last-code."""
from pathlib import Path
import re

import pytest

from tests.test_write_site import isolated_home, site_orch, orch, taught_orch, rig, use_model, answer  # noqa: F401

REAL = Path(__file__).with_name('fixtures').joinpath('queens_head.html').read_text().rstrip('\n')


def repair(html, year=2026):
    from src.core.web_repairs import repair_html
    return repair_html(html, year=year)


@pytest.mark.parametrize('html', [
    REAL,
    REAL.replace('body {', 'body {color:white;'),
    '<style>body{color:white;background:black}.botao{background:gold}</style>'
    '<body><button class="botao">See menu</button><section><h2>Menu</h2></section></body>',
    '<style>body{color:black;background:white}</style><style>.botao{background:gold}</style>'
    '<body><button class="botao">See menu</button><button>View menu</button>'
    '<a href="#menu">Existing link</a><section id="menu"><h2>Menu</h2></section></body>',
])
def test_converted_buttons_keep_button_style_and_final_contrast(html):
    from src.core.web_repairs import _computed_colors, contrast, Document
    fixed, _ = repair(html)
    button_rules = re.findall(r'[^{}]+\{[^{}]*text-decoration:none[^{}]*\}', fixed)
    assert len(button_rules) == 1
    assert 'display:inline-block' in button_rules[0]
    assert 'color:' in button_rules[0]
    styles = [n for n in Document(fixed).elements if n.tag == 'style']
    assert button_rules[0].strip() in fixed[styles[-1].opening_end:styles[-1].end_start]
    checked = 0
    for node, (fg, bg, _) in _computed_colors(fixed).items():
        if node.tag == 'a' and node.words() in ('Ver cardápio', 'See menu', 'View menu'):
            assert contrast(fg, bg) >= 4.5
            checked += 1
    assert checked == html.count('<button')
    assert repair(fixed) == (fixed, [])


def test_unconverted_button_gets_no_anchor_style():
    html = '<style>body{color:black;background:white}</style><body><button>Other</button></body>'
    assert repair(html) == (html, [])


def test_real_page_minimal_edits_and_contrast():
    fixed, changes = repair(REAL)
    assert '&copy; 2026' in fixed
    assert '<a href="#cardapio" class="botao cortex-button-link">Ver cardápio</a>' in fixed
    assert '<div class="cardapio" id="cardapio">' in fixed
    # Strip only the sanctioned edits; everything else must be exact bytes.
    original_style = REAL.split('</style>')[0]
    appended = fixed[len(original_style):fixed.index('</style>')]
    assert appended and '{color:' in appended
    restored = fixed.replace(appended, '', 1).replace('&copy; 2026', '&copy; 2023')
    restored = restored.replace('<a href="#cardapio" class="botao cortex-button-link">Ver cardápio</a>',
                                '<button class="botao">Ver cardápio</button>')
    restored = restored.replace(' id="cardapio"', '')
    assert restored == REAL
    # Idempotence: the same evaluator finds no remaining failing text.
    assert repair(fixed) == (fixed, [])
    assert changes


@pytest.mark.parametrize('html', [
    '<html><style>body{color:#111;background:white}</style><body><p>Good</p></body></html>',
    '<style>body{background:linear-gradient(red, blue);color:red}</style><body><p>Unknown</p></body>',
    '<style>p{color:red!important;background:red}</style><body><p>Unknown</p></body>',
    '<style>body{background:rgba(0,0,0,.5);color:black}</style><body>Unknown</body>',
    '<style>p{color:var(--missing);background:red}</style><body><p>Unknown</p></body>',
    '<style>p{color:red;background:red}</style><body><p style="color:blue">Unknown</p></body>',
    '<style>p{color:red;background:red}</style><body><script>change()</script><p>Unknown</p></body>',
    '<body><form><h2>Menu</h2><button type="submit">See the menu</button></form></body>',
])
def test_safe_noops(html):
    assert repair(html) == (html, [])


@pytest.mark.parametrize('color', ['red', '#f00', '#ff0000', 'rgb(255, 0, 0)', 'var(--accent)'])
def test_colors_and_variables(color):
    html = f'<style>:root{{--accent:red}} body{{background:{color}}} p{{color:{color}}} p span{{color:inherit}}</style><body><p>Fix <span>me</span></p></body>'
    fixed, changes = repair(html)
    assert fixed != html and changes
    assert repair(fixed) == (fixed, [])


@pytest.mark.parametrize('mark', ['©', '&copy;', '&#169;', '(c)', 'Copyright'])
@pytest.mark.parametrize('old,new', [('2023','2026'), ('2019-2023','2019-2026'), ('2020–2023','2020–2026'), ('2024','2026')])
def test_years(mark, old, new):
    html = f'<body><footer>{mark} {old} Pub</footer></body>'
    fixed, changes = repair(html)
    assert fixed == html.replace(old, new) and changes


def test_buttons_existing_unique_id_and_unmatched():
    html = '<body><div id="menu"></div><section><h2>Menu</h2></section><button>See the menu</button><button>Other</button></body>'
    fixed, _ = repair(html)
    assert '<section id="menu-2">' in fixed and 'href="#menu-2"' in fixed
    assert '<button>Other</button>' in fixed
    html = '<body><section id="food"><h2>Cardápio</h2></section><button onclick="go()">Ver cardápio</button></body>'
    assert repair(html) == (html, [])


@pytest.mark.parametrize('lang', ['pt', 'en'])
def test_receipts_and_exact_published_code(isolated_home, lang, monkeypatch):
    from src.core import clock
    from datetime import datetime
    from src.core.save_code import save_web_document
    monkeypatch.setattr(clock, 'local_now', lambda: datetime(2026, 11, 2))
    reply = save_web_document('pub.html', REAL, lang, request_text='create a site in index.html')
    file = next((isolated_home / 'Documents').rglob('index.html'))
    content = file.read_text()
    assert reply.split('```html\n')[1].rsplit('\n```', 1)[0] == content
    fixes = 'Corrigi no código:' if lang == 'pt' else 'Fixed in the code:'
    warning = ('Confira endereço, telefone, preços e horários: são dados de exemplo.' if lang == 'pt' else
               'Check the address, phone, prices and opening hours: they are example data.')
    assert reply.index(fixes) < reply.index(warning) < reply.index('```html')
    assert '2023 → 2026' in reply


@pytest.mark.parametrize('lang', ['pt', 'en'])
def test_warning_only_unsupplied_kinds(isolated_home, lang):
    from src.core.save_code import save_web_document
    html = '<body><p>Endereço: 123 Main St, Londres</p><p>Telefone: 020 1234 5678</p></body>'
    reply = save_web_document('pub.html', html, lang, request_text='create a site')
    expected = ('Confira endereço e telefone: são dados de exemplo.' if lang == 'pt' else
                'Check the address and phone: they are example data.')
    assert expected in reply
    assert 'Corrigi' not in reply and 'Fixed' not in reply
    reply = save_web_document('pub.html', html, lang, request_text='123 Main St, Londres; 020 1234 5678')
    assert 'dados de exemplo' not in reply and 'example data' not in reply


@pytest.mark.parametrize('path', ['sync', 'stream'])
def test_request_reaches_publisher(site_orch, rig, isolated_home, path):
    use_model(site_orch, rig)
    answer(site_orch, 'Crie um site num arquivo home.html', path)
    assert list((isolated_home / 'Documents').rglob('home.html'))


def test_save_last_code_is_exact(isolated_home):
    from src.core.save_code import save_last_code
    save_last_code('save this code', [{'role':'assistant', 'content':'```html\n' + REAL + '\n```'}], 'en')
    file = next((isolated_home / 'Documents').rglob('*.html'))
    assert file.read_bytes() == (REAL + '\n').encode()


def test_approved_plan_original_request(site_orch, isolated_home):
    site_orch.engine_analytical.model_name = 'fake'
    with site_orch._web_request_scope('sim'):
        reply, mode = site_orch._execute_staged_plan({'args': {
            'request_text': 'Crie um site num arquivo index.html; Telefone: 020 1234 5678',
            'passos': [{'id': 'p1', 'tool': 'write_document', 'args': {
                'filename': 'pub.html', 'content': '<body><p>Telefone: 020 1234 5678</p></body>'}}]}})
        assert site_orch._web_request_text() == 'sim'
    assert mode == 'SAVE_CODE' and 'dados de exemplo' not in reply
    assert list((isolated_home / 'Documents').rglob('index.html'))


def test_specificity_order_inheritance_and_exact_context():
    html = ('<html><style>body{color:black;background:white} '
            '#s p{color:blue} .bad{color:white} #s p{color:#800000} '
            '.red{background:#800000} .blue{background:white}</style><body>'
            '<section id="s"><div class="red"><p class="bad">Fix</p></div>'
            '<div class="blue"><p>Good</p></div></section></body></html>')
    fixed, changes = repair(html)
    assert changes == [('contrast', 1)]  # Only the dark-red-on-dark-red context changes.
    assert repair(fixed) == (fixed, [])


def test_styles_multiple_blocks_and_unknown_context_sibling():
    html = ('<style>p{color:red}</style><style>.bad{background:red} '
            '.unknown{background:url(photo.png)}</style><body>'
            '<div class="unknown"><p>Leave</p></div><div class="bad"><p>Fix</p></div></body>')
    fixed, changes = repair(html)
    assert changes == [('contrast', 1)]
    assert fixed.split('</style>')[0] == html.split('</style>')[0]
    assert 'div:nth-of-type(1) > p' not in fixed
    assert repair(fixed) == (fixed, [])


def test_year_does_not_touch_scripts_attributes_or_current_year():
    html = '<body title="Copyright 2023"><script>const x="© 2023";</script><p>© 2026</p></body>'
    assert repair(html) == (html, [])


def test_fixture_every_visible_text_has_required_contrast():
    # Independent fixture-specific oracle: known original cascade + appended
    # positional overrides. Does not use the repairer's selector/cascade code.
    from src.core.web_repairs import Document
    fixed, changes = repair(REAL)
    doc = Document(fixed)
    rules = re.findall(r'([^{}]+)\{color:(#[0-9a-f]{6});\}', fixed.split('</style>')[0])
    overrides = {selector.strip(): value for selector, value in rules}
    backgrounds = {'topo': '#008000', 'sobre': '#ffd700', 'cardapio': '#ffffcc',
                   'horarios': '#008000', 'contato': '#ffd700', 'rodape': '#008000'}
    def rgb(value):
        return [int(value[i:i+2], 16) / 255 for i in (1, 3, 5)]
    def lum(value):
        linear = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4 for v in rgb(value)]
        return sum(v * w for v, w in zip(linear, [.2126, .7152, .0722]))
    checked = 0
    for node in doc.elements:
        if not node.text or not any(a.tag == 'body' for a in node.ancestors()):
            continue
        chain = list(reversed(list(node.ancestors())))
        parts = []
        for n in chain:
            peers = [c for c in n.parent.children if c.tag == n.tag] if n.parent else [n]
            parts.append(n.tag + (('#' + n.attrs['id']) if 'id' in n.attrs else '')
                         + f':nth-of-type({peers.index(n)+1})')
        # The converted link's appended author rule inherits its parent's black.
        fg = overrides.get(' > '.join(parts), {'h1':'#008000', 'h2':'#ffd700',
                                              'p':'#ffffcc'}.get(node.tag, '#000000'))
        bg = '#ffd700' if node.tag == 'a' else backgrounds[node.parent.attrs['class']] if node.parent.tag == 'div' else '#ffffcc'
        lo, hi = sorted([lum(fg), lum(bg)])
        assert (hi + .05) / (lo + .05) >= (3 if node.tag in ('h1','h2','h3') else 4.5), node.words()
        checked += 1
    assert checked == 18
    assert ('contrast', 7) in changes


@pytest.mark.parametrize('css', [
    ':root{--accent:black;--Accent:white} body{background:black;color:var(--Accent)}',
    ':root{--accent:red} p{--accent:white;color:var(--accent);background:red}',
    '@media(min-width:768px){p{color:white}} p{color:red;background:red}',
])
def test_uncertain_or_case_sensitive_variables_and_media(css):
    html = f'<html><style>{css}</style><body><p>Leave</p></body></html>'
    assert repair(html) == (html, [])


def test_layout_only_media_does_not_disable_known_colors():
    html = '<style>p{color:red;background:red}@media(min-width:768px){main{display:grid}}</style><body><p>Fix</p></body>'
    fixed, changes = repair(html)
    assert changes == [('contrast', 1)]
    assert repair(fixed) == (fixed, [])


@pytest.mark.parametrize('lang,warning', [('pt','Confira telefone: são dados de exemplo.'),
                                          ('en','Check the phone: it is example data.')])
def test_partial_data_and_formatting(lang, warning):
    from src.core.web_repairs import example_warning
    html = '<body><p>Address: 123 Main St, London</p><p>Phone: +44 (20) 1234 5678</p><p>£10.99</p></body>'
    assert example_warning(html, '123 Main St, London; £10.99', lang) == warning
    assert example_warning(html, '123 Main St, London; +442012345678; £10.99', lang) == ''


def test_override_outweighs_repeated_class_specificity():
    from src.core.web_repairs import _specificity
    html = '<style>.x.x.x.x.x.x{color:red;background:red}</style><body><p class="x">Fix</p></body>'
    fixed, changes = repair(html)
    assert changes == [('contrast', 1)]
    selector = re.findall(r'([^{}]+)\{color:#[0-9a-f]{6};\}', fixed)[-1].strip()
    assert _specificity(selector) > (0, 6, 0)
    assert repair(fixed) == (fixed, [])


def test_mid_gray_needs_true_black_to_meet_body_contrast():
    html = '<style>body{color:#777777;background:#777777}</style><body><p>Fix</p></body>'
    fixed, changes = repair(html)
    assert '{color:#000000;}' in fixed
    assert changes == [('contrast', 1)]
    assert repair(fixed) == (fixed, [])


@pytest.mark.parametrize("inline", [' style=""', ""])
def test_id_occurs_once_in_contrast_override(inline):
    from src.core.web_repairs import _computed_colors, contrast
    html = ('<html><style>#topo{background:#1f3d2b}</style><body>'
            f'<header id="topo"><a href="#x"{inline}>Topo</a></header></body></html>')
    fixed, changes = repair(html)
    selector = re.findall(r'([^{}]+)\{color:#[0-9a-f]{6};\}', fixed)[-1]
    assert selector.count('#topo') == 1
    assert changes == [('contrast', 1)]
    node, (fg, bg, _) = next((n, colors) for n, colors in _computed_colors(fixed).items() if n.tag == 'a')
    assert contrast(fg, bg) >= 4.5
    assert repair(fixed) == (fixed, [])
