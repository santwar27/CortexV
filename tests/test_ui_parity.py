"""The approved reference is immutable; the live UI has not been ported yet."""
import hashlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs/design/designcortex"
HASHES = {
    "index.html": "95592cddefb5af28d299bfc8ee481bd1b43200399100db67c76b1b05b41b3f20",
    "style.css": "1ec35f8199e660cff8653d4fef90c7332d91186f18d7dbfe7f33f72d34af42ca",
    "app.js": "770c959c1e62e344de1a973153a4533e4c1935246d3db673a7db28a936ad2e0a",
    "DESIGN.md": "06a96e39f8581d0c995fed3c71a865e6e33063da22c24c75098df3fdc611b079",
}
MODES = [("conversar", "Conversar", ("#i-chat",)),
         ("criar", "Criar", ("#i-create",)),
         ("continuar", "Continuar", ("#i-continue",))]
SYMBOLS = {
    "i-chat": '<symbol id="i-chat" viewbox="0 0 24 24"><path d="M21 11.5a8.4 8.4 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.4 8.4 0 0 1-3.8-.9L3 21l1.9-5.7a8.4 8.4 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.4 8.4 0 0 1 3.8-.9h.5a8.5 8.5 0 0 1 8 8z"></path></symbol>',
    "i-create": '<symbol id="i-create" viewbox="0 0 24 24"><path d="m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5Z"></path><path d="m20 2 .6 1.4L22 4l-1.4.6L20 6l-.6-1.4L18 4l1.4-.6Z"></path></symbol>',
    "i-continue": '<symbol id="i-continue" viewbox="0 0 24 24"><path d="M3 10a9 9 0 1 1 1.8 7.5M3 4v6h6M12 7v5l3 2"></path></symbol>',
}


def reference():
    return tuple((REFERENCE / name).read_text() for name in ("index.html", "app.js"))


WINDOW_DOTS = '<div class="window-dots" aria-hidden="true"><i></i><i></i><i></i></div>'
TENNIS_BALL_MARK = '<svg class="brand-mark" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9.5" stroke-dasharray="7.5 2"/><path d="M5.3 5.3c3.4 3.6 3.4 9.8 0 13.4"/><path d="M18.7 5.3c-3.4 3.6-3.4 9.8 0 13.4"/><g style="fill:var(--accent);stroke:none"><circle cx="5.3" cy="5.3" r="1.3"/><circle cx="5.3" cy="18.7" r="1.3"/><circle cx="18.7" cy="5.3" r="1.3"/><circle cx="18.7" cy="18.7" r="1.3"/></g></svg>'
TENNIS_BALL_ORBIT = '<symbol id="i-orbit" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9" stroke-dasharray="7 2"/><path d="M5.6 5.6c3.2 3.4 3.2 9.4 0 12.8"/><path d="M18.4 5.6c-3.2 3.4-3.2 9.4 0 12.8"/></symbol>'


def served_reference():
    # Approved live changes leave the frozen reference untouched.
    html, js = reference()
    from tests.ui_parity import DUCK_MARKUP, GESTURE_SELECT
    html = re.sub(r'(<section class="stage"[^>]*>)', lambda match: match[0] + DUCK_MARKUP, html, count=1)
    html = html.replace(WINDOW_DOTS, "")
    html = re.sub(r'<svg class="brand-mark".*?</svg>', TENNIS_BALL_MARK, html)
    html = re.sub(r'<symbol id="i-orbit".*?</symbol>', TENNIS_BALL_ORBIT, html)
    html = re.sub(r'(<select id="state-select">).*?(</select>)',
                  r'\g<1><option value="idle">Em repouso</option>'
                  '<option value="thinking">Pensando</option>'
                  '<option value="searching">Buscando na web</option>'
                  '<option value="reading">Lendo</option>'
                  '<option value="writing">Escrevendo</option>'
                  '<option value="listening">Ouvindo</option>'
                  r'<option value="speaking">Respondendo</option>\g<2>', html)
    html = html.replace('<option value="speaking">Respondendo</option></select>',
                        '<option value="speaking">Respondendo</option></select>' + GESTURE_SELECT)
    return html, js


@pytest.mark.parametrize("markup", [TENNIS_BALL_MARK, TENNIS_BALL_ORBIT],
                         ids=["brand-mark", "i-orbit"])
def test_served_tennis_ball_markup(markup):
    assert markup in (ROOT / "src/ui/templates/novo.html").read_text()


@pytest.mark.parametrize("name,digest", HASHES.items())
def test_frozen(name, digest):
    assert hashlib.sha256((REFERENCE / name).read_bytes()).hexdigest() == digest


def test_reference_and_three_laws():
    from tests.ui_parity import check, extract
    html, js = reference()
    data = extract(html, js)
    assert WINDOW_DOTS in html
    html, js = served_reference()
    assert check(html, js) == []
    assert data["modes"] == MODES
    assert len(data["symbols"]) == 15
    assert {key: data["symbols"][key] for key in SYMBOLS} == SYMBOLS
    assert check("{% if user %}\n" + html + "\n{% endif %}", js) == []
    assert check(html.replace("><", ">\n  <"), js) == []
    assert check(html.replace("Boa noite, Pedro", "{{ greeting }}"), js) == []


@pytest.mark.parametrize("mutation,element", [
    ("path", "i-create"), ("label", "os três modos"),
    ("order", "os três modos"), ("resume", "cartão de retomada"),
    ("central", "Internet lease"), ("symbol", "i-menu"),
    ("brand", "marca"), ("add_dots", "pontos da janela (removidos)"),
    ("brand_aria", "marca"), ("brand_href", "marca"), ("brand_order", "marca"),
    ("old_brand_mark", "marca (bola de tênis)"),
    ("old_orbit", "símbolo i-orbit (bola de tênis)"),
    ("nav", "Início, Projetos e Conversas"), ("attach", "anexos"),
    ("secondary", "Contatos"),
    ("appearance_option_removed", "Aparência e núcleo"),
    ("appearance_option_order", "Aparência e núcleo"),
    ("appearance_intensity_label", "Aparência e núcleo"),
])
def test_mutations(mutation, element):
    from tests.ui_parity import check
    html, js = served_reference()
    if mutation == "path":
        html = html.replace("m12 3 2.5", "m12 4 2.5")
    elif mutation == "label":
        html = html.replace(">Criar</button>", ">Inventar</button>")
    elif mutation == "order":
        buttons = re.findall(r'<button class="mode.*?</button>', html)
        html = html.replace("".join(buttons), "".join(reversed(buttons)))
    elif mutation == "resume":
        html = re.sub(r'<section class="resume-card".*?</section>', "", html)
    elif mutation == "central":
        js = js.replace("title:'Internet lease'", "title:'Internet'")
    elif mutation == "symbol":
        html = re.sub(r'<symbol id="i-menu".*?</symbol>', "", html)
    elif mutation in {"old_brand_mark", "old_orbit"}:
        pattern = (r'<svg class="brand-mark".*?</svg>' if mutation == "old_brand_mark"
                   else r'<symbol id="i-orbit".*?</symbol>')
        old_markup = re.search(pattern, reference()[0])[0]
        html = re.sub(pattern, old_markup, html)
    elif mutation == "brand_order":
        html = html.replace(TENNIS_BALL_MARK + "<span>CORTEX</span>",
                            "<span>CORTEX</span>" + TENNIS_BALL_MARK)
    else:
        old, new = {"brand": (">CORTEX</span>", ">OTHER</span>"),
                    "brand_aria": ('aria-label="CORTEX início"', 'aria-label="OTHER"'),
                    "brand_href": ('href="#inicio"', 'href="#other"'),
                    "add_dots": ('<aside class="sidebar" aria-label="Navegação principal">',
                                 '<aside class="sidebar" aria-label="Navegação principal">' + WINDOW_DOTS),
                    "nav": (">Início</button>", ">Home</button>"),
                    "attach": ('href="#i-attach"', 'href="#i-mic"'),
                    "appearance_option_removed": ('<option value="searching">Buscando na web</option>', ''),
                    "appearance_option_order": (
                        '<option value="searching">Buscando na web</option><option value="reading">Lendo</option>',
                        '<option value="reading">Lendo</option><option value="searching">Buscando na web</option>'),
                    "appearance_intensity_label": ('>Intensidade dos efeitos</label>', '>Intensidade</label>'),
                    "secondary": ('id="system-hub"', 'id="removed-hub"')}[mutation]
        html = html.replace(old, new)
    assert any(element in violation for violation in check(html, js))


def test_inventory_and_screens():
    from tests.ui_parity import INVENTORY, SCREENS, VIEWPORTS, extract
    preserved = re.search(r"^Preservar: (.+)\.$", (REFERENCE / "DESIGN.md").read_text(), re.M)[1]
    assert set(preserved.split("; ")) <= {item["name"] for item in INVENTORY}
    for item in INVENTORY:
        assert item["rule"] in {"exact", "real", "secondary"}
        assert item["anchor"]
        if item["rule"] == "real":
            assert item["source"].strip()
        if item["rule"] == "secondary":
            assert item["today"].strip() and item["placement"].strip()
    data = extract(*reference())
    assert set(SCREENS) == {"home", "chat", "projects", "central", "detail", "appearance"}
    for screen in SCREENS.values():
        for selectors in screen.values():
            for selector in selectors:
                assert data["anchors"][selector], selector
    assert VIEWPORTS == {"desktop": (1440, 900), "mobile": (390, 844)}


def test_rendered_central_without_js():
    from tests.ui_parity import METRICS, check
    html, js = served_reference()
    options = "".join(f'<button id="option-{key}"><span class="option-heading"><span>{label}</span></span></button>'
                      for key, label in METRICS.items())
    html = html.replace('id="system-grid"></div>', f'id="system-grid">{options}</div>')
    assert check(html) == []
    # Stale source labels cannot mask a regression in the served DOM.
    assert any("Internet lease" in v for v in check(html.replace(">Internet lease<", ">Internet<"), js))


def test_reference_demo_leftovers():
    from tests.ui_parity import demo_leftovers
    assert demo_leftovers(reference()[0]) == [
        ".sidebar-bottom text: · demo",
        "#messages aria-label: Conversa de demonstração",
        "#attach aria-label: Selecionar anexo de demonstração",
        "#mic aria-label: Simular escuta por voz",
        "#send aria-label: Enviar mensagem de demonstração",
        ".demo-note text: Demonstração · sem conexão com o CORTEX",
        ".sample-notice text: Prévia visual · valores ilustrativos, sem monitoramento real | Demonstração · sem leitura do sistema",
    ]


def test_honest_relabel():
    from tests.ui_parity import check
    html, js = served_reference()
    for old, new in [
        ("· demo", ""), ("Conversa de demonstração", "Conversa"),
        ("Selecionar anexo de demonstração", "Anexar arquivo"),
        ("Simular escuta por voz", "Gravar voz"),
        ("Enviar mensagem de demonstração", "Enviar mensagem"),
        ("Demonstração · sem conexão com o CORTEX", ""),
        ("Prévia visual · valores ilustrativos, sem monitoramento real", ""),
        ("Demonstração · sem leitura do sistema", ""),
        ('title="Alternar tema"', 'title="Mudar tema"'),
    ]:
        html = html.replace(old, new)
    assert check(html, js) == []
    from tests.ui_parity import demo_leftovers
    assert demo_leftovers(html) == []


def test_send_icon_still_guarded():
    from tests.ui_parity import check
    html, js = served_reference()
    assert any("enviar" in v for v in check(html.replace('href="#i-send"', 'href="#i-mic"'), js))


def test_demo_in_real_slot():
    from tests.ui_parity import demo_leftovers
    html, _ = reference()
    hits = demo_leftovers(html.replace("Boa noite, Pedro", "Boa noite, Pedro (demo)"))
    assert any("#welcome" in hit and "Boa noite, Pedro (demo)" in hit for hit in hits)


@pytest.mark.parametrize("attr,word", [
    ("aria-label", "DEMO"), ("title", "DEMONSTRAÇÃO"),
    ("placeholder", "Simular"), ("alt", "ILUSTRATIVOS"),
])
def test_demo_attributes_and_non_ui_content(attr, word):
    from tests.ui_parity import demo_leftovers
    html = '<script>const text = "demo";</script><style>.demo {}</style><!-- demo -->'
    html += '<div id="controls"><span title="demo">simulação</span></div>'
    html += '<button id="open-appearance" aria-label="demo"><span>demonstração</span></button>'
    html += f'<input id="real" {attr}="{word}"><p>demografia</p>'
    assert demo_leftovers(html) == [f"#real {attr}: {word}"]
