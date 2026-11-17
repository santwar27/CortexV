"""Static preservation contract, not a renderer or a data-authenticity test."""
import re
from html import escape
from html.parser import HTMLParser
from pathlib import Path

REFERENCE = Path(__file__).resolve().parents[1] / "docs/design/designcortex"
VIEWPORTS = {"desktop": (1440, 900), "mobile": (390, 844)}
DEMO_WORDING = re.compile(r"demonstraç|\bdemo\b|simula|ilustrativ", re.I)
TENNIS_BALL_MARK = '<svg class="brand-mark" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9.5" stroke-dasharray="7.5 2"/><path d="M5.3 5.3c3.4 3.6 3.4 9.8 0 13.4"/><path d="M18.7 5.3c-3.4 3.6-3.4 9.8 0 13.4"/><g style="fill:var(--accent);stroke:none"><circle cx="5.3" cy="5.3" r="1.3"/><circle cx="5.3" cy="18.7" r="1.3"/><circle cx="18.7" cy="5.3" r="1.3"/><circle cx="18.7" cy="18.7" r="1.3"/></g></svg>'
TENNIS_BALL_ORBIT = '<symbol id="i-orbit" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9" stroke-dasharray="7 2"/><path d="M5.6 5.6c3.2 3.4 3.2 9.4 0 12.8"/><path d="M18.4 5.6c-3.2 3.4-3.2 9.4 0 12.8"/></symbol>'
# Preview options approved by the user, 05/11, task 132.
PREVIEW_STATE_OPTIONS = (
    '<option value="idle">Em repouso</option>'
    '<option value="thinking">Pensando</option>'
    '<option value="searching">Buscando na web</option>'
    '<option value="reading">Lendo</option>'
    '<option value="writing">Escrevendo</option>'
    '<option value="listening">Ouvindo</option>'
    '<option value="speaking">Respondendo</option>'
)

# Exact demo selector approved by the user, 06/11, task 136.
GESTURE_SELECT = '<label for="gesture-select">Gesto de demonstração</label><select id="gesture-select"><option value=""></option><option value="look">Olhar em volta</option><option value="hop">Pulinho</option><option value="wave">Dar oi</option><option value="stretch">Espreguiçar</option><option value="racket">Raquete</option><option value="turn">Voltinha</option><option value="walk">Andar</option><option value="pirouette">Pirueta</option><option value="flap">Bater as asas</option><option value="stumble">Tropeço</option><option value="dance">Dancinha</option><option value="peek">Esconde-esconde</option><option value="cheer">Comemorar</option><option value="no">Dizer não</option><option value="startle">Susto</option><option value="compliment">Elogio</option><option value="poop">Cocô</option><option value="knock">Toc-toc</option><option value="sleep">Dormir</option></select>'

# Exact decorative corner approved by the user, 06/11, task 135.
DUCK_MARKUP = '<div id="cortex-duck-dock" class="duck-dock" aria-hidden="true">\n        <div id="cortex-duck" class="duck" data-state="idle" data-eyes="open" data-part="none" data-gesture="none" data-frozen="false" data-reduced="false" data-paused="false">\n          <div class="duck-body"><div class="duck-pose">\n            <svg class="duck-racket" viewBox="0 0 1254 1254" shape-rendering="crispEdges" aria-hidden="true"></svg>\n            <img class="duck-source" src="/ui/static/novo/avatar/duckcortex.png" width="1254" height="1254" alt="" draggable="false">\n            <div class="duck-face">\n              <img class="duck-part duck-head" src="/ui/static/novo/avatar/duckcortex.png" width="1254" height="1254" alt="" draggable="false">\n              <span class="duck-effect eye-patch left-eye"></span><span class="duck-effect eye-patch right-eye"></span>\n              <div class="duck-eye-window left-eye-window"><img class="duck-part eye-copy left-eye-copy" src="/ui/static/novo/avatar/duckcortex.png" width="1254" height="1254" alt="" draggable="false"></div>\n              <div class="duck-eye-window right-eye-window"><img class="duck-part eye-copy right-eye-copy" src="/ui/static/novo/avatar/duckcortex.png" width="1254" height="1254" alt="" draggable="false"></div>\n              <span class="duck-effect antenna-glow"></span><span class="duck-effect ear-glow"></span><span class="duck-effect eye-glow left-eye"></span><span class="duck-effect eye-glow right-eye"></span>\n            </div>\n            <img class="duck-part duck-wing" src="/ui/static/novo/avatar/duckcortex.png" width="1254" height="1254" alt="" draggable="false">\n            <span class="duck-effect wing-glow"></span>\n          </div></div>\n        </div>\n      </div>'


def entry(name, rule, anchor, **details):
    return dict(name=name, rule=rule, anchor=tuple(anchor.split("|")), **details)


INVENTORY = [
    entry("marca", "exact", ".brand"),
    entry("nova conversa", "exact", "#new-chat"),
    entry("Início, Projetos e Conversas", "exact", "[data-page]", comparison="nav"),
    entry("projetos CORTEX e Pessoal", "real", ".nav-section:not(.recent)|#projects-panel|#breadcrumb", source="ProjectStore real projects/links; no fixed CORTEX/Pessoal, no seeds; empty slots when absent"),
    entry("conversas recentes", "real", ".recent", source="GET /ui/conversations"),
    entry("indicação local/privado", "real", ".privacy", source="Per-reply answering model and Local or Online; no blanket privacy claim"),
    entry("núcleo", "exact", "#orb"),
    entry("pato do Cortex", "exact", "#cortex-duck-dock"),
    entry("estado do núcleo", "real", "#state-label", source="POST /ui/chat/stream real status stages"),
    entry("saudação", "real", "#welcome", source="User name + Mac local hour: 05–11 Bom dia, 12–17 Boa tarde, otherwise Boa noite; missing name omitted"),
    entry("os três modos e seus símbolos", "exact", "[data-mode]", comparison="modes"),
    entry("cartão de retomada", "real", "#resume-card|#resume", source="ProjectStore recorded progress/next step; empty when none"),
    entry("conversa com avatar", "real", "#messages", source="GET /ui/conversations/{id} and POST /ui/chat/stream; preserve avatar geometry"),
    entry("campo de mensagem", "real", "#composer|#message-input", source="User input to POST /ui/chat/stream"),
    entry("anexos", "exact", "#attach"),
    entry("microfone", "exact", "#mic"),
    entry("enviar", "exact", "#send"),
    entry("indicação de modelo local", "real", ".sidebar-bottom", source="Per-reply answering model and Local or Online, never configured-model guess"),
    entry("anexos — conteúdo", "real", "#file|#attachment-list", source="POST /ui/upload: up to 2 files sharing today's ~12k-character budget; disclose what was cut"),
    entry("microfone — captura", "real", "#mic", source="'hey cortex' listener retained; browser recording -> Cortex LOCAL Whisper; never browser speech recognition"),
    entry("Criar — ação", "real", '[data-mode="criar"]', source="Create project, note, reminder or idea; idea semantics await Pedro"),
    entry("Central do CORTEX", "real", "#settings|#system-hub|#system-grid", source="GET /ui/data: lease, net, services, studier, audit, cost, voice, memory"),
    entry("Retomar central", "exact", '#metric-return|[data-return-hub="appearance"]'),
    entry("painel correspondente", "real", "#metric-detail|#metric-title|#metric-content", source="GET /ui/data selected metric; Memória uses memory"),
    entry("Aparência e núcleo", "exact", "#controls|#open-appearance|#theme"),
]
METRICS = dict(zip(("internet", "network", "services", "studier", "audit", "costs", "voice", "memory"),
                   ("Internet lease", "Net", "Services", "Studier", "Audit log", "Costs", "Voz", "Memória")))
INVENTORY += [entry(label, "exact", "#option-" + key, comparison="central", metric=key)
              for key, label in METRICS.items()]
INVENTORY += [entry("símbolo " + key, "exact", "#" + key) for key in (
    "i-chat", "i-create", "i-continue", "i-home", "i-folder", "i-file", "i-orbit", "i-arrow",
    "i-send", "i-mic", "i-attach", "i-sun", "i-grid", "i-sliders", "i-menu")]
# These anchors reserve hosts, not controls that the prototype never drew.
INVENTORY += [
    entry("Contatos", "secondary", "#system-hub", today="/ui/novo: Central #option-contacts -> static/novo/contatos.js; GET/POST /ui/contacts, POST /ui/contacts/delete", placement="Central grid, full width after Memória; detail in #metric-detail"),
    entry("sair", "secondary", ".top-right", today="/ui/novo: .top-right a.logout -> GET /ui/logout", placement="Header, as today"),
    entry("renomear", "secondary", ".recent", today="/ui/novo: static/novo/projetos.js double-click title; /ui/conversations/{id}/rename", placement="Conversation title in sidebar, as today"),
    entry("apagar", "secondary", ".recent", today="/ui/novo: static/novo/projetos.js .recent-delete and Selecionar; /ui/conversations/{id}/delete and /ui/conversations/delete", placement="Sidebar row and bulk selection, as today"),
    entry("regenerar", "secondary", "#messages", today="/ui/novo: static/novo/conversa.js .regen-button; /ui/conversations/{id}/pop_last", placement="Latest eligible answer, as today"),
    entry("aprovação em segundo plano", "secondary", "#system-hub", today="/ui/novo: static/novo/central.js Fila de fundo; POST /ui/gate/decide (CLI python -m src.gate_review still works)", placement="Audit log detail in Central, beside each whole preview"),
]
COMMON = ("#orb", "#state-label", "[data-mode]", "#composer", "#attach", "#mic", "#send", ".privacy", "#settings")
SIDEBAR = (".brand", "#new-chat", "[data-page]", ".nav-section:not(.recent)", ".recent", ".sidebar-bottom")
SCREENS = {
    "home": dict(clicks=(), visible=COMMON + ("#welcome", "#resume-card", "#messages"), sidebar=SIDEBAR),
    "chat": dict(clicks=("#resume",), visible=COMMON + ("#resume-card", "#messages"), sidebar=SIDEBAR),
    "projects": dict(clicks=('[data-page="projects"]',), mobile_clicks=(".mobile-menu", '[data-page="projects"]'), visible=COMMON + ("#projects-panel",), sidebar=SIDEBAR),
    "central": dict(clicks=("#settings",), visible=("#system-hub", "#open-appearance") + tuple("#option-" + key for key in METRICS)),
    "detail": dict(clicks=("#settings", "#option-memory"), visible=("#metric-detail", "#metric-title", "#metric-content", "#metric-return")),
    "appearance": dict(clicks=("#settings", "#open-appearance"), visible=("#controls", '[data-return-hub="appearance"]')),
}


def clean(text):
    return " ".join(text.split())


class Node:
    def __init__(self, tag, attrs=()):
        self.tag, self.attrs, self.children = tag, dict(attrs), []

    def text(self):
        return clean("".join(c if isinstance(c, str) else c.text() for c in self.children))

    def markup(self):
        # Keep attribute presence; accessibility wording must describe the real control.
        attrs = ((k, "" if k in {"aria-label", "title"} else v) for k, v in sorted(self.attrs.items()))
        attrs = "".join(f' {k}="{escape(clean(v or ""), quote=True)}"' for k, v in attrs)
        children = "".join(escape(clean(c)) if isinstance(c, str) else c.markup() for c in self.children)
        return f"<{self.tag}{attrs}>{children}</{self.tag}>"

    def descendants(self):
        return [self] + [n for c in self.children if isinstance(c, Node) for n in c.descendants()]


class Document(HTMLParser):
    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.stack, self.nodes = [Node("root")], []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        node = Node(tag, attrs)
        self.stack[-1].children.append(node)
        self.nodes.append(node)
        if tag not in {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}:
            self.stack.append(node)

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                break

    def handle_data(self, data):
        self.stack[-1].children.append(data)

    def select(self, selector):
        if ":not(" in selector:
            base, excluded = selector[:-1].split(":not(")
            return [n for n in self.select(base) if n not in self.select(excluded)]
        if selector.startswith("#"):
            return [n for n in self.nodes if n.attrs.get("id") == selector[1:]]
        if selector.startswith("."):
            return [n for n in self.nodes if selector[1:] in (n.attrs.get("class") or "").split()]
        key, _, value = selector[1:-1].partition("=")
        return [n for n in self.nodes if key in n.attrs and (not value or n.attrs[key] == value.strip('"'))]


def demo_leftovers(html):
    """Report demo copy by slot in HTML, including screens currently closed."""
    hits = {}
    def record(where, kind, text):
        text = clean(text or "")
        if DEMO_WORDING.search(text):
            hits.setdefault(f"{where} {kind}", []).append(text)
    def visit(node, where):
        # Preview copy is allowed only here; source code is never UI text.
        if node.tag in {"head", "script", "style", "template"} or node.attrs.get("id") in {"controls", "open-appearance"}:
            return
        if node.attrs.get("id"):
            where = "#" + node.attrs["id"]
        elif node.attrs.get("class"):
            where = "." + ".".join(node.attrs["class"].split())
        for attr in ("aria-label", "title", "placeholder", "alt"):
            record(where, attr, node.attrs.get(attr))
        for child in node.children:
            if isinstance(child, str):
                record(where, "text", child)
            else:
                visit(child, where)
    visit(Document(html).stack[0], "document")
    return [f"{where}: {' | '.join(texts)}" for where, texts in hits.items()]


def extract(html, js=""):
    """Read HTML and the prototype's literal metric table; never execute JS."""
    doc = Document(html)
    central = {}
    # Rendered options take precedence: stale JS must not conceal a changed label.
    for node in doc.nodes:
        if node.attrs.get("id", "").startswith("option-"):
            labels = [n.text() for n in node.descendants() if n.tag == "span" and not n.attrs]
            central[node.attrs["id"][7:]] = labels[0] if labels else node.text()
    if "button.id='option-'+metric.id" in js and "$('#system-grid').append(button)" in js:
        for key, label in re.findall(r"\{id:'([^']+)',title:'([^']+)'", js):
            central.setdefault(key, label)
    selectors = {a for item in INVENTORY for a in item["anchor"]}
    selectors.update(a for screen in SCREENS.values() for group in screen.values() for a in group)
    anchors = {a: [n.markup() for n in doc.select(a)] for a in selectors}
    for key, label in central.items():
        anchors.setdefault("#option-" + key, [])
        if not anchors["#option-" + key]:
            anchors["#option-" + key] = [label]
    def buttons(attr):
        return [(n.attrs[attr], n.text(), tuple(c.attrs.get("href") for c in n.descendants() if c.tag == "use"))
                for n in doc.select(f"[{attr}]") if n.tag == "button"]
    return dict(ids={n.attrs["id"] for n in doc.nodes if "id" in n.attrs},
                symbols={n.attrs.get("id"): n.markup() for n in doc.nodes if n.tag == "symbol"},
                modes=buttons("data-mode"), nav=buttons("data-page"), central=central, anchors=anchors)


def check(html, js=""):
    """Return named structural violations; screenshots and feed tests remain required."""
    reference_html = (REFERENCE / "index.html").read_text()
    expected = extract(reference_html, (REFERENCE / "app.js").read_text())
    actual, violations = extract(html, js), []
    reference_doc, actual_doc = Document(reference_html), Document(html)
    # Patch exactly one decorative child into the parsed reference, independent
    # of the prototype's whitespace and section attributes.
    duck = Document(DUCK_MARKUP).select('#cortex-duck-dock')[0]
    reference_doc.select('.stage')[0].children.insert(0, duck)
    expected['anchors']['#cortex-duck-dock'] = [duck.markup()]
    dock = actual_doc.select('#cortex-duck-dock')
    stage = actual_doc.select('.stage')
    if dock and (not stage or dock[0] not in stage[0].children):
        violations.append('pato do Cortex: moved outside stage')
    mark = Document(TENNIS_BALL_MARK).select(".brand-mark")[0].markup()
    old_mark = reference_doc.select(".brand-mark")[0].markup()
    # Change only this child in the in-memory reference; keep the full brand exact.
    expected["anchors"][".brand"] = [
        brand.replace(old_mark, mark) for brand in expected["anchors"][".brand"]]
    expected["anchors"]["#i-orbit"] = extract(TENNIS_BALL_ORBIT)["anchors"]["#i-orbit"]
    state_select = reference_doc.select("#state-select")[0]
    old_state_select = state_select.markup()
    state_select.children = Document(PREVIEW_STATE_OPTIONS).stack[0].children
    # Replace only the options; the select attributes and full dialog stay exact.
    expected["anchors"]["#controls"] = [
        controls.replace(old_state_select, state_select.markup() + ''.join(node.markup() for node in Document(GESTURE_SELECT).stack[0].children))
        for controls in expected["anchors"]["#controls"]]
    # Unlike other accessibility copy, the brand label remains frozen.
    if ([clean(n.attrs.get("aria-label") or "") for n in actual_doc.select(".brand")] !=
            [clean(n.attrs.get("aria-label") or "") for n in reference_doc.select(".brand")]):
        violations.append("marca: changed aria-label")
    if [n.markup() for n in actual_doc.select(".brand-mark")] != [mark]:
        violations.append("marca (bola de tênis): changed exact markup")
    if actual["anchors"]["#i-orbit"] != expected["anchors"]["#i-orbit"]:
        violations.append("símbolo i-orbit (bola de tênis): changed exact markup")
    if actual_doc.select(".window-dots"):
        violations.append("pontos da janela (removidos): unexpected .window-dots")
    for item in INVENTORY:
        name, anchors, comparison = item["name"], item["anchor"], item.get("comparison")
        missing = [a for a in anchors if not actual["anchors"].get(a)]
        if missing:
            violations.append(f"{name}: missing {', '.join(missing)}")
        if item["rule"] != "exact":
            continue
        if comparison == "central":
            changed = actual["central"].get(item["metric"]) != item["name"]
        elif comparison:
            changed = actual[comparison] != expected[comparison]
            if comparison == "modes":
                values = {button[0] for button in actual["modes"]}
                for value, label, _ in expected["modes"]:
                    if value not in values:
                        violations.append(f"{name}: missing {label} (data-mode={value})")
        else:
            changed = any(actual["anchors"].get(a) != expected["anchors"][a] for a in anchors)
        if changed:
            violations.append(f"{name}: changed exact markup, label, symbol or order")
    return violations
