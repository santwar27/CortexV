"""Conservative, source-span HTML repairs; no browser, dependencies or reserialising.

This is deliberately a CSS subset, not a browser. Unsupported paint/cascade
features suppress contrast work; publication still preserves the original file.
"""
from dataclasses import dataclass, field
from html import escape, unescape
from html.parser import HTMLParser
import re

from src.core import clock
from src.core.build_guides import normalise


@dataclass(eq=False)
class Element:
    tag: str
    attrs: dict
    start: int
    opening_end: int
    parent: 'Element | None'
    end_start: int = 0
    end: int = 0
    children: list = field(default_factory=list)
    text: list = field(default_factory=list)

    def ancestors(self):
        node = self
        while node:
            yield node
            node = node.parent

    def words(self):
        return unescape(''.join(t[2] for t in sorted(
            t for node in self.walk() for t in node.text)))

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()


class Document(HTMLParser):
    def __init__(self, source):
        super().__init__(convert_charrefs=False)
        self.source = source
        self.lines = [0] + [m.end() for m in re.finditer('\n', source)]
        self.elements = []
        self.stack = []
        self.valid = True
        self.feed(source)
        self.close()
        # EOF may leave a well-nested prefix open after capped generation.
        # Keep mismatch detection; edits still require complete target spans.
        self.valid = self.valid and all(n.tag not in ('style', 'script') for n in self.stack)

    def position(self):
        line, col = self.getpos()
        return self.lines[line - 1] + col

    def handle_starttag(self, tag, attrs):
        start = self.position()
        parent = self.stack[-1] if self.stack else None
        node = Element(tag, {key: value or '' for key, value in attrs}, start, start + len(self.get_starttag_text()), parent)
        self.elements.append(node)
        if parent:
            parent.children.append(node)
        if tag not in ('area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input',
                       'link', 'meta', 'param', 'source', 'track', 'wbr'):
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if self.stack and self.stack[-1].start == self.position():
            self.stack.pop()

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i].tag == tag:
                if i != len(self.stack) - 1:
                    self.valid = False
                node = self.stack[i]
                node.end_start = self.position()
                end = self.source.find('>', node.end_start)
                node.end = end + 1 if end >= 0 else 0
                del self.stack[i:]
                break

    def handle_data(self, data):
        if self.stack:
            self.stack[-1].text.append((self.position(), self.position() + len(data), data))

    def handle_entityref(self, name):
        self.handle_data('&' + name + ';')

    def handle_charref(self, name):
        self.handle_data('&#' + name + ';')


def _apply(source, edits):
    for start, end, replacement in sorted(edits, reverse=True):
        source = source[:start] + replacement + source[end:]
    return source


_YEAR = re.compile(r'(?:©|&copy;|&#169;|\(c\)|\bCopyright)\s*'
                   r'(?:(?:19|20)\d{2}\s*[-–]\s*)?((?:19|20)\d{2})\b', re.I)


def _years(source, year):
    doc = Document(source)
    edits, changes = [], []
    for node in doc.elements:
        if any(a.tag in ('script', 'style') for a in node.ancestors()):
            continue
        # Join adjacent text/entity spans so &copy; and its year stay together.
        for match in _YEAR.finditer(''.join(t[2] for t in node.text)):
            offset = match.start(1)
            for start, end, raw in node.text:
                if offset < len(raw):
                    if offset + 4 <= len(raw) and match[1] != str(year):
                        edits.append((start + offset, start + offset + 4, str(year)))
                        changes.append(('year', match[1], str(year)))
                    break
                offset -= len(raw)
    return _apply(source, edits), list(dict.fromkeys(changes))


def _words(text):
    return re.findall(r'[a-z0-9]+', normalise(text))


def _buttons(source):
    doc = Document(source)
    # Scripts can register click handlers without onclick attributes.
    if not doc.valid or any(n.tag == 'script' for n in doc.elements):
        return source, [], ''
    button_class = 'cortex-button-link'
    number = 2
    while button_class in source:
        button_class = f'cortex-button-link-{number}'
        number += 1
    ids = {n.attrs.get('id') for n in doc.elements if n.attrs.get('id')}
    headings = []
    for n in doc.elements:
        if n.tag in ('h1', 'h2', 'h3'):
            section = next((a for a in n.ancestors() if a.tag in ('section', 'div', 'article')), None)
            if section:
                headings.append((n, section))
    edits, changes, assigned = [], [], {}
    for node in doc.elements:
        if (node.tag != 'button' or not node.end or node.children
                or node.attrs.get('type', '').lower() == 'submit'
                or any(k.startswith('on') for k in node.attrs)
                or 'form' in node.attrs or 'disabled' in node.attrs
                or any(a.tag == 'form' for a in node.ancestors())):
            continue
        words = _words(node.words())
        # Only navigation phrases, not coincidental substrings (e.g. Delete menu).
        prefixes = ([], ['ver'], ['veja'], ['ver', 'o'], ['ver', 'a'],
                    ['see'], ['see', 'the'], ['view'], ['view', 'the'])
        matches = [(h, s) for h, s in headings
                   if any(words == p + _words(h.words()) for p in prefixes)]
        if len(matches) != 1:
            continue
        heading, section = matches[0]
        target = section.attrs.get('id') or assigned.get(section)
        if not target:
            base = '-'.join(_words(heading.words())) or 'section'
            target, number = base, 2
            while target in ids:
                target = f'{base}-{number}'
                number += 1
            ids.add(target)
            assigned[section] = target
            edits.append((section.opening_end - 1, section.opening_end - 1,
                          f' id="{target}"'))
        elif sum(n.attrs.get('id') == target for n in doc.elements) > 1:
            continue
        opening = source[node.start:node.opening_end]
        # Keep attributes, drop button-only type, and mark only converted links.
        opening = re.sub(r'^<button\b', '<a href="#' + escape(target, quote=True) + '"', opening, flags=re.I)
        opening = re.sub(r'\s+type\s*=\s*(?:"[^"]*"|\x27[^\x27]*\x27|[^\s>]+)', '', opening, flags=re.I)
        if 'class' in node.attrs:
            opening = re.sub(r'\s+class\s*=\s*(?:"[^"]*"|\x27[^\x27]*\x27|[^\s>]+)',
                             lambda _: ' class="' + escape(node.attrs['class'], quote=True) + ' ' + button_class + '"',
                             opening, count=1, flags=re.I)
        else:
            opening = opening[:-1] + f' class="{button_class}">'
        edits.extend([(node.start, node.opening_end, opening), (node.end_start, node.end, '</a>')])
        changes.append(('button', node.words().strip(), heading.words().strip()))
    source = _apply(source, edits)
    if changes:
        # Install before contrast evaluation, so it judges the final inherited
        # color rather than the browser's default blue for an unstyled link.
        source = _append_style(source, f'a.{button_class}'
                               '{display:inline-block;text-decoration:none;color:inherit;}')
    return source, changes, button_class if changes else ''


def _append_style(source, rules):
    doc = Document(source)
    styles = [n for n in doc.elements if n.tag == 'style' and n.end_start]
    if not styles:
        head = next((n for n in doc.elements if n.tag == 'head'), None)
        body = next((n for n in doc.elements if n.tag == 'body'), None)
        at = head.opening_end if head else body.start if body else 0
        return source[:at] + '<style>\n' + rules + '\n</style>' + source[at:]
    at = styles[-1].end_start
    return source[:at] + '\n' + rules + '\n' + source[at:]


# CSS named colors (CSS Color 3 + rebeccapurple); aliases share the same RGB.
_NAMES = dict(item.split(':') for item in '''
aliceblue:f0f8ff antiquewhite:faebd7 aqua:00ffff aquamarine:7fffd4 azure:f0ffff
beige:f5f5dc bisque:ffe4c4 black:000000 blanchedalmond:ffebcd blue:0000ff blueviolet:8a2be2 brown:a52a2a burlywood:deb887
cadetblue:5f9ea0 chartreuse:7fff00 chocolate:d2691e coral:ff7f50 cornflowerblue:6495ed cornsilk:fff8dc crimson:dc143c cyan:00ffff
darkblue:00008b darkcyan:008b8b darkgoldenrod:b8860b darkgray:a9a9a9 darkgrey:a9a9a9 darkgreen:006400 darkkhaki:bdb76b darkmagenta:8b008b darkolivegreen:556b2f darkorange:ff8c00 darkorchid:9932cc darkred:8b0000 darksalmon:e9967a darkseagreen:8fbc8f darkslateblue:483d8b darkslategray:2f4f4f darkslategrey:2f4f4f darkturquoise:00ced1 darkviolet:9400d3
deeppink:ff1493 deepskyblue:00bfff dimgray:696969 dimgrey:696969 dodgerblue:1e90ff firebrick:b22222 floralwhite:fffaf0 forestgreen:228b22 fuchsia:ff00ff gainsboro:dcdcdc ghostwhite:f8f8ff gold:ffd700 goldenrod:daa520 gray:808080 grey:808080 green:008000 greenyellow:adff2f honeydew:f0fff0 hotpink:ff69b4 indianred:cd5c5c indigo:4b0082 ivory:fffff0 khaki:f0e68c lavender:e6e6fa lavenderblush:fff0f5 lawngreen:7cfc00 lemonchiffon:fffacd
lightblue:add8e6 lightcoral:f08080 lightcyan:e0ffff lightgoldenrodyellow:fafad2 lightgray:d3d3d3 lightgrey:d3d3d3 lightgreen:90ee90 lightpink:ffb6c1 lightsalmon:ffa07a lightseagreen:20b2aa lightskyblue:87cefa lightslategray:778899 lightslategrey:778899 lightsteelblue:b0c4de lightyellow:ffffe0 lime:00ff00 limegreen:32cd32 linen:faf0e6 magenta:ff00ff maroon:800000 mediumaquamarine:66cdaa mediumblue:0000cd mediumorchid:ba55d3 mediumpurple:9370db mediumseagreen:3cb371 mediumslateblue:7b68ee mediumspringgreen:00fa9a mediumturquoise:48d1cc mediumvioletred:c71585 midnightblue:191970 mintcream:f5fffa mistyrose:ffe4e1 moccasin:ffe4b5 navajowhite:ffdead navy:000080 oldlace:fdf5e6 olive:808000 olivedrab:6b8e23 orange:ffa500 orangered:ff4500 orchid:da70d6 palegoldenrod:eee8aa palegreen:98fb98 paleturquoise:afeeee palevioletred:db7093 papayawhip:ffefd5 peachpuff:ffdab9 peru:cd853f pink:ffc0cb plum:dda0dd powderblue:b0e0e6 purple:800080 rebeccapurple:663399 red:ff0000 rosybrown:bc8f8f royalblue:4169e1 saddlebrown:8b4513 salmon:fa8072 sandybrown:f4a460 seagreen:2e8b57 seashell:fff5ee sienna:a0522d silver:c0c0c0 skyblue:87ceeb slateblue:6a5acd slategray:708090 slategrey:708090 snow:fffafa springgreen:00ff7f steelblue:4682b4 tan:d2b48c teal:008080 thistle:d8bfd8 tomato:ff6347 turquoise:40e0d0 violet:ee82ee wheat:f5deb3 white:ffffff whitesmoke:f5f5f5 yellow:ffff00 yellowgreen:9acd32
'''.split())


def color(value, variables=None, seen=frozenset()):
    value = value.strip()
    if value.startswith('var('):
        match = re.fullmatch(r'var\((--[\w-]+)\)', value)
        if not match or match[1] in seen:
            return None
        return color((variables or {}).get(match[1], ''), variables, seen | {match[1]})
    value = value.lower()
    value = '#' + _NAMES[value] if value in _NAMES else value
    if re.fullmatch(r'#[0-9a-f]{3}', value):
        value = '#' + ''.join(c * 2 for c in value[1:])
    if re.fullmatch(r'#[0-9a-f]{6}', value):
        return tuple(int(value[i:i + 2], 16) for i in (1, 3, 5))
    match = re.fullmatch(r'rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', value)
    if match and all(int(v) <= 255 for v in match.groups()):
        return tuple(map(int, match.groups()))
    return None


def contrast(a, b):
    def luminance(rgb):
        values = [v / 255 for v in rgb]
        linear = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4 for v in values]
        return sum(v * w for v, w in zip(linear, (.2126, .7152, .0722)))
    x, y = sorted((luminance(a), luminance(b)))
    return (y + .05) / (x + .05)


_ATOM = r'(?:[a-zA-Z][\w-]*)?(?:[.#][\w-]+)*(?::nth-of-type\(\d+\))*'
_SELECTOR = re.compile(_ATOM + r'(?:\s*(?:>\s*|\s)' + _ATOM + r')*')


def _match_atom(node, atom):
    tag = re.match(r'^[a-zA-Z][\w-]*', atom)
    if tag and node.tag != tag[0].lower():
        return False
    for prefix, value in re.findall(r'([.#])([\w-]+)', atom):
        if (value != node.attrs.get('id') if prefix == '#' else value not in node.attrs.get('class', '').split()):
            return False
    nth = re.findall(r':nth-of-type\((\d+)\)', atom)
    if nth:
        peers = [n for n in node.parent.children if n.tag == node.tag] if node.parent else [node]
        if any(peers.index(node) + 1 != int(number) for number in nth):
            return False
    return True


def _matches(node, selector):
    tokens = re.findall(r'[^\s>]+|>', selector)
    def match(n, i):
        if n is None or not _match_atom(n, tokens[i]):
            return False
        if i == 0:
            return True
        if tokens[i - 1] == '>':
            return match(n.parent, i - 2)
        return any(match(a, i - 1) for a in list(n.ancestors())[1:])
    return bool(tokens) and match(node, len(tokens) - 1)


def _specificity(selector):
    return (selector.count('#'), selector.count('.') + selector.count(':nth-of-type'),
            len(re.findall(r'(?:^|[\s>])([a-zA-Z][\w-]*)', selector)))


def _exact_selector(node, required):
    parts = []
    for n in reversed(list(node.ancestors())):
        peers = [c for c in n.parent.children if c.tag == n.tag] if n.parent else [n]
        part = n.tag
        ident = n.attrs.get('id', '')
        if re.fullmatch(r'[a-zA-Z_][\w-]*', ident):
            part += '#' + ident
        parts.append(part + f':nth-of-type({peers.index(n) + 1})')
    selector = ' > '.join(parts)
    specificity = _specificity(selector)
    if specificity[0] < required[0]:
        return None  # An identifier we cannot safely emit; do not claim a fix.
    if specificity <= required:
        # Repeating the same positional condition changes specificity, not
        # which element matches. Compound classes must not beat the repair.
        position = re.search(r':nth-of-type\(\d+\)$', selector)[0]
        selector += position * (required[1] - specificity[1] + 1)
    return selector


def _without_layout_media(css):
    while '@' in css:
        start = css.index('@')
        opening = css.find('{', start)
        if opening < 0 or not re.fullmatch(r'@media[^{};]+', css[start:opening].strip(), re.I):
            return None
        depth, end = 1, opening + 1
        while end < len(css) and depth:
            depth += (css[end] == '{') - (css[end] == '}')
            end += 1
        if depth:
            return None
        body = css[opening + 1:end - 1]
        # Unknown conditional features are deliberately not inferred from their
        # selector. Even an override on an ancestor can change a text's paint.
        if re.search(r'(?:color|background[\w-]*|opacity|filter|mix-blend-mode|all|'
                     r'-webkit-text-fill-color|--[\w-]+)\s*:', body, re.I) or '@' in body:
            return None
        css = css[:start] + css[end:]
    return css


def _color_context(source):
    doc = Document(source)
    styles = [n for n in doc.elements if n.tag == 'style' and n.end_start]
    if not doc.valid or not styles or any(n.tag == 'script' or (n.tag == 'link' and 'stylesheet' in n.attrs.get('rel', ''))
                         for n in doc.elements):
        return None
    css = '\n'.join(source[n.opening_end:n.end_start] for n in styles)
    css = re.sub(r'/\*.*?\*/', '', css, flags=re.S)
    # A layout-only media query cannot change our colors. Conditional paint,
    # imports, layers and malformed CSS require a browser: leave those alone.
    css = _without_layout_media(css)
    if (css is None or any(n.attrs.get('media') or 'scoped' in n.attrs
                           or n.attrs.get('type', 'text/css') != 'text/css' for n in styles)
            or css.count('{') != css.count('}') or re.search(r'\{[^}]*\{', css)):
        return None
    blocks = re.findall(r'([^{}]+)\{([^{}]*)\}', css)
    variables, rules, palette, unsafe_selectors = {}, [], [], []
    for selectors, body in blocks:
        if selectors.strip() == ':root':
            for name, val in re.findall(r'(--[\w-]+)\s*:\s*([^;]+)', body):
                variables[name] = val.strip()
    for selectors, body in blocks:
        declarations = []
        for declaration in body.split(';'):
            if ':' not in declaration:
                continue
            key, value = declaration.split(':', 1)
            key, value = key.strip().lower(), value.strip()
            if key.startswith('--'):
                parsed = color(value, variables)
                if parsed:
                    palette.append(parsed)
                if selectors.strip() != ':root':
                    declarations.append(('custom-variable', value, None))
                continue
            if key not in ('color', 'background', 'background-color', 'background-image',
                           'opacity', 'filter', 'mix-blend-mode', 'all', '-webkit-text-fill-color'):
                continue
            parsed = color(value, variables)
            if parsed:
                palette.append(parsed)
            declarations.append((key, value, parsed))
        if not declarations:
            continue
        for selector in selectors.split(','):
            selector = selector.strip()
            root = selector == ':root'
            if root:
                selector = 'html'
            if not selector or not _SELECTOR.fullmatch(selector):
                # Hover/focus rules taint only their base context; other selector
                # syntax cannot be localised with certainty, so skip the page.
                base = re.sub(r':(?:hover|focus|focus-visible|active|visited)\b', '', selector)
                if base != selector and _SELECTOR.fullmatch(base):
                    unsafe_selectors.append((base, declarations))
                    continue
                return None
            rules.append((selector, (0, 1, 0) if root else _specificity(selector), declarations))
    return doc, rules, list(dict.fromkeys(palette)), unsafe_selectors


def _computed_style(node, computed, rules, unsafe_selectors):
    parent_color, parent_bg, parent_unsafe = computed.get(node.parent, ((0, 0, 0), (255, 255, 255), False))
    foreground = (0, 0, 238) if node.tag == 'a' and 'href' in node.attrs else parent_color
    background = parent_bg
    unsafe = (parent_unsafe or bool(node.attrs.get('style', '').strip()) or 'bgcolor' in node.attrs
              or 'background' in node.attrs or node.tag in ('svg', 'canvas', 'button', 'select', 'textarea')
              or any(k.startswith('on') for k in node.attrs))
    unsafe |= any(_matches(node, selector) for selector, _ in unsafe_selectors)
    matched = sorted((spec, order, declarations) for order, (sel, spec, declarations) in enumerate(rules)
                     if _matches(node, sel))
    for _, _, declarations in matched:
        for key, value, parsed in declarations:
            # Once uncertain, never pretend a later normal rule wins over it.
            if '!important' in value.lower():
                unsafe = True
            elif key == 'color':
                foreground = parent_color if value == 'inherit' else parsed
            elif key in ('background', 'background-color'):
                background = parent_bg if value == 'transparent' else parsed
            elif not (key == 'background-image' and value == 'none'):
                unsafe = True
    unsafe |= foreground is None or background is None
    return foreground, background, unsafe, matched


def _computed_colors(source):
    """Resolve the supported normal-state cascade, including appended repairs."""
    context = _color_context(source)
    computed = {}
    if context:
        doc, rules, _, unsafe_selectors = context
        for node in doc.elements:
            computed[node] = _computed_style(node, computed, rules, unsafe_selectors)[:3]
    return computed


def _contrast_repairs(source, button_class=''):
    context = _color_context(source)
    if context is None:
        return source, []
    doc, rules, palette, unsafe_selectors = context
    computed = {}
    overrides, count = [], 0
    for node in doc.elements:
        foreground, background, unsafe, matched = _computed_style(node, computed, rules, unsafe_selectors)
        backgrounds = [background]
        if button_class and button_class in node.attrs.get('class', '').split():
            # A converted button may have known background-only hover/focus
            # rules. Check every such background rather than skipping its text
            # or choosing a color that only works when the pointer is absent.
            states = [d for sel, d in unsafe_selectors if _matches(node, sel)]
            normal_unsafe = _computed_style(node, computed, rules, [])[2]
            if not normal_unsafe and all(
                    key in ('background', 'background-color') and (parsed or value == 'transparent')
                    for declarations in states for key, value, parsed in declarations):
                unsafe = False
                backgrounds += [parsed for declarations in states for _, _, parsed in declarations if parsed]
        direct = any(unescape(raw).strip() for _, _, raw in node.text)
        threshold = 3 if node.tag in ('h1', 'h2', 'h3') else 4.5
        if (not unsafe and direct and any(a.tag == 'body' for a in node.ancestors())
                and node.tag not in ('script', 'style')
                and any(contrast(foreground, bg) < threshold for bg in backgrounds)):
            score = lambda c: min(contrast(c, bg) for bg in backgrounds)
            candidates = [c for c in palette if score(c) >= threshold]
            chosen = max(candidates or [(17, 17, 17), (255, 255, 255)], key=score)
            # Around #777 neither specified fallback reaches 4.5:1. True
            # black closes that gap; never count a still-unreadable override.
            if score(chosen) < threshold:
                chosen = (0, 0, 0)
            value = '#' + ''.join(f'{v:02x}' for v in chosen)
            required = max((spec for spec, _, _ in matched), default=(0, 0, 0))
            selector = _exact_selector(node, required)
            if selector and score(chosen) >= threshold:
                overrides.append(selector + '{color:' + value + ';}')
                foreground = chosen
                count += 1
        computed[node] = foreground, background, unsafe
    if not overrides:
        return source, []
    return _append_style(source, '\n'.join(overrides)), [('contrast', count)]


def repair_html(source: str, *, year: int | None = None) -> tuple[str, list]:
    """Return fixed source and structured changes, using the injectable local clock."""
    year = clock.local_now().year if year is None else year
    source, years = _years(source, year)
    source, buttons, button_class = _buttons(source)
    source, colors = _contrast_repairs(source, button_class)
    return source, years + colors + buttons


def _join(items, lang):
    return items[0] if len(items) == 1 else ', '.join(items[:-1]) + (' e ' if lang == 'pt' else ' and ') + items[-1]


def fixes_line(changes, lang):
    parts = []
    for change in changes:
        if change[0] == 'year':
            parts.append(('o ano do rodapé' if lang == 'pt' else 'the copyright year') + f' ({change[1]} → {change[2]})')
        elif change[0] == 'contrast':
            n = change[1]
            parts.append((f'o contraste de {n} texto' + ('s' if n != 1 else '')) if lang == 'pt' else
                         f'contrast for {n} text element' + ('s' if n != 1 else ''))
        elif change[0] == 'button':
            parts.append((f'o botão "{change[1]}" (agora leva a "{change[2]}")' if lang == 'pt' else
                          f'the "{change[1]}" button (now links to "{change[2]}")'))
    return (('Corrigi no código: ' if lang == 'pt' else 'Fixed in the code: ') + _join(parts, lang) + '.') if parts else ''


def example_warning(source, request_text, lang):
    """Only recognisable labelled data/currency/time spans; prefer false negatives."""
    doc = Document(source)
    text = '\n'.join(n.words().strip() for n in doc.elements
                     if n.text and not n.children and n.tag not in ('script', 'style', 'title'))
    supplied = ' '.join(_words(request_text))
    patterns = [
        ('endereço', 'address', r'(?:endere[cç]o|address)\s*:\s*([^\n]+)'),
        ('telefone', 'phone', r'(?:telefone|phone|tel)\s*:\s*([+()\d][\d\s()+.-]{6,}\d)'),
        ('preços', 'prices', r'((?:£|\$|€|R\$)\s*\d+(?:[.,]\d{2})?)'),
        ('horários', 'opening hours', r'(\d{1,2}(?::\d{2}|h)(?:\s*(?:am|pm))?\s*(?:às|a|to|[-–])\s*\d{1,2}(?::\d{2}|h)(?:\s*(?:am|pm))?)'),
    ]
    kinds = []
    singular = False
    for pt, en, pattern in patterns:
        values = re.findall(pattern, text, re.I)
        if en == 'phone':
            # Formatting a supplied number with spaces/brackets is not invention.
            missing = any(re.sub(r'\D', '', value) not in re.sub(r'\D', '', request_text)
                          for value in values)
        else:
            missing = any(' '.join(_words(value)) not in supplied for value in values)
        if missing:
            kinds.append(pt if lang == 'pt' else en)
            singular = en in ('address', 'phone') and len(values) == 1
    if not kinds:
        return ''
    return ('Confira ' + _join(kinds, lang) + ': são dados de exemplo.' if lang == 'pt' else
            'Check the ' + _join(kinds, lang)
            + (': it is example data.' if len(kinds) == 1 and singular else ': they are example data.'))
