"""Phone form fonts prevent Safari focus zoom without restricting pinch zoom."""
import re

import pytest

from tests.test_ui_routes import client, ui_app  # noqa: F401
from tests.ui_parity import Document


def css_rules(css):
    """Read rule blocks and their enclosing media queries, including minified CSS."""
    stack = []
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    for match in re.finditer(r"([^{}]*)([{}])", css):
        text, brace = match.groups()
        if brace == "{":
            stack.append(text.strip())
        else:
            selector = stack.pop()
            if not selector.startswith("@"):
                yield selector, dict(re.findall(r"([\w-]+)\s*:\s*([^;]+)", text)), tuple(stack)


def selector_matches(selector, ancestry):
    """Match the tag/class/:root descendant selectors used by the field font rules."""
    parts = selector.split()
    for tag, classes in reversed(ancestry):
        part = parts[-1]
        tokens = re.findall(r"[.#:]?[\w-]+", part)
        if "".join(tokens) != part:
            return False
        if all(token[1:] in classes if token.startswith(".") else
               tag == "html" if token == ":root" else token == tag
               for token in tokens):
            parts.pop()
            if not parts:
                return True
        elif len(parts) == len(selector.split()):
            return False  # The last compound must match the field itself.
    return False


def field_font_rules(rules, ancestry, width):
    for selectors, declarations, media in rules:
        if "font-size" not in declarations:
            continue
        if any(not query.startswith("@media") for query in media):
            continue
        bounds = [bound for query in media for bound in
                  re.findall(r"(min|max)-width\s*:\s*(\d+)px", query)]
        if any(width > int(px) if kind == "max" else width < int(px) for kind, px in bounds):
            continue
        for selector in selectors.split(","):
            if selector_matches(selector.strip(), ancestry):
                value = declarations["font-size"].strip()
                specificity = (selector.count("#"), len(re.findall(r"[.:][\w-]+", selector)),
                               len(re.findall(r"(?:^|\s)([a-z][\w-]*)", selector)))
                priority = ("!important" in value, *specificity)
                yield priority, value.replace("!important", "").strip(), bounds


def winning_font(candidates):
    return max(enumerate(candidates), key=lambda item: (item[1][0], item[0]))[1]


@pytest.fixture
def served_rules(client):
    response = client.get("/ui/novo", headers={"X-Cortex-Key": "test-key-xyz"})
    assert response.status_code == 200
    rules = []
    for node in Document(response.text).nodes:
        if node.tag == "link" and node.attrs.get("rel") == "stylesheet":
            css = client.get(node.attrs["href"])
            assert css.status_code == 200
            rules.extend(css_rules(css.text))
    assert rules
    return rules


@pytest.mark.parametrize("width", [390, 800])
@pytest.mark.parametrize("tag,classes,desktop_size", [
    ("input", (), None), ("textarea", (), None), ("select", (), None),
    ("textarea", ("composer",), "13px"),
    ("input", ("contacts-form",), "12px"),
])
def test_phone_field_font_wins_cascade(served_rules, width, tag, classes, desktop_size):
    ancestry = [("html", ()), ("body", ()), ("form", classes), (tag, ())]
    candidates = list(field_font_rules(served_rules, ancestry, width))
    assert candidates, f"No explicit font size for {classes} {tag} at {width}px"
    priority, size, bounds = winning_font(candidates)
    assert re.fullmatch(r"\d+(?:\.\d+)?px", size), size
    assert float(size[:-2]) >= 16, f"Winning font for {classes} {tag}: {size} at {width}px"
    assert ("max", "800") in bounds
    # A smaller declaration must never win by moving either stylesheet or rule.
    assert all(other_priority < priority for other_priority, other_size, _ in candidates
               if other_size != size)
    for desktop_width in (801, 1440):
        desktop = list(field_font_rules(served_rules, ancestry, desktop_width))
        assert (winning_font(desktop)[1] if desktop else None) == desktop_size


@pytest.mark.parametrize("path", ["/ui/novo", "/ui/login"])
def test_viewport_keeps_pinch_zoom(client, path):
    response = client.get(path, headers={"X-Cortex-Key": "test-key-xyz"})
    assert response.status_code == 200
    viewports = [node.attrs["content"].lower() for node in Document(response.text).nodes
                 if node.tag == "meta" and node.attrs.get("name") == "viewport"]
    assert len(viewports) == 1
    assert "maximum-scale" not in viewports[0]
    assert not re.search(r"user-scalable\s*=\s*no\b", viewports[0])


def test_login_password_font_is_16px(client):
    response = client.get("/ui/login")
    assert response.status_code == 200
    passwords = [node for node in Document(response.text).nodes
                 if node.tag == "input" and node.attrs.get("type") == "password"]
    assert len(passwords) == 1
    declarations = dict(re.findall(r"([\w-]+)\s*:\s*([^;]+)", passwords[0].attrs["style"]))
    assert declarations["font-size"] == "16px"


def test_phone_hint_stays_on_one_line_without_shrinking_the_typed_text():
    """The 16px phone field made the hint wrap and get cut; only the hint is smaller and never wraps."""
    from pathlib import Path
    css = (Path(__file__).resolve().parents[1] / "src/ui/static/novo/secundario.css").read_text()
    rules = [(selector, values, media) for selector, values, media in css_rules(css)
             if selector == ".composer textarea::placeholder"]
    plain = [values for _, values, media in rules if not media]
    phone = [values for _, values, media in rules if media == ("@media (max-width: 800px)",)]
    assert plain == [{"white-space": "nowrap", "overflow": "hidden", "text-overflow": "ellipsis"}]
    assert phone == [{"font-size": "13px"}]
    # The field itself is never made smaller than 16px on the phone by this rule.
    assert all("textarea::placeholder" in selector for selector, values, _ in css_rules(css)
               if values.get("font-size") == "13px" and "textarea" in selector)
