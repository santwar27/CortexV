"""An open phone menu consumes the first workspace tap without activating content."""
from pathlib import Path
import re

from tests.test_ui_mobile_zoom import css_rules


ROOT = Path(__file__).resolve().parents[1]
SHELL = ROOT / "src/ui/static/novo/shell.js"
CSS = ROOT / "src/ui/static/novo/secundario.css"
OVERLAY = ".sidebar.open + .workspace::after"


def function_body(source, name):
    match = re.search(r"function " + name + r"\([^)]*\) \{\n(.*?)\n\}", source, re.S)
    assert match, f"Missing {name} function"
    return match[1]


def overlay_rules():
    rules = [rule for rule in css_rules(CSS.read_text()) if ".workspace::after" in rule[0]]
    assert rules, "Missing workspace overlay"
    for selectors, _, media in rules:
        assert selectors.endswith(OVERLAY), "Overlay must require the open sidebar"
        assert media == ("@media (max-width: 800px)",), "Overlay must be phone-only"
    return rules


def test_overlay_covers_workspace_below_menu_only_on_phone():
    rules = overlay_rules()
    base = next((declarations for selector, declarations, _ in rules if selector == OVERLAY), None)
    assert base is not None
    for key, value in {"content": "''", "position": "fixed", "inset": "0",
                       "z-index": "4", "pointer-events": "auto", "user-select": "none"}.items():
        assert base.get(key) == value
    # The existing sidebar is at z-index 5; the dimmer must not cover menu items.
    assert re.search(r"\.sidebar\{[^}]*z-index:5", (CSS.parent / "style.css").read_text())


def test_overlay_uses_light_dimming_for_both_themes():
    rules = overlay_rules()
    colors = []
    for selector in (OVERLAY, "body.light " + OVERLAY):
        declarations = next((d for s, d, _ in rules if s == selector), None)
        assert declarations is not None, f"Missing theme rule: {selector}"
        color = re.fullmatch(r"rgba\(0, 0, 0, (0?\.\d+)\)", declarations.get("background", ""))
        assert color and 0 < float(color[1]) <= .25
        colors.append(float(color[1]))
    assert colors[1] < colors[0]


def test_reduced_motion_overlay_has_no_transition_or_animation():
    rules = overlay_rules()
    reduced = next((d for s, d, _ in rules if s == "body.reduced-motion " + OVERLAY), None)
    assert reduced is not None
    assert reduced.get("transition") == reduced.get("animation") == "none"


def test_workspace_capture_consumes_only_open_phone_menu_taps():
    source = SHELL.read_text()
    handler = function_body(source, "consumeMenuTap")
    guard = "if (!matchMedia('(max-width: 800px)').matches || !$('.sidebar').classList.contains('open')) return;"
    assert guard in handler
    assert handler.index(guard) < handler.index("e.preventDefault();") < handler.index("e.stopPropagation();")
    # Keep the overlay up through pointerdown; closing then would let the click through.
    assert "if (e.type === 'click') setMenuOpen(false);" in handler
    assert handler.count("setMenuOpen(") == 1
    assert ".focus(" not in handler
    for event in ("pointerdown", "click"):
        assert f"$('.workspace').addEventListener('{event}', consumeMenuTap, true);" in source


def test_menu_state_and_aria_share_all_open_close_paths():
    source = SHELL.read_text()
    setter = function_body(source, "setMenuOpen")
    assert "$('.sidebar').classList.toggle('open', open);" in setter
    assert "$('.mobile-menu').setAttribute('aria-expanded', String(open));" in setter
    assert "setMenuOpen($('.sidebar').classList.contains('open'));" in source
    assert ("$('.mobile-menu').addEventListener('click', () =>\n"
            "  setMenuOpen(!$('.sidebar').classList.contains('open')));") in source
    assert "if (e.key === 'Escape') setMenuOpen(false);" in source
    assert "setMenuOpen(false);" in function_body(source, "navigate")
    # No state mutation may bypass the shared aria update.
    assert re.findall(r"\$\('\.sidebar'\)\.classList\.(?:toggle|remove|add)\([^;]+;", source) == [
        "$('.sidebar').classList.toggle('open', open);"]
