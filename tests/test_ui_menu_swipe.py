"""Static contract for phone menu swipes alongside the existing tap shield."""
import re

import pytest

from tests.test_ui_menu_tap import CSS, SHELL, function_body
from tests.test_ui_mobile_zoom import css_rules


@pytest.mark.parametrize("event,handler", [
    ("touchstart", "startMenuSwipe"), ("touchmove", "moveMenuSwipe"),
    ("touchend", "endMenuSwipe"), ("touchcancel", "cancelMenuSwipe"),
])
def test_touch_lifecycle_captures_menu_and_workspace_passively(event, handler):
    source = SHELL.read_text()
    assert (f"document.addEventListener('{event}', {handler}, "
            "{ capture: true, passive: true });") in source
    assert "preventDefault" not in function_body(source, handler)


def test_start_requires_phone_single_touch_and_left_half_only_when_closed():
    start = function_body(SHELL.read_text(), "startMenuSwipe")
    assert "!matchMedia('(max-width: 800px)').matches" in start
    assert "e.touches.length !== 1" in start
    assert "const touch = e.touches[0];" in start
    assert "const open = $('.sidebar').classList.contains('open');" in start
    assert "if (!open && touch.clientX >= window.innerWidth / 2) return;" in start
    assert "id: touch.identifier" in start
    assert "x: touch.clientX, y: touch.clientY, open" in start


def test_start_excludes_editors_dialogs_and_horizontal_scroll_ancestors():
    source = SHELL.read_text()
    start = function_body(source, "startMenuSwipe")
    assert "document.querySelector('dialog[open]')" in start
    assert "e.target.closest('input, textarea, select, [contenteditable]')" in start
    assert "hasHorizontalMenuScroll(e.target)" in start
    scroll = function_body(source, "hasHorizontalMenuScroll")
    assert "node = node.parentElement" in scroll
    assert "node.scrollWidth > node.clientWidth" in scroll
    assert "getComputedStyle(node).overflowX" in scroll
    assert "overflowX === 'auto' || overflowX === 'scroll'" in scroll
    # Check the workspace itself before stopping the ancestor walk.
    assert scroll.index("node.scrollWidth") < scroll.index("node.matches('.workspace')")


def test_vertical_intent_multitouch_and_cancel_abandon_the_gesture():
    source = SHELL.read_text()
    move = function_body(source, "moveMenuSwipe")
    assert "e.touches.length !== 1" in move
    assert "touch.identifier === menuSwipe.id" in move
    assert "Math.abs(dy) >= 10 && Math.abs(dx) <= 1.5 * Math.abs(dy)" in move
    assert "menuSwipe = null;" in move
    assert "menuSwipe = null;" in function_body(source, "cancelMenuSwipe")


def test_end_checks_distance_direction_and_uses_shared_menu_setter():
    end = function_body(SHELL.read_text(), "endMenuSwipe")
    assert "menuSwipe = null;" in end
    assert "!matchMedia('(max-width: 800px)').matches" in end
    assert "document.querySelector('dialog[open]')" in end
    assert "e.touches.length" in end
    assert "$('.sidebar').classList.contains('open') !== swipe.open" in end
    assert "e.changedTouches" in end and "touch.identifier === swipe.id" in end
    assert "if (Math.abs(dx) <= 1.5 * Math.abs(dy)) return;" in end
    assert re.search(r"if \(!swipe.open && dx >= 60\) \{\s*setMenuOpen\(true\);", end)
    assert re.search(r"else if \(swipe.open && dx <= -60\) \{\s*setMenuOpen\(false\);", end)
    assert end.count("menuSwipeClick = true;") == 2


def test_swipe_click_is_consumed_before_menu_or_workspace_can_activate():
    source = SHELL.read_text()
    guard = function_body(source, "consumeMenuSwipeClick")
    assert "!menuSwipeClick" in guard
    assert "!matchMedia('(max-width: 800px)').matches" in guard
    assert "e.preventDefault();" in guard and "e.stopPropagation();" in guard
    assert "menuSwipeClick = false;" in guard
    assert "document.addEventListener('click', consumeMenuSwipeClick, true);" in source
    # A new interaction must not inherit suppression when no click followed a swipe.
    reset = function_body(source, "resetMenuSwipeClick")
    assert "menuSwipeClick = false;" in reset
    for event in ("pointerdown", "keydown"):
        assert (f"document.addEventListener('{event}', resetMenuSwipeClick, "
                "{ capture: true, passive: true });") in source


def test_swipes_do_not_disable_native_panning():
    source = SHELL.read_text()
    assert "startMenuSwipe" in source
    for name in ("startMenuSwipe", "moveMenuSwipe", "endMenuSwipe", "cancelMenuSwipe"):
        body = function_body(source, name)
        assert "preventDefault" not in body
        assert "stopPropagation" not in body
    for path in (CSS, CSS.parent / "style.css"):
        for selectors, declarations, _ in css_rules(path.read_text()):
            assert declarations.get("touch-action", "").strip() != "none", selectors
