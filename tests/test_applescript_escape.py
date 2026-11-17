"""
M4-A3: one canonical AppleScript string-literal escape.

The correct escape (backslash FIRST, then double quote) existed twice
(reminders_client, notes_client) while macos_controller's three call
sites only escaped the quote — a name ending in `\` produced `\\"`
(literal backslash + END of the literal) and everything after ran as
AppleScript code, including `do shell script`. These tests pin the
canonical helper and prove the controller actually uses it.
"""
from src.integrations.applescript import escape


class TestEscape:
    def test_plain_name_unchanged(self):
        assert escape("Spotify") == "Spotify"

    def test_double_quotes_escaped(self):
        assert escape('Find My "Mac"') == 'Find My \\"Mac\\"'

    def test_trailing_backslash_doubled(self):
        # The injection primitive: `x\` must become `x\\` (a literal
        # backslash) — NOT leak a lone `\` that eats the closing quote.
        assert escape("x\\") == "x\\\\"

    def test_backslash_before_quote(self):
        # `x\"` — backslash escaped FIRST, so the quote's own escape
        # survives: x\\\" (literal backslash + literal quote).
        assert escape('x\\"') == 'x\\\\\\"'

    def test_shell_injection_payload_stays_inert(self):
        payload = 'x\\" & (do shell script "rm -rf ~") & "'
        out = escape(payload)
        # Every original backslash and quote is escaped — nothing can
        # terminate the surrounding AppleScript literal.
        assert '\\\\' in out
        assert '"' not in out.replace('\\"', "")

    def test_none_and_empty(self):
        assert escape("") == ""
        assert escape(None) == ""


class TestControllerUsesEscape:
    """The three S2/S3 surfaces must embed adversarial names inertly."""

    def _capture(self, monkeypatch):
        from src.tools import macos_controller
        captured = {}

        def fake_osascript(script, timeout=5, lang="pt"):
            captured["script"] = script
            # "true" satisfies close_window's running check (M4-A14a) so
            # the flow reaches the activate+Cmd+W script we want to inspect;
            # the other surfaces ignore the output.
            return True, "true"
        monkeypatch.setattr(macos_controller, "_osascript", fake_osascript)
        return macos_controller.MacOSController(), captured

    def test_quit_app_escapes_trailing_backslash(self, monkeypatch):
        ctrl, captured = self._capture(monkeypatch)
        ctrl.quit_app('Evil\\')
        assert 'tell application "Evil\\\\" to quit' == captured["script"]

    def test_quit_app_lookup_is_escaped_too(self, monkeypatch):
        from src.tools import macos_controller
        scripts = []

        def fake_osascript(script, timeout=5, lang="pt"):
            scripts.append(script)
            return True, ""
        monkeypatch.setattr(macos_controller, "_osascript", fake_osascript)
        macos_controller.MacOSController().quit_app('Evil\\')
        assert scripts[0] == 'id of application "Evil\\\\"'

    def test_close_window_escapes_quote(self, monkeypatch):
        ctrl, captured = self._capture(monkeypatch)
        ctrl.close_window('My "App"')
        assert 'tell application "My \\"App\\"" to activate' in captured["script"]

    def test_switch_to_app_escapes_payload(self, monkeypatch):
        ctrl, captured = self._capture(monkeypatch)
        ctrl.switch_to_app('x\\" & (do shell script "id") & "')
        script = captured["script"]
        # The payload must stay INSIDE the literal: no naked `\"` sequence
        # survives un-doubled, so the literal never terminates early.
        assert 'do shell script' in script          # payload present…
        assert '\\\\\\"' in script                  # …but escaped inert


class TestClientsShareTheHelper:
    def test_reminders_and_notes_delegate(self):
        from src.integrations import notes_client, reminders_client
        assert reminders_client._escape is escape
        assert notes_client._escape is escape
