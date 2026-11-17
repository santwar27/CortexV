"""
notes_client — AppleScript construction + output parsing (mocked osascript).

We never spawn a real osascript here: subprocess.run is patched so we can
assert on the generated AppleScript and parse canned stdout.
"""
import subprocess

import pytest

from src.integrations import notes_client as nc


@pytest.fixture
def run_calls(monkeypatch):
    """Capture the AppleScript passed to osascript; return canned stdout."""
    calls = {"scripts": [], "stdout": ""}

    def fake_run(args, capture_output, text, timeout):
        # args == ["osascript", "-e", <script>]
        calls["scripts"].append(args[2])
        return subprocess.CompletedProcess(args, 0, stdout=calls["stdout"], stderr="")

    monkeypatch.setattr("src.integrations.applescript.subprocess.run", fake_run)
    return calls


class TestCreateNote:
    def test_builds_titled_html_body(self, run_calls):
        nc.create_note("Lista de compras", "leite\novos")
        script = run_calls["scripts"][0]
        assert 'tell application "Notes"' in script
        assert "make new note with properties" in script
        # title is bolded (becomes the Notes title), body newline → <br>
        assert "<b>Lista de compras</b>" in script
        assert "leite<br>ovos" in script

    def test_escapes_quotes_and_html(self, run_calls):
        nc.create_note('Diz "olá" & <b>', "")
        script = run_calls["scripts"][0]
        # HTML-escaped (no raw < or unescaped quote breaking the AS string)
        assert "&lt;b&gt;" in script
        assert "&quot;" in script or "&#x27;" in script or "olá" in script

    def test_empty_title_raises(self, run_calls):
        with pytest.raises(nc.NotesUnavailable):
            nc.create_note("   ")

    def test_returns_summary(self, run_calls):
        out = nc.create_note("Título", "corpo")
        assert out["title"] == "Título"
        assert out["has_body"] is True
        assert out["folder"] is None

    def test_folder_targets_and_creates_when_missing(self, run_calls):
        out = nc.create_note("Pauta", "itens", folder="Trabalho")
        script = run_calls["scripts"][0]
        # Folder is created when missing, then the note targets it.
        assert 'if not (exists folder "Trabalho")' in script
        assert 'make new folder with properties {name:"Trabalho"}' in script
        assert 'make new note at folder "Trabalho"' in script
        assert out["folder"] == "Trabalho"

    def test_no_folder_keeps_default_script(self, run_calls):
        nc.create_note("Pauta", "itens", folder="   ")
        script = run_calls["scripts"][0]
        assert "folder" not in script  # blank folder → default-folder script


class TestAppendToNote:
    def test_finds_exact_then_contains_and_appends(self, run_calls):
        run_calls["stdout"] = "Mercado\n"
        out = nc.append_to_note("Mercado", "pão\nleite")
        script = run_calls["scripts"][0]
        assert 'notes whose name is "Mercado"' in script
        assert 'notes whose name contains "Mercado"' in script
        assert "last item of matches" in script
        # appended as an HTML block, newline → <br>
        assert "pão<br>leite" in script
        assert "set body of theNote to (body of theNote)" in script
        assert out["title"] == "Mercado"

    def test_returns_real_matched_title(self, run_calls):
        run_calls["stdout"] = "Mercado da semana\n"
        out = nc.append_to_note("mercado", "pão")
        assert out["title"] == "Mercado da semana"

    def test_no_match_raises(self, run_calls):
        run_calls["stdout"] = ""
        with pytest.raises(nc.NotesUnavailable, match="No note found"):
            nc.append_to_note("Inexistente", "x")

    def test_empty_args_raise(self, run_calls):
        with pytest.raises(nc.NotesUnavailable):
            nc.append_to_note("", "x")
        with pytest.raises(nc.NotesUnavailable):
            nc.append_to_note("Título", "   ")

    def test_escapes_html_in_text(self, run_calls):
        run_calls["stdout"] = "T\n"
        nc.append_to_note("T", '<script>"x"</script>')
        script = run_calls["scripts"][0]
        assert "&lt;script&gt;" in script


class TestListNotes:
    def test_parses_titles(self, run_calls):
        run_calls["stdout"] = "Compras\nIdeias\n\nViagem\n"
        notes = nc.list_notes()
        assert [n["title"] for n in notes] == ["Compras", "Ideias", "Viagem"]

    def test_limit(self, run_calls):
        run_calls["stdout"] = "\n".join(f"n{i}" for i in range(20))
        assert len(nc.list_notes(limit=5)) == 5

    def test_empty(self, run_calls):
        run_calls["stdout"] = ""
        assert nc.list_notes() == []

    def test_skips_trash_and_caps_in_applescript(self, run_calls):
        nc.list_notes(limit=7)
        script = run_calls["scripts"][0]
        assert '(name of f) is not "Recently Deleted"' in script
        assert '(name of f) is not "Apagados Recentemente"' in script
        assert "cnt < 7" in script        # limit enforced in AppleScript
        assert "exit repeat" in script    # stops early, not O(all notes)


class TestSearchNotes:
    def test_query_in_script_and_parses(self, run_calls):
        run_calls["stdout"] = "Viagem Itália\n"
        hits = nc.search_notes("viagem")
        script = run_calls["scripts"][0]
        assert script.count('contains "viagem"') == 2  # name + plaintext
        assert "plaintext contains" in script           # not raw HTML body
        assert "body contains" not in script
        assert '(name of f) is not "Recently Deleted"' in script  # skip trash
        assert hits == [{"title": "Viagem Itália"}]

    def test_empty_query_short_circuits(self, run_calls):
        assert nc.search_notes("  ") == []
        assert run_calls["scripts"] == []  # never hit osascript


class TestRunErrors:
    def test_nonzero_exit_raises(self, monkeypatch):
        def fake_run(args, capture_output, text, timeout):
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="not allowed")
        monkeypatch.setattr("src.integrations.applescript.subprocess.run", fake_run)
        with pytest.raises(nc.NotesUnavailable, match="not allowed"):
            nc.list_notes()

    def test_timeout_raises(self, monkeypatch):
        def fake_run(args, capture_output, text, timeout):
            raise subprocess.TimeoutExpired(args, timeout)
        monkeypatch.setattr("src.integrations.applescript.subprocess.run", fake_run)
        with pytest.raises(nc.NotesUnavailable):
            nc.list_notes()
