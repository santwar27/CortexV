"""Spotlight-backed file search (find_file, s81).

The search itself is `mdfind` — not mocked away entirely, but driven through a
stubbed subprocess so the tests are deterministic and never depend on what is
actually indexed on the machine. What matters here is the safety envelope:
results are confined to $HOME, credential/key paths are filtered OUT of the
results, output is bounded, and failures degrade to a reason string instead of
raising.
"""
import subprocess
from unittest.mock import patch

import pytest

from src.core.dispatch.heuristics import REAL_TOOL_HINT_RE, TOOL_HINT_RE
from src.tools.file_finder import (
    FileFinder,
    _human_size,
    _is_excluded,
    format_results,
)


def _proc(stdout="", returncode=0, stderr=""):
    return subprocess.CompletedProcess(
        args=["mdfind"], returncode=returncode, stdout=stdout, stderr=stderr)


class TestExclusions:
    """Credential stores and Cortex's own state must never surface in results,
    even when Spotlight has them indexed."""

    @pytest.mark.parametrize("path", [
        "/Users/x/.ssh/id_rsa",
        "/Users/x/.gnupg/secring.gpg",
        "/Users/x/.aws/credentials",
        "/Users/x/project/data/credentials/token.json",
        "/Users/x/Library/Keychains/login.keychain",
        "/Users/x/.cortex/palace.db",
        "/Users/x/proj/.env",
        "/Users/x/proj/node_modules/left-pad/index.js",
        "/Users/x/proj/.git/config",
        "/Users/x/.Trash/old.pdf",
    ])
    def test_sensitive_and_noise_paths_excluded(self, path):
        assert _is_excluded(path) is True

    @pytest.mark.parametrize("path", [
        "/Users/x/Downloads/id_ed25519",
        "/Users/x/Downloads/id_rsa",
        "/Users/x/Downloads/client.pem",
        "/Users/x/Documents/cert.p12",
        "/Users/x/.netrc",
        "/Users/x/Downloads/Chrome Passwords.csv",
        "/Users/x/Documents/senhas.csv",
        "/Users/x/Documents/Cofre.kdbx",
        "/Users/x/.codex/auth.json",
        "/Users/x/.config/openai/token",
        "/Users/x/Library/Mobile Documents/com~apple~CloudDocs/backup/id_rsa",
    ])
    def test_key_material_is_excluded_by_name_anywhere(self, path):
        """Only their usual FOLDERS were excluded: the same files in Downloads
        came back as ordinary hits (s88, Codex lote 3)."""
        assert _is_excluded(path) is True

    @pytest.mark.parametrize("path", [
        "/Users/x/Documents/curriculo.pdf",
        "/Users/x/Desktop/contrato.docx",
        "/Users/x/Downloads/nota fiscal.pdf",
        "/Users/x/Documents/Apresentação.key",          # a Keynote deck
        "/Users/x/Downloads/id_ed25519.pub",            # a public key
        "/Users/x/Documents/Password policy.pdf",
    ])
    def test_ordinary_files_kept(self, path):
        assert _is_excluded(path) is False

    @pytest.mark.parametrize("path", [
        "/Users/x/Library/Mobile Documents/com~apple~CloudDocs/college/essay.pdf",
        "/Users/x/Library/Mobile Documents/com~apple~CloudDocs/notas.md",
    ])
    def test_icloud_drive_is_not_app_data(self, path):
        """iCloud Drive lives inside ~/Library, so the blanket "/Library/" rule
        hid the user's own synced documents — a college PDF was neither
        findable nor readable (s83)."""
        assert _is_excluded(path) is False

    @pytest.mark.parametrize("path", [
        "/Users/x/Library/Caches/com.apple.Safari/cache.db",
        "/Users/x/Library/Application Support/Notion/state.json",
        "/Users/x/Library/Keychains/login.keychain",
        # the exception is iCloud Drive itself, not everything Apple syncs
        "/Users/x/Library/Mobile Documents/com~apple~Pages/draft.pages",
        # and every other rule still applies INSIDE iCloud
        "/Users/x/Library/Mobile Documents/com~apple~CloudDocs/proj/.env",
        "/Users/x/Library/Mobile Documents/com~apple~CloudDocs/p/.git/config",
    ])
    def test_the_icloud_exception_is_narrow(self, path):
        assert _is_excluded(path) is True

    def test_excluded_hit_is_dropped_from_results(self, tmp_path):
        finder = FileFinder(home=str(tmp_path))
        out = "/Users/x/.ssh/id_rsa\n/Users/x/Documents/cv.pdf\n"
        with patch("subprocess.run", return_value=_proc(out)):
            hits, err = finder.search("cv")
        assert err is None
        assert [h.name for h in hits] == ["cv.pdf"]   # the key never appears


class TestSearchBehaviour:
    def test_confines_search_to_home(self, tmp_path):
        finder = FileFinder(home=str(tmp_path))
        with patch("subprocess.run", return_value=_proc("")) as run:
            finder.search("x")
        args = run.call_args[0][0]
        assert args[0] == "mdfind"
        assert "-onlyin" in args and str(tmp_path) in args

    def test_name_search_by_default(self, tmp_path):
        finder = FileFinder(home=str(tmp_path))
        with patch("subprocess.run", return_value=_proc("")) as run:
            finder.search("curriculo")
        assert "-name" in run.call_args[0][0]

    def test_content_search_drops_name_flag(self, tmp_path):
        finder = FileFinder(home=str(tmp_path))
        with patch("subprocess.run", return_value=_proc("")) as run:
            finder.search("orçamento", content=True)
        assert "-name" not in run.call_args[0][0]

    def test_limit_is_capped(self, tmp_path):
        finder = FileFinder(home=str(tmp_path))
        out = "\n".join(f"/Users/x/Documents/f{i}.txt" for i in range(200))
        with patch("subprocess.run", return_value=_proc(out)):
            hits, _ = finder.search("f", limit=5)
        assert len(hits) == 5

    def test_empty_query_rejected_without_running_mdfind(self, tmp_path):
        finder = FileFinder(home=str(tmp_path))
        with patch("subprocess.run") as run:
            hits, err = finder.search("   ")
        assert hits == [] and err
        run.assert_not_called()


class TestFailSoft:
    """A failing search returns a reason — it must never raise into a turn."""

    def test_missing_mdfind(self, tmp_path):
        finder = FileFinder(home=str(tmp_path))
        with patch("subprocess.run", side_effect=FileNotFoundError()):
            hits, err = finder.search("x")
        assert hits == [] and "mdfind" in err

    def test_timeout(self, tmp_path):
        finder = FileFinder(home=str(tmp_path))
        with patch("subprocess.run",
                   side_effect=subprocess.TimeoutExpired("mdfind", 10)):
            hits, err = finder.search("x")
        assert hits == [] and "excedeu" in err

    def test_nonzero_exit(self, tmp_path):
        finder = FileFinder(home=str(tmp_path))
        with patch("subprocess.run", return_value=_proc("", 1, "boom")):
            hits, err = finder.search("x")
        assert hits == [] and err

    def test_unexpected_exception(self, tmp_path):
        finder = FileFinder(home=str(tmp_path))
        with patch("subprocess.run", side_effect=RuntimeError("nope")):
            hits, err = finder.search("x")
        assert hits == [] and "RuntimeError" in err

    def test_vanished_file_still_listed(self, tmp_path):
        # Indexed but deleted between index and stat → no crash, no metadata.
        finder = FileFinder(home=str(tmp_path))
        with patch("subprocess.run",
                   return_value=_proc("/Users/x/Documents/gone.pdf\n")):
            hits, err = finder.search("gone")
        assert err is None and hits[0].name == "gone.pdf"
        assert hits[0].size is None


class TestFormatting:
    def test_no_results_is_explicit(self):
        assert "Nenhum arquivo" in format_results([], "cv", None)

    def test_error_is_surfaced_as_system_error(self):
        out = format_results([], "cv", "timeout")
        assert out.startswith("[SYSTEM ERROR]")

    def test_lists_hits_with_paths(self, tmp_path):
        finder = FileFinder(home=str(tmp_path))
        with patch("subprocess.run",
                   return_value=_proc("/Users/x/Documents/cv.pdf\n")):
            hits, err = finder.search("cv")
        out = format_results(hits, "cv", err)
        assert "cv.pdf" in out and "/Users/x/Documents/cv.pdf" in out

    def test_human_size(self):
        assert _human_size(0) == "0B"
        assert _human_size(900) == "900B"
        assert _human_size(2048) == "2.0KB"
        assert _human_size(5 * 1024 ** 2) == "5.0MB"


class TestHintRegexes:
    """The two hint regexes gate different things (pure-chat veto vs. tool
    menu). find_file needs BOTH — wiring only one is the bug this guards."""

    FINDS = [
        "cadê meu currículo", "cadê o contrato",
        "onde está o pdf do contrato", "onde tá o meu cv",
        "acha o arquivo de impostos", "procura minhas fotos da viagem",
        "encontra a planilha", "where is my resume", "find the invoice",
    ]
    NOT_FINDS = [
        "onde fica o Japão",          # geography, not files
        "onde fica a Torre Eiffel",
        "procura saber mais sobre isso",
        "acha que vai chover",
        "find out what happened",
        "me explica o que é RAG",
        "qual a capital do Brasil",
    ]

    @pytest.mark.parametrize("q", FINDS)
    def test_file_search_hits_both_regexes(self, q):
        assert TOOL_HINT_RE.search(q), f"{q!r} must veto pure-chat"
        assert REAL_TOOL_HINT_RE.search(q), f"{q!r} must offer the tool menu"

    @pytest.mark.parametrize("q", NOT_FINDS)
    def test_non_file_phrasings_do_not_match(self, q):
        assert not TOOL_HINT_RE.search(q), f"{q!r} is a false positive"
        assert not REAL_TOOL_HINT_RE.search(q), f"{q!r} is a false positive"


class TestToolWiring:
    """find_file must be reachable from BOTH tool entry points (the
    two-entry-points lesson) and classified read-only."""

    def test_classified_s1(self):
        from src.core.tool_risk import S1, risk_level
        assert risk_level("find_file") == S1

    def test_declared_in_tool_specs(self):
        from src.core.tool_specs import TOOL_SPECS
        names = [s["function"]["name"] for s in TOOL_SPECS]
        assert "find_file" in names

    def test_both_entry_points_dispatch_it(self):
        import inspect
        from src.core.staging_mixin import StagingMixin
        for fn in (StagingMixin._execute_tool,
                   StagingMixin._execute_tool_unstaged_call):
            assert "find_file" in inspect.getsource(fn)

    def test_runner_requires_a_query(self):
        from src.core.staging_mixin import StagingMixin
        obj = StagingMixin.__new__(StagingMixin)
        assert "[SYSTEM ERROR]" in obj._run_find_file({})

    def test_runner_returns_paths_not_contents(self, tmp_path):
        from src.core.staging_mixin import StagingMixin
        obj = StagingMixin.__new__(StagingMixin)
        with patch("subprocess.run",
                   return_value=_proc("/Users/x/Documents/cv.pdf\n")):
            out = obj._run_find_file({"query": "cv"})
        assert "cv.pdf" in out
