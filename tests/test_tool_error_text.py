"""A tool's error is a sentence, never the exception (s88, Codex lote 8).

The tool follow-up is told to report a tool error verbatim. Measured on the
real 8B: given "[SYSTEM ERROR] Failed to parse document: PdfReadError('EOF
marker not found')…", it answered "O erro foi: EOF marker not found." 6 times
in 6 — read aloud by the TTS. So the result the model gets holds only what
Pedro may hear; the exception goes to the log. Each source that feeds the
follow-up is pinned here with an exception carrying a marker.
"""
import subprocess

import pytest

MARK = "INTERNAL_MARKER_7f3a"


class TestTheDocumentTools:

    def test_a_document_that_fails_to_parse(self, tmp_path, monkeypatch):
        from src.tools.local_reader import LocalFileReader
        reader = LocalFileReader(base_dir=str(tmp_path))
        (tmp_path / "relatorio.pdf").write_bytes(b"%PDF-1.4 broken")

        def boom(_path):
            raise ValueError(f"{MARK} EOF marker not found")
        monkeypatch.setattr(reader, "_read_pdf", boom)
        out = reader.read_document("relatorio.pdf")
        assert out.startswith("[SYSTEM ERROR]") and MARK not in out

    def test_a_document_that_fails_to_write(self, tmp_path, monkeypatch):
        from src.tools import local_writer
        writer = local_writer.LocalFileWriter(base_dir=str(tmp_path))

        def boom(*a, **k):
            raise OSError(f"{MARK} disk full")
        monkeypatch.setattr(local_writer, "open", boom, raising=False)
        out = writer.write_document("notas.txt", "texto")
        assert out.startswith("[SYSTEM ERROR]") and MARK not in out


class TestTheSearchTools:

    def test_a_failed_web_search(self, monkeypatch):
        from src.tools import web_search

        class _Boom:
            def __enter__(self):
                raise RuntimeError(f"{MARK} ratelimit 202")

            def __exit__(self, *a):
                return False
        monkeypatch.setattr(web_search, "DDGS", _Boom)
        out = web_search.WebSearcher()._search_ddg("clima", 3)
        assert out.startswith("[SYSTEM ERROR") and MARK not in out

    def test_a_failed_mdfind(self, monkeypatch, tmp_path):
        from src.tools.file_finder import FileFinder, format_results
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
            a[0], 1, stdout="", stderr=f"{MARK} spotlight index corrupt"))
        hits, error = FileFinder(home=str(tmp_path)).search("contrato")
        assert hits == [] and MARK not in format_results(hits, "contrato", error)


@pytest.fixture
def orch(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from tests.test_dispatch_parity import _build_real_orch
    o = _build_real_orch(tmp_path)
    o._PENDING_ACTION_FILE = str(tmp_path / "pending.json")
    o.background = False
    o._reply_lang = "pt"
    return o


class TestTheGoogleTools:

    @pytest.mark.parametrize("exc", ["unavailable", "other"])
    def test_a_failed_calendar_read(self, orch, monkeypatch, exc):
        from src.integrations import calendar_client

        def boom(**_k):
            if exc == "unavailable":
                raise calendar_client.CalendarUnavailable(f"{MARK} insufficient scopes")
            raise RuntimeError(f"{MARK} HttpError 500")
        monkeypatch.setattr(calendar_client, "list_events", boom)
        out = orch._execute_tool("list_events", {})
        assert out and MARK not in out

    @pytest.mark.parametrize("exc", ["unavailable", "other"])
    def test_a_failed_inbox_read(self, orch, monkeypatch, exc):
        from src.integrations import gmail_client

        def boom(**_k):
            if exc == "unavailable":
                raise gmail_client.GmailUnavailable(f"{MARK} token revoked")
            raise RuntimeError(f"{MARK} HttpError 500")
        monkeypatch.setattr(gmail_client, "list_recent", boom)
        reply, _mode = orch._handle_list_emails(count=3)
        assert reply and MARK not in reply
