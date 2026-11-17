"""Tests for inovação #3 Bloco A — "responde esse email".

Two layers, both AppleScript-free (the osascript is mocked — it is not
unit-testable, same convention as test_macos_controller):

  - `src/tools/screen_context.py` — the deterministic extraction: sender
    parsing, the Mail-selection reader, frontmost app.
  - `ScreenContextMixin._handle_reply_email` — routing the extracted data
    into a drafted reply STAGED through the send_email gate (no send without
    "sim"), and the soberania contract that the email content is drafted via
    a raw `ollama.generate` (never `engine.chat`, so it never touches Palace).
"""
import os
from unittest.mock import MagicMock, patch

import pytest

from src.core import screen_context_mixin
from src.core.i18n import I18nMixin
from src.core.screen_context_mixin import ScreenContextMixin
from src.tools import screen_context


# =====================================================================
# screen_context.py — extraction
# =====================================================================

class TestParseSender:
    @pytest.mark.parametrize("raw, name, email", [
        ("Pedro Silva <pedro@x.com>", "Pedro Silva", "pedro@x.com"),
        ('"Pedro" <pedro@x.com>', "Pedro", "pedro@x.com"),
        ("pedro@x.com", "pedro@x.com", "pedro@x.com"),
        ("<solo@x.com>", "solo@x.com", "solo@x.com"),
        ("Just A Name", "Just A Name", ""),
        ("  spaced <s@x.com>  ", "spaced", "s@x.com"),
    ])
    def test_parse(self, raw, name, email):
        assert screen_context._parse_sender(raw) == (name, email)


class TestMailSelection:
    def _patch(self, ret):
        return patch("src.tools.screen_context._osascript", return_value=ret)

    def test_parsed_selection(self):
        sep = screen_context._SEP
        out = f"Ana <ana@x.com>{sep}Reunião amanhã{sep}Podemos às 10h?"
        with self._patch((True, out)):
            data, err = screen_context.mail_selection()
        assert err is None
        assert data == {
            "sender_name": "Ana",
            "sender_email": "ana@x.com",
            "subject": "Reunião amanhã",
            "content": "Podemos às 10h?",
        }

    def test_no_selection_returns_none_none(self):
        with self._patch((True, "")):
            data, err = screen_context.mail_selection()
        assert data is None and err is None

    def test_permission_error_surfaces(self):
        with self._patch((False, "permission denied. Open System Settings…")):
            data, err = screen_context.mail_selection()
        assert data is None
        assert "permission denied" in err

    def test_malformed_output_is_no_selection(self):
        # osascript returned something without the field separators — treat
        # as no usable selection rather than guessing at a wrong recipient.
        with self._patch((True, "just one blob no separators")):
            data, err = screen_context.mail_selection()
        assert data is None and err is None

    def test_content_with_separators_keeps_only_three_fields(self):
        sep = screen_context._SEP
        # A body that itself contains the separator must not break parsing —
        # split(_, 2) keeps everything after the 2nd sep as the content.
        out = f"a@x.com{sep}Subj{sep}line1{sep}line2"
        with self._patch((True, out)):
            data, _ = screen_context.mail_selection()
        assert data["content"] == f"line1{sep}line2"


class TestBrowserPage:
    def _patch(self, ret):
        return patch("src.tools.screen_context._osascript", return_value=ret)

    def test_safari_parsed(self):
        sep = screen_context._SEP
        out = f"https://x.com/a{sep}Título da Página{sep}Corpo da página aqui."
        with self._patch((True, out)):
            data, err = screen_context.browser_page("Safari")
        assert err is None
        assert data == {
            "url": "https://x.com/a",
            "title": "Título da Página",
            "text": "Corpo da página aqui.",
            "source": "Safari",
        }

    def test_chromium_parsed(self):
        sep = screen_context._SEP
        out = f"https://y.com{sep}Y{sep}texto"
        with self._patch((True, out)):
            data, err = screen_context.browser_page("Google Chrome")
        assert err is None
        assert data["source"] == "Google Chrome"
        assert data["text"] == "texto"

    def test_non_browser_returns_none_none(self):
        # No osascript should run for a non-browser — but even if it did,
        # the function returns (None, None) before calling it.
        data, err = screen_context.browser_page("TextEdit")
        assert data is None and err is None

    def test_js_permission_error_surfaces(self):
        with self._patch((False, "permission denied…")):
            data, err = screen_context.browser_page("Safari")
        assert data is None
        assert "permission denied" in err

    def test_is_browser(self):
        assert screen_context.is_browser("Safari")
        assert screen_context.is_browser("Brave Browser")
        assert not screen_context.is_browser("TextEdit")
        assert not screen_context.is_browser(None)


class TestSelectedText:
    def _patch(self, ret):
        return patch("src.tools.screen_context._osascript", return_value=ret)

    def test_has_selection(self):
        with self._patch((True, "  selecionado  ")):
            text, err = screen_context.selected_text()
        assert text == "selecionado" and err is None

    def test_no_selection(self):
        with self._patch((True, "")):
            text, err = screen_context.selected_text()
        assert text is None and err is None

    def test_error(self):
        with self._patch((False, "permission denied")):
            text, err = screen_context.selected_text()
        assert text is None and "permission denied" in err


class TestActiveContent:
    def test_browser_wins(self, monkeypatch):
        monkeypatch.setattr(screen_context, "frontmost_app",
                            lambda lang="pt": "Safari")
        monkeypatch.setattr(
            screen_context, "browser_page",
            lambda app, lang="pt": ({"text": "página", "title": "T",
                                     "source": "Safari"}, None))
        data, err = screen_context.active_content()
        assert err is None and data["text"] == "página"

    def test_falls_back_to_selection(self, monkeypatch):
        # A non-browser, non-doc app (Slack) → straight to the selection path;
        # front_document returns (None, None) without any osascript call.
        monkeypatch.setattr(screen_context, "frontmost_app",
                            lambda lang="pt": "Slack")
        monkeypatch.setattr(screen_context, "selected_text",
                            lambda lang="pt": ("trecho", None))
        data, err = screen_context.active_content()
        assert err is None
        assert data["text"] == "trecho" and data["source"] == "Slack"

    def test_browser_empty_falls_back_to_selection(self, monkeypatch):
        monkeypatch.setattr(screen_context, "frontmost_app",
                            lambda lang="pt": "Safari")
        monkeypatch.setattr(screen_context, "browser_page",
                            lambda app, lang="pt": (None, None))
        monkeypatch.setattr(screen_context, "selected_text",
                            lambda lang="pt": ("sel", None))
        data, err = screen_context.active_content()
        assert data["text"] == "sel"

    def test_nothing(self, monkeypatch):
        monkeypatch.setattr(screen_context, "frontmost_app",
                            lambda lang="pt": "Slack")
        monkeypatch.setattr(screen_context, "selected_text",
                            lambda lang="pt": (None, None))
        data, err = screen_context.active_content()
        assert data is None and err is None

    def test_open_doc_beats_selection(self, monkeypatch):
        # Preview/TextEdit open → the doc text wins over any selection
        # (selection is the universal LAST resort).
        monkeypatch.setattr(screen_context, "frontmost_app",
                            lambda lang="pt": "Preview")
        monkeypatch.setattr(
            screen_context, "front_document",
            lambda app, lang="pt": ({"text": "conteúdo do PDF", "title": "f.pdf",
                                     "source": "Preview", "url": ""}, None))
        # selected_text must NOT be consulted when the doc has text.
        monkeypatch.setattr(screen_context, "selected_text",
                            lambda lang="pt": (_ for _ in ()).throw(
                                AssertionError("selection should not run")))
        data, err = screen_context.active_content()
        assert err is None
        assert data["text"] == "conteúdo do PDF" and data["source"] == "Preview"

    def test_doc_error_falls_through_to_selection(self, monkeypatch):
        monkeypatch.setattr(screen_context, "frontmost_app",
                            lambda lang="pt": "Preview")
        monkeypatch.setattr(screen_context, "front_document",
                            lambda app, lang="pt": (None, "no permission"))
        monkeypatch.setattr(screen_context, "selected_text",
                            lambda lang="pt": ("selecionado", None))
        data, err = screen_context.active_content()
        assert err is None and data["text"] == "selecionado"

    def test_doc_error_surfaces_when_no_selection(self, monkeypatch):
        monkeypatch.setattr(screen_context, "frontmost_app",
                            lambda lang="pt": "Preview")
        monkeypatch.setattr(screen_context, "front_document",
                            lambda app, lang="pt": (None, "no permission"))
        monkeypatch.setattr(screen_context, "selected_text",
                            lambda lang="pt": (None, None))
        data, err = screen_context.active_content()
        assert data is None and err == "no permission"

    def test_browser_error_surfaces_when_no_selection(self, monkeypatch):
        # Browser JS error + nothing selected → surface the actionable
        # browser error (enable JS), not a generic "no content".
        monkeypatch.setattr(screen_context, "frontmost_app",
                            lambda lang="pt": "Safari")
        monkeypatch.setattr(screen_context, "browser_page",
                            lambda app, lang="pt": (None, "enable JS"))
        monkeypatch.setattr(screen_context, "selected_text",
                            lambda lang="pt": (None, None))
        data, err = screen_context.active_content()
        assert data is None and err == "enable JS"

    def test_browser_error_falls_through_to_selection(self, monkeypatch):
        # The robustness win: browser JS is off, but the user has text
        # selected → use the selection instead of erroring out.
        monkeypatch.setattr(screen_context, "frontmost_app",
                            lambda lang="pt": "Safari")
        monkeypatch.setattr(screen_context, "browser_page",
                            lambda app, lang="pt": (None, "enable JS"))
        monkeypatch.setattr(screen_context, "selected_text",
                            lambda lang="pt": ("highlighted", None))
        data, err = screen_context.active_content()
        assert err is None
        assert data["text"] == "highlighted" and data["source"] == "Safari"


class TestFrontDocument:
    def _patch(self, ret):
        return patch("src.tools.screen_context._osascript", return_value=ret)

    def test_is_doc_app(self):
        assert screen_context.is_doc_app("Preview")
        assert screen_context.is_doc_app("TextEdit")
        assert not screen_context.is_doc_app("Safari")
        assert not screen_context.is_doc_app("Slack")
        assert not screen_context.is_doc_app(None)

    def test_non_doc_app_returns_none_none_without_osascript(self):
        # Must short-circuit BEFORE any osascript for an unknown app.
        with patch("src.tools.screen_context._osascript",
                   side_effect=AssertionError("osascript should not run")):
            data, err = screen_context.front_document("Slack")
        assert data is None and err is None

    def test_textedit_returns_text_directly(self):
        sep = screen_context._SEP
        with self._patch((True, f"Carta.txt{sep}Olá Pedro, tudo bem?")):
            data, err = screen_context.front_document("TextEdit")
        assert err is None
        assert data["text"] == "Olá Pedro, tudo bem?"
        assert data["title"] == "Carta.txt"
        assert data["source"] == "TextEdit"

    def test_textedit_no_documents(self):
        with self._patch((True, "")):
            data, err = screen_context.front_document("TextEdit")
        assert data is None and err is None

    def test_textedit_permission_error(self):
        with self._patch((False, "not authorised")):
            data, err = screen_context.front_document("TextEdit")
        assert data is None and "not authorised" in err

    def test_preview_reads_file_off_disk(self, monkeypatch):
        with self._patch((True, "/Users/pedro/Downloads/artigo.pdf")), \
             patch("src.tools.screen_context.os.path.exists",
                   return_value=True), \
             patch("src.tools.local_reader.LocalFileReader.read_path",
                   return_value="Texto extraído do PDF."):
            data, err = screen_context.front_document("Preview")
        assert err is None
        assert data["text"] == "Texto extraído do PDF."
        assert data["title"] == "artigo.pdf"
        assert data["source"] == "Preview"

    def test_preview_missing_file_returns_none(self):
        with self._patch((True, "/Users/pedro/gone.pdf")), \
             patch("src.tools.screen_context.os.path.exists",
                   return_value=False):
            data, err = screen_context.front_document("Preview")
        assert data is None and err is None

    def test_preview_parse_error_falls_through(self):
        # A parser failure / unsupported type must yield (None, None) so the
        # caller drops to the selection path, not surface a parser error.
        with self._patch((True, "/Users/pedro/x.pdf")), \
             patch("src.tools.screen_context.os.path.exists",
                   return_value=True), \
             patch("src.tools.local_reader.LocalFileReader.read_path",
                   return_value="[SYSTEM ERROR] Failed to parse document: boom"):
            data, err = screen_context.front_document("Preview")
        assert data is None and err is None

    def test_normalise_doc_path_posix_passthrough(self):
        assert screen_context._normalise_doc_path(
            "/Users/pedro/a.pdf") == "/Users/pedro/a.pdf"

    def test_normalise_doc_path_hfs_to_posix(self):
        assert screen_context._normalise_doc_path(
            "Macintosh HD:Users:pedro:a.pdf") == "/Users/pedro/a.pdf"


class TestCaptureScreen:
    def test_returns_b64_and_deletes_temp(self, monkeypatch):
        # _run "captures" by writing bytes to the temp path it's given.
        written = {}

        def fake_run(args, **kw):
            path = args[-1]
            with open(path, "wb") as f:
                f.write(b"\x89PNG fake")
            written["path"] = path
            return True, ""

        monkeypatch.setattr(screen_context, "_run", fake_run)
        b64, err = screen_context.capture_screen_b64()
        assert err is None and b64
        import base64 as _b64
        assert _b64.b64decode(b64) == b"\x89PNG fake"
        # The transient capture must NOT survive (soberania).
        assert not __import__("os").path.exists(written["path"])

    def test_capture_failure_surfaces(self, monkeypatch):
        monkeypatch.setattr(screen_context, "_run",
                            lambda args, **kw: (False, "screen recording denied"))
        b64, err = screen_context.capture_screen_b64()
        assert b64 is None and "screen recording denied" in err


class TestFrontmostApp:
    def test_happy(self):
        with patch("src.tools.screen_context._osascript",
                   return_value=(True, "Mail")):
            assert screen_context.frontmost_app() == "Mail"

    def test_error_returns_none(self):
        with patch("src.tools.screen_context._osascript",
                   return_value=(False, "boom")):
            assert screen_context.frontmost_app() is None

    def test_empty_returns_none(self):
        with patch("src.tools.screen_context._osascript",
                   return_value=(True, "  ")):
            assert screen_context.frontmost_app() is None


# =====================================================================
# _handle_reply_email — routing + staging + soberania
# =====================================================================

class _Harness(ScreenContextMixin, I18nMixin):
    """Minimal orchestrator stand-in carrying just the StagingMixin state
    the handler reaches into."""

    def __init__(self, lang="pt"):
        self._reply_lang = lang
        self._pending_action = None
        self.persisted = 0
        self.engine_analytical = MagicMock()
        self.engine_analytical.model_name = "cortex-lite"
        self.vision = MagicMock()

    def _persist_pending_action(self):
        self.persisted += 1

    def _human_confirmation_prompt(self, pending):
        return f"Confirma enviar pra {pending['args']['to']}? (sim/não)"


@pytest.fixture
def harness(monkeypatch, tmp_path):
    # Isolate from a real /tmp/cortex_screen_off the dev may have toggled —
    # point the flag at a path that doesn't exist unless a test creates it.
    monkeypatch.setenv("CORTEX_SCREEN_OFF_FLAG", str(tmp_path / "no_flag"))
    return _Harness()


def _mail(**over):
    data = {
        "sender_name": "Ana",
        "sender_email": "ana@x.com",
        "subject": "Reunião",
        "content": "Podemos às 10h?",
    }
    data.update(over)
    return data


class TestReplyEmailHandler:
    def _run(self, harness, *, selection, draft="Claro, às 10h funciona."):
        # The handler resolves `from src.tools import screen_context` then
        # `screen_context.mail_selection(...)` — patch that attribute.
        with patch.object(screen_context, "mail_selection",
                          return_value=selection), \
             patch.object(screen_context_mixin, "ollama") as ollama_mock:
            ollama_mock.generate.return_value = {"response": draft}
            reply, mode = harness._handle_reply_email()
        return reply, mode, ollama_mock

    def test_happy_path_stages_send_email(self, harness):
        reply, mode, ollama_mock = self._run(
            harness, selection=(_mail(), None))
        assert mode == "AGENT_AWAITING_CONFIRMATION"
        assert harness._pending_action == {
            "type": "send_email",
            "args": {
                "to": "ana@x.com",
                "subject": "Re: Reunião",
                "body": "Claro, às 10h funciona.",
            },
        }
        assert harness.persisted == 1

    def test_soberania_uses_generate_not_engine_chat(self, harness):
        # The email content must be drafted via raw ollama.generate (no
        # memory load/persist) — engine.chat (which loads Palace) is never
        # touched. This is the regra-1 contract for transient screen content.
        self._run(harness, selection=(_mail(), None))
        harness.engine_analytical.chat.assert_not_called()
        harness.engine_analytical.chat_stream.assert_not_called()

    def test_subject_not_double_prefixed(self, harness):
        self._run(harness, selection=(_mail(subject="Re: Reunião"), None))
        assert harness._pending_action["args"]["subject"] == "Re: Reunião"

    def test_empty_subject_gets_placeholder_not_dangling_re(self, harness):
        self._run(harness, selection=(_mail(subject=""), None))
        assert harness._pending_action["args"]["subject"] == "Re: (sem assunto)"

    def test_no_selection(self, harness):
        reply, mode, _ = self._run(harness, selection=(None, None))
        assert mode == "MAIL_REPLY_NO_SELECTION"
        assert harness._pending_action is None

    def test_read_error(self, harness):
        reply, mode, _ = self._run(
            harness, selection=(None, "permission denied"))
        assert mode == "MAIL_REPLY_ERROR"
        assert harness._pending_action is None

    def test_no_sender_address(self, harness):
        reply, mode, _ = self._run(
            harness, selection=(_mail(sender_email=""), None))
        assert mode == "MAIL_REPLY_NO_SENDER"
        assert harness._pending_action is None

    def test_empty_draft_fails_soft(self, harness):
        reply, mode, _ = self._run(
            harness, selection=(_mail(), None), draft="   ")
        assert mode == "MAIL_REPLY_DRAFT_FAILED"
        assert harness._pending_action is None

    def test_kill_switch_env_refuses_before_reading(self, harness, monkeypatch):
        # CORTEX_SCREEN_CONTEXT=0 must refuse WITHOUT ever reading the
        # window — the boot-time switch. mail_selection is rigged to blow up
        # so the test fails loudly if the read happens before the gate.
        monkeypatch.setenv("CORTEX_SCREEN_CONTEXT", "0")
        with patch.object(screen_context, "mail_selection",
                          side_effect=AssertionError("read happened!")):
            reply, mode = harness._handle_reply_email()
        assert mode == "SCREEN_CONTEXT_OFF"
        assert harness._pending_action is None

    def test_flag_file_refuses_before_reading(self, harness, monkeypatch,
                                              tmp_path):
        # The INSTANT button: a flag file present → refuse, no env var, no
        # restart. OFF wins over an explicitly-ON env var.
        flag = tmp_path / "screen_off"
        flag.write_text("")
        monkeypatch.setenv("CORTEX_SCREEN_OFF_FLAG", str(flag))
        monkeypatch.setenv("CORTEX_SCREEN_CONTEXT", "1")
        with patch.object(screen_context, "mail_selection",
                          side_effect=AssertionError("read happened!")):
            reply, mode = harness._handle_reply_email()
        assert mode == "SCREEN_CONTEXT_OFF"
        assert harness._pending_action is None

    def test_flag_file_absent_allows(self, harness, monkeypatch, tmp_path):
        monkeypatch.setenv("CORTEX_SCREEN_OFF_FLAG",
                           str(tmp_path / "absent"))
        reply, mode, _ = self._run(harness, selection=(_mail(), None))
        assert mode == "AGENT_AWAITING_CONFIRMATION"

    @pytest.mark.parametrize("val", ["1", "true", "on", "yes", ""])
    def test_kill_switch_enabled_values(self, harness, monkeypatch, val):
        if val:
            monkeypatch.setenv("CORTEX_SCREEN_CONTEXT", val)
        else:
            monkeypatch.delenv("CORTEX_SCREEN_CONTEXT", raising=False)
        reply, mode, _ = self._run(harness, selection=(_mail(), None))
        assert mode == "AGENT_AWAITING_CONFIRMATION"

    def test_generation_exception_fails_soft(self, harness):
        with patch("src.tools.screen_context.mail_selection",
                   return_value=(_mail(), None)), \
             patch.object(screen_context_mixin, "ollama") as ollama_mock:
            ollama_mock.generate.side_effect = RuntimeError("ollama down")
            reply, mode = harness._handle_reply_email()
        assert mode == "MAIL_REPLY_DRAFT_FAILED"
        assert harness._pending_action is None


class TestSummariseHandler:
    def _run(self, harness, *, content, summary="Resumo dos pontos."):
        with patch.object(screen_context, "active_content",
                          return_value=content), \
             patch.object(screen_context_mixin, "ollama") as ollama_mock:
            ollama_mock.generate.return_value = {"response": summary}
            reply, mode = harness._handle_summarise_screen()
        return reply, mode, ollama_mock

    def test_happy_path_returns_summary(self, harness):
        data = {"source": "Safari", "title": "Artigo", "text": "blah blah"}
        reply, mode, _ = self._run(harness, content=(data, None))
        assert mode == "SCREEN_SUMMARY"
        assert reply == "Resumo dos pontos."

    def test_soberania_no_engine_chat(self, harness):
        data = {"source": "Safari", "title": "T", "text": "x"}
        self._run(harness, content=(data, None))
        harness.engine_analytical.chat.assert_not_called()
        harness.engine_analytical.chat_stream.assert_not_called()

    def test_no_content(self, harness):
        reply, mode, _ = self._run(harness, content=(None, None))
        assert mode == "SCREEN_SUMMARY_NO_CONTENT"

    def test_read_error(self, harness):
        reply, mode, _ = self._run(harness, content=(None, "permission denied"))
        assert mode == "SCREEN_SUMMARY_ERROR"

    def test_empty_summary_fails_soft(self, harness):
        data = {"source": "Safari", "title": "T", "text": "x"}
        reply, mode, _ = self._run(harness, content=(data, None), summary="  ")
        assert mode == "SCREEN_SUMMARY_FAILED"

    def test_kill_switch_refuses_before_reading(self, harness, monkeypatch):
        monkeypatch.setenv("CORTEX_SCREEN_CONTEXT", "0")
        with patch.object(screen_context, "active_content",
                          side_effect=AssertionError("read happened!")):
            reply, mode = harness._handle_summarise_screen()
        assert mode == "SCREEN_CONTEXT_OFF"


class TestWhatIsThisHandler:
    def test_text_path_explains(self, harness):
        data = {"source": "Safari", "title": "Doc", "text": "conteúdo"}
        with patch.object(screen_context, "active_content",
                          return_value=(data, None)), \
             patch.object(screen_context_mixin, "ollama") as ollama_mock:
            ollama_mock.generate.return_value = {"response": "É um documento X."}
            reply, mode = harness._handle_what_is_this()
        assert mode == "WHAT_IS_THIS"
        assert reply == "É um documento X."
        # Text path must NOT touch the vision model.
        harness.vision.analyze_image.assert_not_called()

    def test_text_path_soberania_no_engine_chat(self, harness):
        data = {"source": "Safari", "title": "T", "text": "x"}
        with patch.object(screen_context, "active_content",
                          return_value=(data, None)), \
             patch.object(screen_context_mixin, "ollama") as ollama_mock:
            ollama_mock.generate.return_value = {"response": "ok"}
            harness._handle_what_is_this()
        harness.engine_analytical.chat.assert_not_called()

    def test_vision_fallback_when_no_text(self, harness):
        with patch.object(screen_context, "active_content",
                          return_value=(None, None)), \
             patch.object(screen_context, "capture_screen_b64",
                          return_value=("BASE64", None)):
            harness.vision.analyze_image.return_value = "A login screen."
            reply, mode = harness._handle_what_is_this()
        assert mode == "WHAT_IS_THIS_VISION"
        assert "A login screen." in reply
        harness.vision.analyze_image.assert_called_once()

    def test_vision_capture_error(self, harness):
        with patch.object(screen_context, "active_content",
                          return_value=(None, None)), \
             patch.object(screen_context, "capture_screen_b64",
                          return_value=(None, "screen recording denied")):
            reply, mode = harness._handle_what_is_this()
        assert mode == "WHAT_IS_THIS_ERROR"

    def test_vision_system_error_fails_soft(self, harness):
        with patch.object(screen_context, "active_content",
                          return_value=(None, None)), \
             patch.object(screen_context, "capture_screen_b64",
                          return_value=("B64", None)):
            harness.vision.analyze_image.return_value = "[SYSTEM ERROR] boom"
            reply, mode = harness._handle_what_is_this()
        assert mode == "WHAT_IS_THIS_FAILED"

    def test_kill_switch_refuses(self, harness, monkeypatch):
        monkeypatch.setenv("CORTEX_SCREEN_CONTEXT", "0")
        with patch.object(screen_context, "active_content",
                          side_effect=AssertionError("read happened!")):
            reply, mode = harness._handle_what_is_this()
        assert mode == "SCREEN_CONTEXT_OFF"


class TestScreenEyesToggle:
    """Voice toggle of the screen-reading kill-switch — the same flag file the
    `cortex-eyes.sh` button / hotkey flips. The `harness` fixture already points
    CORTEX_SCREEN_OFF_FLAG at a tmp path that starts absent."""

    def _flag(self):
        return screen_context_mixin._screen_off_flag_path()

    def test_off_raises_flag(self, harness):
        reply, mode = harness._handle_screen_eyes_off()
        assert mode == "SCREEN_EYES_OFF"
        assert os.path.exists(self._flag())

    def test_on_clears_flag(self, harness):
        open(self._flag(), "w").close()  # start OFF
        reply, mode = harness._handle_screen_eyes_on()
        assert mode == "SCREEN_EYES_ON"
        assert not os.path.exists(self._flag())

    @pytest.mark.parametrize("phrase", [
        "volta a ver minha tela",
        "Volte a ver minha tela.",   # s76 cont.31 live: imperative "volte"
        "voltar a ver minha tela",
        "pode voltar a ver minha tela",
        "volte a olhar a tela",
    ])
    def test_on_regex_accepts_volta_volte_variants(self, phrase):
        from src.core.dispatch.macos import MAC_EYES_ON_RE
        assert MAC_EYES_ON_RE.match(phrase)

    def test_off_then_on_round_trip_flips_enabled(self, harness):
        # The toggle must actually drive _screen_context_enabled() — the gate
        # every read handler checks. OFF → disabled; ON → enabled again.
        harness._handle_screen_eyes_off()
        assert screen_context_mixin._screen_context_enabled() is False
        harness._handle_screen_eyes_on()
        assert screen_context_mixin._screen_context_enabled() is True

    def test_off_is_idempotent(self, harness):
        harness._handle_screen_eyes_off()
        reply, mode = harness._handle_screen_eyes_off()  # already off
        assert mode == "SCREEN_EYES_OFF"
        assert os.path.exists(self._flag())

    def test_on_is_idempotent_when_already_on(self, harness):
        # Flag absent (already on) — clearing it again must not error.
        reply, mode = harness._handle_screen_eyes_on()
        assert mode == "SCREEN_EYES_ON"
        assert not os.path.exists(self._flag())

    def test_off_does_not_gate_on_kill_switch(self, harness, monkeypatch):
        # The control handlers must NOT refuse when reading is off — turning
        # the eyes OFF can never be blocked by the eyes being off.
        monkeypatch.setenv("CORTEX_SCREEN_CONTEXT", "0")
        reply, mode = harness._handle_screen_eyes_off()
        assert mode == "SCREEN_EYES_OFF"

    def test_on_works_while_off(self, harness, monkeypatch):
        # ON is the only hands-free way back — it must work with the flag
        # present AND the env off, never short-circuiting to a refusal.
        open(self._flag(), "w").close()
        monkeypatch.setenv("CORTEX_SCREEN_CONTEXT", "0")
        reply, mode = harness._handle_screen_eyes_on()
        assert mode == "SCREEN_EYES_ON"
        assert not os.path.exists(self._flag())

    def test_off_fs_error_fails_soft(self, harness, monkeypatch, tmp_path):
        # Flag under a path that can't be created → honest error, not a crash.
        monkeypatch.setenv("CORTEX_SCREEN_OFF_FLAG",
                           str(tmp_path / "nope" / "flag"))
        reply, mode = harness._handle_screen_eyes_off()
        assert mode == "SCREEN_EYES_ERROR"
