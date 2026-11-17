"""read_file_at — reading a file found anywhere in $HOME (s83).

Two things are under test: the gate that decides WHICH paths may ever be
read (shared with the file search, so what cannot be found cannot be read
by typing the path instead), and the staging contract — ask once per file,
then answer the question rather than dumping the text.
"""
import os

import pytest

from src.core.orchestrator import CortexOrchestrator
from src.core.tool_risk import build_preview, is_destructive
from src.tools.file_finder import is_readable_path


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


class TestPathGate:

    def test_accepts_a_document_in_the_home_folder(self, home):
        f = home / "notes.md"
        f.write_text("hello")
        ok, resolved = is_readable_path(str(f))
        assert ok and resolved == os.path.realpath(str(f))

    @pytest.mark.parametrize("name", ["notes.md", "doc.txt", "a.pdf", "b.docx"])
    def test_accepts_every_advertised_format(self, home, name):
        (home / name).write_text("x")
        assert is_readable_path(str(home / name))[0]

    def test_refuses_outside_the_home_folder(self, home):
        assert not is_readable_path("/etc/passwd")[0]

    @pytest.mark.parametrize("rel", [".ssh/id_rsa", ".aws/credentials",
                                     ".cortex/palace.db", "Library/x.txt"])
    def test_refuses_the_protected_locations(self, home, rel):
        p = home / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("secret")
        ok, why = is_readable_path(str(p))
        assert not ok and "protected" in why

    def test_refuses_an_env_file(self, home):
        (home / ".env").write_text("CORTEX_API_KEY=xxx")
        assert not is_readable_path(str(home / ".env"))[0]

    def test_accepts_a_document_in_icloud_drive(self, home):
        d = home / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "college"
        d.mkdir(parents=True)
        (d / "homework.pdf").write_text("x")
        assert is_readable_path(str(d / "homework.pdf"))[0]

    def test_still_refuses_the_rest_of_library(self, home):
        d = home / "Library" / "Caches"
        d.mkdir(parents=True)
        (d / "cache.txt").write_text("x")
        assert not is_readable_path(str(d / "cache.txt"))[0]

    def test_says_when_an_icloud_file_is_not_downloaded(self, home):
        d = home / "Library" / "Mobile Documents" / "com~apple~CloudDocs"
        d.mkdir(parents=True)
        (d / ".essay.pdf.icloud").write_text("stub")   # optimised storage
        ok, why = is_readable_path(str(d / "essay.pdf"))
        assert not ok and "not downloaded" in why

    def test_refuses_a_relative_path(self, home):
        assert not is_readable_path("notes.md")[0]

    def test_refuses_an_unsupported_format(self, home):
        (home / "script.py").write_text("print('x')")
        ok, why = is_readable_path(str(home / "script.py"))
        assert not ok and "unsupported" in why

    def test_refuses_a_missing_file(self, home):
        assert not is_readable_path(str(home / "ghost.pdf"))[0]

    def test_follows_no_symlink_out_of_the_home_folder(self, home, tmp_path):
        outside = tmp_path.parent / "outside_secret.txt"
        outside.write_text("not yours")
        link = home / "innocent.txt"
        try:
            link.symlink_to(outside)
        except OSError:                       # pragma: no cover - platform
            pytest.skip("symlinks unavailable")
        assert not is_readable_path(str(link))[0]


class TestGate:
    """The confirmation exists because file content is the classic indirect
    injection vector — the human gate is the defence that actually holds."""

    def test_read_file_at_needs_the_human_gate(self):
        assert is_destructive("read_file_at")

    def test_preview_names_the_file_and_says_it_is_read_only(self):
        human = build_preview("read_file_at",
                              {"path": "/Users/p/Downloads/contrato.pdf"},
                              lang="pt")["human"]
        assert "contrato.pdf" in human
        assert "/Users/p/Downloads" in human
        assert "Só leitura" in human

    def test_preview_en(self):
        human = build_preview("read_file_at",
                              {"path": "/Users/p/Downloads/contrato.pdf"},
                              lang="en")["human"]
        assert human.startswith("May I open and read 'contrato.pdf'?")


@pytest.fixture
def orch(tmp_path, monkeypatch):
    o = CortexOrchestrator.__new__(CortexOrchestrator)
    o._pending_action = None
    o._PENDING_ACTION_FILE = str(tmp_path / "pending_action.json")
    o.background = False
    o.background_origin = "background"
    o._reply_lang = "pt"
    return o


class TestStagingContract:

    def _doc(self, home):
        f = home / "leia.md"
        f.write_text("o conteudo")
        return str(f)

    def test_first_read_stages_and_does_not_open_the_file(self, orch, home,
                                                          monkeypatch):
        opened = []
        monkeypatch.setattr(orch, "reader", type("R", (), {
            "read_path": lambda _self, p: opened.append(p) or "x"})(),
                            raising=False)
        path = self._doc(home)
        orch._remember_seen_path(path)          # a search put it on the table
        reply = orch._execute_tool("read_file_at", {"path": path})
        assert "CONFIRMATION REQUIRED" in reply
        assert orch._pending_action["type"] == "read_file_at"
        assert opened == []                   # nothing was read yet

    def test_an_approved_file_is_not_asked_about_again(self, orch, home,
                                                       monkeypatch):
        path = self._doc(home)
        monkeypatch.setattr(orch, "reader", type("R", (), {
            "read_path": lambda _self, p: "o conteudo"})(), raising=False)
        monkeypatch.setattr(orch, "engine_analytical", object(), raising=False)
        monkeypatch.setattr(orch, "_synthesise",
                            lambda *a, **k: ("resposta", "LOCAL_RAG"),
                            raising=False)
        orch._remember_seen_path(path)
        orch._remember_read_approval(os.path.realpath(path))
        reply = orch._execute_tool("read_file_at", {"path": path})
        assert reply == "resposta"
        assert orch._pending_action is None    # never staged

    def test_a_refused_path_never_reaches_the_reader(self, orch, home,
                                                     monkeypatch):
        opened = []
        monkeypatch.setattr(orch, "reader", type("R", (), {
            "read_path": lambda _self, p: opened.append(p) or "x"})(),
                            raising=False)
        orch._remember_seen_path("/etc/passwd")   # even if it IS on the table
        reply, mode = orch._run_read_file_at({"path": "/etc/passwd"})
        assert mode == "AGENT_ACTION_ERROR"
        assert "não posso ler" in reply
        assert opened == []

    def test_a_scanned_pdf_is_reported_not_answered(self, orch, home,
                                                    monkeypatch):
        from src.tools.local_reader import PDF_NO_TEXT
        (home / "scan.pdf").write_text("%PDF-fake")
        monkeypatch.setattr(orch, "reader", type("R", (), {
            "read_path": lambda _self, p: PDF_NO_TEXT})(), raising=False)
        called = []
        monkeypatch.setattr(orch, "_synthesise",
                            lambda *a, **k: called.append(1) or ("x", "y"),
                            raising=False)
        orch._remember_seen_path(str(home / "scan.pdf"))
        reply, mode = orch._run_read_file_at({"path": str(home / "scan.pdf")})
        assert mode == "FILE_ERROR"
        assert "scan" in reply
        assert called == []                   # the model is never asked to
                                              # answer from an empty read


class TestScannedPdf:
    """A scan has no text layer, so pypdf returns "" — and an empty read is
    not silence. Three of Pedro's four PDFs came back with zero characters."""

    def test_a_pdf_with_no_text_layer_says_so(self, tmp_path):
        from pypdf import PdfWriter
        from src.tools.local_reader import LocalFileReader, PDF_NO_TEXT
        w = PdfWriter()
        w.add_blank_page(width=200, height=200)      # an image-only page has
        out = tmp_path / "scan.pdf"                  # the same empty text layer
        with open(out, "wb") as fh:
            w.write(fh)
        assert LocalFileReader(base_dir=str(tmp_path)).read_path(str(out)) == PDF_NO_TEXT

    def test_a_pdf_with_text_is_unaffected(self, tmp_path, monkeypatch):
        from src.tools import local_reader
        class _Page:
            def extract_text(self):
                return "conteudo real"
        monkeypatch.setattr(local_reader, "PdfReader",
                            lambda _p: type("R", (), {"pages": [_Page()]})())
        out = tmp_path / "text.pdf"
        out.write_bytes(b"%PDF-")
        assert "conteudo real" in local_reader.LocalFileReader(
            base_dir=str(tmp_path)).read_path(str(out))


class TestOcrFallback:
    """A scan has no text layer; macOS reads it on-device (s83). pyobjc was
    already a dependency, so this cost no new download and the page never
    leaves the Mac."""

    def _pdf(self, tmp_path):
        from pypdf import PdfWriter
        w = PdfWriter()
        w.add_blank_page(width=200, height=200)
        out = tmp_path / "scan.pdf"
        with open(out, "wb") as fh:
            w.write(fh)
        return str(out)

    def test_ocr_text_is_used_and_labelled_as_ocr(self, tmp_path, monkeypatch):
        from src.tools import local_reader
        monkeypatch.setattr("src.tools.ocr.ocr_pdf",
                            lambda *_a, **_k: "texto lido da imagem")
        out = local_reader.LocalFileReader(
            base_dir=str(tmp_path)).read_path(self._pdf(tmp_path))
        assert "texto lido da imagem" in out
        assert "OCR" in out            # the model must know it may misread

    def test_no_text_anywhere_still_says_so(self, tmp_path, monkeypatch):
        from src.tools import local_reader
        monkeypatch.setattr("src.tools.ocr.ocr_pdf", lambda *_a, **_k: "")
        out = local_reader.LocalFileReader(
            base_dir=str(tmp_path)).read_path(self._pdf(tmp_path))
        assert out == local_reader.PDF_NO_TEXT

    def test_ocr_can_be_switched_off(self, tmp_path, monkeypatch):
        from src.tools import local_reader
        monkeypatch.setenv("CORTEX_OCR", "0")
        called = []
        monkeypatch.setattr("src.tools.ocr.ocr_pdf",
                            lambda *_a, **_k: called.append(1) or "x")
        out = local_reader.LocalFileReader(
            base_dir=str(tmp_path)).read_path(self._pdf(tmp_path))
        assert out == local_reader.PDF_NO_TEXT and called == []

    def test_ocr_is_fail_soft_on_a_bad_file(self, tmp_path):
        from src.tools.ocr import ocr_pdf
        assert ocr_pdf(str(tmp_path / "does_not_exist.pdf")) == ""


def _pages(monkeypatch, texts):
    """A PDF whose pages carry exactly these text layers."""
    from src.tools import local_reader

    class _Page:
        def __init__(self, text):
            self._text = text

        def extract_text(self):
            return self._text

    monkeypatch.setattr(local_reader, "PdfReader", lambda _p: type(
        "R", (), {"pages": [_Page(t) for t in texts]})())


class TestBarcodeOnlyScans:
    """Pearson exam scans carry a text layer with only the barcode in it —
    "*P64979A0112*" on each of twelve pages. Not empty, so the empty-only OCR
    trigger never fired and the model was handed barcodes (s85)."""

    def _read(self, tmp_path):
        from src.tools.local_reader import LocalFileReader
        out = tmp_path / "exam.pdf"
        out.write_bytes(b"%PDF-")
        return LocalFileReader(base_dir=str(tmp_path)).read_path(str(out))

    def test_barcode_pages_are_read_by_ocr(self, tmp_path, monkeypatch):
        _pages(monkeypatch, [f"*P64979A0{n}12*" for n in range(1, 13)])
        monkeypatch.setattr("src.tools.ocr.ocr_pdf", lambda *_a, **_k:
                            "Using Text A, identify two ways that businesses "
                            "benefit from flexible working. " * 30)
        out = self._read(tmp_path)
        assert "flexible working" in out and "OCR" in out

    def test_a_cv_with_a_blank_second_page_keeps_its_text(self, tmp_path,
                                                          monkeypatch):
        """Half thin is not mostly thin — measured on Pedro's CVs."""
        _pages(monkeypatch, ["Pedro Silva — experience " * 40, ""])
        called = []
        monkeypatch.setattr("src.tools.ocr.ocr_pdf",
                            lambda *_a, **_k: called.append(1) or "x")
        out = self._read(tmp_path)
        assert "experience" in out and called == []

    def test_ocr_that_reads_less_does_not_replace_the_text(self, tmp_path,
                                                           monkeypatch):
        _pages(monkeypatch, ["Slide one", "Slide two", "Slide three"])
        monkeypatch.setattr("src.tools.ocr.ocr_pdf", lambda *_a, **_k: "Slide")
        out = self._read(tmp_path)
        assert "Slide three" in out and "OCR" not in out


class TestOcrMerge:
    """Vision drops whole lines of a clean page, and WHICH lines depends on
    the scale: 8 of 15 question stems at 2x, 8 at 1x, 10 at 1.25x — the union
    recovered all 15 (s85)."""

    def test_a_line_one_pass_missed_is_taken_from_another(self):
        from src.tools.ocr import merge_passes
        at_2x = [("SECTION B", (0.4, 0.90, 0.1, 0.02)),
                 ("6 Text B includes both facts", (0.1, 0.60, 0.3, 0.02))]
        at_1x = [("SECTION B", (0.4, 0.901, 0.1, 0.02)),
                 ("5 What does each of these quotations", (0.1, 0.83, 0.5, 0.02))]
        assert merge_passes([at_2x, at_1x]) == [
            "SECTION B",
            "5 What does each of these quotations",
            "6 Text B includes both facts",
        ]

    def test_the_first_pass_wins_a_line_both_read(self):
        from src.tools.ocr import merge_passes
        lines = merge_passes([[("Bresley College", (0.1, 0.5, 0.3, 0.02))],
                              [("Bres1ey Co11ege", (0.1, 0.5, 0.3, 0.02))]])
        assert lines == ["Bresley College"]

    def test_a_line_split_in_two_by_one_pass_is_not_doubled(self):
        from src.tools.ocr import merge_passes
        whole = [("Language feature (1)", (0.1, 0.5, 0.8, 0.02))]
        split = [("Language feature", (0.1, 0.5, 0.3, 0.02)),
                 ("(1)", (0.85, 0.5, 0.05, 0.02))]
        assert merge_passes([whole, split]) == ["Language feature (1)"]

    def test_a_row_reads_left_to_right(self):
        from src.tools.ocr import merge_passes
        row = [("(1)", (0.85, 0.5, 0.05, 0.02)),
               ("Example", (0.1, 0.502, 0.2, 0.02))]
        assert merge_passes([row]) == ["Example (1)"]

    def test_blank_readings_are_dropped(self):
        from src.tools.ocr import merge_passes
        assert merge_passes([[("  ", (0.1, 0.5, 0.3, 0.02))]]) == []


class TestModelWindow:
    """The live window is 8192; retain the document cap measured at 4096: a
    long document does not overflow loudly — Ollama drops the START and the
    model answers from the tail. Measured: 6.000 characters answered the
    question correctly, 9.000 and 15.000 invented an organisation."""

    def test_short_text_is_untouched(self):
        from src.core.prompt_safety import clip_to_window
        assert clip_to_window("curto") == ("curto", False)

    def test_long_text_is_cut_and_flagged(self):
        from src.core.prompt_safety import MAX_DOC_CHARS, clip_to_window
        body, clipped = clip_to_window("x" * (MAX_DOC_CHARS + 500))
        assert clipped and len(body) == MAX_DOC_CHARS

    def test_the_user_is_told_when_the_read_was_partial(self):
        from src.core.orchestrator import _partial_read_note
        assert _partial_read_note("whole") == ""
        assert "primeira parte" in _partial_read_note("clipped")
        assert "busquei" in _partial_read_note("retrieved")

    def test_a_long_file_answer_carries_the_note(self, orch, home, monkeypatch):
        import numpy as np
        from src.core.prompt_safety import MAX_DOC_CHARS
        # Fake the embedder, not the selection: the wiring is what is under
        # test, and a unit test must not need Ollama to be up.
        monkeypatch.setattr("src.core.doc_retrieval._default_embed",
                            lambda text: np.asarray(
                                [float(len(text)), 1.0], dtype=np.float32))
        (home / "longo.md").write_text("y")
        long_text = "\n\n".join(f"Parágrafo {i}. " + "texto " * 60
                                 for i in range(20))
        monkeypatch.setattr(orch, "reader", type("R", (), {
            "read_path": lambda _self, p: long_text})(),
            raising=False)
        seen = {}
        def _synth(_engine, injected, lang, mode):
            seen["chars"] = len(injected)
            return ("resposta", "LOCAL_RAG")
        monkeypatch.setattr(orch, "_synthesise", _synth, raising=False)
        monkeypatch.setattr(orch, "engine_analytical", object(), raising=False)
        orch._remember_seen_path(str(home / "longo.md"))
        reply, _ = orch._run_read_file_at({"path": str(home / "longo.md")})
        # Whichever way the text was chosen, the answer must SAY the document
        # did not fit whole — that is the point of the note.
        assert ("busquei" in reply) or ("primeira parte" in reply)
        assert seen["chars"] < MAX_DOC_CHARS + 2000   # the fence + guard only


class TestReadThisFileRoute:
    """Asked to read a file it had just found, the 8B answered ABOUT the
    document without calling any tool (s83) — inventing a question 6 that
    wasn't there. A regex route cannot forget to open the file."""

    @pytest.mark.parametrize("text", [
        "lê esse arquivo",
        "leia este pdf",
        "abre esse documento",
        "o que diz esse pdf",
        "resume esse arquivo",
        "me explica esse documento",
        "read this file and tell me the deadline",
        "what's in this document",
    ])
    def test_matches_the_read_phrasings(self, text):
        from src.core.dispatch.triggers import READ_THIS_FILE_RE
        assert READ_THIS_FILE_RE.match(text), text

    @pytest.mark.parametrize("text", [
        "quero ler um livro hoje",
        # two intents — the planner's job, not a plain read (parity test)
        "lê o documento agentes e me manda um resumo por email",
        "abre o Notion",
        "abre o Spotify",
        "me explica o que é RAG",
    ])
    def test_does_not_eat_other_commands(self, text):
        from src.core.dispatch.triggers import READ_THIS_FILE_RE
        assert not READ_THIS_FILE_RE.match(text), text

    def test_a_bare_verb_needs_a_file_on_the_table(self, orch, home):
        """"abre o punctuation" is a read command only because a file called
        Punctuation… has actually been seen."""
        from src.core.dispatch.triggers import READ_VERB_RE
        assert READ_VERB_RE.match("lê o punctuation")
        assert not orch._names_a_known_file("lê o punctuation")
        f = home / "Punctuation from Pearson.pdf"
        f.write_text("x")
        orch._remember_seen_path(str(f))
        assert orch._names_a_known_file("lê o punctuation")
        assert not orch._names_a_known_file("lê o contrato")

    def test_route_stages_the_file_that_is_on_the_table(self, orch, home):
        f = home / "edital.pdf"
        f.write_text("x")
        orch._remember_seen_path(str(f))
        reply, mode = orch._handle_read_this_file("lê esse arquivo e me diz o prazo")
        assert mode == "AGENT_AWAITING_CONFIRMATION"
        assert "edital.pdf" in reply
        assert orch._pending_action["args"]["path"] == str(f)
        # the whole sentence travels as the question
        assert "prazo" in orch._pending_action["args"]["question"]

    def test_route_asks_which_when_several_are_open(self, orch, home):
        for name in ("um.pdf", "dois.pdf"):
            (home / name).write_text("x")
            orch._remember_seen_path(str(home / name))
        reply, mode = orch._handle_read_this_file("lê esse arquivo")
        assert mode == "FILE_ERROR"
        assert "um.pdf" in reply and "dois.pdf" in reply

    def test_route_asks_for_a_name_when_nothing_is_open(self, orch):
        reply, mode = orch._handle_read_this_file("lê esse arquivo")
        assert mode == "FILE_ERROR" and "qual arquivo" in reply

    def test_a_named_file_wins_over_asking(self, orch, home):
        for name in ("Punctuation from Pearson.pdf", "edital.pdf"):
            (home / name).write_text("x")
            orch._remember_seen_path(str(home / name))
        orch._handle_read_this_file("lê o punctuation e me diz a questão 6")
        assert "Punctuation" in orch._pending_action["args"]["path"]

    def test_candidates_skip_what_cannot_be_read(self, orch, home):
        (home / "script.py").write_text("x")
        (home / "nota.md").write_text("x")
        orch._remember_seen_path(str(home / "script.py"))
        orch._remember_seen_path(str(home / "nota.md"))
        names = [os.path.basename(c) for c in orch.readable_candidates()]
        assert names == ["nota.md"]


class TestInventedPaths:
    """Offered a tool that takes a path, the 8B answered "cadê o pdf de
    punctuation?" by inventing /Users/.../punctuation.pdf (s83)."""

    def test_an_unseen_path_is_refused(self, orch, home, monkeypatch):
        opened = []
        monkeypatch.setattr(orch, "reader", type("R", (), {
            "read_path": lambda _self, p: opened.append(p) or "x"})(),
            raising=False)
        reply, mode = orch._run_read_file_at({"path": "/Users/x/inventado.pdf"})
        assert mode == "AGENT_ACTION_ERROR"
        assert opened == []

    def test_the_tool_is_hidden_until_a_path_exists(self, orch):
        names = [s["function"]["name"] for s in orch._drop_unusable_file_read(
            [{"function": {"name": "read_file_at"}},
             {"function": {"name": "find_file"}}])]
        assert names == ["find_file"]
        orch._remember_seen_path("/Users/x/real.pdf")
        names = [s["function"]["name"] for s in orch._drop_unusable_file_read(
            [{"function": {"name": "read_file_at"}}])]
        assert names == ["read_file_at"]

    def test_a_path_the_user_types_counts_as_seen(self, orch, home):
        (home / "contrato.pdf").write_text("x")
        orch.remember_paths_in(f"lê {home}/contrato.pdf pra mim")
        assert orch._path_was_seen(str(home / "contrato.pdf"))

    def test_a_path_that_does_not_exist_is_not_remembered(self, orch, home):
        orch.remember_paths_in(f"lê {home}/fantasma.pdf pra mim")
        assert not orch._path_was_seen(str(home / "fantasma.pdf"))


class TestPastedPaths:
    """A path dragged into a terminal, or pasted from one, arrives escaped:
    "Mobile\\ Documents", "com\\~apple\\~CloudDocs", "29\\:09\\:26" — and in
    the same paste a backslash where a space belonged. Harvesting stopped at
    the first escaped space, so the file was never on the table, the read
    route never fired, and the 8B leaked a raw tool call to the screen (s83).
    """

    def test_resolves_shell_escaped_spaces_and_tildes(self, home):
        from src.tools.file_finder import normalise_user_path
        d = home / "Mobile Documents" / "com~apple~CloudDocs" / "college homework"
        d.mkdir(parents=True)
        (d / "Punctuation from Pearson.pdf").write_text("x")
        raw = (str(home) + r"/Mobile\ Documents/com\~apple\~CloudDocs"
               r"/college\ homework/Punctuation\ from\ Pearson.pdf")
        assert normalise_user_path(raw) == str(d / "Punctuation from Pearson.pdf")

    def test_recovers_a_filename_mangled_by_a_lost_space(self, home):
        """"from\\Pearson.pdf" — the backslash ate the space. Same folder,
        same characters ignoring spaces: that is enough to identify it."""
        from src.tools.file_finder import normalise_user_path
        (home / "Punctuation from Pearson.pdf").write_text("x")
        raw = str(home) + r"/Punctuation\ from\Pearson.pdf"
        assert normalise_user_path(raw) == str(home / "Punctuation from Pearson.pdf")

    def test_does_not_wander_to_another_file(self, home):
        from src.tools.file_finder import normalise_user_path
        (home / "contrato.pdf").write_text("x")
        assert normalise_user_path(str(home / "recibo.pdf")) is None

    def test_a_pasted_path_reaches_the_seen_list(self, orch, home):
        (home / "Punctuation from Pearson.pdf").write_text("x")
        orch.remember_paths_in(
            "lê " + str(home) + r"/Punctuation\ from\ Pearson.pdf e me diz a questão 5")
        assert orch.readable_candidates() == [
            str(home / "Punctuation from Pearson.pdf")]


class TestLeakedToolCall:
    """The 8B emitted a tool call as TEXT and it reached the screen. The
    sanitiser already caught {"name": …}; the leak carried
    {"type":"function"} first, one key earlier than the pattern allowed."""

    @pytest.mark.parametrize("leak", [
        '{"type":"function","name":"read_file_at","parameters":{"path":"/Users/x/a.pdf"}}',
        '{"type": "function", "name": "find_file", "parameters": {"query": "cv"}}',
        '{"name":"send_email","arguments":{"to":"x"}}',
    ])
    def test_a_tool_call_never_reaches_the_user(self, leak):
        from src.core.engine import OllamaEngine
        out = OllamaEngine._sanitise_response(leak)
        assert "read_file_at" not in out and "find_file" not in out
        assert "send_email" not in out

    def test_ordinary_prose_is_untouched(self):
        from src.core.engine import OllamaEngine
        prose = "A vírgula separa itens de uma lista."
        assert OllamaEngine._sanitise_response(prose) == prose


class TestLanguageMirroring:
    """Cortex answers in the language it was asked in. Pedro wrote "read
    /Users/…/com~apple~CloudDocs/… and answer the question number five" and
    was answered in Portuguese: "com", inside the iCloud folder name, matched
    the Portuguese keyword list (s83). A path is a name, not language."""

    def test_a_path_does_not_decide_the_language(self):
        from src.core.i18n import _detect_lang
        english = (r"read /Users/p/Library/Mobile\ Documents/"
                   r"com\~apple\~CloudDocs/college\ homework/x.pdf"
                   r" and answer the question number five")
        assert _detect_lang(english) == "en"

    def test_a_portuguese_sentence_with_a_path_stays_portuguese(self):
        from src.core.i18n import _detect_lang
        assert _detect_lang(
            r"lê /Users/p/Library/Mobile\ Documents/x.pdf e me responda a questão 5"
        ) == "pt"

    def test_a_url_does_not_decide_the_language(self):
        from src.core.i18n import _detect_lang
        assert _detect_lang(
            "summarise https://exemplo.com.br/noticias/uma-coisa for me") == "en"

    @pytest.mark.parametrize("text", [
        "leia o arquivo e me diz o prazo",
        "abre o Notion",
        "trava a tela",
        "dorme o mac",
        "tira print",
        "cadê o pdf punctuation?",
        "hiberna o mac",
        "escreve no pdf",                 # s85: the write-on-PDF follow-up
        "escreve as respostas no pdf",
        "preenche a prova",
        "ambos",
    ])
    def test_everyday_portuguese_commands_are_portuguese(self, text):
        """The list had the lease verbs but not the ones Pedro uses daily for
        files, the screen and the Mac — so they were answered in English."""
        from src.core.i18n import _detect_lang
        assert _detect_lang(text) == "pt", text

    @pytest.mark.parametrize("text", [
        "read this file and tell me the deadline",
        "lock the screen",
        "put the mac to sleep",
        "take a screenshot",
        "open Notion",
        "where is my cv",
        "summarise this document",
        "write them on the pdf",
        "fill in the pdf",
        "both",
    ])
    def test_english_is_still_english(self, text):
        from src.core.i18n import _detect_lang
        assert _detect_lang(text) == "en", text


class TestOcrScalesSetting:
    """A typo in CORTEX_OCR_SCALES raised at import and turned OCR off in
    silence on the read path (s85 review)."""

    @pytest.mark.parametrize("raw,want", [
        ("2,1,1.25", (2.0, 1.0, 1.25)),
        ("2,x", (2.0,)),
        ("x", (2.0,)),
        ("", (2.0,)),
        ("2,100", (2.0,)),            # absurd scale skipped
    ])
    def test_bad_values_are_skipped(self, raw, want):
        from src.tools.ocr import _parse_scales
        assert _parse_scales(raw) == want
