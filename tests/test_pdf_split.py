"""Typed page selection: one PDF, explicit choice, exact approval, contained tools."""
from datetime import date
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pypdf import PdfReader, PdfWriter

from src.core import effect_store, pdf_info, pdf_publish, pdf_runner, pdf_sandbox
from src.core.staging_mixin import Delivery
from src.core.tool_risk import S3, build_preview, risk_level
from src.tools.local_reader import LocalFileReader
from tests.test_claim_remove_failure import _turn
from tests.test_dispatch_parity import _build_real_orch


def write_pdf(path, pages=12):
    writer = PdfWriter()
    for number in range(1, pages + 1):
        writer.add_blank_page(width=100 + number, height=100)
    writer.write(path)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "Documents").mkdir()
    path = tmp_path / "Downloads/prova.pdf"
    path.parent.mkdir()
    write_pdf(path)
    o = _build_real_orch(tmp_path)
    o.background, o._reply_lang = False, "pt"
    o.set_turn_origin("ui", "conversation")
    o.reader = LocalFileReader(str(tmp_path / "vault"))
    return SimpleNamespace(o=o, path=path, home=tmp_path, store=effect_store.store)


@pytest.fixture
def fake(setup, monkeypatch):
    s = setup
    s.inspect = Mock(side_effect=lambda path, **kw: pdf_info.PdfInfo(
        "ok", len(PdfReader(path).pages), False, (), (), 0, ""))
    monkeypatch.setattr(pdf_info, "inspect", s.inspect)

    def run(job, tool, args, *, cancelled):
        assert callable(cancelled)
        [effect] = s.store.in_states(["executando"])
        assert effect.action == "pdf_split"
        if tool == "pdfseparate":
            assert args[:1] == ["-f"] and args[2] == "-l"
            assert Path(args[4]).parent == job.input_dir
            assert Path(args[5]).parent == job.output_dir
            reader = PdfReader(args[4])
            for number in range(int(args[1]), int(args[3]) + 1):
                writer = PdfWriter()
                writer.add_page(reader.pages[number - 1])
                writer.write(args[5].replace("%d", str(number)))
        elif tool == "pdfunite":
            assert all(Path(p).parent == job.output_dir for p in args)
            writer = PdfWriter()
            for path in args[:-1]:
                writer.append(path)
            writer.write(args[-1])
        else:
            assert tool == "pdfinfo" and Path(args[0]).parent == job.output_dir
            pages = len(PdfReader(args[0]).pages)
            return pdf_runner.ToolRun("exited", 0, .1, f"Pages: {pages}\n".encode(), "", [])
        return pdf_runner.ToolRun("exited", 0, .1, b"", "", [])

    def publish(job, files, name):
        assert files == [name + ".pdf"]
        s.widths = [int(p.mediabox.width) for p in PdfReader(job.output_dir / files[0]).pages]
        return pdf_publish.Published("published", str(s.home / "Documents/Cortex/PDF" / name),
                                     tuple(files), 1, "")

    s.run, s.publish = Mock(side_effect=run), Mock(side_effect=publish)
    monkeypatch.setattr(pdf_runner, "run_tool", s.run)
    monkeypatch.setattr(pdf_publish, "publish", s.publish)
    return s


def phrase(mode="extract", lang="pt", spec=None):
    verb = {("extract", "pt"): "separa", ("remove", "pt"): "remove", ("ask", "pt"): "tira",
            ("extract", "en"): "extract", ("remove", "en"): "remove", ("ask", "en"): "take out"}[mode, lang]
    return f"{verb} {spec or ('pages 3 to 5' if lang == 'en' else 'as páginas 3 a 5')} " + (
        "from" if lang == "en" else "de") + " ~/Downloads/prova.pdf"


def stage(s, stream=False, lang="pt", mode="extract", spec=None):
    reply, result = _turn(s.o, phrase(mode, lang, spec), stream, lang)
    assert result == "AGENT_AWAITING_CONFIRMATION"
    assert s.o._pending_action["type"] == "pdf_split"
    return reply


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("verb,mode", [(v, "extract") for v in
    ("separa", "separe", "extrai", "extraia", "extract", "  SEPARA")] +
    [(v, "remove") for v in ("remove", "apaga", "apague", "delete")])
@pytest.mark.parametrize("spec,join,lang,label,selected", [
    ("a página 4", "de", "pt", "4", [4]), ("página 4", "do", "pt", "4", [4]),
    ("as páginas 3 a 5", "da", "pt", "3-5", [3, 4, 5]),
    ("páginas 3-5", "de", "pt", "3-5", [3, 4, 5]),
    ("as páginas 3, 5 e 7", "de", "pt", "3, 5, 7", [3, 5, 7]),
    ("3 a 5 e 8", "de", "pt", "3-5, 8", [3, 4, 5, 8]),
    ("8 e 5, 3 a 4 e 3", "de", "pt", "3-5, 8", [3, 4, 5, 8]),
    ("page 4", "of", "en", "4", [4]), ("pages 3 to 5", "from", "en", "3-5", [3, 4, 5]),
    ("pages 3-5", "from", "en", "3-5", [3, 4, 5]),
    ("pages 3, 5 and 7", "of", "en", "3, 5, 7", [3, 5, 7]),
])
def test_phrase_forms(fake, stream, verb, mode, spec, join, lang, label, selected):
    text = f"{verb} {spec} {join} ~/Downloads/prova.pdf!? "
    assert _turn(fake.o, text, stream)[1] == "AGENT_AWAITING_CONFIRMATION"
    action = fake.o._pending_action
    args = action["args"]
    assert args["mode"] == mode and args["lang"] == lang and args["label"] == label
    assert args["keep"] == (selected if mode == "extract" else [p for p in range(1, 13) if p not in selected])
    assert args["input"] == {"path": str(fake.path), "pages": 12,
        "sha256": hashlib.sha256(fake.path.read_bytes()).hexdigest()}
    assert json.loads(Path(fake.o._PENDING_ACTION_FILE).read_text()) == action
    fake.run.assert_not_called()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("text", ["por favor separa páginas 3-5 de ~/Downloads/prova.pdf",
    "separa páginas 3-5 de ~/Downloads/prova.pdf por favor",
    "separa páginas 3-5 de ~/Downloads/prova.pdf e ~/Downloads/prova.pdf",
    "separa páginas 3-5 com ~/Downloads/prova.pdf"])
def test_extra_prose(fake, stream, text):
    from src.core import path_route
    assert fake.o._handle_pdf_split_phrase(text, path_route.find_paths(text)) is None
    _turn(fake.o, text, stream)
    # The existing single-file path route stages a read of this PDF.
    assert fake.o._pending_action["type"] == "read_file_at"
    fake.inspect.assert_not_called()


@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("spec", ["0", "5-3", "2 a 0", "three", "3 e", "3,,5", "3.5", "-2", "3 5"])
def test_bad_specs(fake, lang, spec):
    assert _turn(fake.o, phrase(lang=lang, spec=spec), False)[0] == (
        "Não entendi as páginas. Exemplo: separa as páginas 3 a 5 de ~/x.pdf." if lang == "pt" else
        "I didn't understand the pages. Example: extract pages 3 to 5 from ~/x.pdf.")
    assert fake.o._pending_action is None


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("case", ["folder", "missing", "nonpdf", "unreadable", "voice", "background", "size", "names"])
def test_refusals(fake, monkeypatch, stream, case):
    text = phrase()
    if case == "folder":
        text = text.replace("/prova.pdf", "")
        expected = "Só separo páginas de um arquivo .pdf."
    elif case == "missing":
        text = text.replace("prova.pdf", "missing.pdf")
        expected = "missing.pdf"
    elif case == "nonpdf":
        fake.path.with_suffix(".txt").write_text("fictitious")
        text = text.replace("prova.pdf", "prova.txt")
        expected = "Só separo páginas de um arquivo .pdf."
    elif case == "unreadable":
        monkeypatch.setattr("src.tools.file_finder.is_readable_path", lambda p: (False, "blocked"))
        expected = "não posso ler prova.pdf: blocked."
    elif case in ("voice", "background"):
        if case == "voice":
            fake.o.set_turn_origin("voice", "conversation")
        else:
            fake.o.background = True
        expected = "Separar páginas só pedindo digitado, no terminal ou na tela."
    elif case == "size":
        monkeypatch.setattr(pdf_runner, "MAX_INPUT_BYTES", 1)
        expected = "prova.pdf passa de 50 MB — grande demais para separar páginas."
    else:
        monkeypatch.setattr(pdf_publish, "next_free", lambda name: None)
        expected = "Já existem 99 pastas com esse nome em ~/Documents/Cortex/PDF — apague algumas e peça de novo."
    assert expected in _turn(fake.o, text, stream)[0]
    assert fake.o._pending_action is None and fake.store.recent() == []
    fake.run.assert_not_called()


@pytest.mark.parametrize("status,expected", [
    ("protected", "prova.pdf tem senha — não separo páginas de PDF com senha."),
    ("broken", "prova.pdf parece danificado — não consegui abrir."),
    ("unavailable", "As ferramentas de PDF (Poppler) não estão disponíveis neste Mac."),
    *[(status, "Não consegui conferir prova.pdf — nada foi feito.") for status in ("error", "timeout", "stopped", "too_big")]])
def test_inspection_refusals(fake, status, expected):
    fake.inspect.side_effect = None
    fake.inspect.return_value = pdf_info.PdfInfo(status, None, None, (), (), 0, "")
    assert _turn(fake.o, phrase(), False)[0] == expected
    assert fake.o._pending_action is None and fake.store.recent() == []


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("status", ["ok", "protected"])
def test_encrypted_pdf_uses_inspection_status(fake, stream, lang, status):
    fake.inspect.side_effect = None
    fake.inspect.return_value = pdf_info.PdfInfo(status, 12, True, ("copy",), (), 0, "")
    if status == "ok":
        reply = stage(fake, stream, lang)
        assert reply == build_preview("pdf_split", fake.o._pending_action["args"], lang=lang)["human"]
    else:
        assert _turn(fake.o, phrase(lang=lang), stream, lang) == ((
            "prova.pdf tem senha — não separo páginas de PDF com senha." if lang == "pt" else
            "prova.pdf has a password — I don't split password-protected PDFs."), "PDF_SPLIT_REFUSED")
        assert fake.o._pending_action is None
    assert fake.store.recent() == []
    fake.run.assert_not_called()


@pytest.mark.parametrize("stream", [False, True])
def test_question_reuses_resolved_path_with_escaped_spaces(fake, stream):
    fake.path = fake.path.rename(fake.path.with_name("a b c d e f g h i prova.pdf"))
    typed = "~/Downloads/" + fake.path.name.replace(" ", "\\ ")
    text = phrase("ask").replace("~/Downloads/prova.pdf", typed)
    assert _turn(fake.o, text, stream)[1] == "PDF_SPLIT_QUESTION"
    assert _turn(fake.o, "1", stream)[1] == "AGENT_AWAITING_CONFIRMATION"
    assert fake.o._pending_action["args"]["input"]["path"] == str(fake.path)
    assert _turn(fake.o, "sim", stream)[1] == "AGENT_ACTION_DONE"


@pytest.mark.parametrize("stream", [False, True])
def test_empty_turn_clears_question(fake, stream):
    _turn(fake.o, phrase("ask"), stream)
    assert _turn(fake.o, "   ", stream)[1] == "INPUT_ERROR"
    assert fake.o._pdf_split_question is None
    _turn(fake.o, "1", stream)
    assert fake.o._pending_action is None
    fake.run.assert_not_called()


@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("mode,spec,expected_pt,expected_en", [
    ("extract", "15", "prova.pdf tem 12 páginas — a página 15 não existe.", "prova.pdf has 12 pages — page 15 doesn't exist."),
    ("remove", "1-12", "Sem essas páginas não sobra nada — nada foi feito.", "Without those pages nothing is left — nothing was done."),
    ("extract", "1-120", "O resultado teria 120 páginas — consigo até 99 por vez.", "The result would have 120 pages — I can do up to 99 at a time."),
    ("remove", "121", "O resultado teria 120 páginas — consigo até 99 por vez.", "The result would have 120 pages — I can do up to 99 at a time."),
])
def test_page_bounds(fake, lang, mode, spec, expected_pt, expected_en):
    if spec in ("1-120", "121"):
        fake.inspect.side_effect = lambda *a, **kw: pdf_info.PdfInfo("ok", 121, False, (), (), 0, "")
    assert _turn(fake.o, phrase(mode, lang, spec), False)[0] == (expected_pt if lang == "pt" else expected_en)
    assert fake.o._pending_action is None and fake.store.recent() == []


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("verb,lang", [("tira", "pt"), ("tire", "pt"), ("take out", "en")])
@pytest.mark.parametrize("answer,mode", [("1", "extract"), (" 1. ", "extract"), ("2", "remove")])
def test_question_and_answer(fake, stream, verb, lang, answer, mode):
    text = phrase("ask", lang).replace("take out" if lang == "en" else "tira", verb)
    reply, _ = _turn(fake.o, text, stream)
    assert reply == ("Em prova.pdf (12 páginas), você quer:\n1 — um PDF só com as páginas 3-5\n"
        "2 — uma cópia sem as páginas 3-5\nResponda 1 ou 2." if lang == "pt" else
        "In prova.pdf (12 pages), do you want:\n1 — a PDF with only pages 3-5\n"
        "2 — a copy without pages 3-5\nReply 1 or 2.")
    assert fake.o._pending_action is None
    assert _turn(fake.o, answer, stream)[1] == "AGENT_AWAITING_CONFIRMATION"
    assert fake.o._pending_action["args"]["mode"] == mode
    assert fake.o._pending_action["args"]["lang"] == lang
    fake.run.assert_not_called()
    assert _turn(fake.o, "sim", stream)[1] == "AGENT_ACTION_DONE"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang", ["pt", "en"])
def test_question_two_page_source_stays_plural(fake, stream, lang):
    write_pdf(fake.path, 2)
    assert _turn(fake.o, phrase("ask", lang, "2"), stream, lang) == ((
        "Em prova.pdf (2 páginas), você quer:\n1 — um PDF só com as páginas 2\n"
        "2 — uma cópia sem as páginas 2\nResponda 1 ou 2." if lang == "pt" else
        "In prova.pdf (2 pages), do you want:\n1 — a PDF with only pages 2\n"
        "2 — a copy without pages 2\nReply 1 or 2."), "PDF_SPLIT_QUESTION")
    assert fake.o._pending_action is None
    fake.run.assert_not_called()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang", ["pt", "en"])
def test_question_one_page_source_is_not_feasible(fake, stream, lang):
    write_pdf(fake.path, 1)
    assert _turn(fake.o, phrase("ask", lang, "1"), stream, lang) == ((
        "Sem essas páginas não sobra nada — nada foi feito." if lang == "pt" else
        "Without those pages nothing is left — nothing was done."), "PDF_SPLIT_REFUSED")
    assert fake.o._pdf_split_question is None and fake.o._pending_action is None
    fake.run.assert_not_called()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("case", ["1 e 2", "um", "o primeiro", "oi", "voice", "elsewhere", "expired"])
def test_question_is_one_turn_scoped(fake, monkeypatch, stream, case):
    _turn(fake.o, phrase("ask"), stream)
    if case == "expired":
        import src.core.orchestrator as module
        now = module.time.time()
        monkeypatch.setattr(module.time, "time", lambda: now + 901)
    elif case in ("voice", "elsewhere"):
        fake.o.set_turn_origin("voice" if case == "voice" else "ui",
                               "conversation" if case == "voice" else "elsewhere")
    _turn(fake.o, "1" if case in ("voice", "elsewhere", "expired") else case, stream)
    assert fake.o._pending_action is None
    if case not in ("voice", "elsewhere"):
        _turn(fake.o, "1", stream)
        assert fake.o._pending_action is None
    fake.run.assert_not_called()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("mode", ["extract", "remove"])
def test_preview_and_success(fake, stream, lang, mode):
    action = ("páginas 3-5" if mode == "extract" else "sem as páginas 3-5") if lang == "pt" else (
        "pages 3-5" if mode == "extract" else "without pages 3-5")
    name = pdf_publish.folder_name("prova.pdf", action, date.today())
    (fake.home / "Documents/Cortex/PDF" / name).mkdir(parents=True)
    name += " (2)"
    count = 3 if mode == "extract" else 9
    first = ("Fazer um PDF só com as páginas 3-5 de prova.pdf?" if mode == "extract" else
             "Fazer uma cópia de prova.pdf sem as páginas 3-5?") if lang == "pt" else (
             "Make a PDF with only pages 3-5 of prova.pdf?" if mode == "extract" else
             "Make a copy of prova.pdf without pages 3-5?")
    expected = first + (f"\nOrigem: prova.pdf (12 páginas) — {fake.path.parent}\n"
        f"Resultado: um PDF de {count} páginas em ~/Documents/Cortex/PDF/{name}/\n"
        "O original não muda.\nResponda 'sim' (exatamente essa palavra) para fazer ou 'não' para cancelar."
        if lang == "pt" else f"\nSource: prova.pdf (12 pages) — {fake.path.parent}\n"
        f"Result: a PDF of {count} pages in ~/Documents/Cortex/PDF/{name}/\n"
        "The original doesn't change.\nReply 'yes' (exactly that word) to do it or 'no' to cancel.")
    assert stage(fake, stream, lang, mode) == expected
    args = fake.o._pending_action["args"]
    action_id = fake.o._pending_action["id"]
    assert risk_level("pdf_split") == S3
    preview = build_preview("pdf_split", args, lang=lang)
    assert preview["llm"] == "[SYSTEM ERROR] pdf_split requires the typed phrase."
    assert preview["cancel_label"] == ("a separação das páginas" if lang == "pt" else "split the pages")
    before = fake.path.read_bytes()
    done = (f"Pronto: um PDF de {count} páginas em ~/Documents/Cortex/PDF/{name}/{name}.pdf. O original não mudou."
        if lang == "pt" else f"Done: a PDF of {count} pages in ~/Documents/Cortex/PDF/{name}/{name}.pdf. The original didn't change.")
    assert _turn(fake.o, "yes" if lang == "en" else "sim", stream, lang) == (done, "AGENT_ACTION_DONE")
    assert fake.widths == [100 + p for p in args["keep"]]
    assert fake.path.read_bytes() == before
    [effect] = fake.store.recent()
    assert effect.id == action_id and effect.state == "feito"
    assert effect.args == {"input": str(fake.path), "keep": args["keep"], "name": name}
    calls = fake.run.call_count
    _turn(fake.o, "sim", stream)
    assert fake.run.call_count == calls and len(fake.store.recent()) == 1


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("source_pages,page", [(12, 4), (1, 1)])
def test_single_page_extract_preview(fake, stream, lang, source_pages, page):
    write_pdf(fake.path, source_pages)
    action = f"páginas {page}" if lang == "pt" else f"pages {page}"
    name = pdf_publish.folder_name("prova.pdf", action, date.today())
    source_count = ("1 página" if source_pages == 1 else "12 páginas") if lang == "pt" else (
        "1 page" if source_pages == 1 else "12 pages")
    assert stage(fake, stream, lang, spec=f"a página {page}" if lang == "pt" else f"page {page}") == (
        f"Fazer um PDF só com as páginas {page} de prova.pdf?\n"
        f"Origem: prova.pdf ({source_count}) — {fake.path.parent}\n"
        f"Resultado: um PDF de 1 página em ~/Documents/Cortex/PDF/{name}/\n"
        "O original não muda.\nResponda 'sim' (exatamente essa palavra) para fazer ou 'não' para cancelar."
        if lang == "pt" else
        f"Make a PDF with only pages {page} of prova.pdf?\n"
        f"Source: prova.pdf ({source_count}) — {fake.path.parent}\n"
        f"Result: a PDF of 1 page in ~/Documents/Cortex/PDF/{name}/\n"
        "The original doesn't change.\nReply 'yes' (exactly that word) to do it or 'no' to cancel.")
    fake.run.assert_not_called()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("source_pages,page", [(12, 4), (1, 1)])
def test_single_page_extract_done(fake, stream, lang, source_pages, page):
    write_pdf(fake.path, source_pages)
    stage(fake, stream, lang, spec=str(page))
    name = fake.o._pending_action["args"]["name"]
    destination = f"~/Documents/Cortex/PDF/{name}/{name}.pdf"
    assert _turn(fake.o, "sim", stream, lang) == ((
        f"Pronto: um PDF de 1 página em {destination}. O original não mudou." if lang == "pt" else
        f"Done: a PDF of 1 page in {destination}. The original didn't change."), "AGENT_ACTION_DONE")
    assert fake.widths == [100 + page]
    assert fake.store.recent()[0].state == "feito"


@pytest.mark.parametrize("args", [None, [], [1], "odd", 1, True,
    {}, {"input": None}, {"input": 3, "keep": 7},
    {"input": {"path": None, "pages": True}, "keep": "3", "name": 3},
    {"input": {"path": [], "pages": "12"}, "keep": [None, {}, True], "mode": []}])
@pytest.mark.parametrize("lang", ["pt", "en"])
def test_preview_total(args, lang):
    assert isinstance(build_preview("pdf_split", args, lang=lang)["human"], str)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("word", ["ok", "pode", "sim, pode", "yes please"])
def test_exact_approval(fake, stream, word):
    stage(fake, stream)
    assert _turn(fake.o, word, stream) == (
        "Para separar as páginas, responda exatamente 'sim' (ou 'não' para cancelar).", "AWAITING_CONFIRMATION")
    fake.run.assert_not_called()
    assert fake.store.recent() == []
    assert _turn(fake.o, "sim", stream)[1] == "AGENT_ACTION_DONE"


@pytest.mark.parametrize("stream", [False, True])
def test_changed_before_approval(fake, stream):
    stage(fake, stream)
    write_pdf(fake.path, 6)
    assert _turn(fake.o, "sim", stream) == (
        "prova.pdf mudou desde a prévia — peça de novo. Nada foi feito.", "PDF_SPLIT_REFUSED")
    assert fake.store.recent() == []
    fake.run.assert_not_called()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("origin", [("voice", "conversation"), ("ui", "elsewhere"), ("terminal", None)])
def test_approval_origin(fake, stream, origin):
    stage(fake, stream)
    fake.o.set_turn_origin(*origin)
    assert _turn(fake.o, "sim", stream)[1] == "AGENT_ACTION_ELSEWHERE"
    fake.run.assert_not_called()


def test_runs_and_order(fake):
    stage(fake, spec="8 e 3 a 5 e 1")
    assert _turn(fake.o, "sim", False)[1] == "AGENT_ACTION_DONE"
    calls = fake.run.call_args_list
    assert [c.args[2][:4] for c in calls if c.args[1] == "pdfseparate"] == [
        ["-f", "1", "-l", "1"], ["-f", "3", "-l", "5"], ["-f", "8", "-l", "8"]]
    unite = next(c.args[2] for c in calls if c.args[1] == "pdfunite")
    assert [Path(p).name for p in unite[:-1]] == [f"page-{p}.pdf" for p in (1, 3, 4, 5, 8)]
    assert fake.widths == [101, 103, 104, 105, 108]


@pytest.mark.parametrize("stream", [False, True])
def test_path_case_and_terminal_success(fake, stream):
    fake.path = fake.path.rename(fake.path.with_name("Prova.pdf"))
    fake.o.set_turn_origin("terminal")
    text = phrase().replace("separa", "  SEPARA").replace("prova.pdf", "Prova.pdf")
    assert _turn(fake.o, text, stream)[1] == "AGENT_AWAITING_CONFIRMATION"
    assert fake.o._pending_action["args"]["input"]["path"] == str(fake.path)
    fake.o._pending_action = fake.o._load_pending_action()
    assert _turn(fake.o, "sim", stream)[1] == "AGENT_ACTION_DONE"
    assert fake.store.recent()[0].origin == {"channel": "terminal", "conversation": None}


@pytest.mark.parametrize("mode,spec", [("extract", "1-99"), ("remove", "100-120")])
def test_99_page_boundary(fake, mode, spec):
    write_pdf(fake.path, 120)
    stage(fake, mode=mode, spec=spec)
    assert _turn(fake.o, "sim", False)[1] == "AGENT_ACTION_DONE"
    assert fake.widths == list(range(101, 200))


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("case", ["stop_before", "unreadable", "unrecorded"])
def test_preexecution_gates(fake, monkeypatch, stream, case):
    stage(fake, stream)
    if case == "stop_before":
        delivery = Delivery()
        delivery.cancelled.set()
        fake.o.set_turn_delivery(delivery)
        expected = "Parei antes de executar — não fiz nada."
    elif case == "unreadable":
        monkeypatch.setattr("src.tools.file_finder.is_readable_path", lambda p: (False, "blocked"))
        expected = "não posso ler prova.pdf: blocked."
    else:
        monkeypatch.setattr(fake.store, "create", lambda *a, **kw: None)
        expected = "Não consegui registrar essa ação antes de executar — não fiz nada."
    assert _turn(fake.o, "sim", stream)[0] == expected
    fake.run.assert_not_called()
    fake.publish.assert_not_called()
    if case == "stop_before":
        assert fake.store.recent()[0].state == "cancelado"
    else:
        assert fake.store.recent() == []


def test_question_checks_and_revalidates(fake):
    _turn(fake.o, phrase("ask", spec="15"), False)
    assert fake.o._pdf_split_question is None
    _turn(fake.o, phrase("ask"), False)
    write_pdf(fake.path, 2)
    assert _turn(fake.o, "1", False)[0] == "prova.pdf tem 2 páginas — a página 3 não existe."
    assert fake.o._pending_action is None
    assert fake.store.recent() == []


@pytest.mark.parametrize("stream", [False, True])
def test_question_isolated_from_other_conversation(fake, stream):
    _turn(fake.o, phrase("ask"), stream)
    fake.o.set_turn_origin("ui", "elsewhere")
    assert fake.o._pdf_split_question is None
    _turn(fake.o, "1", stream)
    assert fake.o._pending_action is None
    fake.o.set_turn_origin("ui", "conversation")
    assert _turn(fake.o, "2.", stream)[1] == "AGENT_AWAITING_CONFIRMATION"
    assert fake.o._pending_action["args"]["mode"] == "remove"
    fake.run.assert_not_called()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("elapsed,accepted", [(899, True), (900, False)])
def test_question_expiry_boundary(fake, monkeypatch, stream, elapsed, accepted):
    import src.core.orchestrator as module
    now = module.time.time()
    monkeypatch.setattr(module.time, "time", lambda: now)
    _turn(fake.o, phrase("ask"), stream)
    monkeypatch.setattr(module.time, "time", lambda: now + elapsed)
    _turn(fake.o, "1", stream)
    assert (fake.o._pending_action is not None) == accepted
    assert fake.o._pdf_split_question is None
    fake.run.assert_not_called()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("spec", ["1-12", "1-120", "121"])
def test_question_requires_both_choices_to_be_valid(fake, stream, spec):
    if spec != "1-12":
        write_pdf(fake.path, 121)
    assert _turn(fake.o, phrase("ask", spec=spec), stream)[1] == "PDF_SPLIT_REFUSED"
    assert fake.o._pdf_split_question is None and fake.o._pending_action is None
    assert fake.store.recent() == []
    fake.run.assert_not_called()


@pytest.mark.parametrize("spec", ["1-99999999999999999999", "99999999999999999999"])
def test_large_page_numbers_refused_before_expansion(fake, spec):
    assert _turn(fake.o, phrase(spec=spec), False)[1] == "PDF_SPLIT_REFUSED"
    assert fake.o._pending_action is None and fake.store.recent() == []
    fake.run.assert_not_called()


@pytest.mark.parametrize("lang", ["pt", "en"])
def test_cancel_language(fake, lang):
    stage(fake, lang=lang)
    reply, _ = _turn(fake.o, "no" if lang == "en" else "não", False, lang)
    assert ("split the pages" if lang == "en" else "a separação das páginas") in reply
    fake.run.assert_not_called()


def test_background_and_nested_plan_guards(fake):
    assert fake.o._gate_background_action("pdf_split", {}) == "[SYSTEM ERROR] pdf_split requires the typed phrase."
    fake.o._pending_action = {"type": "execute_plan", "args": {"passos": [
        {"tool": "execute_plan", "args": {"passos": [{"tool": "pdf_split", "args": {}}]}}]}}
    fake.o._persist_pending_action()
    assert fake.o._pending_action is None
    assert not Path(fake.o._PENDING_ACTION_FILE).exists()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("case", ["wrong_pages", "duplicate_count", "missing_page", "extra_page", "stopped", "timeout", "occupied", "publish_error", "staged_changed", "stop_after_info"])
def test_execution_failures(fake, monkeypatch, stream, case):
    stage(fake, stream)
    original = fake.run.side_effect
    if case in ("occupied", "publish_error"):
        fake.publish.side_effect = lambda *a: pdf_publish.Published(
            "occupied" if case == "occupied" else "error", None, (), 0, "")
    elif case == "staged_changed":
        stage_copy = pdf_runner.PdfJob.stage
        def change(job, paths):
            staged = stage_copy(job, paths)
            write_pdf(staged[0], 6)
            return staged
        monkeypatch.setattr(pdf_runner.PdfJob, "stage", change)
    else:
        delivery = Delivery()
        fake.o.set_turn_delivery(delivery)
        def fail(job, tool, args, **kw):
            if case in ("stopped", "timeout"):
                return pdf_runner.ToolRun(case, None, .1, b"", "", [])
            if tool == "pdfinfo" and case in ("wrong_pages", "duplicate_count"):
                return pdf_runner.ToolRun("exited", 0, .1,
                    b"Pages: 99\n" if case == "wrong_pages" else b"Pages: 3\nPages: 3\n", "", [])
            result = original(job, tool, args, **kw)
            if tool == "pdfseparate":
                if case == "missing_page":
                    (job.output_dir / "page-4.pdf").unlink()
                elif case == "extra_page":
                    write_pdf(job.output_dir / "page-99.pdf", 1)
            if tool == "pdfinfo" and case == "stop_after_info":
                delivery.cancelled.set()
            return result
        fake.run.side_effect = fail
    assert _turn(fake.o, "sim", stream)[1] == "AGENT_ACTION_ERROR"
    assert fake.store.recent()[0].state == "falhou"
    if case not in ("occupied", "publish_error"):
        fake.publish.assert_not_called()


@pytest.mark.parametrize("entry", ["_execute_tool", "_execute_tool_unstaged", "_skill_macro_stage_step"])
@pytest.mark.parametrize("plan", [False, True])
def test_8b_cannot_call(fake, entry, plan):
    name, args = "pdf_split", {}
    if plan:
        name, args = "execute_plan", {"passos": [{"tool": name, "args": args}]}
    assert "[SYSTEM ERROR]" in getattr(fake.o, entry)(name, args)
    assert fake.o._pending_action is None and fake.store.recent() == []
    fake.run.assert_not_called()


@pytest.mark.parametrize("stream", [False, True])
def test_terminal_restore_cancel_and_background(fake, stream):
    fake.o.set_turn_origin("terminal")
    stage(fake, stream)
    fake.o._pending_action = fake.o._load_pending_action()
    assert "a separação das páginas" in _turn(fake.o, "não", stream)[0]
    stage(fake, stream)
    fake.o.background = True
    assert _turn(fake.o, "sim", stream)[0] == "Separar páginas só pedindo digitado, no terminal ou na tela."
    assert fake.store.recent() == []
    fake.run.assert_not_called()


@pytest.mark.skipif(os.environ.get("CORTEX_REAL_SANDBOX_TESTS") != "1"
                   or not Path(pdf_sandbox.SANDBOX_EXEC).exists()
                   or pdf_sandbox.discover() is None,
                   reason="requires opt-in real sandbox-exec and Poppler")
@pytest.mark.parametrize("stream", [False, True])
def test_real_split(setup, stream):
    if not pdf_sandbox.available():
        pytest.skip("real sandbox unavailable in this environment")
    s = setup
    write_pdf(s.path, 6)
    before = s.path.read_bytes()
    for mode, spec, answer, wanted in [("extract", "2, 4 a 5", None, [2, 4, 5]),
            ("remove", "1", None, [2, 3, 4, 5, 6]), ("ask", "2, 4 a 5", "2", [1, 3, 6])]:
        _turn(s.o, phrase(mode, spec=spec), stream)
        if answer:
            _turn(s.o, answer, stream)
        name = s.o._pending_action["args"]["name"]
        assert _turn(s.o, "sim", stream)[1] == "AGENT_ACTION_DONE"
        folder = s.home / "Documents/Cortex/PDF" / name
        assert list(folder.iterdir()) == [folder / (name + ".pdf")]
        assert [int(p.mediabox.width) for p in PdfReader(folder / (name + ".pdf")).pages] == [100 + p for p in wanted]
        assert s.path.read_bytes() == before
    assert len(s.store.recent()) == 3
    assert all(e.action == "pdf_split" and e.state == "feito" for e in s.store.recent())
