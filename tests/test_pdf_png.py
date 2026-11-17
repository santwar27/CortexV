"""Typed PDF page rendering: exact approval, bounded PNGs, exclusive publication."""
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import struct
from unittest.mock import Mock
import zlib

import pytest
from pypdf import PdfReader

from src.core import path_route, pdf_info, pdf_publish, pdf_runner, pdf_sandbox
from src.core.staging_mixin import Delivery
from src.core.tool_risk import S3, build_preview, risk_level
from tests.test_claim_remove_failure import _turn
from tests.test_pdf_split import setup, write_pdf


def png(width=2048, height=1024):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress((b"\0" * (width + 1)) * height)) + chunk(b"IEND", b""))


@pytest.fixture
def fake(setup, monkeypatch):
    s = setup
    s.inspect = Mock(side_effect=lambda path, **kw: pdf_info.PdfInfo(
        "ok", len(PdfReader(path).pages), False, (), (), 0, ""))
    monkeypatch.setattr(pdf_info, "inspect", s.inspect)

    def run(job, tool, args, *, cancelled):
        assert callable(cancelled)
        [effect] = s.store.in_states(["executando"])
        assert effect.action == "pdf_png"
        assert tool == "pdftoppm"
        assert args[:10] == ["-f", args[1], "-l", args[1], "-singlefile", "-png", "-r", "150", "-scale-to", "2048"]
        assert len(args) == 12 and Path(args[10]).parent == job.input_dir
        assert Path(args[11]).parent == job.output_dir
        (Path(args[11] + ".png")).write_bytes(png())
        return pdf_runner.ToolRun("exited", 0, .1, b"", "", [])

    def publish(job, files, name):
        s.files = {p: (job.output_dir / p).read_bytes() for p in files}
        return pdf_publish.Published("published", str(s.home / "Documents/Cortex/PDF" / name),
                                     tuple(files), sum(map(len, s.files.values())), "")

    s.run, s.publish = Mock(side_effect=run), Mock(side_effect=publish)
    monkeypatch.setattr(pdf_runner, "run_tool", s.run)
    monkeypatch.setattr(pdf_publish, "publish", s.publish)
    return s


def phrase(lang="pt", spec=None):
    return (f"export {spec or 'pages 3 to 5'} from ~/Downloads/prova.pdf as images" if lang == "en" else
            f"salva {spec or 'as páginas 3 a 5'} de ~/Downloads/prova.pdf como imagem")


def stage(s, stream=False, lang="pt", spec=None):
    reply, result = _turn(s.o, phrase(lang, spec), stream, lang)
    assert result == "AGENT_AWAITING_CONFIRMATION"
    assert s.o._pending_action["type"] == "pdf_png"
    return reply


PT_FORMS = [f"{verb}{spec} {join} {{path}} {tail}" for verb in
    ("", "salva ", "salve ", "transforma ", "transforme ", "converte ", "converta ", "exporta ", "exporte ")
    for spec, join in [("a página 4", "de"), ("as páginas 3 a 5", "do"), ("3-5 e 8", "da")]
    for tail in ("como imagem", "como imagens", "em imagem", "em imagens", "em png", "como png")]
EN_FORMS = [f"{verb}pages 3-5 {join} {{path}} {tail}" for verb in ("", "save ", "export ", "convert ")
    for join in ("of", "from") for tail in ("as image", "as images", "as png", "to png")]
FRONT_FORMS = [f"{word} {join} as páginas 3 a 5 de {{path}}" for word in ("imagem", "imagens", "png", "pngs")
    for join in ("da", "das")] + [f"{word} of pages 3-5 {join} {{path}}" for word in
    ("image", "images", "png") for join in ("of", "from")]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("template", PT_FORMS + EN_FORMS + FRONT_FORMS)
def test_phrase_forms(fake, stream, template):
    text = "  " + template.format(path="~/Downloads/prova.pdf") + "!? "
    # Case-insensitive words must not change the filesystem path.
    text = text.replace("pages", "PAGES").replace("salva", "SALVA")
    assert _turn(fake.o, text, stream)[1] == "AGENT_AWAITING_CONFIRMATION"
    action = fake.o._pending_action
    args = action["args"]
    assert set(args) == {"input", "keep", "label", "name", "lang"}
    assert args["input"] == {"path": str(fake.path), "pages": 12,
        "sha256": hashlib.sha256(fake.path.read_bytes()).hexdigest()}
    assert args["keep"] == ([4] if "página 4" in text else [3, 4, 5, 8] if "e 8" in text else [3, 4, 5])
    assert args["lang"] == ("en" if template in EN_FORMS + FRONT_FORMS[8:] else "pt")
    assert json.loads(Path(fake.o._PENDING_ACTION_FILE).read_text()) == action
    fake.run.assert_not_called()


@pytest.mark.parametrize("spec,label,pages", [("página 4", "4", [4]), ("page 4", "4", [4]),
    ("8 e 5, 3 a 4 e 3", "3-5, 8", [3, 4, 5, 8]), ("pages 3, 5 and 7", "3, 5, 7", [3, 5, 7])])
def test_page_parser(fake, spec, label, pages):
    stage(fake, spec=spec)
    assert fake.o._pending_action["args"]["label"] == label
    assert fake.o._pending_action["args"]["keep"] == pages


@pytest.mark.parametrize("text", ["a página 4 de {path}", "separa as páginas 3 de {path}",
    "salva a página 4 de {path}", "por favor salva a página 4 de {path} como imagem",
    "salva a página 4 de {path} como imagem por favor", "imagem da página 4 de {path} depois",
    "salva a página 4 com {path} como imagem", "imagem da página 4 de {path} e {path}"])
def test_other_routes(fake, text):
    text = text.format(path="~/Downloads/prova.pdf")
    assert fake.o._handle_pdf_png_phrase(text, path_route.find_paths(text)) is None
    if text.startswith("separa"):
        assert _turn(fake.o, text, False)[1] == "AGENT_AWAITING_CONFIRMATION"
        assert fake.o._pending_action["type"] == "pdf_split"


@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("spec", ["0", "5-3", "2 a 0", "three", "3 e", "3,,5", "3.5", "-2", "3 5"])
def test_bad_specs(fake, lang, spec):
    assert _turn(fake.o, phrase(lang, spec), False)[0] == (
        "Não entendi as páginas. Exemplo: salva as páginas 3 a 5 de ~/x.pdf como imagem." if lang == "pt" else
        "I didn't understand the pages. Example: save pages 3 to 5 from ~/x.pdf as images.")
    assert fake.o._pending_action is None


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("count", [1, 3])
def test_preview_success_once(fake, stream, lang, count):
    label = "4" if count == 1 else "3-5"
    action = (f"imagem da página {label}" if count == 1 else f"imagens das páginas {label}") if lang == "pt" else (
        f"image of page {label}" if count == 1 else f"images of pages {label}")
    name = pdf_publish.folder_name("prova.pdf", action, date.today())
    (fake.home / "Documents/Cortex/PDF" / name).mkdir(parents=True)
    name += " (2)"
    spec = ("a página 4" if count == 1 else "as páginas 3 a 5") if lang == "pt" else (
        "page 4" if count == 1 else "pages 3-5")
    pages = ("a página 4" if count == 1 else "as páginas 3-5") if lang == "pt" else (
        "page 4" if count == 1 else "pages 3-5")
    noun = ("imagem" if count == 1 else "imagens") if lang == "pt" else ("image" if count == 1 else "images")
    expected = (f"Salvar como imagem (PNG, 2048 px) {pages} de prova.pdf?\n"
        f"Origem: prova.pdf (12 páginas) — {fake.path.parent}\n"
        f"Resultado: {count} {noun} em ~/Documents/Cortex/PDF/{name}/\n"
        "O original não muda.\nResponda 'sim' (exatamente essa palavra) para salvar ou 'não' para cancelar."
        if lang == "pt" else f"Save as image (PNG, 2048 px) {pages} of prova.pdf?\n"
        f"Source: prova.pdf (12 pages) — {fake.path.parent}\n"
        f"Result: {count} {noun} in ~/Documents/Cortex/PDF/{name}/\n"
        "The original doesn't change.\nReply 'yes' (exactly that word) to save them or 'no' to cancel.")
    assert stage(fake, stream, lang, spec) == expected
    args = fake.o._pending_action["args"]
    action_id = fake.o._pending_action["id"]
    assert risk_level("pdf_png") == S3
    preview = build_preview("pdf_png", args, lang=lang)
    assert preview["llm"] == "[SYSTEM ERROR] pdf_png requires the typed phrase."
    assert preview["cancel_label"] == ("a gravação das imagens" if lang == "pt" else "save the images")
    before = fake.path.read_bytes()
    done = (f"Pronto: {count} {noun} em ~/Documents/Cortex/PDF/{name}/. O original não mudou." if lang == "pt" else
            f"Done: {count} {noun} in ~/Documents/Cortex/PDF/{name}/. The original didn't change.")
    assert _turn(fake.o, "sim" if lang == "pt" else "yes", stream, lang) == (done, "AGENT_ACTION_DONE")
    assert fake.path.read_bytes() == before and fake.run.call_count == count
    assert list(fake.files) == [f"{'página' if lang == 'pt' else 'page'} {p}.png" for p in args["keep"]]
    for call, page in zip(fake.run.call_args_list, args["keep"]):
        job, tool, argv = call.args
        assert argv == ["-f", str(page), "-l", str(page), "-singlefile", "-png", "-r", "150", "-scale-to", "2048",
                        str(job.input_dir / "input-1.pdf"), str(job.output_dir / f"{'página' if lang == 'pt' else 'page'} {page}")]
    [effect] = fake.store.recent()
    assert effect.id == action_id and effect.state == "feito"
    assert effect.args == {"input": str(fake.path), "pages": args["keep"], "name": name}
    _turn(fake.o, "sim", stream, lang)
    assert fake.run.call_count == count and len(fake.store.recent()) == 1


@pytest.mark.parametrize("args", [None, [], [1], "odd", 1, True, {}, {"input": None},
    {"input": 3, "keep": 7}, {"input": {"path": None, "pages": True}, "keep": "3", "name": 3},
    {"input": {"path": [], "pages": "12"}, "keep": [None, {}, True], "label": []}])
@pytest.mark.parametrize("lang", ["pt", "en"])
def test_preview_total(args, lang):
    preview = build_preview("pdf_png", args, lang=lang)
    assert isinstance(preview["human"], str)
    assert preview["llm"] == "[SYSTEM ERROR] pdf_png requires the typed phrase."


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("case", ["folder", "missing", "nonpdf", "unreadable", "voice", "background", "size", "names", "above"])
def test_refusals(fake, monkeypatch, stream, lang, case):
    text = phrase(lang)
    if case in ("folder", "nonpdf"):
        if case == "nonpdf":
            fake.path.with_suffix(".txt").write_text("fictional")
        text = text.replace("/prova.pdf", "" if case == "folder" else "/prova.txt")
        expected = "Só salvo páginas de um arquivo .pdf como imagem." if lang == "pt" else "I only save pages from a .pdf file as images."
    elif case == "missing":
        text = text.replace("prova.pdf", "missing.pdf")
        expected = "missing.pdf"
    elif case == "unreadable":
        monkeypatch.setattr("src.tools.file_finder.is_readable_path", lambda p: (False, "blocked"))
        expected = "não posso ler prova.pdf: blocked." if lang == "pt" else "I can't read prova.pdf: blocked."
    elif case in ("voice", "background"):
        if case == "voice":
            fake.o.set_turn_origin("voice", "conversation")
        else:
            fake.o.background = True
        expected = ("Salvar páginas como imagem só pedindo digitado, no terminal ou na tela." if lang == "pt" else
                    "Saving pages as images only works when typed, in the terminal or on the screen.")
    elif case == "size":
        monkeypatch.setattr(pdf_runner, "MAX_INPUT_BYTES", 1)
        expected = ("prova.pdf passa de 50 MB — grande demais para salvar páginas como imagem." if lang == "pt" else
                    "prova.pdf exceeds 50 MB — too large to save pages as images.")
    elif case == "above":
        text = phrase(lang, "13")
        expected = ("prova.pdf tem 12 páginas — a página 13 não existe." if lang == "pt" else
                    "prova.pdf has 12 pages — page 13 doesn't exist.")
    else:
        monkeypatch.setattr(pdf_publish, "next_free", lambda name: None)
        expected = ("Já existem 99 pastas com esse nome em ~/Documents/Cortex/PDF — apague algumas e peça de novo." if lang == "pt" else
                    "There are already 99 folders with that name in ~/Documents/Cortex/PDF — delete some and ask again.")
    assert expected in _turn(fake.o, text, stream, lang)[0]
    assert fake.o._pending_action is None and fake.store.recent() == []
    fake.run.assert_not_called()


@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("status", ["protected", "broken", "unavailable", "error", "timeout", "stopped", "too_big", "ok"])
def test_inspection(fake, lang, status):
    fake.inspect.side_effect = None
    fake.inspect.return_value = pdf_info.PdfInfo(status, 12, True, ("copy",), (), 0, "")
    replies = {
        "protected": ("prova.pdf tem senha — não salvo páginas de PDF com senha como imagem.",
                      "prova.pdf has a password — I don't save pages from password-protected PDFs as images."),
        "broken": ("prova.pdf parece danificado — não consegui abrir.", "prova.pdf looks damaged — I couldn't open it."),
        "unavailable": ("As ferramentas de PDF (Poppler) não estão disponíveis neste Mac.", "The PDF tools (Poppler) aren't available on this Mac.")}
    if status == "ok":  # owner restrictions alone are allowed
        stage(fake, lang=lang)
    else:
        expected = replies.get(status, ("Não consegui conferir prova.pdf — nada foi feito.", "I couldn't check prova.pdf — nothing was done."))
        assert _turn(fake.o, phrase(lang), False, lang) == (expected[lang == "en"], "PDF_PNG_REFUSED")
        assert fake.o._pending_action is None
    assert fake.store.recent() == []
    fake.run.assert_not_called()


@pytest.mark.parametrize("count", [10, 11, 12])
@pytest.mark.parametrize("lang", ["pt", "en"])
def test_page_limit(fake, count, lang):
    if count == 10:
        stage(fake, lang=lang, spec="1-10")
        assert _turn(fake.o, "sim", False, lang)[1] == "AGENT_ACTION_DONE"
        assert fake.run.call_count == 10
    else:
        assert _turn(fake.o, phrase(lang, f"1-{count}"), False, lang) == ((
            f"Faço até 10 páginas por vez como imagem — você pediu {count}." if lang == "pt" else
            f"I do up to 10 pages at a time as images — you asked for {count}."), "PDF_PNG_REFUSED")
        assert fake.o._pending_action is None and fake.store.recent() == []
        fake.run.assert_not_called()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("word", ["ok", "pode", "sim, pode", "yes please", "sim..", "yes!!", "'sim'"])
def test_exact_approval(fake, stream, lang, word):
    stage(fake, stream, lang)
    assert _turn(fake.o, word, stream, lang) == ((
        "Para salvar as imagens, responda exatamente 'sim' (ou 'não' para cancelar)." if lang == "pt" else
        "To save the images, reply exactly 'yes' (or 'no' to cancel)."), "AWAITING_CONFIRMATION")
    fake.run.assert_not_called()
    assert fake.store.recent() == []


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("word", ["sim", "Sim.", "sim!", "yes", "yes.", "yes!"])
def test_approval_runs_once(fake, stream, word):
    lang = "pt" if word.lower().startswith("sim") else "en"
    stage(fake, stream, lang, spec="4")
    assert _turn(fake.o, word, stream, lang)[1] == "AGENT_ACTION_DONE"
    fake.run.assert_called_once()
    fake.publish.assert_called_once()
    [effect] = fake.store.recent()
    assert effect.action == "pdf_png" and effect.state == "feito"
    _turn(fake.o, word, stream, lang)
    fake.run.assert_called_once()
    fake.publish.assert_called_once()
    assert len(fake.store.recent()) == 1


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("word", ["sim", "Sim.", "yes!"])
@pytest.mark.parametrize("origin", [("voice", "conversation"), ("ui", "elsewhere"), ("terminal", None)])
def test_origin(fake, stream, word, origin):
    stage(fake, stream)
    fake.o.set_turn_origin(*origin)
    assert _turn(fake.o, word, stream)[1] == "AGENT_ACTION_ELSEWHERE"
    assert fake.store.recent() == []
    fake.run.assert_not_called()


@pytest.mark.parametrize("stream", [False, True])
def test_changed_before_approval(fake, stream):
    stage(fake, stream)
    write_pdf(fake.path, 6)
    assert _turn(fake.o, "sim", stream) == (
        "prova.pdf mudou desde a prévia — peça de novo. Nada foi feito.", "PDF_PNG_REFUSED")
    assert fake.store.recent() == []
    fake.run.assert_not_called()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("case", ["size", "not_png", "bad_ihdr", "truncated", "missing", "extra", "linked", "directory",
    "stopped", "timeout", "exit", "occupied", "publish_error", "staged_changed", "stop_after_render"])
def test_execution_failures(fake, monkeypatch, stream, case):
    stage(fake, stream)
    original = fake.run.side_effect
    delivery = Delivery()
    fake.o.set_turn_delivery(delivery)
    if case in ("occupied", "publish_error"):
        fake.publish.side_effect = lambda *a: pdf_publish.Published("occupied" if case == "occupied" else "error", None, (), 0, "")
    elif case == "staged_changed":
        stage_copy = pdf_runner.PdfJob.stage
        def change(job, paths):
            staged = stage_copy(job, paths)
            write_pdf(staged[0], 6)
            return staged
        monkeypatch.setattr(pdf_runner.PdfJob, "stage", change)
    else:
        def fail(job, tool, args, **kw):
            if case in ("stopped", "timeout", "exit"):
                return pdf_runner.ToolRun("exited" if case == "exit" else case, 1, .1, b"", "", [])
            result = original(job, tool, args, **kw)
            output = Path(args[-1] + ".png")
            if case in ("size", "not_png", "bad_ihdr", "truncated"):
                output.write_bytes({"size": png(1024, 512), "not_png": b"not a PNG", "bad_ihdr": png().replace(b"IHDR", b"BAD!"),
                                    "truncated": png()[:20]}[case])
            elif case in ("missing", "linked", "directory"):
                output.unlink()
                if case == "linked":
                    output.symlink_to(fake.path)
                elif case == "directory":
                    output.mkdir()
            elif case == "extra":
                (job.output_dir / "unexpected.png").write_bytes(png())
            elif case == "stop_after_render":
                delivery.cancelled.set()
            return result
        fake.run.side_effect = fail
    reply, mode = _turn(fake.o, "sim", stream)
    assert mode == "AGENT_ACTION_ERROR"
    assert fake.store.recent()[0].state == "falhou"
    if case == "occupied":
        assert "apareceu enquanto isso" in reply
    elif case in ("stopped", "stop_after_render"):
        assert reply == "Parei — nada foi gravado."
    elif case != "staged_changed":
        assert reply == "Não consegui salvar as imagens — nada foi gravado."
    if case not in ("occupied", "publish_error"):
        fake.publish.assert_not_called()


@pytest.mark.parametrize("entry", ["_execute_tool", "_execute_tool_unstaged", "_skill_macro_stage_step", "_gate_background_action"])
@pytest.mark.parametrize("plan", [False, True])
def test_8b_guard(fake, entry, plan):
    name, args = "pdf_png", {}
    if plan:
        name, args = "execute_plan", {"passos": [{"tool": "execute_plan", "args": {"passos": [{"tool": name, "args": args}]}}]}
    assert getattr(fake.o, entry)(name, args).endswith("[SYSTEM ERROR] pdf_png requires the typed phrase.")
    assert fake.o._pending_action is None and fake.store.recent() == []
    fake.run.assert_not_called()


def test_nested_plan_cannot_persist(fake):
    fake.o._pending_action = {"type": "execute_plan", "args": {"passos": [{"tool": "pdf_png", "args": {}}]}}
    fake.o._persist_pending_action()
    assert fake.o._pending_action is None and not Path(fake.o._PENDING_ACTION_FILE).exists()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("case", ["stop_before", "unreadable", "unrecorded", "background"])
def test_preexecution(fake, monkeypatch, stream, case):
    stage(fake, stream)
    if case == "stop_before":
        delivery = Delivery()
        delivery.cancelled.set()
        fake.o.set_turn_delivery(delivery)
        expected = "Parei antes de executar — não fiz nada."
    elif case == "unreadable":
        monkeypatch.setattr("src.tools.file_finder.is_readable_path", lambda p: (False, "blocked"))
        expected = "não posso ler prova.pdf: blocked."
    elif case == "background":
        fake.o.background = True
        expected = "Salvar páginas como imagem só pedindo digitado, no terminal ou na tela."
    else:
        monkeypatch.setattr(fake.store, "create", lambda *a, **kw: None)
        expected = "Não consegui registrar essa ação antes de executar — não fiz nada."
    assert _turn(fake.o, "sim", stream)[0] == expected
    fake.run.assert_not_called()
    fake.publish.assert_not_called()
    assert fake.store.recent()[0].state == "cancelado" if case == "stop_before" else fake.store.recent() == []


@pytest.mark.parametrize("lang", ["pt", "en"])
def test_terminal_restore_cancel(fake, lang):
    fake.o.set_turn_origin("terminal")
    stage(fake, lang=lang)
    fake.o._pending_action = fake.o._load_pending_action()
    reply, _ = _turn(fake.o, "não" if lang == "pt" else "no", False, lang)
    assert ("a gravação das imagens" if lang == "pt" else "save the images") in reply
    stage(fake, lang=lang)
    assert _turn(fake.o, "sim", False, lang)[1] == "AGENT_ACTION_DONE"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang", ["pt", "en"])
def test_single_page_source_and_front_phrase(fake, stream, lang):
    write_pdf(fake.path, 1)
    text = ("imagem da página 1 de ~/Downloads/prova.pdf" if lang == "pt" else
            "image of page 1 from ~/Downloads/prova.pdf")
    reply, mode = _turn(fake.o, text, stream, lang)
    assert mode == "AGENT_AWAITING_CONFIRMATION"
    assert ("(1 página)" if lang == "pt" else "(1 page)") in reply
    assert _turn(fake.o, "sim", stream, lang)[1] == "AGENT_ACTION_DONE"
    assert fake.run.call_count == 1


@pytest.mark.parametrize("spec", ["1-99999999999999999999", "99999999999999999999"])
def test_huge_page_numbers(fake, spec):
    assert _turn(fake.o, phrase(spec=spec), False)[1] == "PDF_PNG_REFUSED"
    assert fake.o._pending_action is None and fake.store.recent() == []
    fake.run.assert_not_called()


def test_sorted_deduplicated_output(fake):
    stage(fake, spec="8 e 5, 3 a 4 e 3")
    assert _turn(fake.o, "sim", False)[1] == "AGENT_ACTION_DONE"
    assert [call.args[2][1] for call in fake.run.call_args_list] == ["3", "4", "5", "8"]
    assert list(fake.files) == [f"página {p}.png" for p in (3, 4, 5, 8)]


@pytest.mark.parametrize("width,height,valid", [(2048, 1024, True), (1024, 2048, True),
    (2047, 1000, True), (1000, 2049, True), (2046, 1000, False), (2050, 1000, False),
    (0, 2048, False)])
def test_png_dimensions(fake, width, height, valid):
    stage(fake, spec="4")
    original = fake.run.side_effect
    def render(job, tool, args, **kw):
        result = original(job, tool, args, **kw)
        Path(args[-1] + ".png").write_bytes(png(width, height))
        return result
    fake.run.side_effect = render
    assert _turn(fake.o, "sim", False)[1] == ("AGENT_ACTION_DONE" if valid else "AGENT_ACTION_ERROR")
    assert fake.store.recent()[0].state == ("feito" if valid else "falhou")
    if not valid:
        fake.publish.assert_not_called()


@pytest.mark.parametrize("patch", [{"keep": list(range(1, 12))}, {"keep": []}, {"keep": [True]},
    {"keep": [0]}, {"keep": [13]}, {"keep": [3, 3]}, {"keep": [5, 3]}, {"keep": "3"},
    {"input": None}, {"input": {"path": "x.pdf", "sha256": "bad", "pages": 12}},
    {"name": "../escape"}, {"name": "."}, {"name": None}])
def test_forged_pending_arguments(fake, patch):
    stage(fake)
    args = dict(fake.o._pending_action["args"], **patch)
    assert fake.o._run_pdf_png(args) == ("Não consegui salvar as imagens — nada foi gravado.", "AGENT_ACTION_ERROR")
    assert fake.store.recent() == []
    fake.run.assert_not_called()


def test_resolved_path_changed_before_approval(fake, monkeypatch):
    stage(fake)
    monkeypatch.setattr("src.tools.file_finder.is_readable_path", lambda p: (True, str(fake.path.with_name("other.pdf"))))
    assert _turn(fake.o, "sim", False) == (
        "prova.pdf mudou desde a prévia — peça de novo. Nada foi feito.", "PDF_PNG_REFUSED")
    assert fake.store.recent() == []
    fake.run.assert_not_called()


@pytest.mark.skipif(os.environ.get("CORTEX_REAL_SANDBOX_TESTS") != "1"
                   or not Path(pdf_sandbox.SANDBOX_EXEC).exists() or pdf_sandbox.discover() is None,
                   reason="requires opt-in real sandbox-exec and Poppler")
@pytest.mark.parametrize("stream", [False, True])
def test_real_png(setup, stream):
    if not pdf_sandbox.available():
        pytest.skip("real sandbox unavailable in this environment")
    s = setup
    write_pdf(s.path, 3)
    before = s.path.read_bytes()
    stage(s, stream, spec="as páginas 1 e 3")
    name = s.o._pending_action["args"]["name"]
    assert _turn(s.o, "sim", stream)[1] == "AGENT_ACTION_DONE"
    folder = s.home / "Documents/Cortex/PDF" / name
    assert {p.name for p in folder.iterdir()} == {"página 1.png", "página 3.png"}
    for path in folder.iterdir():
        data = path.read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR"
        assert abs(max(struct.unpack(">II", data[16:24])) - 2048) <= 1
    assert s.path.read_bytes() == before
    [effect] = s.store.recent()
    assert effect.action == "pdf_png" and effect.state == "feito"
