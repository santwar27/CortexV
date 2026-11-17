"""Inventory first, typed approval, bounded extraction of real PDF images."""
from datetime import date
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest
from pypdf import PdfReader, PdfWriter
from pypdf.generic import (DecodedStreamObject, DictionaryObject,
                           NameObject, NumberObject)

from src.core import path_route, pdf_info, pdf_publish, pdf_runner, pdf_sandbox
from src.core.staging_mixin import Delivery
from src.core.tool_risk import S3, build_preview, risk_level
from tests.test_claim_remove_failure import _turn
from tests.test_pdf_split import setup, write_pdf


HEADER = ("page num type width height color comp bpc enc interp object ID x-ppi y-ppi size ratio\n"
          "--------------------------------------------------------------------------------------------\n")


def row(page=3, num=0, kind="image", width=200, height=200, obj=10, enc="jpeg"):
    return f"{page} {num} {kind} {width} {height} rgb 3 8 {enc} no {obj} 0 72 72 1.2K 10%"


def inventory(rows):
    return (HEADER + "\n".join(rows) + "\n").encode()


def jpeg200():
    """A baseline 200x200 gray JPEG: 625 constant 8x8 blocks, no codec needed."""
    import struct
    def segment(marker, data):
        return b"\xff" + bytes([marker]) + struct.pack(">H", len(data) + 2) + data
    # Each block is DC difference zero (code 0), then AC end-of-block (code 0).
    # The final partial byte is padded with ones, per JPEG's entropy coding.
    huffman = b"\x01" + b"\0" * 15 + b"\0"
    return (b"\xff\xd8" + segment(0xdb, b"\0" + b"\x01" * 64)
            + segment(0xc0, bytes([8, 0, 200, 0, 200, 1, 1, 0x11, 0]))
            + segment(0xc4, b"\0" + huffman + b"\x10" + huffman)
            + segment(0xda, bytes([1, 1, 0, 0, 63, 0]))
            + b"\0" * 156 + b"\x3f\xff\xd9")


def phrase(lang="pt", spec=None):
    return (("extract the images" + (f" from {spec}" if spec else "") + " of ") if lang == "en" else
            ("tira as imagens" + (f" das {spec}" if spec else "") + " de ")) + "~/Downloads/prova.pdf"


@pytest.fixture
def fake(setup, monkeypatch):
    s = setup
    s.rows = [row(), row(num=1, kind="smask", obj=11), row(num=2, width=32, obj=12),
              row(num=3, obj=10), row(page=4, num=4, obj=13, enc="image")]
    s.inspect = Mock(side_effect=lambda path, **kw: pdf_info.PdfInfo(
        "ok", len(PdfReader(path).pages), False, (), (), 0, ""))
    monkeypatch.setattr(pdf_info, "inspect", s.inspect)

    def run(job, tool, args, *, cancelled):
        assert tool == "pdfimages" and callable(cancelled)
        listing = args[0] == "-list"
        flags = args[1:-1] if listing else args[3:-2]
        assert not flags or flags == ["-f", flags[1], "-l", flags[3]]
        selected = [r.split() for r in s.rows if not flags or int(flags[1]) <= int(r.split()[0]) <= int(flags[3])]
        selected = [dict(enumerate(r)) for r in selected]
        for num, r in enumerate(selected):
            r[1] = str(num)
        assert Path(args[-1] if listing else args[-2]).parent == job.input_dir
        if listing:
            return pdf_runner.ToolRun("exited", 0, .1,
                inventory([" ".join(r.values()) for r in selected]), "", [])
        assert args[:3] == ["-j", "-png", "-p"]
        # Three fixed extraction flags precede optional -f/-l.
        assert len(s.store.in_states(["executando"])) == 1
        for r in selected:
            ext = ".jpg" if r[8] == "jpeg" else ".png"
            Path(args[-1] + f"-{int(r[0]):03d}-{int(r[1]):03d}" + ext).write_bytes(
                b"\xff\xd8fictional jpeg" if ext == ".jpg" else b"\x89PNG\r\n\x1a\nfictional png")
        return pdf_runner.ToolRun("exited", 0, .1, b"", "", [])

    def publish(job, files, name):
        s.files = {p: (job.output_dir / p).read_bytes() for p in files}
        s.private_files = {p.name for p in job.output_dir.iterdir()}
        return pdf_publish.Published("published", str(s.home / "Documents/Cortex/PDF" / name),
                                     tuple(files), sum(map(len, s.files.values())), "")

    s.run, s.publish = Mock(side_effect=run), Mock(side_effect=publish)
    monkeypatch.setattr(pdf_runner, "run_tool", s.run)
    monkeypatch.setattr(pdf_publish, "publish", s.publish)
    return s


def stage(s, stream=False, lang="pt", spec=None):
    reply, mode = _turn(s.o, phrase(lang, spec), stream, lang)
    assert mode == "AGENT_AWAITING_CONFIRMATION"
    assert s.o._pending_action["type"] == "pdf_images"
    return reply


PT = [f"{v} {a}{n}{spec} {join} {{path}}" for v in
      ("tira", "tire", "extrai", "extraia", "salva", "salve", "pega", "pegue")
      for a in ("", "as ", "todas as ") for n in ("imagens", "fotos", "figuras")
      for spec, join in (("", "de"), (" das páginas 3 a 5", "do"), (" das 3, 5 e 8", "da"))]
EN = [f"{v} {a}{n}{spec} {join} {{path}}" for v in ("extract", "save", "get")
      for a in ("", "all ", "the ", "all the ") for n in ("images", "pictures")
      for spec, join in (("", "from"), (" from pages 3-5", "of"), (" of 3, 5 and 8", "from"))]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("template", PT + EN)
def test_phrases(fake, stream, template):
    text = "  " + template.format(path="~/Downloads/prova.pdf").replace("tira", "TIRA") + "!? "
    assert _turn(fake.o, text, stream)[1] == "AGENT_AWAITING_CONFIRMATION"
    action = fake.o._pending_action
    assert action["type"] == "pdf_images" and action["args"]["lang"] == ("en" if template in EN else "pt")
    assert json.loads(Path(fake.o._PENDING_ACTION_FILE).read_text()) == action
    assert all(c.args[2][0] == "-list" for c in fake.run.call_args_list)
    assert fake.store.recent() == []


def test_split_and_nonphrases(fake):
    assert _turn(fake.o, "tira as páginas 3 a 5 de ~/Downloads/prova.pdf", False)[1] == "PDF_SPLIT_QUESTION"
    for text in ("por favor tira as imagens de {path}", "tira as imagens de {path} depois",
                 "tira as imagens com {path}", "tira as imagens de {path} e {path}"):
        text = text.format(path="~/Downloads/prova.pdf")
        assert fake.o._handle_pdf_images_phrase(text, path_route.find_paths(text)) is None


@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("spec", [None, "pages 3-5", "page 3"])
@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("stream", [False, True])
def test_preview_and_success(fake, lang, spec, count, stream):
    fake.rows = [row(num=i, obj=10+i, enc="jpeg" if i == 0 else "image") for i in range(count)]
    if lang == "pt" and spec:
        spec = spec.replace("pages", "páginas").replace("page", "página")
    label = "3-5" if spec and "5" in spec else "3" if spec else ""
    action = ("imagens extraídas" + (f" das páginas {label}" if label else "")) if lang == "pt" else (
        "extracted images" + (f" of pages {label}" if label else ""))
    base = pdf_publish.folder_name("prova.pdf", action, date.today())
    folder = fake.home / "Documents/Cortex/PDF" / base
    folder.mkdir(parents=True)
    name = base + " (2)"
    noun = ("imagem" if count == 1 else "imagens") if lang == "pt" else ("image" if count == 1 else "images")
    expected = (f"Tirar as imagens{' das páginas ' + label if label else ''} de prova.pdf?\n"
        f"Origem: prova.pdf (12 páginas) — {fake.path.parent}\n"
        f"Encontrei {count} {noun} (fora máscaras e ícones menores que 64 px). JPEG fica .jpg, o resto .png.\n"
        f"Resultado: {count} {'arquivo' if count == 1 else 'arquivos'} em ~/Documents/Cortex/PDF/{name}/\n"
        "O original não muda.\nResponda 'sim' (exatamente essa palavra) para tirar ou 'não' para cancelar."
        if lang == "pt" else f"Take the images out of {'pages ' + label + ' of ' if label else ''}prova.pdf?\n"
        f"Source: prova.pdf (12 pages) — {fake.path.parent}\n"
        f"Found {count} {noun} (leaving out masks and icons under 64 px). JPEG stays .jpg, the rest .png.\n"
        f"Result: {count} {'file' if count == 1 else 'files'} in ~/Documents/Cortex/PDF/{name}/\n"
        "The original doesn't change.\nReply 'yes' (exactly that word) to take them out or 'no' to cancel.")
    assert stage(fake, stream, lang, spec) == expected
    args = fake.o._pending_action["args"]
    action_id = fake.o._pending_action["id"]
    assert args["input"] == {"path": str(fake.path), "pages": 12,
                             "sha256": hashlib.sha256(fake.path.read_bytes()).hexdigest()}
    assert risk_level("pdf_images") == S3
    preview = build_preview("pdf_images", args, lang=lang)
    assert preview["llm"] == "[SYSTEM ERROR] pdf_images requires the typed phrase."
    assert preview["cancel_label"] == ("a extração das imagens" if lang == "pt" else "take the images out")
    before = fake.path.read_bytes()
    done = (f"Pronto: {count} {noun} em ~/Documents/Cortex/PDF/{name}/. O original não mudou." if lang == "pt" else
            f"Done: {count} {noun} in ~/Documents/Cortex/PDF/{name}/. The original didn't change.")
    assert _turn(fake.o, "sim", stream, lang) == (done, "AGENT_ACTION_DONE")
    assert fake.path.read_bytes() == before
    assert list(fake.files) == [f"{'página' if lang == 'pt' else 'page'} 3 - {i+1}{'.jpg' if i == 0 else '.png'}" for i in range(count)]
    [effect] = fake.store.recent()
    assert effect.id == action_id and effect.action == "pdf_images" and effect.state == "feito"
    assert effect.args == {"input": str(fake.path), "pages": args["runs"] or "all", "name": name}


@pytest.mark.parametrize("args", [None, [], 1, True, "odd", {}, {"input": None},
    {"input": {"path": [], "pages": True}, "keep": [None], "label": []}])
@pytest.mark.parametrize("lang", ["pt", "en"])
def test_preview_total(args, lang):
    preview = build_preview("pdf_images", args, lang=lang)
    assert isinstance(preview["human"], str)
    assert preview["llm"] == "[SYSTEM ERROR] pdf_images requires the typed phrase."


def test_filter_and_runs(fake):
    fake.rows += [row(page=8, num=5, obj=10), row(page=8, num=6, obj=20),
                  row(page=8, num=7, kind="mask", obj=21), row(page=8, num=8, kind="stencil", obj=22),
                  row(page=8, num=9, height=63, obj=23), row(page=8, num=10, width=64, height=64, obj=24)]
    stage(fake, spec="8 e 3 a 4 e 3")
    args = fake.o._pending_action["args"]
    assert args["runs"] == [[3, 4], [8, 8]] and args["label"] == "3-4, 8"
    assert args["keep"] == [[3, 0, 0], [4, 4, 0], [8, 1, 1], [8, 5, 1]]
    assert _turn(fake.o, "Sim.", False)[1] == "AGENT_ACTION_DONE"
    assert list(fake.files) == ["página 3 - 1.jpg", "página 4 - 1.png", "página 8 - 1.jpg", "página 8 - 2.jpg"]
    assert fake.private_files > set(fake.files)  # discarded masks, icons and duplicates stay private
    expected = []
    for listing in (True, False):
        for k, (a, b) in enumerate(args["runs"]):
            expected.append((listing, k, ["-f", str(a), "-l", str(b)]))
    # Stage lists, approval lists, then extraction, each in document order.
    assert fake.run.call_count == 6
    for call, (listing, k, flags) in zip(fake.run.call_args_list, expected[:2] * 2 + expected[2:]):
        job, tool, argv = call.args
        assert tool == "pdfimages"
        assert argv == (["-list"] + flags + [str(job.input_dir / "input-1.pdf")] if listing else
                        ["-j", "-png", "-p"] + flags + [str(job.input_dir / "input-1.pdf"), str(job.output_dir / f"r{k}")])


@pytest.mark.parametrize("count", [0, 50, 51])
@pytest.mark.parametrize("lang", ["pt", "en"])
def test_limits(fake, count, lang):
    fake.rows = [row(num=i, obj=10+i) for i in range(count)]
    reply, mode = _turn(fake.o, phrase(lang), False, lang)
    if count == 50:
        assert mode == "AGENT_AWAITING_CONFIRMATION"
        assert _turn(fake.o, "yes!", False, lang)[1] == "AGENT_ACTION_DONE"
        assert len(fake.files) == 50
    else:
        assert mode == "PDF_IMAGES_REFUSED" and fake.o._pending_action is None
        assert reply == (("Não achei imagens em prova.pdf (fora máscaras e ícones pequenos) — nada a fazer." if lang == "pt" else
                          "I found no images in prova.pdf (leaving out masks and tiny icons) — nothing to do.") if count == 0 else
            ("prova.pdf tem 51 imagens — consigo até 50 por vez. Escolha páginas, por exemplo: tira as imagens das páginas 3 a 5 de ~/x.pdf." if lang == "pt" else
             "prova.pdf has 51 images — I can do up to 50 at a time. Pick pages, e.g.: extract the images from pages 3 to 5 of ~/x.pdf."))
        assert fake.store.recent() == []
        assert all(c.args[2][0] == "-list" for c in fake.run.call_args_list)


@pytest.mark.parametrize("count", [100, 101])
@pytest.mark.parametrize("lang", ["pt", "en"])
def test_row_limit(fake, count, lang):
    fake.rows = [row(num=i, obj=10 if i == 0 else 11, kind="image" if i == 0 else "smask") for i in range(count)]
    reply, mode = _turn(fake.o, phrase(lang), False, lang)
    if count == 100:
        assert mode == "AGENT_AWAITING_CONFIRMATION"
        assert _turn(fake.o, "sim", False, lang)[1] == "AGENT_ACTION_DONE"
    else:
        assert (reply, mode) == (("Essas páginas têm itens demais para tirar de uma vez — escolha menos páginas." if lang == "pt" else
                                "Those pages hold too many items to take at once — pick fewer pages."), "PDF_IMAGES_REFUSED")
        assert fake.store.recent() == []


@pytest.mark.parametrize("bad", [b"", b"garbage", inventory([row() + " extra"]),
    inventory([row().replace("200 200", "oops 200")]), inventory([row(kind="other")]),
    inventory([row(num=1)]), inventory([row(page=13)]), inventory([row(), row(num=0)]),
    inventory([row().replace("10 0 72", "x 0 72")])])
def test_malformed_inventory(fake, bad):
    fake.run.side_effect = lambda *a, **kw: pdf_runner.ToolRun("exited", 0, .1, bad, "", [])
    assert _turn(fake.o, phrase(), False) == ("Não consegui conferir as imagens de prova.pdf — nada foi feito.", "PDF_IMAGES_REFUSED")
    assert fake.o._pending_action is None and fake.store.recent() == []


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("word", ["Sim.", "yes!"])
def test_approval_once(fake, stream, word):
    stage(fake, stream)
    assert _turn(fake.o, word, stream)[1] == "AGENT_ACTION_DONE"
    assert fake.run.call_count == 3
    _turn(fake.o, word, stream)
    assert fake.run.call_count == 3 and len(fake.store.recent()) == 1


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("word", ["sim, pode", "ok", "yes please", "sim.."])
def test_loose_approval(fake, stream, word):
    stage(fake, stream)
    assert _turn(fake.o, word, stream) == ("Para tirar as imagens, responda exatamente 'sim' (ou 'não' para cancelar).", "AWAITING_CONFIRMATION")
    assert fake.run.call_count == 1 and fake.store.recent() == []


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("origin", [("voice", "conversation"), ("ui", "elsewhere"), ("terminal", None)])
def test_wrong_origin(fake, stream, origin):
    stage(fake, stream)
    fake.o.set_turn_origin(*origin)
    assert _turn(fake.o, "sim", stream)[1] == "AGENT_ACTION_ELSEWHERE"
    assert fake.run.call_count == 1 and fake.store.recent() == []


@pytest.mark.parametrize("stream", [False, True])
def test_changed_file(fake, stream):
    stage(fake, stream)
    write_pdf(fake.path, 2)
    assert _turn(fake.o, "sim", stream) == ("prova.pdf mudou desde a prévia — peça de novo. Nada foi feito.", "PDF_IMAGES_REFUSED")
    assert fake.run.call_count == 1 and fake.store.recent() == []


@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("case", ["folder", "nonpdf", "missing", "unreadable", "voice", "background", "size", "above", "bad_spec", "names"])
def test_refusals(fake, monkeypatch, lang, case):
    text = phrase(lang)
    if case in ("folder", "nonpdf"):
        fake.path.with_suffix(".txt").write_text("fictional")
        text = text.replace("/prova.pdf", "" if case == "folder" else "/prova.txt")
    elif case == "missing":
        text = text.replace("prova.pdf", "missing.pdf")
    elif case == "unreadable":
        monkeypatch.setattr("src.tools.file_finder.is_readable_path", lambda p: (False, "blocked"))
    elif case == "voice":
        fake.o.set_turn_origin("voice", "conversation")
    elif case == "background":
        fake.o.background = True
    elif case == "size":
        monkeypatch.setattr(pdf_runner, "MAX_INPUT_BYTES", 1)
    elif case in ("above", "bad_spec"):
        text = phrase(lang, "13" if case == "above" else "3,,5")
    else:
        monkeypatch.setattr(pdf_publish, "next_free", lambda name: None)
    reply, mode = _turn(fake.o, text, False, lang)
    assert mode != "AGENT_AWAITING_CONFIRMATION" and fake.o._pending_action is None
    if case in ("voice", "background"):
        assert reply == ("Tirar imagens do PDF só pedindo digitado, no terminal ou na tela." if lang == "pt" else
                         "Taking images out of a PDF only works when typed, in the terminal or on the screen.")
    assert fake.store.recent() == []
    assert all(c.args[2][0] == "-list" for c in fake.run.call_args_list)


@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("status", ["protected", "broken", "unavailable", "error", "timeout", "stopped", "too_big", "ok"])
def test_inspection(fake, status, lang):
    fake.inspect.side_effect = None
    fake.inspect.return_value = pdf_info.PdfInfo(status, 12, True, ("copy",), (), 0, "")
    reply, mode = _turn(fake.o, phrase(lang), False, lang)
    assert mode == ("AGENT_AWAITING_CONFIRMATION" if status == "ok" else "PDF_IMAGES_REFUSED")
    assert fake.store.recent() == []
    if status != "ok":
        fake.run.assert_not_called()
        replies = {
            "protected": ("prova.pdf tem senha — não tiro imagens de PDF com senha.",
                          "prova.pdf has a password — I don't take images out of password-protected PDFs."),
            "broken": ("prova.pdf parece danificado — não consegui abrir.", "prova.pdf looks damaged — I couldn't open it."),
            "unavailable": ("As ferramentas de PDF (Poppler) não estão disponíveis neste Mac.",
                            "The PDF tools (Poppler) aren't available on this Mac.")}
        assert reply == replies.get(status, ("Não consegui conferir prova.pdf — nada foi feito.",
                                             "I couldn't check prova.pdf — nothing was done."))[lang == "en"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("case", ["inventory", "signature", "extension", "missing", "linked", "directory", "ambiguous", "occupied", "publish_error",
    "stopped", "timeout", "exit", "staged_changed", "stop_after_extract"])
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
            copied = stage_copy(job, paths)
            write_pdf(copied[0], 2)
            return copied
        monkeypatch.setattr(pdf_runner.PdfJob, "stage", change)
    else:
        def fail(job, tool, args, **kw):
            if case in ("stopped", "timeout", "exit"):
                return pdf_runner.ToolRun("exited" if case == "exit" else case, 1, .1, b"", "", [])
            if case == "inventory":
                fake.rows[0] = row(width=201)
            result = original(job, tool, args, **kw)
            if args[0] != "-list":
                output = Path(args[-1] + "-003-000.jpg")
                if case == "signature":
                    output.write_bytes(b"not jpeg")
                elif case == "extension":
                    output.unlink()
                    output.with_suffix(".png").write_bytes(b"\x89PNG\r\n\x1a\n")
                elif case in ("missing", "linked", "directory"):
                    output.unlink()
                    if case == "linked":
                        output.symlink_to(fake.path)
                    elif case == "directory":
                        output.mkdir()
                elif case == "ambiguous":
                    output.with_suffix(".png").write_bytes(b"\x89PNG\r\n\x1a\n")
                elif case == "stop_after_extract":
                    delivery.cancelled.set()
            return result
        fake.run.side_effect = fail
    reply, mode = _turn(fake.o, "sim", stream)
    assert mode == "AGENT_ACTION_ERROR" and fake.store.recent()[0].state == "falhou"
    if case == "occupied":
        assert "apareceu enquanto isso" in reply
    elif case in ("stopped", "stop_after_extract"):
        assert reply == "Parei — nada foi gravado."
    elif case != "staged_changed":
        assert reply == "Não consegui tirar as imagens — nada foi gravado."
    if case not in ("occupied", "publish_error"):
        fake.publish.assert_not_called()
    if case == "inventory":
        assert all(c.args[2][0] == "-list" for c in fake.run.call_args_list)


@pytest.mark.parametrize("entry", ["_execute_tool", "_execute_tool_unstaged", "_skill_macro_stage_step", "_gate_background_action"])
@pytest.mark.parametrize("plan", [False, True])
def test_8b_guard(fake, entry, plan):
    name, args = "pdf_images", {}
    if plan:
        name, args = "execute_plan", {"passos": [{"tool": "execute_plan", "args": {"passos": [{"tool": name, "args": args}]}}]}
    assert getattr(fake.o, entry)(name, args).endswith("[SYSTEM ERROR] pdf_images requires the typed phrase.")
    assert fake.o._pending_action is None and fake.store.recent() == []
    fake.run.assert_not_called()


@pytest.mark.parametrize("lang", ["pt", "en"])
def test_cancel_restore_terminal(fake, lang):
    fake.o.set_turn_origin("terminal")
    stage(fake, lang=lang)
    fake.o._pending_action = fake.o._load_pending_action()
    assert ("a extração das imagens" if lang == "pt" else "take the images out") in _turn(fake.o, "no", False, lang)[0]
    stage(fake, lang=lang)
    assert _turn(fake.o, "yes!", False, lang)[1] == "AGENT_ACTION_DONE"


@pytest.mark.parametrize("case", ["stop", "background", "unrecorded", "unreadable", "resolved"])
def test_preexecution(fake, monkeypatch, case):
    stage(fake)
    if case == "stop":
        delivery = Delivery()
        delivery.cancelled.set()
        fake.o.set_turn_delivery(delivery)
    elif case == "background":
        fake.o.background = True
    elif case == "unrecorded":
        monkeypatch.setattr(fake.store, "create", lambda *a, **kw: None)
    else:
        monkeypatch.setattr("src.tools.file_finder.is_readable_path", lambda p: (case == "resolved", str(fake.path.with_name("other.pdf"))))
    assert _turn(fake.o, "sim", False)[1] != "AGENT_ACTION_DONE"
    assert fake.run.call_count == 1
    fake.publish.assert_not_called()
    assert fake.store.recent()[0].state == "cancelado" if case == "stop" else fake.store.recent() == []


@pytest.mark.parametrize("patch", [{"runs": [[0, 3]]}, {"runs": [[5, 3]]}, {"runs": [[3, 13]]},
    {"runs": [[3, 5], [4, 6]]}, {"keep": []}, {"keep": [[3, True, 0]]}, {"keep": [[3, 0, 99]]},
    {"inventory": []}, {"input": None}, {"name": "../escape"}, {"name": None}])
def test_forged_args(fake, patch):
    stage(fake)
    assert fake.o._run_pdf_images(dict(fake.o._pending_action["args"], **patch)) == (
        "Não consegui tirar as imagens — nada foi gravado.", "AGENT_ACTION_ERROR")
    assert fake.run.call_count == 1 and fake.store.recent() == []


def test_whole_file_argv(fake):
    stage(fake)
    assert fake.o._pending_action["args"]["runs"] == []
    assert _turn(fake.o, "sim", False)[1] == "AGENT_ACTION_DONE"
    assert fake.run.call_count == 3
    for call in fake.run.call_args_list:
        job, tool, argv = call.args
        staged = str(job.input_dir / "input-1.pdf")
        assert argv == (["-list", staged] if argv[0] == "-list" else
                        ["-j", "-png", "-p", staged, str(job.output_dir / "r0")])


@pytest.mark.parametrize("stream", [False, True])
def test_english_reminder(fake, stream):
    stage(fake, stream, "en")
    assert _turn(fake.o, "ok", stream, "en") == (
        "To take the images out, reply exactly 'yes' (or 'no' to cancel).", "AWAITING_CONFIRMATION")
    assert fake.run.call_count == 1 and fake.store.recent() == []


def test_duplicate_generation_and_page_order(fake):
    fake.rows = [row(), row(page=4, num=1), row(page=4, num=2).replace("10 0 72", "10 1 72")]
    stage(fake)
    assert fake.o._pending_action["args"]["keep"] == [[3, 0, 0], [4, 2, 0]]
    assert _turn(fake.o, "sim", False)[1] == "AGENT_ACTION_DONE"
    assert list(fake.files) == ["página 3 - 1.jpg", "página 4 - 1.jpg"]


@pytest.mark.parametrize("lang", ["pt", "en"])
def test_only_masks_and_tiny(fake, lang):
    fake.rows = [row(kind="smask"), row(num=1, kind="mask"), row(num=2, kind="stencil"),
                 row(num=3, width=63), row(num=4, height=63)]
    reply, mode = _turn(fake.o, phrase(lang), False, lang)
    assert reply == ("Não achei imagens em prova.pdf (fora máscaras e ícones pequenos) — nada a fazer." if lang == "pt" else
                     "I found no images in prova.pdf (leaving out masks and tiny icons) — nothing to do.")
    assert mode == "PDF_IMAGES_REFUSED" and fake.o._pending_action is None
    assert fake.run.call_count == 1 and fake.store.recent() == []


def test_total_rows_across_runs(fake):
    fake.rows = [row(page=3 if i < 50 else 8, num=i, kind="image" if i == 0 else "smask") for i in range(101)]
    assert _turn(fake.o, phrase(spec="3 e 8"), False) == (
        "Essas páginas têm itens demais para tirar de uma vez — escolha menos páginas.", "PDF_IMAGES_REFUSED")
    assert fake.run.call_count == 2 and fake.store.recent() == []


@pytest.mark.parametrize("status", ["timeout", "stopped", "unavailable", "too_big", "error", "exited"])
def test_inventory_failure_at_staging(fake, status):
    fake.run.side_effect = lambda *a, **kw: pdf_runner.ToolRun(status, 1, .1, b"", "", [])
    assert _turn(fake.o, phrase(), False) == ("Não consegui conferir as imagens de prova.pdf — nada foi feito.", "PDF_IMAGES_REFUSED")
    assert fake.o._pending_action is None and fake.store.recent() == []
    fake.publish.assert_not_called()


@pytest.mark.parametrize("case", ["stopped", "timeout", "exit", "png_signature"])
def test_extraction_failure(fake, case):
    stage(fake)
    original = fake.run.side_effect
    def fail(job, tool, argv, **kw):
        if argv[0] == "-list":
            return original(job, tool, argv, **kw)
        if case != "png_signature":
            return pdf_runner.ToolRun("exited" if case == "exit" else case, 1, .1, b"", "", [])
        result = original(job, tool, argv, **kw)
        Path(argv[-1] + "-004-004.png").write_bytes(b"not png")
        return result
    fake.run.side_effect = fail
    reply, mode = _turn(fake.o, "sim", False)
    assert reply == ("Parei — nada foi gravado." if case == "stopped" else "Não consegui tirar as imagens — nada foi gravado.")
    assert mode == "AGENT_ACTION_ERROR" and fake.store.recent()[0].state == "falhou"
    fake.publish.assert_not_called()


def test_nested_plan_not_persisted(fake):
    fake.o._pending_action = {"type": "execute_plan", "args": {"passos": [{"tool": "pdf_images", "args": {}}]}}
    fake.o._persist_pending_action()
    assert fake.o._pending_action is None and not Path(fake.o._PENDING_ACTION_FILE).exists()


@pytest.mark.parametrize("stream", [False, True])
def test_single_page_source(fake, stream):
    write_pdf(fake.path, 1)
    fake.rows = [row(page=1)]
    assert "(1 página)" in stage(fake, stream)
    assert _turn(fake.o, "sim", stream)[1] == "AGENT_ACTION_DONE"


@pytest.mark.skipif(os.environ.get("CORTEX_REAL_SANDBOX_TESTS") != "1",
                   reason="requires opt-in real sandbox-exec and Poppler")
@pytest.mark.parametrize("stream", [False, True])
def test_real_images(setup, stream):
    if not pdf_sandbox.available():
        pytest.skip("real sandbox unavailable in this environment")
    # Fixed, fictitious pixels; no input from the user's documents.
    writer = PdfWriter()
    page = writer.add_blank_page(width=600, height=600)
    resources = DictionaryObject()
    for name, size, jpeg in (("J", 200, True), ("P", 200, False), ("Tiny", 32, False)):
        image = DecodedStreamObject()
        image.update({NameObject("/Type"): NameObject("/XObject"), NameObject("/Subtype"): NameObject("/Image"),
            NameObject("/Width"): NumberObject(size), NameObject("/Height"): NumberObject(size),
            NameObject("/ColorSpace"): NameObject("/DeviceRGB"), NameObject("/BitsPerComponent"): NumberObject(8)})
        if jpeg:
            image[NameObject("/ColorSpace")] = NameObject("/DeviceGray")
            image.set_data(jpeg200())
            image[NameObject("/Filter")] = NameObject("/DCTDecode")
        else:
            image.set_data(bytes((20, 60, 90)) * size * size)
        resources[NameObject("/" + name)] = writer._add_object(image)
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/XObject"): resources})
    stream_obj = DecodedStreamObject()
    stream_obj.set_data(b"q 200 0 0 200 0 0 cm /J Do Q q 200 0 0 200 200 0 cm /P Do Q "
                        b"q 32 0 0 32 0 250 cm /Tiny Do Q q 200 0 0 200 0 350 cm /J Do Q")
    page[NameObject("/Contents")] = writer._add_object(stream_obj)
    writer.write(setup.path)
    before = setup.path.read_bytes()
    stage(setup, stream)
    name = setup.o._pending_action["args"]["name"]
    assert _turn(setup.o, "sim", stream)[1] == "AGENT_ACTION_DONE"
    folder = setup.home / "Documents/Cortex/PDF" / name
    assert {p.name for p in folder.iterdir()} == {"página 1 - 1.jpg", "página 1 - 2.png"}
    assert setup.path.read_bytes() == before
    [effect] = setup.store.recent()
    assert effect.action == "pdf_images" and effect.state == "feito"
