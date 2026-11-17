"""Typed PDF merge: exact approval, contained work, exclusive publication, effects."""
from datetime import date
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pypdf import PdfReader, PdfWriter

from src.core import effect_store, path_route, pdf_info, pdf_publish, pdf_runner, pdf_sandbox
from src.core.staging_mixin import Delivery
from src.core.tool_risk import S3, build_preview, risk_level
from src.tools.local_reader import LocalFileReader
from tests.test_claim_remove_failure import _turn
from tests.test_dispatch_parity import _build_real_orch


def write_pdf(path, pages=2, width=100):
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=width, height=100)
    writer.write(path)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "Documents").mkdir()
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    paths = [downloads / "prova.pdf", downloads / "textos.pdf"]
    for i, path in enumerate(paths):
        write_pdf(path, width=100 + i)
    o = _build_real_orch(tmp_path)
    o.background, o._reply_lang = False, "pt"
    o.set_turn_origin("ui", "conversation")
    o.reader = LocalFileReader(str(tmp_path / "vault"))
    return SimpleNamespace(o=o, paths=paths, store=effect_store.store, home=tmp_path)


@pytest.fixture
def fake(setup, monkeypatch):
    s = setup
    s.inspect = Mock(side_effect=lambda path, **kw: pdf_info.PdfInfo(
        "ok", len(PdfReader(path).pages), False, (), (), 0, ""))
    monkeypatch.setattr(pdf_info, "inspect", s.inspect)

    def run(job, tool, args, *, cancelled):
        assert callable(cancelled)
        [effect] = s.store.in_states(["executando"])
        assert effect.action == "pdf_merge"
        if tool == "pdfunite":
            assert all(Path(p).parent == job.input_dir for p in args[:-1])
            assert Path(args[-1]).parent == job.output_dir
            writer = PdfWriter()
            for path in args[:-1]:
                writer.append(path)
            writer.write(args[-1])
            return pdf_runner.ToolRun("exited", 0, .1, b"", "", [])
        assert tool == "pdfinfo" and Path(args[0]).parent == job.output_dir
        pages = len(PdfReader(args[0]).pages)
        return pdf_runner.ToolRun("exited", 0, .1, f"Pages: {pages}\n".encode(), "", [])

    def publish(job, files, name):
        assert files == [name + ".pdf"]
        assert (job.output_dir / files[0]).is_file()
        return pdf_publish.Published("published", str(s.home / "Documents/Cortex/PDF" / name),
                                     tuple(files), 1, "")

    s.run = Mock(side_effect=run)
    s.publish = Mock(side_effect=publish)
    monkeypatch.setattr(pdf_runner, "run_tool", s.run)
    monkeypatch.setattr(pdf_publish, "publish", s.publish)
    yield s


def phrase(lang="pt"):
    return ("merge ~/Downloads/prova.pdf and ~/Downloads/textos.pdf" if lang == "en"
            else "junta ~/Downloads/prova.pdf e ~/Downloads/textos.pdf")


def stage(s, stream=False, lang="pt"):
    reply, mode = _turn(s.o, phrase(lang), stream, lang)
    assert mode == "AGENT_AWAITING_CONFIRMATION"
    assert s.o._pending_action["type"] == "pdf_merge"
    return reply


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("verb,sep,lang", [(v, sep, "en" if v == "merge" else "pt")
    for v in ("junta", "junte", "juntar", "merge") for sep in (" e ", ", ", " and ", ",")])
def test_phrase(fake, stream, verb, sep, lang):
    text = verb + " ~/Downloads/prova.pdf" + sep + "~/Downloads/textos.pdf!? "
    assert _turn(fake.o, text, stream)[1] == "AGENT_AWAITING_CONFIRMATION"
    action = fake.o._pending_action
    assert action["type"] == "pdf_merge" and action["args"]["lang"] == lang
    assert action["args"]["inputs"] == [
        {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "pages": 2}
        for p in fake.paths]
    assert json.loads(Path(fake.o._PENDING_ACTION_FILE).read_text()) == action
    fake.run.assert_not_called()
    fake.publish.assert_not_called()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("verb,lang", [("Junta", "pt"), ("JUNTA", "pt"),
                                      ("  junta", "pt"), ("Merge", "en"), ("MERGE", "en")])
def test_phrase_prefix_preserves_path_case(fake, stream, verb, lang):
    paths = [p.with_name(p.name.capitalize()) for p in fake.paths]
    for original, renamed in zip(fake.paths, paths):
        original.rename(renamed)
    text = verb + " ~/Downloads/Prova.pdf e ~/Downloads/Textos.pdf"
    assert _turn(fake.o, text, stream)[1] == "AGENT_AWAITING_CONFIRMATION"
    action = fake.o._pending_action
    assert action["type"] == "pdf_merge" and action["args"]["lang"] == lang
    assert [item["path"] for item in action["args"]["inputs"]] == [str(p) for p in paths]
    fake.run.assert_not_called()
    fake.publish.assert_not_called()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("text", ["por favor junta ~/Downloads/prova.pdf e ~/Downloads/textos.pdf",
    "junta ~/Downloads/prova.pdf com ~/Downloads/textos.pdf",
    "junta ~/Downloads/prova.pdf e ~/Downloads/textos.pdf por favor",
    "junta ~/Downloads/prova.pdf ~/Downloads/textos.pdf"])
def test_extra_prose_keeps_old_route(fake, stream, text):
    assert _turn(fake.o, text, stream)[1] == "FILE_ERROR"
    assert fake.o._pending_action is None
    fake.inspect.assert_not_called()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("case", ["one", "four", "folder", "missing", "nonpdf", "unreadable",
                                  "voice", "background"])
def test_refusals(fake, monkeypatch, stream, case):
    text = phrase()
    expected = "Junto de 2 a 3 PDFs por vez: junta ~/a.pdf e ~/b.pdf."
    if case == "one":
        text = "junta ~/Downloads/prova.pdf"
    elif case == "four":
        text += " e ~/Downloads/prova.pdf e ~/Downloads/textos.pdf"
    elif case == "folder":
        text = "junta ~/Downloads e ~/Downloads/prova.pdf"
    elif case == "missing":
        text = "junta ~/Downloads/missing.pdf e ~/Downloads/prova.pdf"
        expected = fake.o._say_path_missing(path_route.find_paths(text)[0])[0]
    elif case == "nonpdf":
        (fake.paths[0].parent / "a.txt").write_text("fictitious")
        text = "junta ~/Downloads/a.txt e ~/Downloads/prova.pdf"
        expected = "Só junto arquivos .pdf."
    elif case == "unreadable":
        monkeypatch.setattr("src.tools.file_finder.is_readable_path", lambda p: (False, "blocked"))
        expected = "não posso ler prova.pdf: blocked."
    elif case == "voice":
        fake.o.set_turn_origin("voice", "conversation")
        expected = "Juntar PDFs só pedindo digitado, no terminal ou na tela."
    else:
        fake.o.background = True
        expected = "Juntar PDFs só pedindo digitado, no terminal ou na tela."
    assert _turn(fake.o, text, stream)[0] == expected
    assert fake.o._pending_action is None and fake.store.recent() == []
    fake.inspect.assert_not_called()


@pytest.mark.parametrize("status,expected", [
    ("protected", "prova.pdf tem senha — não junto PDF com senha."),
    ("broken", "prova.pdf parece danificado — não consegui abrir."),
    ("unavailable", "As ferramentas de PDF (Poppler) não estão disponíveis neste Mac."),
    ("error", "Não consegui conferir prova.pdf — nada foi feito."),
    ("timeout", "Não consegui conferir prova.pdf — nada foi feito.")])
def test_inspection_refusals(fake, status, expected):
    fake.inspect.side_effect = None
    fake.inspect.return_value = pdf_info.PdfInfo(status, None, None, (), (), 0, "")
    assert _turn(fake.o, phrase(), False)[0] == expected
    assert fake.o._pending_action is None and fake.store.recent() == []


@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("args, expected_lines, pages, name", [
    ({}, [], 0, ""),
    ({"inputs": [1, {"path": "/x/a.pdf"}], "name": 3},
     ["1.  — ", "2. a.pdf — /x"], 0, "3"),
    *[({"inputs": value}, [], 0, "") for value in (None, 1, "foreign", {"pages": 2})],
    ({"inputs": [{"path": "/x/a.pdf", "pages": 2},
                 {"path": "/x/b.pdf", "pages": "3"},
                 {"path": None, "pages": True}, {"pages": 1.5}]},
     None, 2, ""),
])
def test_preview_malformed_args(lang, args, expected_lines, pages, name):
    preview = build_preview("pdf_merge", args, lang=lang)["human"]
    lines = preview.splitlines()
    if expected_lines is None:
        expected_lines = [f"1. a.pdf (2 {'páginas' if lang == 'pt' else 'pages'}) — /x",
                          "2. b.pdf — /x", "3.  — ", "4.  — "]
    assert lines[1:1 + len(expected_lines)] == expected_lines
    result = (f"Resultado: um PDF de {pages} páginas em " if lang == "pt"
              else f"Result: a PDF of {pages} pages in ")
    assert lines[1 + len(expected_lines)] == result + f"~/Documents/Cortex/PDF/{name}/"
    assert preview.endswith(
        "Responda 'sim' (exatamente essa palavra) para juntar ou 'não' para cancelar."
        if lang == "pt" else
        "Reply 'yes' (exactly that word) to merge them or 'no' to cancel.")


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang", ["pt", "en"])
def test_preview_and_success(fake, stream, lang):
    name = pdf_publish.folder_name("prova.pdf", "merged" if lang == "en" else "juntado", date.today())
    (fake.home / "Documents/Cortex/PDF" / name).mkdir(parents=True)
    name += " (2)"
    preview = stage(fake, stream, lang)
    folder = str(fake.paths[0].parent)
    if lang == "pt":
        expected = (f"Juntar estes PDFs, nesta ordem?\n1. prova.pdf (2 páginas) — {folder}\n"
                    f"2. textos.pdf (2 páginas) — {folder}\n"
                    f"Resultado: um PDF de 4 páginas em ~/Documents/Cortex/PDF/{name}/\n"
                    "Os originais não mudam.\n"
                    "Responda 'sim' (exatamente essa palavra) para juntar ou 'não' para cancelar.")
        done = (f"Juntei os 2 PDFs (4 páginas) em ~/Documents/Cortex/PDF/{name}/{name}.pdf. "
                "Os originais não mudaram.")
    else:
        expected = (f"Merge these PDFs, in this order?\n1. prova.pdf (2 pages) — {folder}\n"
                    f"2. textos.pdf (2 pages) — {folder}\n"
                    f"Result: a PDF of 4 pages in ~/Documents/Cortex/PDF/{name}/\n"
                    "The originals don't change.\n"
                    "Reply 'yes' (exactly that word) to merge them or 'no' to cancel.")
        done = (f"Merged the 2 PDFs (4 pages) into ~/Documents/Cortex/PDF/{name}/{name}.pdf. "
                "The originals didn't change.")
    assert preview == expected
    action_id = fake.o._pending_action["id"]
    assert risk_level("pdf_merge") == S3
    assert build_preview("pdf_merge", fake.o._pending_action["args"], lang=lang)["llm"] == (
        "[SYSTEM ERROR] pdf_merge requires the typed phrase.")
    before = [p.read_bytes() for p in fake.paths]
    delivery = Delivery()
    fake.o.set_turn_delivery(delivery)
    assert _turn(fake.o, "yes" if lang == "en" else "sim", stream, lang) == (done, "AGENT_ACTION_DONE")
    assert fake.run.call_count == 2
    fake.publish.assert_called_once()
    assert [p.read_bytes() for p in fake.paths] == before
    [effect] = fake.store.recent()
    assert effect.state == "feito" and effect.action == "pdf_merge"
    assert effect.id == action_id
    assert effect.args == {"inputs": [str(p) for p in fake.paths], "name": name}
    assert effect.origin == {"channel": "ui", "conversation": "conversation", "turn": delivery.id}
    for engine in (fake.o.engine_analytical, fake.o.engine_creative):
        engine.chat.assert_not_called()
        engine.chat_with_tools.assert_not_called()
    _turn(fake.o, "sim", stream)
    assert fake.run.call_count == 2 and len(fake.store.recent()) == 1


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("word", ["ok", "pode", "sim, pode"])
def test_exact_approval(fake, stream, word):
    stage(fake, stream)
    assert _turn(fake.o, word, stream) == (
        "Para juntar os PDFs, responda exatamente 'sim' (ou 'não' para cancelar).",
        "AWAITING_CONFIRMATION")
    fake.run.assert_not_called()
    assert fake.store.recent() == []
    assert _turn(fake.o, "sim", stream)[1] == "AGENT_ACTION_DONE"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("word", ["Sim.", "yes!", "sim!", "sim."])
def test_punctuated_approval_runs_once(fake, stream, word):
    stage(fake, stream)
    assert _turn(fake.o, word, stream)[1] == "AGENT_ACTION_DONE"
    assert [call.args[1] for call in fake.run.call_args_list] == ["pdfunite", "pdfinfo"]
    fake.publish.assert_called_once()
    [effect] = fake.store.recent()
    assert effect.action == "pdf_merge" and effect.state == "feito"
    _turn(fake.o, word, stream)
    assert fake.run.call_count == 2
    fake.publish.assert_called_once()
    assert len(fake.store.recent()) == 1


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("channel,conversation", [("voice", "conversation"), ("ui", "elsewhere"),
                                                  ("terminal", "conversation")])
def test_wrong_origin(fake, stream, channel, conversation):
    stage(fake, stream)
    fake.o.set_turn_origin(channel, conversation)
    assert _turn(fake.o, "sim", stream)[1] == "AGENT_ACTION_ELSEWHERE"
    fake.run.assert_not_called()
    assert fake.store.recent() == []


@pytest.mark.parametrize("stream", [False, True])
def test_changed_before_approval(fake, stream):
    stage(fake, stream)
    write_pdf(fake.paths[0], width=150)
    assert _turn(fake.o, "sim", stream) == (
        "prova.pdf mudou desde a prévia — peça de novo. Nada foi feito.", "PDF_MERGE_REFUSED")
    assert fake.store.recent() == []
    fake.run.assert_not_called()
    fake.publish.assert_not_called()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("case", ["occupied", "stopped", "timeout", "too_big", "wrong_pages",
    "missing_output", "nonzero", "invalid", "unavailable", "error", "raise", "staged_changed"])
def test_execution_failures(fake, monkeypatch, stream, case):
    stage(fake, stream)
    name = fake.o._pending_action["args"]["name"]
    expected = "Não consegui juntar os PDFs — nada foi gravado."
    if case in ("occupied", "invalid", "unavailable", "error"):
        fake.publish.side_effect = None
        fake.publish.return_value = pdf_publish.Published(case, None, (), 0, "")
        if case == "occupied":
            expected = f"A pasta {name} apareceu enquanto isso — peça de novo para eu propor outro nome. Nada foi gravado."
    elif case == "raise":
        fake.run.side_effect = RuntimeError("boom")
        expected = "Não sei se deu certo (boom). Confere antes de pedir de novo — não vou repetir sozinho."
    elif case == "staged_changed":
        original = pdf_runner.PdfJob.stage
        def substitute(job, paths):
            staged = original(job, paths)
            write_pdf(staged[0], width=170)
            return staged
        monkeypatch.setattr(pdf_runner.PdfJob, "stage", substitute)
        expected = "prova.pdf mudou desde a prévia — peça de novo. Nada foi feito."
    else:
        original = fake.run.side_effect
        def fail(job, tool, args, **kw):
            if case == "wrong_pages":
                if tool == "pdfinfo":
                    return pdf_runner.ToolRun("exited", 0, .1, b"Pages: 99\n", "", [])
                return original(job, tool, args, **kw)
            return pdf_runner.ToolRun(
                case if case in ("stopped", "timeout", "too_big") else "exited",
                1 if case == "nonzero" else 0, .1, b"", "", [])
        fake.run.side_effect = fail
        if case == "stopped":
            expected = "Parei — nada foi gravado."
    assert _turn(fake.o, "sim", stream) == (expected, "AGENT_ACTION_ERROR")
    [effect] = fake.store.recent()
    assert effect.state == ("desconhecido" if case == "raise" else "falhou")
    if case not in ("occupied", "invalid", "unavailable", "error"):
        fake.publish.assert_not_called()
    if case == "raise":
        fake.run.assert_called_once()
        _turn(fake.o, "sim", stream)
        fake.run.assert_called_once()


@pytest.mark.parametrize("entry", ["_execute_tool", "_execute_tool_unstaged", "_execute_tool_unstaged_call",
                                   "_skill_macro_stage_step", "_gate_background_action"])
@pytest.mark.parametrize("plan", [False, True])
def test_model_cannot_stage_or_run(fake, entry, plan):
    name, args = "pdf_merge", {"inputs": [], "name": "invented"}
    if plan:
        name, args = "execute_plan", {"passos": [{"tool": name, "args": args}]}
    assert "[SYSTEM ERROR] pdf_merge requires the typed phrase." in getattr(fake.o, entry)(name, args)
    assert fake.o._pending_action is None and fake.store.recent() == []
    fake.run.assert_not_called()


def test_duplicate_and_order(fake):
    text = "junta ~/Downloads/textos.pdf e ~/Downloads/prova.pdf e ~/Downloads/textos.pdf"
    assert _turn(fake.o, text, False)[1] == "AGENT_AWAITING_CONFIRMATION"
    assert [x["path"] for x in fake.o._pending_action["args"]["inputs"]] == [
        str(fake.paths[1]), str(fake.paths[0]), str(fake.paths[1])]
    def check(job, files, name):
        pages = PdfReader(job.output_dir / files[0]).pages
        assert [int(p.mediabox.width) for p in pages] == [101, 101, 100, 100, 101, 101]
        return pdf_publish.Published("published", "unused", tuple(files), 1, "")
    fake.publish.side_effect = check
    assert _turn(fake.o, "sim", False)[1] == "AGENT_ACTION_DONE"


def test_size_limit_and_no_names(fake, monkeypatch):
    monkeypatch.setattr(pdf_runner, "MAX_INPUT_BYTES", sum(p.stat().st_size for p in fake.paths) - 1)
    assert _turn(fake.o, phrase(), False)[0] == "Os PDFs passam de 50 MB juntos — grande demais para juntar."
    assert fake.o._pending_action is None
    monkeypatch.setattr(pdf_runner, "MAX_INPUT_BYTES", 50 * 1024 * 1024)
    monkeypatch.setattr(pdf_publish, "next_free", lambda name: None)
    assert _turn(fake.o, phrase(), False)[0] == (
        "Já existem 99 pastas com esse nome em ~/Documents/Cortex/PDF — apague algumas e peça de novo.")
    assert fake.o._pending_action is None and fake.store.recent() == []


@pytest.mark.parametrize("stream", [False, True])
def test_terminal_and_restored_preview(fake, stream):
    fake.o.set_turn_origin("terminal")
    stage(fake, stream)
    fake.o._pending_action = fake.o._load_pending_action()
    assert _turn(fake.o, "sim", stream)[1] == "AGENT_ACTION_DONE"
    [effect] = fake.store.recent()
    assert effect.origin == {"channel": "terminal", "conversation": None}


@pytest.mark.parametrize("case", ["background", "unreadable", "unrecorded", "stop_before", "stop_after_info"])
def test_execution_gates(fake, monkeypatch, case):
    stage(fake)
    delivery = Delivery()
    fake.o.set_turn_delivery(delivery)
    if case == "background":
        fake.o.background = True
        expected = "Juntar PDFs só pedindo digitado, no terminal ou na tela."
    elif case == "unreadable":
        monkeypatch.setattr("src.tools.file_finder.is_readable_path", lambda p: (False, "blocked"))
        expected = "não posso ler prova.pdf: blocked."
    elif case == "unrecorded":
        monkeypatch.setattr(fake.store, "create", lambda *a, **kw: None)
        expected = "Não consegui registrar essa ação antes de executar — não fiz nada."
    elif case == "stop_before":
        delivery.cancelled.set()
        expected = "Parei antes de executar — não fiz nada."
    else:
        original = fake.run.side_effect
        def stop(job, tool, args, **kw):
            result = original(job, tool, args, **kw)
            if tool == "pdfinfo":
                delivery.cancelled.set()
                assert kw["cancelled"]()
            return result
        fake.run.side_effect = stop
        expected = "Parei — nada foi gravado."
    assert _turn(fake.o, "sim", False)[0] == expected
    fake.publish.assert_not_called()
    if case != "stop_after_info":
        fake.run.assert_not_called()
    effects = fake.store.recent()
    if case in ("stop_before", "stop_after_info"):
        assert len(effects) == 1
        assert effects[0].state == ("cancelado" if case == "stop_before" else "falhou")
    else:
        assert effects == []


@pytest.mark.parametrize("stdout,status,code", [(b"", "exited", 0),
    (b"Pages: 4\nPages: 4\n", "exited", 0), (b"Pages: 4\n", "exited", 1),
    (b"Pages: 4\n", "timeout", None), (b"Pages: four\n", "exited", 0)])
def test_output_inspection_must_succeed(fake, stdout, status, code):
    stage(fake)
    original = fake.run.side_effect
    def inspect(job, tool, args, **kw):
        if tool == "pdfinfo":
            return pdf_runner.ToolRun(status, code, .1, stdout, "", [])
        return original(job, tool, args, **kw)
    fake.run.side_effect = inspect
    assert _turn(fake.o, "sim", False) == (
        "Não consegui juntar os PDFs — nada foi gravado.", "AGENT_ACTION_ERROR")
    fake.publish.assert_not_called()
    assert fake.store.recent()[0].state == "falhou"


def test_publication_size_failure_and_private_cleanup(fake):
    stage(fake)
    roots = []
    def refuse(job, files, name):
        roots.append(job.root)
        return pdf_publish.Published("too_big", None, (), 0, "")
    fake.publish.side_effect = refuse
    assert _turn(fake.o, "sim", False)[1] == "AGENT_ACTION_ERROR"
    assert roots and not roots[0].exists()
    assert fake.store.recent()[0].state == "falhou"


def test_model_plan_cannot_persist(fake):
    fake.o._pending_action = {"type": "execute_plan", "args": {"passos": [
        {"tool": "execute_plan", "args": {"passos": [{"tool": "pdf_merge", "args": {}}]}}]}}
    fake.o._persist_pending_action()
    assert fake.o._pending_action is None
    assert not Path(fake.o._PENDING_ACTION_FILE).exists()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang", ["pt", "en"])
def test_cancel(fake, stream, lang):
    stage(fake, stream, lang)
    reply, _ = _turn(fake.o, "no" if lang == "en" else "não", stream, lang)
    assert ("merge the PDFs" if lang == "en" else "a junção dos PDFs") in reply
    fake.run.assert_not_called()
    assert fake.store.recent() == []


@pytest.mark.skipif(os.environ.get("CORTEX_REAL_SANDBOX_TESTS") != "1"
                   or not Path(pdf_sandbox.SANDBOX_EXEC).exists()
                   or pdf_sandbox.discover() is None,
                   reason="requires opt-in real sandbox-exec and Poppler")
@pytest.mark.parametrize("stream", [False, True])
def test_real_merge(setup, stream):
    if not pdf_sandbox.available():
        pytest.skip("real sandbox unavailable in this environment")
    s = setup
    s.o.set_turn_origin("terminal", "terminal")
    write_pdf(s.paths[1], pages=3, width=101)
    before = [p.read_bytes() for p in s.paths]
    stage(s, stream)
    assert _turn(s.o, "sim", stream)[1] == "AGENT_ACTION_DONE"
    [folder] = list((s.home / "Documents/Cortex/PDF").iterdir())
    [output] = list(folder.iterdir())
    assert output.name == folder.name + ".pdf"
    assert len(PdfReader(output).pages) == 5
    assert [p.read_bytes() for p in s.paths] == before
    [effect] = s.store.recent()
    assert effect.action == "pdf_merge" and effect.state == "feito"
