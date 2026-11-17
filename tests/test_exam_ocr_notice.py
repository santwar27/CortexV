"""Only the reader's leading OCR label justifies an exam OCR warning."""
import os

import pytest

from src.core.orchestrator import CortexOrchestrator
from src.tools.local_reader import LocalFileReader
from tests.test_exam_reader import BOOKLET, PAPER


OCR_RULE = (
    "- The files were scanned and read by OCR: an odd character is a "
    "misreading, not a word.\n"
)
NOTES = {
    "pt": "(Os PDFs são escaneados e foram lidos por OCR — confira as citações no papel.)",
    "en": "(The PDFs are scans read by OCR — check the quotations against the paper.)",
}
LANGUAGES = {
    "pt": "Answer in Brazilian Portuguese, but copy quotations from the text "
          "exactly as they are, in English.",
    "en": "Answer in English.",
}


@pytest.fixture
def exam_run(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    exam_dir = tmp_path / "Downloads" / "college" / "13-10"
    exam_dir.mkdir(parents=True)
    (exam_dir / "question.pdf").write_bytes(b"%PDF-q")
    (exam_dir / "text.pdf").write_bytes(b"%PDF-t")
    orch = CortexOrchestrator.__new__(CortexOrchestrator)
    orch._pending_action = None
    orch._pending_query = None
    orch._PENDING_ACTION_FILE = str(tmp_path / "pending_action.json")
    orch.background = False
    orch.background_origin = "background"
    orch.engine_analytical = type("E", (), {"model_name": "cortex-lite"})()
    calls, emitted, entries = [], [], []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        user = kwargs["messages"][1]["content"]
        number = user.rsplit("Answer question ", 1)[1].rstrip(".")
        return {"message": {"content": f"resposta {number}"}}

    monkeypatch.setattr("src.core.staging_mixin.ollama.chat", fake_chat)
    monkeypatch.setattr("src.core.staging_mixin.audit.log",
                        lambda *args, **kwargs: entries.append((args, kwargs)))

    def run(ocr, lang="pt", labelled="question.pdf"):
        contents = {"question.pdf": PAPER, "text.pdf": BOOKLET}
        if ocr:
            contents[labelled] = LocalFileReader._ocr_labelled(contents[labelled])
        orch.reader = type("R", (), {
            "read_path": lambda _s, path: contents[os.path.basename(path)]})()
        orch._reply_lang = lang
        orch.set_token_emitter(emitted.append)
        _, mode = orch._handle_user_paths(
            f"two pdf in {exam_dir} , answer question 3")
        assert mode == "AGENT_AWAITING_CONFIRMATION"
        reply, mode = orch._execute_pending_action()
        orch.set_token_emitter(None)
        assert mode == "LOCAL_RAG"
        return reply, calls, emitted, entries

    return run


@pytest.mark.parametrize("ocr", [False, True])
@pytest.mark.parametrize("lang", ["pt", "en"])
def test_note_and_prompt_follow_the_reader(exam_run, ocr, lang):
    reply, calls, emitted, _ = exam_run(ocr, lang)
    assert len(calls) == 1
    assert "**3.** resposta 3" in reply
    assert "".join(emitted) == reply
    assert (NOTES[lang] in reply) is ocr
    assert NOTES["en" if lang == "pt" else "pt"] not in reply
    system = CortexOrchestrator._EXAM_SYSTEM
    assert system.count(OCR_RULE) == 1
    expected = system if ocr else system.replace(OCR_RULE, "")
    for call in calls:
        prompt = call["messages"][0]["content"]
        assert prompt == expected + "\n- " + LANGUAGES[lang]
        assert ("scanned and read by OCR" in prompt) is ocr


@pytest.mark.parametrize("ocr", [False, True])
def test_streaming_has_exactly_the_note_that_was_said(exam_run, ocr):
    reply, _, emitted, _ = exam_run(ocr)
    assert "".join(emitted) == reply
    assert (NOTES["pt"] in "".join(emitted)) is ocr


@pytest.mark.parametrize("ocr", [False, True])
def test_exam_start_audits_ocr(exam_run, ocr):
    _, _, _, entries = exam_run(ocr)
    starts = [kwargs["payload"] for args, kwargs in entries
              if args == ("–", "exam", "start")]
    assert len(starts) == 1
    assert starts[0] == {
        "questions": 1, "texts": "A, B",
        "text_chars": {"A": len("Text A\nArticle\nFlexible working makes workers more productive."),
                       "B": len("Text B\nWebpage\nBresley College offers a wellbeing programme.")},
        "ocr": ocr,
    }


def test_ocr_in_the_booklet_also_counts(exam_run):
    reply, calls, _, _ = exam_run(True, labelled="text.pdf")
    assert NOTES["pt"] in reply
    assert calls[0]["messages"][0]["content"] == (
        CortexOrchestrator._EXAM_SYSTEM + "\n- " + LANGUAGES["pt"])


def test_the_label_is_unchanged_and_only_a_leading_label_counts():
    from src.tools.local_reader import read_by_ocr

    labelled = LocalFileReader._ocr_labelled("x")
    assert labelled == (
        "[SYSTEM NOTICE: This PDF is a scan with no text layer. The text "
        "below was read by on-device OCR and may contain recognition "
        "errors — quote it with care.]\n\nx")
    assert read_by_ocr(labelled)
    assert not read_by_ocr(PAPER)
    assert not read_by_ocr("A document quoting a notice:\n" + labelled)
