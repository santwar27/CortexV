"""A "sim" that nothing is waiting for (s85, fila item 3).

Measured against the real 8B before: right after Cortex's own question —
"write the answers on a copy of the PDF?", "which file?" — a "sim" got a
bare "Sim." back and nothing happened, 14 times in 14. A bare "yes" got
"I couldn't understand", 6 in 6. Thanks and acks ("valeu", "show") are
conversation, not confirmations, and stay with the chat.
"""
import os

import pytest

from src.core.orchestrator import CortexOrchestrator


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def orch(tmp_path, home):
    from tests.test_dispatch_parity import _build_real_orch
    o = _build_real_orch(tmp_path)
    o._PENDING_ACTION_FILE = str(tmp_path / "pending.json")
    o.background = False
    return o


@pytest.fixture
def folder(home):
    d = home / "Downloads" / "13-10"
    d.mkdir(parents=True)
    (d / "question.pdf").write_bytes(b"%PDF-q")
    (d / "text.pdf").write_bytes(b"%PDF-t")
    return d


class TestNothingPending:

    @pytest.mark.parametrize("yes,lang_word", [("sim", "nada pendente"),
                                               ("Sim!", "nada pendente"),
                                               ("yes", "nothing waiting")])
    def test_an_explicit_yes_is_told_so(self, orch, yes, lang_word):
        reply, mode = orch._process_core(yes)
        assert mode == "NOTHING_PENDING" and lang_word in reply
        orch.engine_analytical.chat.assert_not_called()

    @pytest.mark.parametrize("ack", ["valeu", "show", "beleza", "kk", "ok"])
    def test_thanks_and_acks_stay_with_the_chat(self, orch, ack):
        _reply, mode = orch._process_core(ack)
        assert mode != "NOTHING_PENDING"

    def test_a_staged_action_is_still_confirmed(self, orch, monkeypatch):
        ran = []
        monkeypatch.setattr(orch, "_run_action",
                            lambda name, args, effect_id=None: ran.append(name) or ("feito", "AGENT_ACTION_DONE"),
                            raising=False)
        orch._pending_action = {"type": "create_reminder", "args": {"title": "x"}}
        orch._persist_pending_action()          # stamps staged_at, as staging does
        _reply, mode = orch._process_core("sim")
        assert ran == ["create_reminder"] and mode == "AGENT_ACTION_DONE"

    def test_no_is_unchanged(self, orch):
        _reply, mode = orch._process_core("não")
        assert mode != "NOTHING_PENDING"


class TestAnsweringCortexsQuestion:

    def test_yes_after_a_listing_asks_which(self, orch, folder):
        _r, mode = orch._process_core(f"o que tem em {folder}?")
        assert mode == "FILE_LIST"
        reply, mode = orch._process_core("sim")
        assert mode == "FILE_LIST"
        assert reply == "Qual deles? question.pdf ou text.pdf? Ou os dois?"
        # and the question still stands: "os dois" now reads them
        _r, mode = orch._process_core("os dois")
        assert mode == "AGENT_AWAITING_CONFIRMATION"

    def test_yes_after_the_pdf_offer_asks_to_write_the_copy(self, orch, folder):
        paper = str(folder / "question.pdf")
        orch._remember_seen_path(paper)
        orch._last_exam = {"when": __import__("time").monotonic(), "paper": paper,
                           "answers": {1: "a"}, "lang": "pt"}
        orch._offer = {"kind": "write_pdf"}
        reply, mode = orch._process_core("sim")
        assert mode == "AGENT_AWAITING_CONFIRMATION"
        assert orch._pending_action["type"] == "write_answers_pdf"
        assert "question - respostas.pdf" in reply

    def test_even_a_casual_yes_answers_the_question(self, orch, folder):
        orch._process_core(f"o que tem em {folder}?")
        _reply, mode = orch._process_core("beleza")
        assert mode == "FILE_LIST"

    def test_the_question_lasts_one_turn(self, orch, folder):
        orch._process_core(f"o que tem em {folder}?")
        orch._process_core("quem foi einstein")             # something else
        _reply, mode = orch._process_core("sim")
        assert mode == "NOTHING_PENDING"


class TestBothEntryPoints:

    def test_the_stream_shortcuts_drop_the_question_too(self):
        """Reflex and pure chat bypass _process_core on the voice path; a
        question left over there would answer a "sim" two turns later."""
        from pathlib import Path
        source = (Path(__file__).resolve().parents[1]
                  / "src" / "core" / "orchestrator.py").read_text()
        assert source.count("self._offer = None  # a question lasts ONE turn") == 2

    def test_yes_never_takes_the_pure_chat_shortcut(self):
        assert "_APPROVAL_RE" in CortexOrchestrator._PURE_CHAT_BLOCKING_RES

    @pytest.mark.parametrize("ack", ["valeu", "vlw"])
    def test_a_farewell_word_answers_the_pdf_offer_on_the_stream_too(self, orch, folder, ack):
        """"valeu" is a goodbye to the reflex: the stream said "Até mais!" and
        dropped the offer that process() took (s88, Codex lote 1)."""
        paper = str(folder / "question.pdf")
        orch._remember_seen_path(paper)
        orch._last_exam = {"when": __import__("time").monotonic(), "paper": paper,
                           "answers": {1: "a"}, "lang": "pt"}
        orch._offer = {"kind": "write_pdf"}
        *_, (_reply, _final, mode, _lang) = orch.process_stream(ack, None)
        assert mode == "AGENT_AWAITING_CONFIRMATION"
        assert orch._pending_action["type"] == "write_answers_pdf"

    def test_a_farewell_after_a_listing_asks_which_on_the_stream(self, orch, folder):
        orch._process_core(f"o que tem em {folder}?")
        *_, (_reply, _final, mode, _lang) = orch.process_stream("valeu", None)
        assert mode == "FILE_LIST"

    def test_with_no_question_it_is_still_a_goodbye(self, orch):
        *_, (reply, _final, mode, _lang) = orch.process_stream("valeu", None)
        assert (reply, mode) == ("Até mais!", "REFLEX")


class TestEveryFileListAsksTheSameQuestion:
    """s85 review: only the folder listing left "which one?" standing — a
    message naming two files and "not found, but it has…" did not."""

    def test_two_files_named_then_both(self, orch, folder):
        reply, mode = orch._process_core(
            f"estão aqui {folder}/question.pdf e {folder}/text.pdf")
        assert mode == "FILE_ERROR" and '"os dois"' in reply
        _r, mode = orch._process_core("os dois")
        assert mode == "AGENT_AWAITING_CONFIRMATION"
        assert orch._pending_action["type"] == "read_files"

    def test_not_found_then_yes_asks_which(self, orch, folder):
        reply, _ = orch._process_core(f"lê {folder}/answers.pdf")
        assert "Quer ler algum?" in reply
        reply, mode = orch._process_core("sim")
        assert mode == "FILE_LIST" and "Ou os dois?" in reply
