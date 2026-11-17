"""A path in the message never reaches the 8B on its own (s85).

Baseline, measured end to end against the real 8B before this route existed:
18 of 18 turns carrying a path pretended to a read that never happened —
invented contents for a typed `read_file_at(...)`, invented contents for a
file that does not exist, "I'll read them" for a folder. The filesystem can
settle a path in microseconds, so it is settled before the model.
"""
import os

import pytest

from src.core import path_route
from src.core.orchestrator import CortexOrchestrator
from src.tools.file_finder import folder_gate, normalise_user_path, readable_files_in


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def exam(home):
    """The real case of 13/10, in miniature: a folder with two PDFs."""
    d = home / "Downloads" / "college" / "13-10"
    d.mkdir(parents=True)
    (d / "question.pdf").write_bytes(b"%PDF-" + b"q" * 2048)
    (d / "text.pdf").write_bytes(b"%PDF-" + b"t" * 1024)
    return d


@pytest.fixture
def orch(tmp_path):
    o = CortexOrchestrator.__new__(CortexOrchestrator)
    o._pending_action = None
    o._pending_query = None
    o._PENDING_ACTION_FILE = str(tmp_path / "pending_action.json")
    o.background = False
    o.background_origin = "background"
    o._reply_lang = "pt"
    return o


class TestWhatCountsAsAPath:

    @pytest.mark.parametrize("text", [
        "isso vale and/or aquilo",
        "a 100 km/h",
        "a prova é dia 13/10",
        "suporte 24/7",
        "meia xícara, 1 / 2",
        "olha https://site.com/a/b/c.pdf",
        "o que faz rm -rf /tmp/x ?",      # a system path is a shell question
        "cat /etc/passwd",
        "abre /Users",                      # one segment
        "w/o açúcar",
    ])
    def test_ordinary_slashes_are_not_paths(self, home, text):
        assert not path_route.mentions_path(text)
        assert path_route.find_paths(text) == []

    def test_a_folder_is_a_folder(self, exam):
        [m] = path_route.find_paths(
            f"cortex, there is two pdf in this folder {exam} , read them")
        assert m.kind == path_route.FOLDER and m.path == str(exam)

    def test_a_typed_tool_call_is_a_file(self, exam):
        """Verbatim shape of what Pedro typed on 13/10."""
        [m] = path_route.find_paths(
            f'read_file_at(path="{exam}/text.pdf", '
            'question=“extracts the information”)')
        assert m.kind == path_route.FILE
        assert m.path == str(exam / "text.pdf")

    def test_a_file_that_is_not_there_is_missing(self, exam):
        [m] = path_route.find_paths(f"read {exam}/answers.pdf and tell me")
        assert m.kind == path_route.MISSING
        assert m.path == str(exam / "answers.pdf")

    @pytest.mark.parametrize("wrap", ["lê {}.", "(veja {})", "lê {}, por favor",
                                      "lê `{}`", "'{}'"])
    def test_the_sentence_around_it_is_not_part_of_it(self, exam, wrap):
        [m] = path_route.find_paths(wrap.format(exam / "text.pdf"))
        assert m.kind == path_route.FILE

    def test_tilde_works(self, exam):
        [m] = path_route.find_paths("o que tem em ~/Downloads/college/13-10?")
        assert m.kind == path_route.FOLDER

    def test_a_path_pasted_with_plain_spaces(self, home):
        d = home / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "college homework"
        d.mkdir(parents=True)
        (d / "Punctuation from Pearson.pdf").write_text("x")
        [m] = path_route.find_paths(
            f"lê {d}/Punctuation from Pearson.pdf e responde a 5")
        assert m.kind == path_route.FILE
        assert m.path == str(d / "Punctuation from Pearson.pdf")

    @pytest.mark.parametrize("wrap", ["lê {}", 'lê "{}"', "lê {} e responde"])
    def test_a_folder_named_like_the_start_of_the_file_does_not_win(self, home, wrap):
        """~/Downloads/Meu exists beside "Meu arquivo.pdf": the cut "…/Meu"
        resolved and the full name was never tried (s88, Codex lote 1)."""
        d = home / "Downloads"
        (d / "Meu").mkdir(parents=True)
        (d / "Meu arquivo.pdf").write_text("x")
        [m] = path_route.find_paths(wrap.format("~/Downloads/Meu arquivo.pdf"))
        assert m.kind == path_route.FILE
        assert m.path == str(d / "Meu arquivo.pdf")

    def test_the_folder_alone_is_still_the_folder(self, home):
        d = home / "Downloads"
        (d / "Meu").mkdir(parents=True)
        (d / "Meu arquivo.pdf").write_text("x")
        [m] = path_route.find_paths("lista ~/Downloads/Meu e me diz o que tem")
        assert m.kind == path_route.FOLDER and m.path == str(d / "Meu")

    def test_a_shell_escaped_folder(self, home):
        d = home / "Mobile Documents" / "college homework"
        d.mkdir(parents=True)
        raw = str(home) + r"/Mobile\ Documents/college\ homework"
        [m] = path_route.find_paths(f"lista {raw}")
        assert m.kind == path_route.FOLDER and m.path == str(d)

    def test_the_same_path_twice_is_one_mention(self, exam):
        assert len(path_route.find_paths(f"{exam} e de novo {exam}")) == 1

    def test_mentions_path_agrees_with_find_paths(self, exam):
        """The streaming probe and the route must never disagree: a path the
        probe misses streams to chat, a path the route misses falls through."""
        for text in (f"lê {exam}/text.pdf", f"lê {exam}/nada.pdf", str(exam),
                     "and/or", "13/10"):
            assert path_route.mentions_path(text) == bool(path_route.find_paths(text))


class TestFolderHelpers:

    def test_the_squash_match_never_returns_a_folder(self, exam):
        """".../college/13-10" squash-matched the FOLDER 13-10 and came back
        as "the file" — the bug this route found on its first run."""
        assert normalise_user_path(str(exam)) is None

    def test_lists_only_what_could_be_read(self, exam):
        (exam / "notas.py").write_text("x")
        (exam / ".escondido.pdf").write_text("x")
        (exam / "sub").mkdir()
        names = [os.path.basename(p) for p, _ in readable_files_in(str(exam))]
        assert names == ["question.pdf", "text.pdf"]

    @pytest.mark.parametrize("rel", [".ssh", ".aws", ".cortex", "Library/Caches"])
    def test_a_protected_folder_is_not_listed(self, home, rel):
        d = home / rel
        d.mkdir(parents=True)
        ok, why = folder_gate(str(d))
        assert not ok and "protected" in why

    def test_a_folder_outside_home_is_not_listed(self, home):
        ok, why = folder_gate("/etc")
        assert not ok and "outside" in why


class TestTheReplies:

    def test_a_file_stages_the_read_and_opens_nothing(self, orch, exam,
                                                      monkeypatch):
        opened = []
        monkeypatch.setattr(orch, "reader", type("R", (), {
            "read_path": lambda _s, p: opened.append(p) or "x"})(),
            raising=False)
        reply, mode = orch._handle_user_paths(
            f'read_file_at(path="{exam}/text.pdf", question=“extracts”)')
        assert mode == "AGENT_AWAITING_CONFIRMATION"
        assert "text.pdf" in reply
        assert orch._pending_action["args"]["path"] == str(exam / "text.pdf")
        # the question travels without the long path in it
        assert str(exam) not in orch._pending_action["args"]["question"]
        assert opened == []

    def test_an_approved_file_is_read_without_asking_again(self, orch, exam,
                                                           monkeypatch):
        f = str(exam / "text.pdf")
        orch._remember_read_approval(os.path.realpath(f))
        monkeypatch.setattr(orch, "_run_read_file_at",
                            lambda args: ("resposta", "LOCAL_RAG"),
                            raising=False)
        assert orch._handle_user_paths(f"lê {f}") == ("resposta", "LOCAL_RAG")
        assert orch._pending_action is None

    def test_a_file_that_cannot_be_read_says_why(self, orch, exam):
        (exam / "script.py").write_text("x")
        reply, mode = orch._handle_user_paths(f"lê {exam}/script.py")
        assert mode == "FILE_ERROR" and "não posso ler script.py" in reply
        assert orch._pending_action is None

    def test_a_folder_lists_its_files_and_asks(self, orch, exam):
        reply, mode = orch._handle_user_paths(
            f"cortex, there is two pdf in this folder {exam}")
        assert mode == "FILE_LIST"
        assert "question.pdf" in reply and "text.pdf" in reply
        assert "Qual deles" in reply and '"os dois"' in reply
        assert orch._pending_action is None
        # listed means on the table: "lê o text" can pick it next turn
        assert orch._names_a_known_file("lê o text")

    def test_a_folder_reply_in_english(self, orch, exam):
        orch._reply_lang = "en"
        reply, _ = orch._handle_user_paths(f"what is in {exam}?")
        assert reply.startswith("The folder 13-10 has 2 files I can read")
        assert '(or "both")' in reply

    def test_an_empty_folder_says_so(self, orch, home):
        d = home / "vazia"
        d.mkdir()
        reply, mode = orch._handle_user_paths(f"lê {d}")
        assert mode == "FILE_LIST" and "não tem nenhum arquivo" in reply

    def test_a_protected_folder_is_refused_not_listed(self, orch, home):
        d = home / ".ssh"
        d.mkdir()
        (d / "chave.txt").write_text("segredo")
        reply, mode = orch._handle_user_paths(f"lista {d}")
        assert mode == "FILE_ERROR" and "chave.txt" not in reply

    def test_a_missing_file_says_so_and_shows_the_folder(self, orch, exam):
        reply, mode = orch._handle_user_paths(
            f"cortex, read the pdf {exam}/answers.pdf and tell me what it says")
        assert mode == "FILE_ERROR"
        assert "não achei answers.pdf" in reply
        assert "question.pdf" in reply and "text.pdf" in reply

    def test_a_missing_folder_says_so(self, orch, home):
        reply, mode = orch._handle_user_paths(f"lê {home}/nao/existe/x.pdf")
        assert mode == "FILE_ERROR" and "não achei" in reply

    def test_a_missing_file_in_a_protected_folder_lists_nothing(self, orch, home):
        d = home / ".ssh"
        d.mkdir()
        (d / "chave.txt").write_text("segredo")
        reply, _ = orch._handle_user_paths(f"lê {d}/nada.txt")
        assert "chave.txt" not in reply

    def test_two_files_ask_which(self, orch, exam):
        reply, mode = orch._handle_user_paths(
            f"{exam}/question.pdf e {exam}/text.pdf")
        assert mode == "FILE_ERROR"
        assert "question.pdf" in reply and "text.pdf" in reply
        assert orch._pending_action is None

    def test_no_path_is_none(self, orch):
        assert orch._handle_user_paths("me conta uma piada") is None


class TestBothEntryPoints:
    """The cross-cutting lesson: the route lives in _process_core, and
    process_stream reaches it only if its pure-chat shortcut steps aside."""

    def test_the_streaming_shortcut_steps_aside_for_a_path(self, orch, exam):
        # Before the gate/realtime probes: those need a live gatekeeper and
        # the 8B, and a path must never get that far.
        assert orch._is_pure_chat(f"read_file_at(path=\"{exam}/text.pdf\")") is False
        assert orch._is_pure_chat(f"there is two pdf in this folder {exam}") is False

    def test_process_core_answers_a_path_without_the_model(self, tmp_path,
                                                           exam):
        from tests.test_dispatch_parity import _build_real_orch
        o = _build_real_orch(tmp_path)
        o._PENDING_ACTION_FILE = str(tmp_path / "pending.json")
        o.background = False
        cases = {
            f"cortex, there is two pdf in this folder {exam}": "FILE_LIST",
            f"cortex, there is two pdf in this folder {exam} , read them":
                "AGENT_AWAITING_CONFIRMATION",
            f'read_file_at(path="{exam}/text.pdf", question=“x”)':
                "AGENT_AWAITING_CONFIRMATION",
            f"cortex, read the pdf {exam}/answers.pdf and tell me": "FILE_ERROR",
        }
        for text, want in cases.items():
            o._pending_action = None
            _reply, mode = o._process_core(text)
            assert mode == want, text
        o.engine_analytical.chat.assert_not_called()
        o.engine_analytical.chat_with_tools.assert_not_called()

    def test_wired_in_both(self):
        from pathlib import Path
        source = (Path(__file__).resolve().parents[1]
                  / "src" / "core" / "orchestrator.py").read_text()
        assert source.count("if self._mentions_files(own):") == 1    # the veto
        assert source.count("path_route.mentions_path(text)") == 1   # inside it
        assert source.count("self._handle_user_paths(own)") == 1     # the route
        # both read only what the user wrote — never a fenced attachment (s85 review)
        assert source.count("own = self._pedros_words(") == 3
