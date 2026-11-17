"""A file asked for by NAME, without a path (s87, item 4.3).

Measured before (real 8B, process and process_stream): 1 useful reply in 28.
"lê meu currículo" never searched; "cadê meu currículo?" once named
Cortex_comandos.txt as the CV; "abre meu perfil britânico" tried to open an
app; "lê o perfil britânico" invented a document; "read my resume" did
nothing. When the 8B searched, it searched by CONTENT (10 of 10), and the one
PDF called "Punctuation" was buried under code files; by name, "currículo"
finds 0 files and "cv" 37. Now: by name, with Pedro's synonyms (16/10).
"""
import os
import time
from types import SimpleNamespace

import pytest

from src.core import named_file as nf
from src.core.named_file import LOCATE, OPEN, READ
from src.tools import file_finder
from src.tools.file_finder import FileHit


class TestTheWords:

    @pytest.mark.parametrize("text, verb, terms, suffix, strong", [
        ("lê meu currículo", READ, ("curriculo",), None, True),
        ("cadê meu currículo?", LOCATE, ("curriculo",), None, True),
        ("cadê o pdf de punctuation?", LOCATE, ("punctuation",), ".pdf", True),
        ("cadê o pdf de punctuation? lê e responde a questão 5", READ, ("punctuation",), ".pdf", True),
        ("abre meu perfil britânico", OPEN, ("perfil", "britanico"), None, True),
        ("lê o perfil britânico", READ, ("perfil", "britanico"), None, False),
        ("read my resume", READ, ("resume",), None, True),
        ("onde está o meu cv?", LOCATE, ("cv",), None, True),
        ("pode ler pra mim o meu cv?", READ, ("cv",), None, True),
        ("abre o pdf de punctuation", OPEN, ("punctuation",), ".pdf", True),
    ])
    def test_a_request(self, text, verb, terms, suffix, strong):
        r = nf.parse_request(text)
        assert (r.verb, r.terms, r.suffix, r.strong) == (verb, terms, suffix, strong)

    @pytest.mark.parametrize("text", [
        "abre o spotify",              # an app: "abre" needs "meu" or a type word
        "lê o arquivo",                # no name — the read-this-file route's
        "lê isso",
        "me conta uma história",
        "onde fica o Japão?",
        "lê meu email",                # the email tools' (through the 8B)
        "lê minhas notas",
        "lê a tela",
        "cadê minha agenda?",
    ])
    def test_not_a_request(self, text):
        assert nf.parse_request(text) is None

    @pytest.mark.parametrize("text, pick", [
        ("2", (1, None)), ("o segundo", (1, None)), ("Segunda.", (1, None)),
        ("abre o primeiro", (0, OPEN)), ("lê o 3", (2, READ)),
        ("read the third one", (2, READ)), ("o número 4", (3, None)),
    ])
    def test_a_pick(self, text, pick):
        assert nf.parse_pick(text) == pick

    @pytest.mark.parametrize("text", ["sim", "o arquivo", "2 e 3", "o segundo é melhor"])
    def test_not_a_pick(self, text):
        assert nf.parse_pick(text) is None

    def test_a_type_word_makes_it_a_file(self):
        assert nf.parse_request("lê o pdf do email do banco").terms == ("email", "banco")

    def test_long_input_is_instant(self):
        """The lesson of 16/10: a regex that backtracks freezes the reply."""
        text = "lê meu " + "a " * 20000 + "?"
        t0 = time.perf_counter()
        nf.parse_request(text)
        nf.parse_pick(text)
        assert time.perf_counter() - t0 < 0.1


class TestNamesAndSynonyms:

    @pytest.mark.parametrize("term, name, hit", [
        ("curriculo", "Pedro_CV_one_page.docx", True),        # Pedro's synonym
        ("curriculo", "pedro's resume.pdf", True),
        ("cv", "opencv_notes.pdf", False),                     # a word, not a piece
        ("britanico", "perfil_britanico.md", True),
        ("britanic", "perfil_britanico.md", True),             # a word's start
        ("prova", "Maths exam paper.pdf", True),
        ("trabalho", "History homework 3.docx", True),
        ("perfil", "LinkedIn profile.pdf", True),
        ("punctuation", "Punctuation from Pearson.pdf", True),
    ])
    def test_name_matches(self, term, name, hit):
        assert nf.name_matches(term, name) is hit

    def test_every_term_must_be_in_the_name(self):
        r = nf.parse_request("abre meu perfil britânico")
        hits = [nf.Found("/h/perfil_dev.md", "perfil_dev.md", "2026-05-20"),
                nf.Found("/h/perfil_britanico.md", "perfil_britanico.md", "2026-10-16")]
        assert [h.name for h in nf.rank(r, hits)] == ["perfil_britanico.md"]

    def test_newest_first_and_no_duplicates(self):
        r = nf.parse_request("lê meu cv")
        hits = [nf.Found("/h/a/CV_old.pdf", "CV_old.pdf", "2025-01-01"),
                nf.Found("/h/b/Pedro_CV.docx", "Pedro_CV.docx", "2026-01-31"),
                nf.Found("/h/a/CV_old.pdf", "CV_old.pdf", "2025-01-01")]
        assert [h.name for h in nf.rank(r, hits)] == ["Pedro_CV.docx", "CV_old.pdf"]

    def test_the_type_word_filters(self):
        r = nf.parse_request("cadê o pdf de punctuation?")
        hits = [nf.Found("/h/punctuation.py", "punctuation.py", "2026-10-01"),
                nf.Found("/h/Punctuation from Pearson.pdf", "Punctuation from Pearson.pdf", "2026-09-29")]
        assert [h.name for h in nf.rank(r, hits)] == ["Punctuation from Pearson.pdf"]


# ----------------------------------------------------------------------
# the route, in both entry points — Spotlight replaced by a fake over a
# temporary home holding real files (the readability gate runs for real)
# ----------------------------------------------------------------------
FILES = {
    "Documents/CV/Pedro_CV_one_page.docx": "2026-01-31",
    "Documents/old/pedro's resume.pdf": "2025-03-09",
    "Documents/CV/Pedro_CV_Amex.docx": "2026-01-15",
    "college/29-09/Punctuation from Pearson.pdf": "2026-09-29",
    "cortex_project/src/punctuation.py": "2026-10-01",
    "notes/perfil_britanico.md": "2026-10-16",
    "notes/perfil_dev.md": "2026-05-20",
}


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    for rel, day in FILES.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        stamp = time.mktime(time.strptime(day, "%Y-%m-%d"))
        os.utime(path, (stamp, stamp))
    searched = []

    def fake_search(query, limit=10, content=False):
        searched.append((query, content))
        hits = []
        for rel in FILES:
            path = str(tmp_path / rel)
            if query.lower() in os.path.basename(path).lower():
                hits.append(file_finder.FileFinder._describe(path))
        return hits[:limit], None
    monkeypatch.setattr(file_finder.finder, "search", fake_search)
    return SimpleNamespace(root=tmp_path, searched=searched)


@pytest.fixture
def orch(tmp_path, home):
    from tests.test_dispatch_parity import _build_real_orch
    o = _build_real_orch(tmp_path)
    o._PENDING_ACTION_FILE = str(tmp_path / "pending.json")
    o.background = False
    o.macos.open_file.return_value = "Abrindo perfil_britanico.md."
    return o


def staged_path(orch):
    return (orch._pending_action or {}).get("args", {}).get("path")


class TestTheRoute:

    def test_several_cvs_are_listed_newest_first(self, orch, home):
        reply, mode = orch._process_core("lê meu currículo")
        assert mode == "FILE_LIST"
        lines = [l for l in reply.splitlines() if l[:2] in ("1.", "2.", "3.")]
        assert [l.split(" — ")[0] for l in lines] == [
            "1. Pedro_CV_one_page.docx", "2. Pedro_CV_Amex.docx", "3. pedro's resume.pdf"]
        assert "procurei também: curriculum, cv, resume" in reply
        assert all(content is False for _q, content in home.searched)   # by NAME
        orch.engine_analytical.chat.assert_not_called()

    def test_the_number_picks_and_reads_with_confirmation(self, orch, home):
        orch._process_core("lê meu currículo")
        _reply, mode = orch._process_core("o segundo")
        assert mode == "AGENT_AWAITING_CONFIRMATION"
        assert staged_path(orch).endswith("Pedro_CV_Amex.docx")

    def test_one_file_read_goes_straight_to_the_confirmation(self, orch):
        _reply, mode = orch._process_core("cadê o pdf de punctuation? lê e responde a questão 5")
        assert mode == "AGENT_AWAITING_CONFIRMATION"
        assert staged_path(orch).endswith("Punctuation from Pearson.pdf")
        assert "questão 5" in orch._pending_action["args"]["question"]

    def test_where_is_it_then_yes_reads_it(self, orch):
        reply, mode = orch._process_core("cadê o pdf de punctuation?")
        assert mode == "FILE_LIST"
        assert "~/college/29-09: Punctuation from Pearson.pdf" in reply
        assert "Quer que eu leia? (sim)" in reply
        _reply, mode = orch._process_core("sim")
        assert mode == "AGENT_AWAITING_CONFIRMATION"           # not NOTHING_PENDING
        assert staged_path(orch).endswith("Punctuation from Pearson.pdf")

    def test_open_opens_the_file_not_an_app(self, orch, home):
        _reply, mode = orch._process_core("abre meu perfil britânico")
        assert mode == "FILE_OPEN"
        orch.macos.open_file.assert_called_once()
        assert orch.macos.open_file.call_args[0][0] == str(home.root / "notes/perfil_britanico.md")
        orch.macos.open_app.assert_not_called()

    def test_an_app_is_still_an_app(self, orch):
        _reply, mode = orch._process_core("abre o spotify")
        assert mode != "FILE_OPEN"
        orch.macos.open_file.assert_not_called()

    def test_the_email_route_keeps_its_words(self, orch, home):
        _reply, mode = orch._process_core("lê meu email")
        assert not mode.startswith("FILE_")
        assert home.searched == []

    def test_nothing_found_says_so(self, orch):
        reply, mode = orch._process_core("lê meu boletim")
        assert mode == "FILE_ERROR"
        assert reply.startswith('Não achei nenhum arquivo com "boletim" no nome')
        orch.engine_analytical.chat.assert_not_called()

    def test_not_surely_a_file_goes_on_to_the_other_routes(self, orch):
        _reply, mode = orch._process_core("acha o melhor restaurante")
        assert mode != "FILE_ERROR"

    def test_two_requests_are_the_planners(self, orch, home):
        """Read AND email: the planner splits it; this route would lose the email."""
        from src.core.dispatch.heuristics import multi_intent_match
        text = "lê o documento agentes e me manda um resumo por email"
        assert multi_intent_match(text)
        assert orch._handle_named_file(text) is None
        assert home.searched == []

    def test_in_english(self, orch):
        reply, mode = orch._process_core("read my resume")
        assert mode == "FILE_LIST" and reply.startswith("I found 3 files")

    def test_the_list_lasts_one_turn(self, orch):
        orch._process_core("lê meu currículo")
        orch._process_core("que dia é hoje?")
        _reply, mode = orch._process_core("2")
        assert mode != "AGENT_AWAITING_CONFIRMATION"

    def test_a_number_beyond_the_list_is_not_a_pick(self, orch):
        orch._process_core("lê meu currículo")
        _reply, mode = orch._process_core("o quinto")
        assert mode != "AGENT_AWAITING_CONFIRMATION"

    def test_open_the_first_one_from_the_list(self, orch, home):
        orch._process_core("cadê meu cv")
        _reply, mode = orch._process_core("abre o primeiro")
        assert mode == "FILE_OPEN"
        assert orch.macos.open_file.call_args[0][0].endswith("Pedro_CV_one_page.docx")


class TestBothEntryPoints:

    def _stream(self, orch, text):
        parts, mode = [], None
        for chunk, final, m, _lang in orch.process_stream(text):
            parts.append(chunk)
            if final:
                mode = m
        return "".join(parts), mode

    def test_the_stream_lists_and_picks(self, orch):
        _r, mode = self._stream(orch, "lê meu currículo")
        assert mode == "FILE_LIST"
        _r, mode = self._stream(orch, "1")
        assert mode == "AGENT_AWAITING_CONFIRMATION"
        assert staged_path(orch).endswith("Pedro_CV_one_page.docx")
        orch.engine_analytical.chat_stream.assert_not_called()

    def test_the_stream_yes_to_where_is_it(self, orch):
        self._stream(orch, "cadê o pdf de punctuation?")
        _r, mode = self._stream(orch, "sim")
        assert mode == "AGENT_AWAITING_CONFIRMATION"

    def test_pure_chat_vetoes(self, orch):
        assert orch._is_pure_chat("lê meu currículo") is False
        assert orch._is_pure_chat("read my resume") is False
        orch._offer = {"kind": "find_pick", "paths": ["/a", "/b"], "verb": READ}
        assert orch._is_pure_chat("o segundo") is False
