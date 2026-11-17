"""
Apple Notes dispatch — regexes + orchestrator handlers (s40, Claude-B).

Notes are S2 (local, recoverable) → executed directly, no sim/não gate.
Regex-first, no TOOL_SPECS entry (the 8B can't fabricate a note tool).
"""
from types import SimpleNamespace

import pytest

from src.core.dispatch.notes import (
    NOTE_APPEND_RE,
    NOTE_APPEND_TAIL_RE,
    NOTE_CREATE_FOLDER_RE,
    NOTE_CREATE_RE,
    NOTE_LIST_RE,
    NOTE_SEARCH_RE,
)


class TestNoteCreateRegex:
    @pytest.mark.parametrize("text, body", [
        ("anota comprar leite e ovos", "comprar leite e ovos"),
        ("anota aí: ligar pro dentista", "ligar pro dentista"),
        ("anota aí ligar pro dentista", "ligar pro dentista"),
        ("anote que a reunião é quinta", "a reunião é quinta"),
        ("anotar pagar o aluguel", "pagar o aluguel"),
        ("cria uma nota: lista de compras", "lista de compras"),
        ("cria nota sobre o projeto X", "o projeto X"),
        ("nova nota reunião de sexta", "reunião de sexta"),
        ("faz uma nota com as ideias", "as ideias"),
        ("adiciona uma nota lembrar de pagar", "lembrar de pagar"),
        ("create a note about the trip", "the trip"),
        ("new note: groceries", "groceries"),
        ("note down call mom", "call mom"),
    ])
    def test_matches_and_captures_body(self, text, body):
        m = NOTE_CREATE_RE.match(text)
        assert m, f"should match: {text}"
        assert m.group("body").strip() == body

    @pytest.mark.parametrize("text", [
        "anotado com sucesso",        # past participle, not a command
        "que nota você me dá",        # asking for a grade
        "nota fiscal da compra",      # 'nota' noun, no create verb
        "notável esse resultado",     # 'notável', unrelated
        "quero saber a nota do jogo",
        "anota",                      # no body
    ])
    def test_does_not_overmatch(self, text):
        assert not NOTE_CREATE_RE.match(text), f"false positive: {text}"


class TestNoteCreateFolderRegex:
    @pytest.mark.parametrize("text, folder, body", [
        ("anota na pasta Trabalho: revisar contrato", "Trabalho", "revisar contrato"),
        ("cria uma nota na pasta Ideias — app de treino", "Ideias", "app de treino"),
        ("cria nota na pasta Casa: trocar lâmpada", "Casa", "trocar lâmpada"),
        ("create a note in folder Work: ship the fix", "Work", "ship the fix"),
        ("new note in the folder Trips: pack list", "Trips", "pack list"),
    ])
    def test_matches_and_captures(self, text, folder, body):
        m = NOTE_CREATE_FOLDER_RE.match(text)
        assert m, f"should match: {text}"
        assert m.group("folder").strip() == folder
        assert m.group("body").strip() == body

    @pytest.mark.parametrize("text", [
        "anota comprar leite",               # plain create, no folder
        "anota na pasta Trabalho",           # no separator + body
        "lista as notas da pasta Trabalho",  # list, not create
    ])
    def test_does_not_overmatch(self, text):
        assert not NOTE_CREATE_FOLDER_RE.match(text), f"false positive: {text}"


class TestNoteAppendRegexes:
    @pytest.mark.parametrize("text, title, content", [
        ("acrescenta na nota Mercado: pão integral", "Mercado", "pão integral"),
        ("adiciona à nota Ideias — testar Open-Meteo", "Ideias", "testar Open-Meteo"),
        ("coloca na nota Viagem: passaporte", "Viagem", "passaporte"),
        ("append to note Work: review PR", "Work", "review PR"),
        ("add to the note Groceries: milk", "Groceries", "milk"),
    ])
    def test_title_first_shape(self, text, title, content):
        m = NOTE_APPEND_RE.match(text)
        assert m, f"should match: {text}"
        assert m.group("title").strip() == title
        assert m.group("text").strip() == content

    @pytest.mark.parametrize("text, title, content", [
        ("adiciona pão integral na nota Mercado", "Mercado", "pão integral"),
        ("acrescenta o link do hotel à nota Viagem", "Viagem", "o link do hotel"),
        ("bota isso na nota Lembretes", "Lembretes", "isso"),
        ("append review PR to note Work", "Work", "review PR"),
        ("add milk to the note Groceries", "Groceries", "milk"),
    ])
    def test_text_first_shape(self, text, title, content):
        m = NOTE_APPEND_TAIL_RE.match(text)
        assert m, f"should match: {text}"
        assert m.group("title").strip() == title
        assert (m.group("text") or m.group("text_en")).strip() == content

    @pytest.mark.parametrize("text", [
        "adiciona uma nota lembrar de pagar",   # create, not append
        "anota comprar leite",                  # create
        "adiciona leite na lista de compras",   # reminders/list, not a note
    ])
    def test_does_not_overmatch(self, text):
        assert not NOTE_APPEND_RE.match(text), f"false positive: {text}"
        assert not NOTE_APPEND_TAIL_RE.match(text), f"false positive: {text}"


class TestNoteListRegex:
    @pytest.mark.parametrize("text", [
        "minhas notas", "lista as notas", "lista minhas notas",
        "mostra minhas notas", "quais são minhas notas",
        "list my notes", "show notes", "list all notes",
    ])
    def test_matches(self, text):
        assert NOTE_LIST_RE.match(text), text

    @pytest.mark.parametrize("text", [
        "anota minhas ideias",        # create, not list
        "qual a nota do filme",
        "notas musicais",
    ])
    def test_does_not_overmatch(self, text):
        assert not NOTE_LIST_RE.match(text), f"false positive: {text}"


class TestNoteSearchRegex:
    @pytest.mark.parametrize("text, query", [
        ("acha a nota sobre o projeto", "o projeto"),
        ("procura nota de compras", "compras"),
        ("busca a nota que fala sobre férias", "férias"),
        ("encontra a nota X", "X"),
        ("search notes for budget", "budget"),
        ("find my notes about taxes", "taxes"),
    ])
    def test_matches_and_captures_query(self, text, query):
        m = NOTE_SEARCH_RE.match(text)
        assert m, f"should match: {text}"
        assert m.group("query").strip() == query

    @pytest.mark.parametrize("text", [
        "anota procura emprego",      # create, not search
        "minhas notas",
    ])
    def test_does_not_overmatch(self, text):
        assert not NOTE_SEARCH_RE.match(text), f"false positive: {text}"


class TestNoteHandlers:
    @pytest.fixture
    def orch(self):
        from src.core.orchestrator import CortexOrchestrator
        o = CortexOrchestrator.__new__(CortexOrchestrator)
        o._pending_action = None
        self.created = []
        self.appended = []
        o.notes = SimpleNamespace(
            create_note=lambda title, body="", folder=None:
                self.created.append((title, body, folder)),
            list_notes=lambda limit=10: [{"title": "Compras"}, {"title": "Ideias"}],
            search_notes=lambda q, limit=10: (
                [{"title": "Viagem Itália"}] if "viagem" in q.lower() else []
            ),
            append_to_note=lambda title, text: (
                self.appended.append((title, text)) or {"title": title}
            ),
        )
        return o

    def test_create_uses_first_line_as_title(self, orch):
        reply, mode = orch._handle_note_create("Reunião\npauta: orçamento")
        assert mode == "NOTE_CREATE"
        assert 'Anotado: "Reunião".' == reply
        assert self.created == [("Reunião", "pauta: orçamento", None)]

    def test_create_single_line(self, orch):
        reply, mode = orch._handle_note_create("comprar leite")
        assert self.created == [("comprar leite", "", None)]
        assert "comprar leite" in reply

    def test_create_with_folder(self, orch):
        reply, mode = orch._handle_note_create("Ideias do projeto", folder="Trabalho")
        assert mode == "NOTE_CREATE"
        assert self.created == [("Ideias do projeto", "", "Trabalho")]
        assert "Trabalho" in reply

    def test_append_hit(self, orch):
        reply, mode = orch._handle_note_append("Mercado", "pão")
        assert mode == "NOTE_APPEND"
        assert self.appended == [("Mercado", "pão")]
        assert "Mercado" in reply

    def test_append_miss_is_honest_not_a_create(self, orch):
        def miss(title, text):
            from src.integrations.notes_client import NotesUnavailable
            raise NotesUnavailable(f'No note found matching "{title}".')
        orch.notes = SimpleNamespace(append_to_note=miss)
        reply, mode = orch._handle_note_append("Inexistente", "x")
        assert mode == "NOTE_APPEND"
        assert "não achei" in reply.lower()

    def test_append_empty_falls_through(self, orch):
        assert orch._handle_note_append("", "texto") is None
        assert orch._handle_note_append("título", "  ") is None
        assert self.appended == []

    def test_create_empty_falls_through(self, orch):
        assert orch._handle_note_create("   ") is None
        assert self.created == []

    def test_create_failsoft_on_error(self, orch):
        def boom(title, body="", folder=None):
            from src.integrations.notes_client import NotesUnavailable
            raise NotesUnavailable("Notes locked")
        orch.notes = SimpleNamespace(create_note=boom)
        reply, mode = orch._handle_note_create("x")
        assert mode == "NOTE_ERROR"
        assert "não consegui criar" in reply.lower()

    def test_list(self, orch):
        reply, mode = orch._handle_note_list()
        assert mode == "NOTE_LIST"
        assert "Compras" in reply and "Ideias" in reply

    def test_list_empty(self, orch):
        orch.notes = SimpleNamespace(list_notes=lambda limit=10: [])
        reply, mode = orch._handle_note_list()
        assert mode == "NOTE_LIST"
        assert "não tem notas" in reply.lower()

    def test_search_hit(self, orch):
        reply, mode = orch._handle_note_search("viagem")
        assert mode == "NOTE_SEARCH"
        assert "Viagem Itália" in reply

    def test_search_miss(self, orch):
        reply, mode = orch._handle_note_search("xyz")
        assert mode == "NOTE_SEARCH"
        assert "não achei" in reply.lower()

    def test_search_empty_falls_through(self, orch):
        assert orch._handle_note_search("  ") is None
