"""Apple Notes dispatch behaviour (s40, Claude-B). S2 — executes directly."""
from evals.core import EvalCase, EvalKind


def _create(name, text):
    return EvalCase(
        name=name, kind=EvalKind.DISPATCH, category="notes",
        input=text, expected_mode="NOTE_CREATE", expected_matcher="NOTE_CREATE_RE",
    )


CASES: list[EvalCase] = [
    # ----- create -----
    _create("note_create_anota", "anota comprar leite e ovos"),
    _create("note_create_anota_ai", "anota aí: ligar pro dentista"),
    _create("note_create_anote_que", "anote que a reunião é quinta"),
    _create("note_create_cria_nota", "cria uma nota: lista de compras"),
    _create("note_create_nova_nota", "nova nota reunião de sexta"),
    _create("note_create_en", "create a note about the trip"),
    # ----- list -----
    EvalCase(
        name="note_list_pt", kind=EvalKind.DISPATCH, category="notes",
        input="minhas notas",
        expected_mode="NOTE_LIST", expected_matcher="NOTE_LIST_RE",
    ),
    EvalCase(
        name="note_list_en", kind=EvalKind.DISPATCH, category="notes",
        input="list my notes",
        expected_mode="NOTE_LIST", expected_matcher="NOTE_LIST_RE",
    ),
    # ----- search -----
    EvalCase(
        name="note_search_pt", kind=EvalKind.DISPATCH, category="notes",
        input="acha a nota sobre o projeto",
        expected_mode="NOTE_SEARCH", expected_matcher="NOTE_SEARCH_RE",
    ),
    EvalCase(
        name="note_search_en", kind=EvalKind.DISPATCH, category="notes",
        input="find my notes about taxes",
        expected_mode="NOTE_SEARCH", expected_matcher="NOTE_SEARCH_RE",
    ),
    # ----- v1.1 (s73): append + folder targeting -----
    EvalCase(
        name="note_append_title_first_pt", kind=EvalKind.DISPATCH, category="notes",
        input="acrescenta na nota Mercado: pão integral",
        expected_mode="NOTE_APPEND", expected_matcher="NOTE_APPEND_RE",
    ),
    EvalCase(
        name="note_append_text_first_pt", kind=EvalKind.DISPATCH, category="notes",
        input="adiciona pão integral na nota Mercado",
        expected_mode="NOTE_APPEND", expected_matcher="NOTE_APPEND_TAIL_RE",
    ),
    EvalCase(
        name="note_append_en", kind=EvalKind.DISPATCH, category="notes",
        input="add milk to the note Groceries",
        expected_mode="NOTE_APPEND", expected_matcher="NOTE_APPEND_TAIL_RE",
    ),
    EvalCase(
        name="note_create_folder_pt", kind=EvalKind.DISPATCH, category="notes",
        input="anota na pasta Trabalho: revisar contrato",
        expected_mode="NOTE_CREATE", expected_matcher="NOTE_CREATE_FOLDER_RE",
        description="Folder create must win over plain create (route order).",
    ),
    EvalCase(
        name="note_create_folder_en", kind=EvalKind.DISPATCH, category="notes",
        input="create a note in folder Work: ship the fix",
        expected_mode="NOTE_CREATE", expected_matcher="NOTE_CREATE_FOLDER_RE",
    ),
    # ----- must NOT fire notes (route elsewhere / OFFLINE) -----
    EvalCase(
        name="note_neg_nota_fiscal", kind=EvalKind.DISPATCH, category="notes",
        input="nota fiscal da compra",
        expected_mode="OFFLINE",
        description="'nota' noun without a create verb must not fire NOTE_CREATE.",
    ),
    EvalCase(
        name="note_neg_anotado", kind=EvalKind.DISPATCH, category="notes",
        input="anotado com sucesso",
        expected_mode="OFFLINE",
        description="Past participle 'anotado' is not a create command.",
    ),
]
