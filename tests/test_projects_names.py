"""Task 122: code-owned names and active-project renames, without a model."""
import pytest

from src.core import projects
from tests.test_project_routes import orch, store, turn
from tests.test_ui_routes import client, ui_app


PEDRO = "vamos criar um projeto, um website para um pub, the queens head"
CLEAN = "website para um pub, the queens head"
PT_NOTICE = ' Para trocar o nome, diga "renomeia o projeto: <nome novo>".'
EN_NOTICE = ' To change the name, say "rename project: <new name>".'


@pytest.mark.parametrize("stream", [False, True])
def test_chip_sentence_creates_clean_name_once(orch, store, stream):
    reply, mode = turn(orch, "projeto: " + PEDRO, stream)
    assert (reply, mode) == (
        f'Projeto “{CLEAN}” criado e ativo. Esta conversa entrou nele.' + PT_NOTICE,
        "PROJECT_ACTIVE")
    first = store.active()
    assert first.name == CLEAN
    assert store.project_for_conversation("A").id == first.id
    assert turn(orch, "projeto: " + PEDRO, stream) == (
        f'Projeto “{CLEAN}” ativo. Esta conversa entrou nele.', "PROJECT_ACTIVE")
    assert [p.id for p in store.list()] == [first.id]


@pytest.mark.parametrize("frame", [
    "vamos criar um projeto, ", "vamos fazer o novo projeto: ",
    "vamos começar projeto - ", "vamos comecar um projeto de ",
    "vamos abrir um novo projeto para ", "quero criar um projeto chamado ",
    "quero fazer projeto: ", "cria um projeto, ", "criar projeto de ",
    "um projeto de ", "um projeto para ", "um projeto pra ",
    "um projeto sobre ", "um projeto chamado ", "novo projeto: ",
    "let's create a project: ", "let's start a new project - ",
    "let’s make a project called ", "a project for ", "a project about ",
    "a project called ", "new project: ",
])
def test_lead_in_forms_before_find_and_create(store, frame):
    p = store.create_or_get(frame + "website para The Queens Head")
    assert p.name == "website para The Queens Head"
    assert store.find(frame + "website para The Queens Head").id == p.id
    assert store.create_or_get("website para The Queens Head").id == p.id


@pytest.mark.parametrize("name,expected", [
    ("cortex", "cortex"), ("The Queens Head", "The Queens Head"),
    ("novo projeto: The Queens Head", "The Queens Head"),
    ("um website para o pub", "website para o pub"),
    ("uma aplicação", "aplicação"), ("o site", "site"), ("a página", "página"),
    ("a website", "website"), ("an app", "app"), ("the website", "website"),
    ("vamos criar um projeto", "vamos criar um projeto"),
    ("new project:", "new project:"),
    ("projeto do coração", "projeto do coração"),
    ("vamos criar um projetor", "vamos criar um projetor"),
])
def test_plain_names_articles_and_empty_fallback(orch, store, name, expected):
    assert turn(orch, "projeto: " + name)[1] == "PROJECT_ACTIVE"
    assert store.active().name == expected


@pytest.mark.parametrize("legacy", [False, True])
def test_existing_clean_or_legacy_name_is_reused(orch, store, legacy):
    # Seed an actual pre-task record, including the old parser's 80-char cap.
    raw = "vamos criar um projeto, " + "um website para um pub, " + "The Queens Head " * 4
    cleaned = ("website para um pub, " + "The Queens Head " * 4).strip()[:80]
    name = raw[:80] if legacy else cleaned
    assert store.available()
    with store._conn:
        store._conn.execute(
            "INSERT INTO projects (name, name_key, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (name, projects._key(name), projects._now(), projects._now()))
    old = store.list()[0]
    assert store.find(raw).id == old.id
    assert store.create_or_get(raw).id == old.id
    reply, mode = turn(orch, "projeto: " + raw, True)
    assert mode == "PROJECT_ACTIVE" and "criado" not in reply and PT_NOTICE not in reply
    assert store.active().name == name and len(store.list()) == 1


@pytest.mark.parametrize("prefix,name,notice", [
    ("projeto", "um dois tres quatro cinco", PT_NOTICE),
    ("project", "one two three four five", EN_NOTICE),
    ("project", "Queens, Head", EN_NOTICE),
    ("projeto", "um dois tres quatro", ""),
])
def test_notice_threshold_and_creation_only(orch, prefix, name, notice):
    reply, _ = turn(orch, f"{prefix}: {name}")
    assert (PT_NOTICE in reply or EN_NOTICE in reply) == bool(notice)
    if notice:
        assert reply.endswith(notice)
    reply, _ = turn(orch, f"{prefix}: {name}", True)
    assert PT_NOTICE not in reply and EN_NOTICE not in reply


RENAMES = [
    (f"{verb} o projeto{separator}", "pt")
    for verb in ("renomeia", "renomear", "renomeie")
    for separator in (" para ", ": ")
] + [
    (f"{verb} o nome do projeto{separator}", "pt")
    for verb in ("troca", "muda")
    for separator in (" para ", ": ")
] + [
    (f"rename {article}project{separator}", "en")
    for article in ("", "the ")
    for separator in (" to ", ": ")
] + [("renomear projeto: ", "pt")]  # Keep the original Task 122 form.


@pytest.mark.parametrize("command,lang", RENAMES)
def test_rename_separator_forms_route_literal_name(command, lang):
    assert projects.parse_command(command + "Website The Queens Head") == (
        "renamed", "Website The Queens Head", lang)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("command,lang", RENAMES)
def test_rename_active_preserves_record_and_relations(orch, store, stream, command, lang):
    p = store.create_or_get("Antigo", "Objetivo literal")
    other = store.create_or_get("Outro")
    store.link_conversation("A", other.id)  # Rename ACTIVE, not the linked project.
    store.link_conversation("B", p.id)
    store.set_active(p.id)
    entry = store.add_entry(p.id, "feito", "feito literal", "ui", "B")
    obs = store.observe(p.id, "written", "write_file", "pub.html", "ui", "B")
    orch._pending_query = "busca pendente"
    assert orch._is_pure_chat(command + "The Queens Head") is False
    expected = ("Project renamed" if lang == "en" else "Projeto renomeado")
    assert turn(orch, command + "The Queens Head", stream) == (
        f"{expected}: Antigo → The Queens Head.", "PROJECT_RENAMED")
    renamed = store.active()
    assert renamed.id == p.id and renamed.name == "The Queens Head"
    assert renamed.objective == p.objective and renamed.created_at == p.created_at
    assert renamed.updated_at > p.updated_at
    assert store.find("the queens head").id == p.id and store.find("Antigo") is None
    assert store.entries(p.id) == [entry] and store.observations(p.id) == [obs]
    assert store.conversations(p.id) == ["B"]
    assert store.project_for_conversation("B").name == renamed.name
    assert store.project_for_conversation("A").id == other.id
    assert orch._pending_query is None


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("command,lang", RENAMES)
def test_rename_none_taken_and_same_key(orch, store, stream, command, lang):
    expected = ('No active project. Say "project: name" first.' if lang == "en" else
                'Nenhum projeto ativo. Diga "projeto: nome" primeiro.')
    assert turn(orch, command + "Novo", stream) == (expected, "PROJECT_NONE")
    p = store.create_or_get("Antigo")
    taken = store.create_or_get("Currículo")
    store.set_active(p.id)
    expected = ("A project called CURRICULO already exists." if lang == "en" else
                "Já existe um projeto chamado CURRICULO.")
    assert turn(orch, command + "CURRICULO", stream) == (expected, "PROJECT_TAKEN")
    assert store.active().name == p.name and store.get(taken.id).name == taken.name
    assert turn(orch, command + "ANTIGO", stream)[1] == "PROJECT_RENAMED"
    assert store.active().id == p.id and store.active().name == "ANTIGO"


@pytest.mark.parametrize("text", [
    *(command.rstrip() for command, _ in RENAMES),
    "renomeia o projeto", "renomear o projeto", "renomeie o projeto",
    "troca o nome do projeto", "muda o nome do projeto",
    "rename project", "rename the project",
    "renomeia o projeto:", "renomear projeto:  ", "muda o nome do projeto para ",
    "muda o nome do projeto:", "rename project: ", "rename the project to ",
    "quero renomear projeto: Casa", "please rename project: House",
    "o projeto da casa está atrasado", "vamos criar um projeto, um website",
    "a project for a pub", "rename the project today", "renomeia o projeto antigo",
    "muda o nome do projetopara Casa",
    "renomeia o projetopara Casa", "renomeia o projeto paraiso",
    "rename project today",
])
def test_non_commands_and_empty_rename_are_not_routed(orch, text):
    assert projects.parse_command(text) is None
    assert orch._handle_projects(text) is None


def test_rename_failure_rolls_back(orch, store):
    p = store.create_or_get("Antigo")
    store.set_active(p.id)
    with store._conn:
        store._conn.execute("CREATE TRIGGER fail_rename BEFORE UPDATE ON projects "
                            "BEGIN SELECT RAISE(ABORT, 'disk'); END")
    assert turn(orch, "renomeia o projeto: Novo", True) == (
        "Não consegui guardar o projeto — tenta de novo.", "PROJECT_ERROR")
    assert store.active() == store.get(p.id) and store.active().name == "Antigo"
    assert not store._conn.in_transaction


def test_rename_is_literal_and_visible_to_other_connections(orch, store):
    turn(orch, "projeto: Antigo")
    other = projects.ProjectStore(store.db_path)
    try:
        pid = other.active().id
        assert turn(orch, "Cortex, RENAME PROJECT:   a website  for a pub", True) == (
            "Project renamed: Antigo → a website for a pub.", "PROJECT_RENAMED")
        assert other.active().id == pid and other.active().name == "a website for a pub"
        assert len(other.list()) == 1
    finally:
        if other._conn:
            other._conn.close()


@pytest.mark.parametrize("command", ["renomeia o projeto: Novo", "rename project: New"])
def test_rename_store_read_failure_does_not_look_like_no_active(orch, store, command):
    assert store.available()
    with store._conn:
        store._conn.execute("DROP TABLE active_project")
    reply, mode = turn(orch, command, True)
    assert mode == "PROJECT_ERROR"
    assert reply == ("I couldn't save the project — please try again." if command.startswith("rename") else
                     "Não consegui guardar o projeto — tenta de novo.")


def test_project_list_api_exposes_renamed_name(client, ui_app, orch, store):
    turn(orch, "projeto: Antigo")
    pid = store.active().id
    assert turn(orch, "renomeia o projeto: Novo", True)[1] == "PROJECT_RENAMED"
    data = client.get("/ui/projects", headers={"X-Cortex-Key": "test-key-xyz"}).json()
    assert data["available"] is True
    assert [(p["id"], p["name"]) for p in data["projects"]] == [(pid, "Novo")]
