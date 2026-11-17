"""
Tests for src/core/memory_doctor.py — the proactive Palace anti-poison sweep.

`scan` is pure, so each poison category is tested with synthetic conversation
docs + active facts. `load_palace_data` is exercised fail-soft. The refusal /
error definitions are single-sourced from CortexMemory, so a refusal the write
guard would block is exactly what the doctor flags retroactively.
"""
import pytest

from src.core import memory_doctor as md
from src.core.memory import CortexMemory


def _doc(id, texto):
    return {"id": id, "tipo": "conversation", "texto": texto}


def _turn(user, cortex):
    return f"User: {user}\nCortex: {cortex}"


def _fact(suj, pred, obj):
    return {"sujeito": suj, "predicado": pred, "objeto": obj, "valid_until": None}


# --- refusals (single-sourced from the write guard) ------------------------

def test_flags_stored_refusal():
    doc = _doc(1, _turn("quem descobriu o Brasil?", "Não posso ajudar com isso."))
    out = md.scan([doc], [])
    assert len(out.refusals) == 1 and out.refusals[0].ref == "1"


def test_refusal_definition_matches_guard():
    # Anything CortexMemory._is_refusal flags on the Cortex side, the doctor
    # must flag too — no drift between block-going-forward and hunt-backward.
    reply = "Sinto muito, mas não posso fazer isso."
    assert CortexMemory._is_refusal(reply)
    out = md.scan([_doc(7, _turn("faça X", reply))], [])
    assert len(out.refusals) == 1


def test_normal_answer_not_flagged_as_refusal():
    doc = _doc(2, _turn("quem descobriu o Brasil?", "Pedro Álvares Cabral, em 1500."))
    assert md.scan([doc], []).refusals == []


# --- error markers ---------------------------------------------------------

def test_flags_stored_engine_error():
    doc = _doc(3, _turn("oi", "[OFFLINE] sem conexão"))
    assert len(md.scan([doc], []).errors) == 1


# --- role confusion (s40) --------------------------------------------------

def test_flags_identity_swap():
    doc = _doc(4, _turn("quem é você?", "Eu sou o Pedro, engenheiro de software."))
    assert len(md.scan([doc], []).role_confusion) == 1


def test_flags_profile_misattribution():
    doc = _doc(5, _turn("cadê meu perfil?", "Esta é minha configuração de perfil pessoal."))
    assert len(md.scan([doc], []).role_confusion) == 1


def test_normal_identity_not_flagged():
    doc = _doc(6, _turn("quem é você?", "Sou o Cortex, seu assistente."))
    assert md.scan([doc], []).role_confusion == []


# --- contradictions in active facts ----------------------------------------

def test_flags_contradictory_active_facts():
    facts = [_fact("user", "mora_em", "Londres"), _fact("user", "mora_em", "Lisboa")]
    out = md.scan([], facts)
    assert len(out.contradictions) == 1
    assert "Londres" in out.contradictions[0].snippet
    assert "Lisboa" in out.contradictions[0].snippet


def test_single_value_fact_not_contradiction():
    facts = [_fact("user", "mora_em", "Londres"), _fact("user", "trabalha_em", "escritório")]
    assert md.scan([], facts).contradictions == []


def test_same_value_repeated_not_contradiction():
    facts = [_fact("user", "mora_em", "Londres"), _fact("user", "mora_em", "Londres")]
    assert md.scan([], facts).contradictions == []


# --- duplicates ------------------------------------------------------------

def test_flags_exact_duplicate_turn_once():
    t = _turn("que horas são?", "São 20:00.")
    out = md.scan([_doc(1, t), _doc(2, t), _doc(3, t)], [])
    assert len(out.duplicates) == 1  # reported once, not per extra copy


def test_distinct_turns_not_duplicates():
    out = md.scan([_doc(1, _turn("a", "b")), _doc(2, _turn("c", "d"))], [])
    assert out.duplicates == []


# --- scan bookkeeping ------------------------------------------------------

def test_counts_scanned():
    out = md.scan([_doc(1, _turn("a", "b"))], [_fact("user", "x", "y")])
    assert out.docs_scanned == 1 and out.facts_scanned == 1
    assert out.total() == 0


# --- render ----------------------------------------------------------------

def test_render_clean():
    out = md.render(md.scan([_doc(1, _turn("a", "b"))], []))
    assert "Limpo" in out


def test_render_lists_findings_no_tables():
    doc = _doc(9, _turn("faça X", "Não posso fazer isso."))
    text = md.render(md.scan([doc], []))
    assert "Recusas" in text
    assert "delete_document" in text   # points at the manual remedy
    assert "|" not in text             # bullets, never markdown tables
    assert "Read-only" in text


def test_render_truncates_per_category():
    docs = [_doc(i, _turn("q", "Não posso.")) for i in range(30)]
    text = md.render(md.scan(docs, []), limit=10)
    assert "+20 mais" in text


# --- load_palace_data fail-soft --------------------------------------------

def test_load_missing_db_is_fail_soft(tmp_path):
    path = tmp_path / "nope.db"
    docs, facts = md.load_palace_data(str(path))
    # A non-existent DB path → mode=ro fails to open; the loader degrades to
    # empty lists, never raises.
    assert docs == [] and facts == []
    # M3-A7: and it must NOT have created an empty DB at the mistyped path
    # (the old writable open did: makedirs + CREATE TABLE).
    assert not path.exists()


def test_load_does_not_mutate_the_patient(tmp_path):
    # M3-A7 headline: auditing a suspect/foreign DB leaves it byte-identical.
    # This file is a valid SQLite DB but NOT a palace — the old writable open
    # would run _init_schema and write our schema into the patient.
    import sqlite3
    path = tmp_path / "suspect.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE unrelated (x INTEGER)")
    conn.commit()
    conn.close()
    before = path.read_bytes()
    docs, facts = md.load_palace_data(str(path))
    assert docs == [] and facts == []   # fail-soft: no palace tables there
    assert path.read_bytes() == before  # and the patient was not touched


def test_main_runs_clean(tmp_path, capsys):
    from src.memory.palace import Palace
    Palace(db_path=str(tmp_path / "empty.db")).close()        # a real, empty Palace
    rc = md.main(["--db", str(tmp_path / "empty.db")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Memory doctor" in out and "Limpo" in out


@pytest.mark.parametrize("kind", ["missing", "corrupt", "not_a_palace"])
def test_main_never_calls_an_unread_db_clean(tmp_path, capsys, kind):
    """"✓ Limpo", exit 0, for a DB it could not read (s88, Codex lote 6)."""
    import sqlite3
    path = tmp_path / "x.db"
    if kind == "corrupt":
        path.write_text("lixo")
    elif kind == "not_a_palace":
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE unrelated (x INTEGER)")
        conn.commit()
        conn.close()
    rc = md.main(["--db", str(path)])
    out = capsys.readouterr().out
    assert rc == 2
    assert "Não consegui ler" in out and "Limpo" not in out
