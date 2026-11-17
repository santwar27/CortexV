"""
Tests for src/core/system_status.py + its dispatch regex (Bug B-1, s57).

The handler exists so "status do sistema?" reports real pulled facts instead of
falling through to the LLM+retrieval path (where a weakly-relevant past turn was
woven into a fabricated status — ideas_backlog Bug B). Coverage:
  - assemble() is pure → exercised directly over signal dicts.
  - gather() is fail-soft → a broken source degrades to None, never raises.
  - the regex matches system-status phrasings and NOT casual greetings.
"""
from src.core import system_status as ss
from src.core.dispatch.system_status import SYSTEM_STATUS_RE


# --- assemble (pure) -------------------------------------------------------


def test_assemble_all_healthy():
    out = ss.assemble({
        "jobs": {"total": 3, "stale": []},
        "gate_pending": 0,
        "palace_docs": 56,
    })
    assert "Tudo certo" in out
    assert "56 documentos" in out
    assert "Nenhuma ação pendente" in out
    # speakable: no emoji / bullets / tables
    assert "•" not in out and "|" not in out and "✓" not in out


def test_assemble_flags_stale_jobs():
    out = ss.assemble({
        "jobs": {"total": 3, "stale": ["backup", "weekly"]},
        "gate_pending": 0,
        "palace_docs": 10,
    })
    assert "Atenção" in out
    assert "backup, weekly" in out
    assert "2 trabalhos agendados" in out


def test_assemble_singular_grammar():
    out = ss.assemble({
        "jobs": {"total": 1, "stale": ["studier"]},
        "gate_pending": 1,
        "palace_docs": 1,
    })
    assert "1 trabalho agendado não rodou" in out
    assert "1 documento." in out
    assert "1 ação esperando aprovação" in out


def test_assemble_pending_approvals():
    out = ss.assemble({
        "jobs": {"total": 3, "stale": []},
        "gate_pending": 2,
        "palace_docs": 5,
    })
    assert "2 ações esperando aprovação" in out


def test_assemble_degrades_when_signals_missing():
    # Every source failed (None) → a coherent reply that claims nothing:
    # it used to say "Tudo certo por aqui." (s88, Codex lote 13).
    out = ss.assemble({"jobs": None, "gate_pending": None, "palace_docs": None})
    assert out == "Não consegui verificar o estado do sistema agora."
    out_en = ss.assemble({"jobs": None, "gate_pending": None, "palace_docs": None},
                         lang="en")
    assert out_en == "I couldn't check the system's state just now."


def test_a_failed_job_check_is_not_all_good():
    out = ss.assemble({"jobs": None, "gate_pending": 0, "palace_docs": 10})
    assert "Tudo certo" not in out
    assert out.startswith("Não consegui verificar os trabalhos agendados agora.")
    assert "A memória tem 10 documentos." in out


# --- gather (fail-soft) ----------------------------------------------------


def test_gather_returns_all_keys():
    sig = ss.gather()
    assert set(sig) == {"jobs", "gate_pending", "palace_docs"}


def test_gather_palace_docs_missing_db_is_none(tmp_path):
    assert ss._gather_palace_docs(tmp_path / "nope.db") is None


def test_gather_jobs_failsoft(monkeypatch):
    monkeypatch.setattr(ss.heartbeat, "check", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert ss._gather_jobs() is None


def test_summary_is_a_nonempty_string(monkeypatch):
    monkeypatch.setattr(ss, "gather", lambda: {
        "jobs": {"total": 2, "stale": []}, "gate_pending": 0, "palace_docs": 7,
    })
    out = ss.summary()
    assert isinstance(out, str) and out


# --- regex -----------------------------------------------------------------


def test_regex_matches_status_phrasings():
    for q in [
        "status do sistema",
        "status do cortex",
        "qual o status do sistema",
        "como está o sistema",
        "como tá o sistema",
        "como anda o sistema?",
        "está tudo funcionando?",
        "tá tudo rodando",
        "diagnóstico do sistema",
        "saúde do sistema",
        "system status",
        "status report",
        "are you healthy?",
        "is everything ok?",
        "health check",
        "diagnostics",
    ]:
        assert SYSTEM_STATUS_RE.match(q), f"should match: {q!r}"


def test_regex_does_not_eat_greetings_or_chat():
    for q in [
        "oi cortex",
        "oi, tudo bem?",
        "tudo bem?",
        "como você está?",
        "está tudo bem?",
        "are you working on the report",  # trailing words → fall through
        "qual o status do jogo do Brasil",  # not the system
        "me conta uma fofoca",
    ]:
        assert not SYSTEM_STATUS_RE.match(q), f"should NOT match: {q!r}"
