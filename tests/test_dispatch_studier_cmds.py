"""
Regex battery for the Studier dispatch verbs (src/core/dispatch/studier_cmds.py).

Born from a live miss (s78, 2026-07-09): Pedro's "me de a lista de estudo
que esta na fila" fell through to the LLM, which hallucinated a refusal.
Two gaps: the courteous lead-in ("me dê/dá/mostra …") and "lista DE
estudo(s)". Same lesson as the s24 time-reflex: real phrasing is looser
than the regex author imagines — pin the real phrases.
"""
import pytest

from src.core.dispatch.studier_cmds import (
    LIST_STUDIES_RE,
    SHOW_STUDY_RE,
    STUDY_RE,
)


class TestListStudies:
    @pytest.mark.parametrize("phrase", [
        # a frase real que falhou ao vivo (sem acentos, prefixo cortês)
        "me de a lista de estudo que esta na fila",
        "me dê a lista de estudos",
        "me dá a lista de estudos",
        "me mostra a fila de estudos",
        "lista de estudos",
        "lista de estudo",
        "fila de estudo",
        # cobertura pré-existente que NÃO pode regredir
        "lista os estudos",
        "listar todos os estudos",
        "fila de estudos",
        "estudos pendentes",
        "study queue",
        "list studies",
        "mostra a fila",
    ])
    def test_matches(self, phrase):
        assert LIST_STUDIES_RE.match(phrase), phrase

    @pytest.mark.parametrize("phrase", [
        "me de a lista de compras",
        "lista minhas tarefas",
        "me mostra a tela",
        "estude rede neural",   # STUDIER_ADD, não LIST
        "lista",                # sozinho é ambíguo demais
    ])
    def test_rejects(self, phrase):
        assert not LIST_STUDIES_RE.match(phrase), phrase

    @pytest.mark.parametrize("phrase", [
        "me de a lista de estudo que esta na fila",
        "me mostra a fila de estudos",
        "lista de estudos",
    ])
    def test_show_study_does_not_steal(self, phrase):
        # SHOW_STUDY_RE ("ver/mostrar/show …") is registered BEFORE
        # LIST_STUDIES_RE in the registry — it must not capture these.
        assert not SHOW_STUDY_RE.match(phrase), phrase

    @pytest.mark.parametrize("phrase,mode", [
        ("ver as notas", "NOTE_LIST"),
        ("show my notes", "NOTE_LIST"),
    ])
    def test_show_study_leaves_other_lists_alone(self, phrase, mode):
        """"ver as notas" answered "Estudo 'as notas' não encontrado" (s88,
        Codex lote 9)."""
        from src.core.dispatch.registry import simulate
        assert not SHOW_STUDY_RE.match(phrase), phrase
        m = simulate(phrase)
        assert m is not None and m.mode == mode, phrase

    @pytest.mark.parametrize("phrase,ref", [
        ("ver estudo 1", "1"),
        ("ver estudo notas de rodapé", "notas de rodapé"),
        ("mostrar rede neural", "rede neural"),
    ])
    def test_show_study_still_shows_a_study(self, phrase, ref):
        m = SHOW_STUDY_RE.match(phrase)
        assert m and m.group(1).strip() == ref, phrase

    def test_add_still_captures_topic(self):
        m = STUDY_RE.match("estude rede neural")
        assert m and m.group(1) == "rede neural"
