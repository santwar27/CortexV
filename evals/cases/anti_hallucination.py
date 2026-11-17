"""Hallucination detector behaviour (bloco H sessão 11)."""
from evals.core import EvalCase, EvalKind


CASES: list[EvalCase] = [
    # ------- POSITIVE: must be flagged -------
    EvalCase(
        name="canonical_tocando",
        kind=EvalKind.DETECTOR,
        category="anti_hallucination",
        detector_input="Tocando 'BACKSTAGE' de Matuê.",
        expected_flagged=True,
        description="The exact Matuê hallucination shape from the original bug.",
        tags=["regression"],
    ),
    EvalCase(
        name="paren_pausa",
        kind=EvalKind.DETECTOR,
        category="anti_hallucination",
        detector_input="(pausa) ... agora a música está tocando!",
        expected_flagged=True,
    ),
    EvalCase(
        name="paren_musica_comeca",
        kind=EvalKind.DETECTOR,
        category="anti_hallucination",
        detector_input="(música começa) Boa escolha.",
        expected_flagged=True,
    ),
    EvalCase(
        name="email_enviei",
        kind=EvalKind.DETECTOR,
        category="anti_hallucination",
        detector_input="Enviei o e-mail para João.",
        expected_flagged=True,
    ),
    EvalCase(
        name="english_playing",
        kind=EvalKind.DETECTOR,
        category="anti_hallucination",
        detector_input="Playing 'Bohemian Rhapsody' by Queen.",
        expected_flagged=True,
    ),
    EvalCase(
        name="lembrete_criado",
        kind=EvalKind.DETECTOR,
        category="anti_hallucination",
        detector_input="Lembrete criado: comprar leite.",
        expected_flagged=True,
    ),

    # ------- NEGATIVE: must NOT be flagged -------
    EvalCase(
        name="knowledge_about_queen",
        kind=EvalKind.DETECTOR,
        category="anti_hallucination",
        detector_input="Queen tocou no Live Aid em 1985, um dos shows mais memoráveis.",
        expected_flagged=False,
        description="Past-tense narrative about real history must not trip.",
    ),
    EvalCase(
        name="refusal_no_spotify",
        kind=EvalKind.DETECTOR,
        category="anti_hallucination",
        detector_input="Não consigo tocar música sem o Spotify aberto.",
        expected_flagged=False,
    ),
    EvalCase(
        name="biography_matue",
        kind=EvalKind.DETECTOR,
        category="anti_hallucination",
        detector_input="O Matuê é um rapper brasileiro de São Paulo. Conhecido por trap.",
        expected_flagged=False,
        description="Knowledge query about a music artist must not trip.",
    ),
    EvalCase(
        name="generic_greeting",
        kind=EvalKind.DETECTOR,
        category="anti_hallucination",
        detector_input="Oi! Em que posso ajudar?",
        expected_flagged=False,
    ),
    EvalCase(
        name="empty_string",
        kind=EvalKind.DETECTOR,
        category="anti_hallucination",
        detector_input="",
        expected_flagged=False,
    ),

    # ------- EDGE: multi-line -------
    EvalCase(
        name="action_in_second_paragraph_not_flagged",
        kind=EvalKind.DETECTOR,
        category="anti_hallucination",
        detector_input=(
            "O Matuê tem várias músicas famosas.\n"
            "Tocando 'BACKSTAGE' agora."
        ),
        expected_flagged=False,
        description=(
            "Action claim in a later line should not trip — only "
            "first-line claims are diagnostic. Conservative by design."
        ),
    ),
]
