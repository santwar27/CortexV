"""V13 personality: opt-in real-model cases, never part of offline evals.

Casual cases use production's no-tools chat path; actions use chat_with_tools.
Loss guidance is injected exactly as on a loss turn.
"""
from evals.core import EvalCase, EvalKind
from src.core.text_utils import EMOJI_RE, LAUGH_RE


_SLANG = r"\b(?:irmão|irmao|tmj|cara|k{3,}|mano)\b"
_LAUGHS_OR_EMOJI = rf"(?i)(?:{LAUGH_RE.pattern}|{EMOJI_RE.pattern})"


CASES = [
    EvalCase(
        name="personality_exam_pt", kind=EvalKind.LLM_BEHAVIOR,
        category="llm_behavior", input="passei na prova de cálculo!!",
        language="pt",
        expected_contains_any=["parabéns", "parabens", "boa", "mandou bem", "mandou muito bem", "arrasou",
                               "comemorar", "conquista", "não acredito"],
        expected_not_contains="sinto muito", tags=["personality"],
        description="Celebrate the exam result without a tool or condolences.",
    ),
    EvalCase(
        name="personality_broken_car_pt", kind=EvalKind.LLM_BEHAVIOR,
        category="llm_behavior", input="meu carro quebrou hoje cedo",
        language="pt",
        expected_reply_regex=(
            r"(?is)(?=.*(?:poxa|putz|que (?:chato|merda|pena|situação)|sinto muito|complicado))"
            r"(?=.*(?:quer.{0,40}ajud|posso.{0,40}ajud|se precisar|vamos ver|vamos.{0,30}resolver|"
            r"precisa.{0,30}ajud)).+"
        ),
        forbidden_reply_regex=r"(?i)\bk{3,}\b", tags=["personality"],
        description="Empathy and an offer of concrete help; no tool or k-laugh.",
    ),
    EvalCase(
        name="personality_loss_pt", kind=EvalKind.LLM_BEHAVIOR,
        category="llm_behavior", input="Morreu meu pai",
        language="pt", apply_turn_notes=True,
        expected_contains_any=["sinto muito", "meus sentimentos"],
        forbidden_reply_regex=(
            _LAUGHS_OR_EMOJI + r"|\b(?:piada\w*|brincadeir\w*|zoeir\w*|engraçad\w*|joke\w*|funny)\b"
        ),
        tags=["personality", "loss"],
        description="Existing loss trigger: warm condolence, no laughter, emoji or joke words.",
    ),
    EvalCase(
        name="personality_email_neutral_pt", kind=EvalKind.LLM_TOOL_CALL,
        category="llm_tool_call",
        input="manda um email pro joao@exemplo.com dizendo que vou me atrasar 10 minutos",
        language="pt", expected_tool="send_email",
        expected_arg_equals={"to": "joao@exemplo.com"},
        expected_arg_regex={"body": r"(?is)(?=.*10)(?=.*minut)(?=.*atras).+"},
        forbidden_arg_regex={"body": rf"(?i)(?:{_SLANG}|{LAUGH_RE.pattern}|{EMOJI_RE.pattern})"},
        tags=["personality", "tool_arg_fidelity"],
        description="Compose a neutral email body; no chat slang, laughter or emoji.",
    ),
    EvalCase(
        name="personality_reminder_verbatim_pt", kind=EvalKind.LLM_TOOL_CALL,
        category="llm_tool_call", input="me lembra de pagar a conta de luz amanhã às 9",
        language="pt", expected_tool="create_reminder",
        expected_arg_equals={"title": "pagar a conta de luz"},
        tags=["personality", "tool_arg_fidelity"],
        description="Reminder title stays verbatim, with no persona decorations.",
    ),
    EvalCase(
        name="personality_answer_first_pt", kind=EvalKind.LLM_BEHAVIOR,
        category="llm_behavior", input="quanto é 15% de 80?",
        language="pt",
        expected_first_sentence_regex=r"\b12\b", tags=["personality"],
        description="The first spoken sentence contains 12, with no tool.",
    ),
    EvalCase(
        name="personality_bug_en", kind=EvalKind.LLM_BEHAVIOR,
        category="llm_behavior", input="I finally fixed that bug!",
        language="en",
        expected_contains_any=["nice", "well done", "great", "brilliant", "yes", "nailed", "good",
                               "proper", "sorted", "did it", "cracked", "nice one"],
        forbidden_reply_regex=(
            rf"(?i)(?:{_SLANG}|\b(?:(?:rs){{2,}}|(?:hua){{2,}}|boa|parabéns|parabens|poxa|valeu)\b)"
        ),
        tags=["personality"],
        description="Celebrate in English without Brazilian slang.",
    ),
]
