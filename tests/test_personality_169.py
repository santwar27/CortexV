"""Task 169: prompt isolation, speech-only cleanup and fake-model evals."""
import ast
import inspect
import json
import re
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from evals import runner
from evals.core import EvalCase, EvalKind, EvalStatus, load_all_cases
from evals.cases.llm_personality import CASES
from src.core import prompts
from src.core.text_utils import strip_markdown_for_speech
from src.voice import laugh_try


@pytest.mark.parametrize("version", [*prompts.SYSTEM_PROMPTS, *prompts.DORMANT_PROMPTS])
def test_tool_body_selection(version):
    assert prompts.get_tool_prompt_body(version) == prompts.get_prompt_body(
        12 if version == 13 else version)


@pytest.mark.parametrize("version", ["13", "12", "frontier"])
@pytest.mark.parametrize("path", ["tools", "tools_followup", "tools_retry", "chat", "stream", "retry", "stream_retry"])
def test_real_payload_uses_its_path_body(monkeypatch, version, path):
    from src.core import engine as mod

    monkeypatch.setenv("CORTEX_PROMPT_VERSION", version)
    monkeypatch.setenv("CORTEX_MODEL_PROFILE", "8b")
    now = datetime(2026, 11, 17, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(mod, "datetime", SimpleNamespace(now=lambda: now))
    context = "[KNOWLEDGE] O passaporte de Pedro é AB123456." if path in {"retry", "stream_retry"} else ""
    memory = MagicMock()
    memory.get_user_profile.return_value = "USER PROFILE: exact bytes\n\n"
    memory.retrieve_relevant_context.return_value = context
    # Keep the real refusal recogniser when replacing the memory constructor.
    from src.core.memory import CortexMemory
    monkeypatch.setattr(mod, "CortexMemory", MagicMock(
        return_value=memory, _is_refusal=CortexMemory._is_refusal))
    engine = mod.OllamaEngine()
    engine.push_turn_context("TURN CONTEXT: exact bytes")
    tools = [{"type": "function", "function": {"name": "list_events"}}]
    calls = []
    counts = []

    def fake_ollama(**kwargs):
        calls.append(kwargs)
        reply = "The answer is AB123456."
        if path in {"retry", "stream_retry"} and len(calls) == 1:
            reply = "Desculpe, mas não posso fornecer informações de identificação pessoal."
        message = {"role": "assistant", "content": reply}
        if ((path == "tools_followup" and len(calls) == 1)
                or (path == "tools_retry" and len(calls) <= 2)):
            message = {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "list_events", "arguments": {}}}]}
        count = mod._est_tokens(kwargs["messages"][0]["content"])
        counts.append(count)
        result = {"message": message, "done": True, "prompt_eval_count": count, "eval_count": 7}
        return iter([result]) if kwargs.get("stream") else result

    monkeypatch.setattr(mod.ollama, "chat", fake_ollama)
    mod.turn_metrics.reset()
    try:
        if path.startswith("tools"):
            engine.chat_with_tools("List my events.", tools, lambda *_: "No events.")
        elif path.startswith("stream"):
            list(engine.chat_stream("qual meu passaporte"))
        else:
            engine.chat("qual meu passaporte")
    finally:
        usage = mod.turn_metrics.collect()
    assert len(calls) == (3 if path == "tools_retry" else
                          2 if path in {"tools_followup", "retry", "stream_retry"} else 1)
    for call in calls:
        has_tools = "tools" in call
        tool_flow = path.startswith("tools")  # its no-tools retry keeps the tool body
        assert has_tools or not tool_flow or call is calls[-1]
        body = (prompts.get_tool_prompt_body(engine.prompt_version) if tool_flow
                else prompts.get_prompt_body(engine.prompt_version))
        content = call["messages"][0]["content"]
        assert content == engine._build_system_prompt(body)["content"] + (
            "" if path.startswith("tools") else mod._NO_TOOLS_NOTE)
        assert "- Current date: 2026-11-17 (Tuesday)\n" in content
        assert "- Local timezone:" in content
        assert "USER PROFILE: exact bytes\n\nTURN CONTEXT: exact bytes\n\n" in content
        expected, absent = ((prompts._V13_PERSONA, prompts._V12_PERSONA)
                            if version == "13" and not tool_flow
                            else (prompts._V12_PERSONA, prompts._V13_PERSONA))
        assert expected in content and absent not in content
    assert usage["prompt_eval_count"] == sum(counts)
    assert usage["eval_count"] == 7 * len(calls)
    assert usage["calls"] == len(calls)


def test_persona_is_the_only_change_and_within_budget():
    before, after = prompts.SYSTEM_PROMPT_V12, prompts.SYSTEM_PROMPT_V13
    prefix12, rest12 = before.split("PERSONA:\n")
    prefix13, rest13 = after.split("PERSONA:\n")
    persona12, suffix12 = rest12.split("\n\nTOOLS", 1)
    persona13, suffix13 = rest13.split("\n\nTOOLS", 1)
    assert (prefix12, suffix12) == (prefix13, suffix13)
    assert persona12 != persona13
    assert 0 < len(after) - len(before) <= 600
    assert prompts.resolve_prompt_version("12") == 12
    assert prompts.get_prompt_body(12) == before == prompts.SYSTEM_PROMPT_FRONTIER
    assert prompts.LATEST_PROMPT_VERSION == prompts.resolve_prompt_version(None) == 13


def test_persona_required_boundaries():
    block = prompts._V13_PERSONA
    for phrase in (
        "Casual chat only", "slang", "self-mockery", "never flatter",
        "Sad news: serious and warm", "Loss: no jokes, laughs or emoji",
        "LOSS_NEWS_NOTE", "First sentence carries the answer (TTS)",
        "Never invent facts, events or capabilities", "UK English only, without Brazilian slang",
        "Tool-argument text (email body, event/reminder title, note, document) NEVER uses this voice",
        "Copy user text verbatim", "composed email bodies use neutral tone",
        "no slang, jokes, kkkk or emoji",
        "Address Pedro by name when natural; never 'user', 'Sir' or 'Senhor'",
    ):
        assert phrase in block
    assert not re.search(r"jarvis|spider.?man|peter parker|homem.aranha", block, re.I)


@pytest.mark.parametrize("drift", ["missing", "duplicate"])
def test_persona_replace_guard_rejects_anchor_drift(drift):
    tree = ast.parse(inspect.getsource(prompts))
    start = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Assert)
                 and "V13 anchor" in ast.unparse(n))
    block = ast.Module(body=tree.body[start:start + 3], type_ignores=[])
    before = prompts.SYSTEM_PROMPT_V12
    before = (before.replace(prompts._V12_PERSONA, "missing") if drift == "missing"
              else before + prompts._V12_PERSONA)
    with pytest.raises(AssertionError, match="V13 anchor"):
        exec(compile(block, "<V13 construction>", "exec"), {
            "SYSTEM_PROMPT_V12": before, "_V12_PERSONA": prompts._V12_PERSONA,
            "_V13_PERSONA": prompts._V13_PERSONA,
        })


LAUGHS = ["kkk", "Kkkk", "K" * 100, "rsrs", "Rs" * 50, "haha", "HAHAHA",
          "hahaaaa", "hehe", "heheee", "HeHeHe", "huahua", "HUA" * 50, "LOL", "lMaO"]


@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("token", LAUGHS)
def test_all_laugh_forms_use_selected_language(monkeypatch, lang, token):
    monkeypatch.setenv("CORTEX_SPOKEN_LAUGH_PT", "risada curta")
    monkeypatch.setenv("CORTEX_SPOKEN_LAUGH_EN", "short laugh")
    expected = "short laugh" if lang == "en" else "risada curta"
    assert strip_markdown_for_speech(f"({token})", lang) == f"({expected})"


@pytest.mark.parametrize("lang", ["pt", "en"])
def test_empty_laugh_drops_every_form(monkeypatch, lang):
    monkeypatch.setenv(f"CORTEX_SPOKEN_LAUGH_{lang.upper()}", "")
    assert strip_markdown_for_speech(" ".join(LAUGHS), lang) == ""


@pytest.mark.parametrize("lang,expected", [("pt", "Há, há!"), ("en", "Ha, ha!")])
def test_default_laugh(monkeypatch, lang, expected):
    monkeypatch.delenv(f"CORTEX_SPOKEN_LAUGH_{lang.upper()}", raising=False)
    assert strip_markdown_for_speech("kkkk", lang) == expected


def test_speech_language_fallback(monkeypatch):
    monkeypatch.setenv("CORTEX_SPOKEN_LAUGH_PT", "PT laugh")
    monkeypatch.setenv("CORTEX_SPOKEN_LAUGH_EN", "EN laugh")
    assert strip_markdown_for_speech("kkkk") == "PT laugh"
    assert strip_markdown_for_speech("(_kkkk_)") == "(PT laugh)"
    assert strip_markdown_for_speech("você conseguiu kkkk") == "você conseguiu PT laugh"
    assert strip_markdown_for_speech("You did it lol") == "You did it EN laugh"


def test_word_boundaries_and_markdown_unchanged():
    words = "cursos abcKkkdef lollipop hahahaX Xrsrs huahuana lol123 kk rs ha he"
    assert strip_markdown_for_speech(words) == words
    assert strip_markdown_for_speech(
        "# Title\n**bold** and _soft_ [link](https://example.test)\n- `code`\n```hide```"
    ) == "Title, bold and soft link, code,"


@pytest.mark.parametrize("emoji", ["🫠", "😂", "❤️", "👍🏽", "👨‍👩‍👧‍👦", "🇧🇷",
                                     "1️⃣", "#️⃣", "*️⃣", "☀️", "©️", "↔️",
                                     "🏴\U000e0067\U000e0062\U000e007f"])
def test_emoji_removed_without_changing_screen(emoji):
    original = f"Hello {emoji} world"
    assert strip_markdown_for_speech(original) == "Hello world"
    assert original == f"Hello {emoji} world"
    assert strip_markdown_for_speech("15% de 80 = 12; 1 + 2 = 3") == "15% de 80 = 12; 1 + 2 = 3"


@pytest.mark.parametrize("lang", ["pt", "en"])
def test_laugh_try_prints_then_speaks_once_and_nothing_else(monkeypatch, capsys, lang):
    def forbidden(*a, **k):
        pytest.fail("fake audition must not configure TTS, read files or start a process")

    monkeypatch.setattr(laugh_try, "resolve_tts_config", forbidden)
    monkeypatch.setattr(laugh_try.tempfile, "TemporaryDirectory", forbidden)
    monkeypatch.setattr(laugh_try.subprocess, "run", forbidden)
    calls = []

    def fake_speak(text, language):
        assert capsys.readouterr().out == text + "\n"
        calls.append((text, language))

    laugh_try.main(["--lang", lang], speak=fake_speak)
    assert calls == [(text, lang) for text in laugh_try.CANDIDATES[lang]]
    assert 4 <= len(calls) <= 6


@pytest.mark.parametrize("engine,synth_ok,command", [
    ("say", True, ["say", "-v", "test-pt", "Ha!"]),
    ("piper", True, ["afplay", "fake.wav"]),
    ("piper", False, ["say", "-v", "test-pt", "Ha!"]),
])
def test_shared_tts_preparation(engine, synth_ok, command):
    config = SimpleNamespace(engine=engine, piper_model_for=lambda lang: "fake.onnx",
                             say_voice_for=lambda lang: "test-" + lang)
    calls = []

    def synth(*args):
        calls.append(args)
        return synth_ok

    assert laugh_try.prepare_tts("Ha!", "pt", config, "fake.wav", synth=synth) == command
    assert calls == ([("fake.onnx", "Ha!", "fake.wav")] if engine == "piper" else [])


@pytest.mark.parametrize("lang", ["pt", "en"])
def test_listener_cleans_speech_but_preserves_transcript(monkeypatch, tmp_path, lang):
    from src.voice import listener
    from src.voice.tts_config import TTSConfig

    original = "Boa kkkk 😂!" if lang == "pt" else "Nice lol 😂!"
    monkeypatch.setenv(f"CORTEX_SPOKEN_LAUGH_{lang.upper()}", "audition laugh")
    monkeypatch.setattr(listener, "SILENCE_FLAG", str(tmp_path / "silence"))
    monkeypatch.setattr(listener, "TTS_CONFIG", TTSConfig(
        "say", "unused-pt", "unused-en", "test-pt", "test-en", "test"))
    response = MagicMock()
    response.__enter__.return_value = response
    response.iter_lines.return_value = iter([
        ("data: " + json.dumps(frame)).encode() for frame in (
            {"language": lang}, {"chunk": original},
            {"done": True, "status_mode": "OFFLINE", "reply": original},
        )
    ])
    monkeypatch.setattr(listener.requests, "post", lambda *a, **k: response)
    popen = MagicMock()
    popen.return_value.poll.return_value = 0
    monkeypatch.setattr(listener.subprocess, "Popen", popen)
    status, transcript = listener.stream_and_speak("test", stt_lang=lang)
    assert (status, transcript) == ("OFFLINE", original)
    assert popen.call_count == 1
    assert popen.call_args.args[0] == [
        "say", "-v", "test-" + lang,
        ("Boa" if lang == "pt" else "Nice") + " audition laugh !",
    ]


class FakeEngine:
    model_name = "fake-offline"
    prompt_version = 13

    def __init__(self, reply="Good answer.", tool=None, args=None):
        self.reply, self.tool, self.args = reply, tool, args or {}
        self.contexts = []
        self.current_context = None
        self.paths = []

    def push_turn_context(self, text):
        self.current_context = text

    def pop_turn_context(self):
        self.current_context = None

    def chat(self, *a, **k):
        self.paths.append("chat")
        self.contexts.append(self.current_context)
        return self.reply

    def chat_with_tools(self, text, specs, executor, **kwargs):
        self.paths.append("chat_with_tools")
        if self.tool:
            executor(self.tool, self.args)
        return self.chat(), self.tool


def run_fake(monkeypatch, case, **kwargs):
    engine = FakeEngine(**kwargs)
    monkeypatch.setattr(runner, "_get_llm_engine", lambda: engine)
    return runner._run_one(case), engine


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_personality_eval_uses_production_path(monkeypatch, case):
    _, engine = run_fake(monkeypatch, case, tool=case.expected_tool)
    action = case.expected_tool is not None
    assert case.kind == (EvalKind.LLM_TOOL_CALL if action else EvalKind.LLM_BEHAVIOR)
    assert engine.paths[0] == ("chat_with_tools" if action else "chat")
    assert ("chat_with_tools" in engine.paths) == action


@pytest.mark.parametrize("kind", [EvalKind.LLM_BEHAVIOR, EvalKind.LLM_TOOL_CALL])
@pytest.mark.parametrize("field,pattern,good,bad", [
    ("expected_reply_regex", r"(?i)\bboa\b", "BOA, Pedro!", "Nope."),
    ("forbidden_reply_regex", r"(?i)\blol\b", "Good.", "LOL!"),
    ("expected_first_sentence_regex", r"\b12\b", "12. That's it.", "Let's see. 12."),
])
def test_new_reply_checks_pass_and_fail(monkeypatch, kind, field, pattern, good, bad):
    case = EvalCase(name="reply", kind=kind, category="test", input="test",
                    expect_no_tool=True, drift_tolerant=False, **{field: pattern})
    assert run_fake(monkeypatch, case, reply=good)[0].status == EvalStatus.PASS
    assert run_fake(monkeypatch, case, reply=bad)[0].status == EvalStatus.FAIL


@pytest.mark.parametrize("field,value,good,bad", [
    ("expected_arg_equals", "pagar a conta", "pagar a conta", "Pagar a conta"),
    ("expected_arg_regex", r"\b10 minutos\b", "Atraso de 10 minutos", "atraso"),
    ("forbidden_arg_regex", r"(?i)\bmano\b|😂", "Atraso de 10 minutos", "MANO 😂"),
])
def test_new_argument_checks_pass_and_fail(monkeypatch, field, value, good, bad):
    case = EvalCase(name="arg", kind=EvalKind.LLM_TOOL_CALL, category="test", input="test",
                    expected_tool="send_email", drift_tolerant=False, **{field: {"body": value}})
    assert run_fake(monkeypatch, case, tool="send_email", args={"body": good})[0].status == EvalStatus.PASS
    assert run_fake(monkeypatch, case, tool="send_email", args={"body": bad})[0].status == EvalStatus.FAIL
    if field != "forbidden_arg_regex":
        assert run_fake(monkeypatch, case, tool="send_email")[0].status == EvalStatus.FAIL


@pytest.mark.parametrize("checks,good,bad", [
    ({"expected_contains_any": ["boa", "parabéns"]}, "BOA!", "Hello"),
    ({"expected_contains_all": ["poxa", "ajudar"]}, "Poxa, posso ajudar?", "Poxa."),
    ({"expected_not_contains": "sinto muito"}, "Boa!", "Sinto muito."),
])
def test_no_tool_reply_checks_opt_in_only(monkeypatch, checks, good, bad):
    case = EvalCase(name="no-tool", kind=EvalKind.LLM_TOOL_CALL, category="test", input="x",
                    expect_no_tool=True, check_no_tool_reply=True, drift_tolerant=False, **checks)
    assert run_fake(monkeypatch, case, reply=good)[0].status == EvalStatus.PASS
    assert run_fake(monkeypatch, case, reply=bad)[0].status == EvalStatus.FAIL
    assert run_fake(monkeypatch, replace(case, check_no_tool_reply=False), reply=bad)[0].status == EvalStatus.PASS
    assert run_fake(monkeypatch, case, reply=good, tool="send_email")[0].status == EvalStatus.FAIL


def test_loss_note_is_opt_in_and_cleared(monkeypatch):
    case = next(c for c in CASES if c.name == "personality_loss_pt")
    result, engine = run_fake(monkeypatch, case, reply="Sinto muito, Pedro.")
    assert result.status == EvalStatus.PASS
    assert engine.contexts == [prompts.LOSS_NEWS_NOTE]
    assert engine.current_context is None
    _, engine = run_fake(monkeypatch, replace(case, apply_turn_notes=False))
    assert engine.contexts == [None]


@pytest.mark.parametrize("kind", [EvalKind.LLM_BEHAVIOR, EvalKind.LLM_TOOL_CALL])
def test_loss_note_cleared_on_model_error(monkeypatch, kind):
    case = EvalCase(name="error", category="test", kind=kind, input="Morreu meu pai",
                    expect_no_tool=True, expected_contains_any=["sinto"], apply_turn_notes=True)
    engine = FakeEngine()

    def broken(*a, **k):
        assert engine.current_context == prompts.LOSS_NEWS_NOTE
        raise RuntimeError("fake failure")

    engine.chat = broken
    monkeypatch.setattr(runner, "_get_llm_engine", lambda: engine)
    assert runner._run_one(case).status == EvalStatus.ERROR
    assert engine.current_context is None


@pytest.mark.parametrize("case,good,bad,args", [
    (CASES[0], "BOA, parabéns!", "Sinto muito, parabéns.", None),
    (CASES[1], "Poxa, que chato. Quer uma ajuda?", "Poxa, que chato kkkk. Quer ajuda?", None),
    (CASES[2], "Sinto muito, Pedro.", "Sinto muito 😂 piada lol", None),
    (CASES[3], "", "", {"to": "joao@exemplo.com", "body": "Vou me atrasar 10 minutos."}),
    (CASES[4], "", "", {"title": "pagar a conta de luz"}),
    (CASES[5], "12, Pedro.", "Claro. É 12.", None),
    (CASES[6], "Brilliant, well done!", "Brilliant, irmão!", None),
])
def test_personality_cases_reject_regressions(monkeypatch, case, good, bad, args):
    case = replace(case, drift_tolerant=False)
    assert run_fake(monkeypatch, case, reply=good, tool=case.expected_tool, args=args)[0].status == EvalStatus.PASS
    if args:
        args = dict(args)
        key = "body" if "body" in args else "title"
        args[key] += " irmão 😂"
    assert run_fake(monkeypatch, case, reply=bad, tool=case.expected_tool, args=args)[0].status == EvalStatus.FAIL


def test_new_cases_are_opt_in_and_existing_cases_keep_defaults():
    all_cases = load_all_cases(include_llm=True)
    offline_names = {c.name for c in load_all_cases()}
    assert not offline_names.intersection(c.name for c in CASES)
    for case in all_cases:
        if case.name in {c.name for c in CASES}:
            continue
        assert case.expected_reply_regex is case.forbidden_reply_regex is None
        assert case.expected_first_sentence_regex is None
        assert not (case.check_no_tool_reply or case.apply_turn_notes or case.expected_arg_equals
                    or case.expected_arg_regex or case.forbidden_arg_regex)


@pytest.mark.parametrize("slang", ["rsrs", "huahua", "poxa", "valeu"])
def test_english_case_rejects_pt_laughs_and_slang(monkeypatch, slang):
    result, _ = run_fake(monkeypatch, CASES[-1], reply=f"Nice, {slang}!")
    assert result.status == EvalStatus.DRIFT
