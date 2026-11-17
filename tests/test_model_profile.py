"""C3 (s76): the MODEL_PROFILE flip and the 8B-bound guard inventory.

Three layers under test:
1. The resolver — env parsing, default, unknown-value fallback (guards ON
   is the safe direction).
2. The table contract — guard_active per profile, loud KeyError on typos,
   and the anti-drift check that every `site` pointer still names a real
   symbol in a real file (a refactor that moves a guard must move its
   table entry with it).
3. The wired call sites — under the default 8b profile behaviour is
   byte-identical to before this module existed (the whole rest of the
   suite pins that); under gemma4/frontier each wired guard steps aside.
"""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.core import model_profile
from src.core.dispatch.heuristics import REAL_TOOL_HINT_RE
from src.core.engine import OllamaEngine, _sanitise_stream_line
from src.core.model_profile import GUARDS, PROFILES, guard_active, model_profile as profile
from src.core.orchestrator import CortexOrchestrator
from src.core.tool_specs import TOOL_SPECS

REPO = Path(__file__).resolve().parents[1]


# ── 1. resolver ──────────────────────────────────────────────────────────

def test_default_profile_is_8b(monkeypatch):
    monkeypatch.delenv("CORTEX_MODEL_PROFILE", raising=False)
    assert profile() == "8b"


@pytest.mark.parametrize("value", ["8b", "gemma4", "frontier"])
def test_valid_profiles_resolve(monkeypatch, value):
    monkeypatch.setenv("CORTEX_MODEL_PROFILE", value)
    assert profile() == value


def test_profile_is_case_and_whitespace_tolerant(monkeypatch):
    monkeypatch.setenv("CORTEX_MODEL_PROFILE", "  Gemma4 ")
    assert profile() == "gemma4"


def test_unknown_profile_falls_back_to_8b(monkeypatch):
    monkeypatch.setenv("CORTEX_MODEL_PROFILE", "gpt9")
    assert profile() == "8b"
    # …and the safe direction is guards-ON:
    assert all(guard_active(name) for name in GUARDS)


# ── 2. table contract ────────────────────────────────────────────────────

def test_all_guards_active_under_8b(monkeypatch):
    monkeypatch.delenv("CORTEX_MODEL_PROFILE", raising=False)
    assert all(guard_active(name) for name in GUARDS)


@pytest.mark.parametrize("prof", ["gemma4", "frontier"])
def test_off_in_is_honoured(monkeypatch, prof):
    monkeypatch.setenv("CORTEX_MODEL_PROFILE", prof)
    for name, spec in GUARDS.items():
        assert guard_active(name) == (prof not in spec["off_in"])


def test_unknown_guard_name_raises():
    with pytest.raises(KeyError):
        guard_active("guard_that_does_not_exist")


def test_off_in_only_names_real_profiles():
    for name, spec in GUARDS.items():
        assert spec["off_in"] <= set(PROFILES), name


def test_inventory_only_guards_have_empty_off_in():
    # An unwired guard with a non-empty off_in would LIE: the table would
    # claim a flip that no call site implements.
    for name, spec in GUARDS.items():
        if not spec["wired"]:
            assert spec["off_in"] == frozenset(), name


def test_site_pointers_do_not_drift():
    for name, spec in GUARDS.items():
        path, symbol = spec["site"].split("::")
        source = (REPO / path).read_text()
        assert re.search(rf"\b{re.escape(symbol)}\b", source), (
            f"guard {name!r}: symbol {symbol!r} not found in {path} — "
            f"update the GUARDS table in src/core/model_profile.py"
        )


# ── 3. wired call sites ──────────────────────────────────────────────────

ARTIFACT_LINE = '{"kw": "futebol, copa"}'
ENVELOPE = '{"response": {"text": "olá"}}'


def test_stream_line_strips_artifact_under_8b(monkeypatch):
    monkeypatch.delenv("CORTEX_MODEL_PROFILE", raising=False)
    assert _sanitise_stream_line(ARTIFACT_LINE).strip() == ""


def test_stream_line_passes_through_under_gemma4(monkeypatch):
    monkeypatch.setenv("CORTEX_MODEL_PROFILE", "gemma4")
    assert _sanitise_stream_line(ARTIFACT_LINE) == ARTIFACT_LINE


def test_sanitise_response_unwraps_under_8b(monkeypatch):
    monkeypatch.delenv("CORTEX_MODEL_PROFILE", raising=False)
    assert OllamaEngine._sanitise_response(ENVELOPE) == "olá"


def test_sanitise_response_passes_through_under_frontier(monkeypatch):
    monkeypatch.setenv("CORTEX_MODEL_PROFILE", "frontier")
    assert OllamaEngine._sanitise_response(ENVELOPE) == ENVELOPE


def test_language_directive_present_under_8b(monkeypatch):
    monkeypatch.delenv("CORTEX_MODEL_PROFILE", raising=False)
    assert OllamaEngine._language_directive("pt") == (
        "[Responda em português brasileiro] "
    )
    assert OllamaEngine._language_directive("en") == "[Respond in UK English] "
    assert OllamaEngine._language_directive(None) == ""


def test_language_directive_dropped_under_gemma4(monkeypatch):
    monkeypatch.setenv("CORTEX_MODEL_PROFILE", "gemma4")
    assert OllamaEngine._language_directive("pt") == ""
    assert OllamaEngine._language_directive("en") == ""


def _specs_for(text, seen_paths=None):
    stub = SimpleNamespace(_REAL_TOOL_HINT_RE=REAL_TOOL_HINT_RE,
                           _seen_paths=seen_paths)
    return CortexOrchestrator._tool_specs_for(stub, text)


# read_file_at is held back until the conversation has a path to read (s83):
# offered from the first turn it out-competed find_file and the 8B invented a
# path. The rest of the menu is unchanged.
_MENU_BEFORE_A_PATH = [spec for spec in TOOL_SPECS
                       if spec["function"]["name"] != "read_file_at"]


def test_tool_specs_gate_holds_under_8b(monkeypatch):
    monkeypatch.delenv("CORTEX_MODEL_PROFILE", raising=False)
    # music-only hint → empty menu (the s38 anti-fabrication gate)
    assert _specs_for("amo música brasileira") == []
    # real tool hint → full menu
    assert _specs_for("agenda uma reunião amanhã") == _MENU_BEFORE_A_PATH


def test_tool_specs_gate_opens_under_gemma4(monkeypatch):
    monkeypatch.setenv("CORTEX_MODEL_PROFILE", "gemma4")
    assert _specs_for("amo música brasileira") == _MENU_BEFORE_A_PATH


def test_file_read_joins_the_menu_once_a_path_is_known(monkeypatch):
    monkeypatch.delenv("CORTEX_MODEL_PROFILE", raising=False)
    assert _specs_for("agenda uma reunião amanhã",
                      seen_paths={"/Users/x/a.pdf"}) == TOOL_SPECS


def test_refusal_retry_and_hallucination_sites_consult_guard():
    # The two condition-gated sites in engine.py can't be exercised without
    # a fake ollama round-trip; pin the wiring textually instead (the same
    # technique test_web_search uses for the verify flag).
    source = (REPO / "src/core/engine.py").read_text()
    assert source.count('guard_active("knowledge_refusal_retry")') == 2
    assert 'guard_active("hallucination_detector")' in source
