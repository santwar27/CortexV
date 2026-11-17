"""Versioned system prompt registry + CORTEX_PROMPT_VERSION resolution.

The point of versioning is a no-code rollback: if a prompt edit
regresses, pin CORTEX_PROMPT_VERSION to the previous number. These
tests lock the resolver's fallback behaviour (a typo must never boot
the assistant with no prompt) and the registry invariants.
"""
import ast
import hashlib
import inspect

import pytest

from src.core.prompts import (
    DORMANT_PROMPTS,
    LATEST_PROMPT_VERSION,
    SYSTEM_PROMPTS,
    resolve_prompt_version,
)


class TestRegistryInvariants:
    def test_registry_not_empty(self):
        assert SYSTEM_PROMPTS

    def test_latest_is_max_key(self):
        assert LATEST_PROMPT_VERSION == max(SYSTEM_PROMPTS)

    def test_v13_is_latest(self):
        assert LATEST_PROMPT_VERSION == 13

    def test_keys_are_ints(self):
        assert all(isinstance(k, int) for k in SYSTEM_PROMPTS)

    @pytest.mark.parametrize("version", list(SYSTEM_PROMPTS))
    def test_every_version_is_a_real_prompt(self, version):
        body = SYSTEM_PROMPTS[version]
        assert isinstance(body, str)
        assert body.strip()
        # Identity anchor — guards against an accidentally truncated or
        # blanked version slipping into the registry.
        assert "CORTEX" in body

    def test_v1_present(self):
        assert 1 in SYSTEM_PROMPTS

    def test_v2_present(self):
        assert 2 in SYSTEM_PROMPTS

    def test_v3_present(self):
        assert 3 in SYSTEM_PROMPTS

    def test_v4_present(self):
        assert 4 in SYSTEM_PROMPTS


# =====================================================================
# Rollback guarantee: shipped versions are immutable
# =====================================================================

# Golden hashes (M2-A12) — the single source of truth for EVERY shipped
# prompt body: the numeric versions AND the dormant named ones (frontier).
# The whole point of versioning is that a shipped body today is byte-
# identical to the same body last month, so a one-env-var rollback actually
# rolls back. Editing a shipped body in place trips its hash — copy to a new
# version instead. The completeness test below fails if a NEW body is added
# without a pin, so the net can't silently lapse and the ~30k-char "verify
# by eyeball" diff is gone. LATEST is pinned too now (M2-A12): you iterate by
# creating V<n+1>, never by editing the live body — so pinning it catches an
# UNINTENTIONAL typo to the current default. Update a hash only alongside a
# deliberate, reviewed change to that exact body.
_GOLDEN_SHA256 = {
    1: "5eb39ec174d7a72f9eace57bb51258697f1365f30044b59f76ac7fe8739d347b",
    2: "c334aa19fcd3e07a92840af50660c53b5a23fd9715ee1981a187c148004777a9",
    3: "a42acaa0b023d0d2a0f33554dde67410332ec6e902810793ec1ae640e0bc78e5",
    4: "52c3db918df70a5ab5f9f5a44c440f0a6ec6379ed1192d45137aae1d8e0c1d05",
    5: "e908d8d788f7e1f86aa07baf68d659a218759f3427a4e86a83df74c8c8c3f25f",
    6: "8dc64e2cb30fa0fdae809acfe1449f10d381833a368e7a85629f47a6ff3851a5",
    7: "b8dee5379e2be54025ea4170b82d2bc9104409d8b1e685c1b51f1b9cce836c5b",
    8: "ef4a0f2b76bd08a5103333a81126c1387b673e8c55425fcc3c4944faaa39828f",
    9: "3bc5364aad05512186d8d0725a283aeaf9e08283bf90647e5f0ddb38bbde321b",
    10: "0915faf0cbaf5e40116e7ec912f1f2debdb8f6d88515c5307951f2ef9c9b01c6",
    11: "301ed5d9a2a6a6d98ffaa1c40aea3dc283021f00f8a547f8a1e161bc26a76c87",
    12: "845b83ffa348469c9f1792a38ee4e703402b89d124ff8aede8985feaf597bec9",
    13: "d6e3e55e52725c85364faa4c3bb8ede028d024c5b159fc69d85e10452da7c2a6",
    "frontier": "845b83ffa348469c9f1792a38ee4e703402b89d124ff8aede8985feaf597bec9",
}


def _shipped_body(key):
    """Resolve a golden-map key to its body: int → SYSTEM_PROMPTS, str →
    DORMANT_PROMPTS."""
    return SYSTEM_PROMPTS[key] if isinstance(key, int) else DORMANT_PROMPTS[key]


class TestShippedVersionsImmutable:
    @pytest.mark.parametrize("key", list(_GOLDEN_SHA256))
    def test_byte_identical(self, key):
        actual = hashlib.sha256(_shipped_body(key).encode("utf-8")).hexdigest()
        assert actual == _GOLDEN_SHA256[key], (
            f"prompt body {key!r} changed. Shipped bodies are immutable — "
            f"copy to a new version instead of editing in place (rollback "
            f"depends on byte-identity). Update the hash only with a "
            f"deliberate, reviewed change."
        )

    def test_every_shipped_body_is_pinned(self):
        # Completeness (M2-A12): a new SYSTEM_PROMPTS or DORMANT_PROMPTS entry
        # without a golden hash fails HERE — the immutability net can never
        # silently lapse, and a stale pin (body removed) is caught too.
        shipped = set(SYSTEM_PROMPTS) | set(DORMANT_PROMPTS)
        pinned = set(_GOLDEN_SHA256)
        assert shipped == pinned, (
            f"golden-hash drift — unpinned bodies: {shipped - pinned}; "
            f"stale pins: {pinned - shipped}. Add/remove the hash in "
            f"_GOLDEN_SHA256 alongside the version change."
        )


# =====================================================================
# v2 — anti-over-tooling tightening
# =====================================================================

class TestV2Edits:
    """v2 closes the over-tooling gap the LLM_TOOL_CALL eval flagged.
    These pin the three surgical edits so an accidental rollback to v1
    style on v2 trips a test, not a silent regression."""

    def _v2(self):
        return SYSTEM_PROMPTS[2]

    def test_read_document_has_explicit_never(self):
        # Edit #1 — bullet for read_document fences off general knowledge.
        assert "NEVER call read_document for general" in self._v2()

    def test_answer_directly_has_tiebreaker(self):
        # Edit #2 — explicit default-to-direct when in doubt.
        assert "default to ANSWER DIRECTLY" in self._v2()

    def test_never_simulate_has_overtooling_example(self):
        # Edit #3 — concrete WRONG/RIGHT pair using the live-observed case.
        assert "fotossíntese" in self._v2()

    def test_v1_does_not_have_v2_additions(self):
        # Sanity: the edits really are NEW in v2, not already in v1.
        from src.core.prompts import SYSTEM_PROMPT_V1
        assert "NEVER call read_document for general" not in SYSTEM_PROMPT_V1
        assert "default to ANSWER DIRECTLY" not in SYSTEM_PROMPT_V1
        assert "fotossíntese" not in SYSTEM_PROMPT_V1


# =====================================================================
# v3 — calendar anti-hallucination tightening
# =====================================================================

class TestV3Edits:
    """v3 closes the schedule-hallucination gap flagged by the LLM eval
    `schedule_without_tool_is_honest` (sessão 17/D). Mirror the v2
    pattern: a concrete WRONG/RIGHT pair for the calendar verb."""

    def _v3(self):
        return SYSTEM_PROMPTS[3]

    def test_schedule_has_wrong_example(self):
        # Edit #1 — concrete WRONG for the agenda hallucination case.
        assert "reunião agendada" in self._v3()

    def test_schedule_has_right_example(self):
        # The RIGHT side names the calendar tool explicitly.
        assert "ferramenta de calendário" in self._v3()

    def test_v2_does_not_have_v3_additions(self):
        # Sanity: the new pair is genuinely NEW in v3, not already in v2.
        from src.core.prompts import SYSTEM_PROMPT_V2
        assert "reunião agendada" not in SYSTEM_PROMPT_V2
        assert "ferramenta de calendário" not in SYSTEM_PROMPT_V2

    def test_v3_keeps_v2_edits(self):
        # The v2 edits (read_document NEVER, ANSWER DIRECTLY tie-breaker,
        # fotossíntese pair) must carry forward — v3 ADDS, doesn't replace.
        v3 = self._v3()
        assert "NEVER call read_document for general" in v3
        assert "default to ANSWER DIRECTLY" in v3
        assert "fotossíntese" in v3


# =====================================================================
# v4 — tool-arg fidelity (preserve user text verbatim)
# =====================================================================

class TestV4Edits:
    """v4 closes the tool-arg paraphrase gap flagged live in sessão 20:
    a request to send 'I love you mil milhões' became 'mil milões' in
    the email body, and 'eu te amo' as a subject became 'Í eu te amo'.
    The regex-first path (SEND_EMAIL_STRUCTURED_RE) is the bulletproof
    fix for structured input; v4 hardens the free-form LLM-extraction
    path that catches everything else."""

    def _v4(self):
        return SYSTEM_PROMPTS[4]

    def test_has_tool_arg_fidelity_block(self):
        assert "TOOL ARG FIDELITY" in self._v4()

    def test_has_verbatim_rule(self):
        # The key behavioural directive.
        assert "VERBATIM" in self._v4()
        assert "bit-for-bit" in self._v4()

    def test_has_live_bug_wrong_examples(self):
        # Concrete anchors from the sessão 20 live bug.
        v4 = self._v4()
        assert "mil milões" in v4  # the WRONG one (lost h)
        assert "milhões" in v4     # the corrective reference
        assert "Í eu te amo" in v4  # the WRONG one (added Í)

    def test_v3_does_not_have_v4_additions(self):
        from src.core.prompts import SYSTEM_PROMPT_V3
        assert "TOOL ARG FIDELITY" not in SYSTEM_PROMPT_V3
        assert "bit-for-bit" not in SYSTEM_PROMPT_V3
        assert "mil milões" not in SYSTEM_PROMPT_V3

    def test_v4_keeps_v3_edits(self):
        # Carry-forward sanity: every prior version's edits stay.
        v4 = self._v4()
        # v2 anchors
        assert "NEVER call read_document for general" in v4
        assert "default to ANSWER DIRECTLY" in v4
        assert "fotossíntese" in v4
        # v3 anchor
        assert "reunião agendada" in v4
        assert "ferramenta de calendário" in v4


# =====================================================================
# v5 — the phantom spotify_control bullet is gone (M2-A5)
# =====================================================================

class TestV5Edits:
    """v5 removes the instruction to use a tool that NEVER existed in
    TOOL_SPECS: every version since v1 told the model to call
    `spotify_control`, pushing it to fabricate the very tool-call JSON
    the `_TOOLCALL_JSON_RE` leak guard was built to clean. Music stays
    upstream (regex dispatch); exposing a real spotify_control is the
    registered post-swap decision (M4-B2 option b)."""

    def _v5(self):
        return SYSTEM_PROMPTS[5]

    def test_no_phantom_tool(self):
        assert "spotify_control" not in self._v5()

    def test_music_redirect_note_present(self):
        v5 = self._v5()
        assert "NO music tool" in v5
        assert "toca matue no spotify" in v5

    def test_real_tools_survive_the_surgery(self):
        # The replace() is surgical: every REAL tool bullet stays.
        v5 = self._v5()
        for tool in ("read_document", "write_document", "list_emails",
                     "send_email", "list_events", "create_event",
                     "create_reminder", "web_search"):
            assert tool in v5, f"v5 lost a real tool: {tool}"

    def test_v5_keeps_v4_edits(self):
        v5 = self._v5()
        assert "TOOL ARG FIDELITY" in v5
        assert "bit-for-bit" in v5
        # and the earlier carry-forwards
        assert "default to ANSWER DIRECTLY" in v5
        assert "ferramenta de calendário" in v5

    def test_v4_still_has_the_bullet(self):
        # Rollback target intact: pinning CORTEX_PROMPT_VERSION=4 must
        # reproduce the old behaviour, phantom bullet included.
        assert "spotify_control" in SYSTEM_PROMPTS[4]


_V9_EDITS = (
    (
        "NOT call the matching tool, you MUST say plainly that you couldn't perform it.",
        "NOT call the matching tool, you MUST say plainly that nothing was done. "
        "Never claim you tried.",
    ),
    (
        "  RIGHT: \"Não consegui acionar o Spotify pra tocar isso. "
        "Tenta dizer 'toca matue no spotify'.\"\n",
        "  RIGHT: \"Não entendi esse comando, então nada foi feito.\"\n"
        "  RIGHT (artist/song named by user): \"Não entendi esse comando, então nada foi feito. "
        "Tenta dizer 'toca <artista/música> no spotify'.\"\n",
    ),
    (
        "- Music playback (play/pause/skip/volume/what's playing) is "
        "handled by direct commands UPSTREAM — you have NO music tool. "
        "If a music request reaches you, do not fabricate a tool call; "
        "say plainly you couldn't control Spotify this time and suggest "
        "a direct phrasing like 'toca matue no spotify'.\n",
        "- Music playback (play/pause/skip/volume/what's playing) is "
        "handled by direct commands UPSTREAM — you have NO music tool. "
        "If a music request reaches you, never invent a tool call or claim you tried. "
        "Say: 'Não entendi esse comando, então nada foi feito.' "
        "Any suggestion must use only the artist/song the user named, never an example artist.\n",
    ),
)


class TestV9Edits:
    def test_only_three_authorized_replacements(self):
        expected = SYSTEM_PROMPTS[8]
        for old, new in _V9_EDITS:
            assert expected.count(old) == 1
            expected = expected.replace(old, new, 1)
            assert new in SYSTEM_PROMPTS[9]
            assert old not in SYSTEM_PROMPTS[9]
        assert SYSTEM_PROMPTS[9] == expected
        assert "matue" not in SYSTEM_PROMPTS[9].casefold()

    def test_v8_still_selectable(self):
        assert resolve_prompt_version("8") == 8

    @pytest.mark.parametrize("missing_anchor", [0, 1, 2, None])
    def test_import_asserts_reject_silent_noops_and_fixed_artist(self, missing_anchor):
        from src.core import prompts

        tree = ast.parse(inspect.getsource(prompts))
        # Execute the real V9 construction against a drifted V8, including
        # its import-time asserts, without changing any shipped constant.
        def assigns(node, name):
            return isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets)

        starts = [i for i, node in enumerate(tree.body) if assigns(node, "SYSTEM_PROMPT_V9")]
        assert starts, "V9 must be derived from V8 with guarded replacements"
        end = next(i for i, node in enumerate(tree.body) if assigns(node, "SYSTEM_PROMPT_FRONTIER"))
        block = ast.Module(body=tree.body[starts[0]:end], type_ignores=[])
        v8 = SYSTEM_PROMPTS[8]
        if missing_anchor is None:
            v8 += "\ntoca matue"
        else:
            v8 = v8.replace(_V9_EDITS[missing_anchor][0], "missing anchor", 1)
        with pytest.raises(AssertionError, match="V9"):
            exec(compile(block, prompts.__file__, "exec"), {"SYSTEM_PROMPT_V8": v8})



class TestV10Edits:
    def test_only_the_three_authorized_replacements(self):
        from src.core.prompts import _V10_EDITS

        expected = SYSTEM_PROMPTS[9]
        for old, new in _V10_EDITS:
            assert expected.count(old) == 1
            expected = expected.replace(old, new, 1)
            assert new in SYSTEM_PROMPTS[10] and old not in SYSTEM_PROMPTS[10]
        assert SYSTEM_PROMPTS[10] == expected
        assert "tenta de novo" not in SYSTEM_PROMPTS[10]
        assert "nada foi enviado" in SYSTEM_PROMPTS[10] and "nada foi agendado" in SYSTEM_PROMPTS[10]

    def test_v9_still_selectable(self):
        assert resolve_prompt_version("9") == 9

class TestResolvePromptVersion:
    def test_none_falls_back_to_latest(self):
        assert resolve_prompt_version(None) == LATEST_PROMPT_VERSION

    @pytest.mark.parametrize("raw", ["", "   ", "\t"])
    def test_blank_falls_back_to_latest(self, raw):
        assert resolve_prompt_version(raw) == LATEST_PROMPT_VERSION

    @pytest.mark.parametrize("raw", ["abc", "v1", "1.0", "one", "-"])
    def test_non_integer_falls_back_to_latest(self, raw):
        assert resolve_prompt_version(raw) == LATEST_PROMPT_VERSION

    def test_unknown_version_falls_back_to_latest(self):
        # A number with no registered prompt — operator typo'd a future
        # version that doesn't exist yet.
        assert resolve_prompt_version("999") == LATEST_PROMPT_VERSION

    def test_known_version_selected(self):
        assert resolve_prompt_version("1") == 1

    def test_whitespace_around_integer_tolerated(self):
        assert resolve_prompt_version("  1  ") == 1

    def test_return_is_always_a_registry_key(self):
        # Whatever the input, the result must index SYSTEM_PROMPTS safely.
        for raw in (None, "", "abc", "999", "1", "-5", "0"):
            assert resolve_prompt_version(raw) in SYSTEM_PROMPTS


class TestFrontierDormant:
    """M2-B2 (s66): the lean frontier prompt is a DORMANT scaffold —
    selectable only by its explicit name, never the default by accident."""

    def test_frontier_registered_as_dormant(self):
        from src.core.prompts import DORMANT_PROMPTS, SYSTEM_PROMPT_FRONTIER
        assert DORMANT_PROMPTS["frontier"] is SYSTEM_PROMPT_FRONTIER

    def test_frontier_reaches_the_registry_only_as_v12(self):
        """15/11/2026: Pedro switched it on as V12 after it was measured.
        It must be reachable as that one version and no other."""
        from src.core.prompts import SYSTEM_PROMPT_FRONTIER
        assert [k for k, v in SYSTEM_PROMPTS.items()
                if v == SYSTEM_PROMPT_FRONTIER] == [12]

    def test_unset_env_still_resolves_to_shipped_latest(self):
        assert resolve_prompt_version(None) == LATEST_PROMPT_VERSION

    def test_explicit_name_selects_frontier(self):
        from src.core.prompts import get_prompt_body
        for raw in ("frontier", "FRONTIER", "  frontier  "):
            key = resolve_prompt_version(raw)
            assert key == "frontier"
            assert get_prompt_body(key)

    def test_frontier_is_lean_and_carries_persona(self):
        from src.core.prompts import SYSTEM_PROMPT_FRONTIER, SYSTEM_PROMPTS
        body = SYSTEM_PROMPT_FRONTIER
        # The whole point: persona in, crutch examples out, much smaller.
        assert "JARVIS" in body and "PERSONA" in body
        assert len(body) < len(SYSTEM_PROMPTS[5]) / 2
        # The phantom tool stays dead (M2-A5 must not regress here).
        assert "spotify_control" not in body
        # The two rules that never leave, however lean the prompt gets.
        assert "[KNOWLEDGE]" in body
        assert "VERBATIM" in body

    def test_frontier_follows_v9_to_v11_honesty_rules(self):
        """14/11: edited in place (never live, so no rollback to protect) to
        carry the V9–V11 lessons: no attempted-failure wording, no invisible
        lease gate, music and Mac control have no tool here."""
        from src.core.prompts import SYSTEM_PROMPT_FRONTIER as body
        assert "couldn't" not in body and "lease" not in body
        assert "nothing was done; never claim you tried" in body
        assert "internet access is off and nothing was searched" in body
        assert "Music and Mac control" in body

    def test_get_prompt_body_handles_both_key_kinds(self):
        from src.core.prompts import get_prompt_body
        assert get_prompt_body(1) == SYSTEM_PROMPTS[1]
        assert get_prompt_body("frontier")


class TestV11Edits:
    def test_only_the_two_authorized_replacements(self):
        from src.core.prompts import _V11_EDITS

        expected = SYSTEM_PROMPTS[10]
        for old, new in _V11_EDITS:
            assert expected.count(old) == 1
            expected = expected.replace(old, new, 1)
            assert new in SYSTEM_PROMPTS[11] and old not in SYSTEM_PROMPTS[11]
        assert SYSTEM_PROMPTS[11] == expected

    def test_no_invisible_lease_gate_and_no_mac_tool(self):
        body = SYSTEM_PROMPTS[11]
        assert "lease" not in body and "macos tool" not in body
        assert "you have NO Mac tool" in body
        assert "internet access is off and nothing was searched" in body

    def test_v10_still_selectable(self):
        assert resolve_prompt_version("10") == 10


class TestV12IsFrontier:
    def test_v12_is_the_measured_frontier_body(self):
        from src.core import prompts
        assert prompts.SYSTEM_PROMPTS[12] == prompts.SYSTEM_PROMPT_FRONTIER
        assert prompts.resolve_prompt_version(None) == 13

    def test_v11_stays_selectable_for_rollback(self):
        from src.core import prompts
        assert prompts.resolve_prompt_version("11") == 11
        assert prompts.get_prompt_body(11) == prompts.SYSTEM_PROMPT_V11
