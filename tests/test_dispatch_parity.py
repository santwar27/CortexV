"""
Parity: eval simulator × the REAL `_process_core` (M9-A1, review Fable 5).

`evals/dispatch.py` is a declared duplication of `_process_core`'s routing
sequence — and the drift between the two is the hole every routing bug of
the review crawled through (M1-A1 invisible because the simulator doesn't
execute; M6-A1 invisible because it skips gate/lease). The self-test only
proves the CASES pass, not that the simulator agrees with the orchestrator.

This test converts "remember to update both" into a verified contract: it
feeds every DISPATCH eval case input through `simulate_dispatch` AND
through a real `CortexOrchestrator._process_core` (integrations mocked,
zero network/LLM), then asserts the modes agree under a documented
equivalence map.

The map encodes the simulator's DECLARED approximations (dispatch.py:19-26)
— those are contract, not drift:
  - The simulator reports the route's front-door mode; the real handler
    may stage S3 (AGENT_AWAITING_CONFIRMATION), ask a clarifying question,
    or fail-soft. Family-prefix match or staging modes are accepted.
  - Sim "OFFLINE"/fallthrough means "would head to reflex → skills → LLM";
    the real path may answer via REFLEX or route tools. Any of those
    real modes is accepted.
  - APPROVAL/DENIAL depend on pending state the simulator doesn't model —
    the real orchestrator (no pending) treats them as chat. Accepted.
Anything OUTSIDE the map is parity drift and must fail loudly.
"""
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import pytest

from evals.core import EvalKind, load_all_cases
from evals.dispatch import simulate_dispatch


# ---------------------------------------------------------------------
# Real-orchestrator harness (no Ollama, no network, no real Palace file)
# ---------------------------------------------------------------------

def _build_real_orch(tmp_path):
    from src.core.orchestrator import CortexOrchestrator
    from src.core.gatekeeper import LogicGate
    from src.core.studies_review import StudiesReview
    from src.core.study_queue import StudyQueue
    from src.memory.palace import Palace

    orch = CortexOrchestrator.__new__(CortexOrchestrator)

    for eng_attr in ("engine_analytical", "engine_creative"):
        eng = MagicMock()
        eng.model_name = "cortex-lite"
        eng.chat.return_value = "resposta"
        eng.chat_with_tools.return_value = ("resposta", None)
        setattr(orch, eng_attr, eng)

    orch._palace = Palace(db_path=":memory:")
    orch.study_queue = StudyQueue(path=str(tmp_path / "study_queue.json"))
    orch.studies_review = StudiesReview()
    orch.skill_registry = None

    # Integrations the registry handlers touch — pure stubs. Handler
    # replies become MagicMocks (we only assert MODE), but the mode is
    # always a literal string in the handler source, so parity holds.
    orch.spotify = MagicMock()
    orch.macos = MagicMock()
    orch.notes = MagicMock()
    orch.vision = MagicMock()
    orch.web_searcher = MagicMock()
    orch.reader = MagicMock()
    orch.reader.read_document.return_value = "conteúdo"
    orch.writer = MagicMock()
    orch.writer.write_document.return_value = "salvo"

    orch.gatekeeper = LogicGate()  # state file isolated by conftest
    orch.lease_hours = 4
    orch._intent_cache = OrderedDict()
    orch._pending_action = None
    orch._pending_query = None
    orch._pending_engine = None
    return orch


# ---------------------------------------------------------------------
# Equivalence map — the simulator's declared approximations
# ---------------------------------------------------------------------

# Sim modes whose real handler stages/clarifies instead of acting.
_STAGED_OR_CLARIFY = {
    "AGENT_AWAITING_CONFIRMATION",  # S3 staged, waits for "sim"
}

# The routes whose handler legitimately ends in an S3 confirmation, and the
# action it stages. A staged confirmation used to agree with ANY simulated
# route — "minhas notas" turning into a staged action would have passed
# (s88, Codex lote 5). Measured over the sweep: these four families are the
# only ones that reach it; the test also checks the action staged is theirs.
_STAGES_S3 = {
    "MAC_QUIT_APP": "mac_quit_app",
    "MAC_SLEEP": "mac_sleep",
    "SEND_EMAIL_BY_NAME": "send_email",
    "SEND_EMAIL_STRUCTURED": "send_email",
}

# Routing labels → real outcome modes. The registry documents (registry.py
# "the simulator modes carry the STUDIER_ prefix — routing labels, not the
# handlers' outcome modes") that some declared modes are LABELS for the
# route, while the handler's returned mode depends on runtime state (an
# approve hits STUDY_APPROVE or STUDY_NOT_FOUND depending on the queue).
# This table IS that contract, verified: a handler returning a mode
# outside its label's set is drift.
_LABEL_OUTCOMES = {
    "STUDIER_REVIEW": {"STUDIES_REVIEW"},
    "STUDIER_APPROVE": {"STUDY_APPROVE", "STUDY_NOT_FOUND",
                        "STUDY_AMBIGUOUS", "STUDY_APPROVE_ERROR"},
    "STUDIER_REJECT": {"STUDY_REJECT", "STUDY_NOT_FOUND",
                       "STUDY_AMBIGUOUS", "STUDY_REJECT_ERROR"},
    "STUDIER_SHOW": {"STUDY_SHOW", "STUDY_NOT_FOUND", "STUDY_AMBIGUOUS"},
    "STUDIER_LIST": {"STUDY_LIST"},
    # Cancel (s83): the route is deterministic, the final mode depends on the
    # queue — this harness starts empty, so the handler answers "already
    # empty" (STUDY_LIST) instead of naming what it removed. Same routing
    # decision: CANCEL_STUDY_RE fired.
    "STUDY_CANCELLED": {"STUDY_CANCELLED", "STUDY_LIST"},
    "STUDIER_ADD": {"STUDY_QUEUED", "STUDY_REJECTED_TRANSIENT",
                    "STUDY_REJECTED_DUPLICATE", "INPUT_ERROR"},
    "LIST_EMAILS": {"GMAIL_LIST", "GMAIL_NOT_CONFIGURED", "GMAIL_ERROR"},
    # Weather (s73): the route is deterministic but the FINAL mode depends
    # on lease state, which the simulator deliberately doesn't model — no
    # lease (this harness) → the offer flow; lease + API down → honest
    # degrade. All three are the same routing decision: WEATHER_RE fired.
    "WEATHER": {"WEATHER", "AWAITING_APPROVAL", "WEATHER_ERROR"},
    # Reply-to-email (inovação #3): the route fired; the real outcome depends
    # on the Mail read, which the simulator doesn't model. In this harness
    # subprocess.run is mocked, so the osascript read fails → MAIL_REPLY_ERROR;
    # the happy path stages the send_email (AGENT_AWAITING_CONFIRMATION).
    "MAC_REPLY_EMAIL": {"AGENT_AWAITING_CONFIRMATION", "MAIL_REPLY_ERROR",
                        "MAIL_REPLY_NO_SELECTION", "MAIL_REPLY_NO_SENDER",
                        "MAIL_REPLY_DRAFT_FAILED"},
    # Summarise-the-screen (inovação #3 Bloco B): route fired; the real
    # outcome depends on the window read, which the simulator doesn't model.
    # subprocess.run is mocked here → the read fails → SCREEN_SUMMARY_ERROR.
    "MAC_SUMMARISE": {"SCREEN_SUMMARY", "SCREEN_SUMMARY_ERROR",
                      "SCREEN_SUMMARY_NO_CONTENT", "SCREEN_SUMMARY_FAILED"},
    # What-is-this (inovação #3 Bloco C): text path or vision fallback. With
    # subprocess.run mocked, the read/capture fails → an honest error/no-content.
    "MAC_WHAT_IS": {"WHAT_IS_THIS", "WHAT_IS_THIS_VISION", "WHAT_IS_THIS_ERROR",
                    "WHAT_IS_THIS_NO_CONTENT", "WHAT_IS_THIS_FAILED"},
    # Voice toggle of the screen-reading kill-switch (#3): the handlers flip a
    # flag file (isolated to a tmp path by the conftest fixture) and almost
    # always succeed; the error branch only fires if the FS op raises.
    # Top jobs: the route fired; which branch answers depends on what is in
    # jobs.db. Until s88 these cases read the user's REAL ~/.cortex/jobs.db (328
    # ranked jobs) and so came back JOBS_TOP; the conftest now gives every
    # test an empty tmp one — the same handler's empty branch.
    "JOBS_TOP": {"JOBS_TOP", "JOBS_EMPTY", "JOBS_UNRANKED", "JOBS_ALL_CLOSED"},
    "MAC_EYES_OFF": {"SCREEN_EYES_OFF", "SCREEN_EYES_ERROR"},
    "MAC_EYES_ON": {"SCREEN_EYES_ON", "SCREEN_EYES_ERROR"},
}

# What a sim fallthrough ("OFFLINE") may legitimately become for real —
# the layers the simulator deliberately skips (dispatch.py:19-26).
_FALLTHROUGH_REAL_MODES = {
    "OFFLINE", "REFLEX", "SKILL", "AWAITING_APPROVAL",
}


def _modes_agree(sim_mode: str, real_mode: str) -> bool:
    if sim_mode == real_mode:
        return True
    if sim_mode in _LABEL_OUTCOMES:
        return real_mode in _LABEL_OUTCOMES[sim_mode]
    # Family-prefix agreement: sim "MAC_VOLUME" vs real "MAC_VOLUME" is
    # exact; sim "CALENDAR" vs real "CALENDAR_CLARIFY" (handler asked a
    # question) is the same routing decision.
    if real_mode.startswith(sim_mode):
        return True
    if real_mode in _STAGED_OR_CLARIFY:
        # The route fired; the handler staged S3 for human confirmation —
        # only for the routes that do that (_STAGES_S3).
        return sim_mode in _STAGES_S3
    if sim_mode == "OFFLINE":
        return (
            real_mode in _FALLTHROUGH_REAL_MODES
            or real_mode.startswith("TOOL_")
        )
    if sim_mode in ("APPROVAL", "DENIAL"):
        # No pending state in the harness: real treats these as chat.
        return real_mode in _FALLTHROUGH_REAL_MODES
    if sim_mode == "PLANNER":
        # Real decompose needs the LLM; with ollama failing it falls
        # through to the tool/chat path by design (orchestrator.py:796).
        return (
            real_mode in _FALLTHROUGH_REAL_MODES
            or real_mode.startswith("TOOL_")
            or real_mode in _STAGED_OR_CLARIFY
        )
    return False


def _dispatch_inputs():
    seen = set()
    for case in load_all_cases():
        if case.kind is not EvalKind.DISPATCH:
            continue
        if not case.input or case.input in seen:
            continue
        seen.add(case.input)
        yield pytest.param(case.input, id=case.name)


class TestSimulatorParity:
    """One real `_process_core` call per DISPATCH eval input, compared
    against `simulate_dispatch` under the equivalence map."""

    @pytest.fixture
    def orch(self, tmp_path, monkeypatch):
        orch = _build_real_orch(tmp_path)
        # The LLM realtime classifier is one of the simulator's declared
        # skips — pin it False so a fallthrough stays deterministic.
        monkeypatch.setattr(
            type(orch), "_needs_realtime", lambda self, text: False
        )
        # The send-by-name eval case resolves "lara" through the contact
        # book. Before s69 this silently hit the user's REAL ~/.cortex/
        # contacts.db (which happens to have a Lara) — the conftest now
        # isolates the singleton to a tmp db, so seed the contact the
        # case needs and keep asserting the full staging path.
        from src.integrations.contacts import contact_book
        contact_book.add("lara", "lara@example.com")
        return orch

    @pytest.mark.parametrize("text", _dispatch_inputs())
    def test_parity(self, orch, text, monkeypatch):
        sim = simulate_dispatch(text)

        # No subprocess side effects (silence pkill, macOS osascript via
        # any handler that bypasses the mocked controllers), no real LLM
        # (planner decompose falls through deterministically), and no
        # NETWORK: the email/calendar handlers lazy-import their clients
        # at call time, so patching the module attrs is enough. Without
        # the gmail patch this test hit the REAL Gmail API on the user's
        # machine (caught on first run — the very pollution class the
        # M9-A2 isolation fixture exists for).
        with patch("subprocess.run"), \
             patch("src.core.orchestrator.ollama") as ollama_mock, \
             patch("src.integrations.gmail_client.list_recent",
                   return_value=[]), \
             patch("src.integrations.calendar_client.list_events",
                   return_value=[]):
            ollama_mock.chat.side_effect = RuntimeError("no LLM in parity")
            reply, real_mode = orch._process_core(text)

        assert _modes_agree(sim.mode, real_mode), (
            f"PARITY DRIFT for {text!r}: simulator says mode="
            f"{sim.mode!r} (matcher={sim.matcher!r}), real "
            f"_process_core returned {real_mode!r}. Either "
            f"_process_core gained/reordered a step the simulator "
            f"doesn't know, or the equivalence map in this file needs "
            f"a documented entry."
        )
        if real_mode in _STAGED_OR_CLARIFY and sim.mode in _STAGES_S3:
            staged = (orch._pending_action or {}).get("type")
            assert staged == _STAGES_S3[sim.mode], (
                f"{text!r} staged {staged!r}, not {_STAGES_S3[sim.mode]!r}")


class TestTheComparatorRefuses:
    """Negative cases of _modes_agree itself (s88, Codex lote 5)."""

    @pytest.mark.parametrize("sim, real", [
        ("NOTE_LIST", "AGENT_AWAITING_CONFIRMATION"),
        ("SPOTIFY_PAUSE", "AGENT_AWAITING_CONFIRMATION"),
        ("WEATHER", "AGENT_AWAITING_CONFIRMATION"),
    ])
    def test_a_staged_action_is_not_any_route(self, sim, real):
        assert not _modes_agree(sim, real)

    @pytest.mark.parametrize("sim", sorted(_STAGES_S3))
    def test_the_routes_that_stage_still_agree(self, sim):
        assert _modes_agree(sim, "AGENT_AWAITING_CONFIRMATION")


class TestKnownRoutingBugs:
    """Adversarial cases from the review that the simulator could not see
    before the gate state-change step landed (M6-A1, s63). Planted as
    xfail(strict) in s62; the fix arrived and they were promoted to real
    tests — the lookahead in MAC_QUIT_APP_RE keeps the revoke phrasings
    out of app-quit, and the gatekeeper's revoke patterns gained the verb
    forms Pedro actually says ("desliga", "corta", "fica offline").
    """

    @pytest.fixture
    def orch(self, tmp_path, monkeypatch):
        orch = _build_real_orch(tmp_path)
        monkeypatch.setattr(
            type(orch), "_needs_realtime", lambda self, text: False
        )
        return orch

    @pytest.mark.parametrize("text", [
        "desliga a internet",
        "desliga a internet aí",
        "corta a internet",
        "desligue a internet",
        "desliga a net",
        "desconecta a net",
        "desliga o wi-fi",
        "fica offline",
    ])
    def test_desliga_internet_revokes_lease_not_app_quit(self, orch, text):
        orch.gatekeeper.grant_lease(1)
        assert orch.gatekeeper.is_lease_active()

        with patch("subprocess.run"), \
             patch("src.core.orchestrator.ollama") as ollama_mock:
            ollama_mock.chat.side_effect = RuntimeError("no LLM")
            reply, mode = orch._process_core(text)

        # The contract: the lease is revoked and the user hears it.
        assert mode == "SYSTEM_UPDATE", (
            f"got mode={mode!r} reply={str(reply)[:80]!r}"
        )
        assert not orch.gatekeeper.is_lease_active()
        from src.core.orchestrator import _PROMPTS
        assert reply == _PROMPTS["lease_revoked"]["pt"], (
            f"expected the revoke ack, got {str(reply)[:80]!r}"
        )

    def test_desliga_app_still_quits_with_active_lease(self, orch):
        # The other side of the collision: a real quit verb with an app
        # name must NOT start revoking the lease now.
        orch.gatekeeper.grant_lease(1)

        with patch("subprocess.run"), \
             patch("src.core.orchestrator.ollama") as ollama_mock:
            ollama_mock.chat.side_effect = RuntimeError("no LLM")
            reply, mode = orch._process_core("desliga o Spotify")

        assert mode != "SYSTEM_UPDATE"
        assert orch.gatekeeper.is_lease_active(), (
            "quitting an app must not touch the lease"
        )

    @pytest.mark.parametrize("text", [
        "liga a internet",
        "conecta a net",
        "fica online",
    ])
    def test_liga_internet_grants_lease(self, orch, text):
        # The mirror gap, exposed live (s63): with no bare-grant verb the
        # spoken "Liga a internet!" fell through to plain chat and there
        # was NO voice command to reopen the gate after a revoke.
        assert not orch.gatekeeper.is_lease_active()

        with patch("subprocess.run"), \
             patch("src.core.orchestrator.ollama") as ollama_mock:
            ollama_mock.chat.side_effect = RuntimeError("no LLM")
            reply, mode = orch._process_core(text)

        assert mode == "SYSTEM_UPDATE", (
            f"got mode={mode!r} reply={str(reply)[:80]!r}"
        )
        assert orch.gatekeeper.is_lease_active()
        assert "Internet liberada" in str(reply)

    def test_grant_works_on_streaming_entry_point_too(self, orch):
        assert not orch.gatekeeper.is_lease_active()

        with patch("subprocess.run"), \
             patch("src.core.orchestrator.ollama") as ollama_mock:
            ollama_mock.chat.side_effect = RuntimeError("no LLM")
            chunks = list(orch.process_stream("liga a internet"))

        assert len(chunks) == 1
        reply, is_final, mode, lang = chunks[0]
        assert mode == "SYSTEM_UPDATE"
        assert lang == "pt"
        assert orch.gatekeeper.is_lease_active()
        assert "Internet liberada" in str(reply)

    def test_revoke_works_on_streaming_entry_point_too(self, orch):
        # The 2-entry-points lesson (s24): process_stream's _is_pure_chat
        # probes the gate BEFORE deferring to _process_core. The probe must
        # not mutate (peek=True) — otherwise the deferred pass re-evaluates
        # against an already-revoked lease and answers "já estamos offline"
        # to the very command that turned it off.
        orch.gatekeeper.grant_lease(1)
        assert orch.gatekeeper.is_lease_active()

        with patch("subprocess.run"), \
             patch("src.core.orchestrator.ollama") as ollama_mock:
            ollama_mock.chat.side_effect = RuntimeError("no LLM")
            chunks = list(orch.process_stream("desliga a internet"))

        assert len(chunks) == 1
        reply, is_final, mode, lang = chunks[0]
        assert is_final is True
        assert mode == "SYSTEM_UPDATE"
        assert not orch.gatekeeper.is_lease_active()
        from src.core.orchestrator import _PROMPTS
        assert reply == _PROMPTS["lease_revoked"]["pt"], (
            "the streaming path must ack the revoke, not 'already offline' "
            f"— got {str(reply)[:80]!r}"
        )
