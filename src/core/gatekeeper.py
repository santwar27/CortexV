import re
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Optional
from pydantic import BaseModel

# The "internet-ish" nouns that turn a quit-verb ("desliga", "corta") into a
# lease REVOKE instead of an app action. Single source of truth: the dispatch
# layer (dispatch/macos.py MAC_QUIT_APP_RE) imports this to EXCLUDE the same
# words from its app group — if the two sets drift, "desliga a net" either
# stages quitting an app called "net" or falls through to chat, mute on its
# purpose (M6-A1, review Fable 5). `net\b` does not catch "netflix".
NET_WORDS_RE = r"(?:internet|wi-?fi|net|conex[ãa]o|rede)"

# Module-level so the eval dispatch simulator can classify gate
# state-change phrasings without instantiating a LogicGate (whose
# constructor reads the persisted gate state from disk).
LEASE_PATTERNS = [
    # s63: gained "liga/ligue" verbs and an optional net noun between verb
    # and duration — "conecta a internet por 2 horas" never matched before
    # (the noun broke the verb→digits adjacency the old pattern required).
    r"(?:connect|conect(?:e|a)|reconect(?:e|a)|lig(?:a|ue|ar)|"
    r"fi(?:ca|car|que)\s+online|(?:go|vai|volta)\s+online|"
    r"acesse a internet)"
    r"(?:[, ]+(?:me|please))?"
    r"(?:\s+(?:n[ao]\s+|[àa]\s+|o\s+|the\s+)?" + NET_WORDS_RE + r")?"
    r"\s+(?:for|por|durante|de)?\s*"
    r"(\d+)\s*"
    r"(?:h|hr|hrs|hour|hours|hora|horas)\b",
]

# Bare grant — the spoken mirror of the revoke verbs (s63, the gap the
# M6-A1 live test exposed: "Liga a internet!" fell through to plain chat
# and the 8B hallucinated commands that don't exist either). No duration
# in the phrase → the DEFAULT lease window applies (CORTEX_LEASE_HOURS,
# same env the orchestrator reads). The net noun is REQUIRED: "liga"
# alone is far too generic to mean "open the gate".
GRANT_PATTERNS = [
    r"^(?:lig(?:a|ue|ar)|conect(?:a|e|ar)(?:-me)?|reconect(?:a|e|ar)(?:-me)?)\s+"
    r"(?:n[ao]\s+|[àa]\s+|o\s+|the\s+)?" + NET_WORDS_RE + r"\b",
    # "fica online" (+ conjugations) — mirror of "fica offline".
    r"^fi(?:ca|car|que)\s+online\b",
    r"^(?:go|vai|volta)\s+online\b",
]

REVOKE_PATTERNS = [
    r"^(?:disconnect|desconect(?:ar|a|e)|"
    r"desligue?(?:\s+a)?(?:\s+internet)?|"
    r"corte(?:\s+a)?\s+internet|"
    r"sair do online|go offline|offline now)\b",
    # M6-A1: the user's real phrasings. "desliga"/"corta" never matched
    # above (only "desligue"/"corte" did) — but unlike bare "desligue",
    # these forms are quit-verbs in MAC_QUIT_APP_RE, so the net noun is
    # REQUIRED here: bare "desliga X" must stay an app action.
    r"^(?:deslig(?:a|ar|ue)|cort(?:a|e|ar))\s+"
    r"(?:[oa]\s+|the\s+)?" + NET_WORDS_RE + r"\b",
    # "fica offline" (+ conjugations) — same family as "go offline".
    r"^fi(?:ca|car|que)\s+offline\b",
]

class AccessStatus(BaseModel):
    """Data model for the current gate state."""
    is_authorised: bool
    mode: str 
    expires_at: Optional[datetime] = None

class LogicGate:
    def __init__(self, state_file: str = "data/gate_state.json"):
        self.state_file = state_file
        self.session_expiry: Optional[datetime] = self._load_state()
        
        # Lease grant: tolerant to mixed PT/EN prepositions (e.g.
        # "connect por 4 horas", "conecte for 4h") and to commas /
        # leading "me" / plural variations.
        self.lease_patterns = list(LEASE_PATTERNS)

        # Bare grant (no duration) → default window, same env/default the
        # orchestrator uses for its approval flow.
        self.grant_patterns = list(GRANT_PATTERNS)
        self.default_lease_hours = int(os.getenv("CORTEX_LEASE_HOURS", "4"))

        # Lease revoke: same tolerance.
        self.revoke_patterns = list(REVOKE_PATTERNS)

        self.search_triggers = [
            r"^(?:search|look up|find out)\b",
            r"^(?:pesquise|procure|busque)\b",
            r"(?:search the web|pesquise na web)"
        ]
        # Note: realtime_triggers (hardcoded keywords like "hoje", "atual") were
        # removed in favour of an LLM-based intent classifier in the orchestrator.

    def _load_state(self) -> Optional[datetime]:
        """Loads expiry from a local file to survive reboots."""
        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)
                if 'expires_at' in data and data['expires_at']:
                    return datetime.fromisoformat(data['expires_at'])
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        return None

    def _save_state(self):
        """Persists the expiry state locally in the data folder.

        Atomic (M6-A8): write to a tmp file then os.replace — a crash
        mid-write must not leave half a JSON behind. _load_state recovers
        fail-closed from a corrupt file, but that silently DROPS an active
        lease; tmp+rename means the file is always the old or the new
        state, never garbage."""
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)

        data = {
            'expires_at': self.session_expiry.isoformat() if self.session_expiry else None
        }
        # A tmp of its own per write (s88, Codex lote 10): with one fixed
        # ".tmp", two writers renamed the same file and the second rename hit
        # FileNotFoundError after its state had already changed in memory.
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.state_file) or ".",
                                   prefix=os.path.basename(self.state_file) + ".",
                                   suffix=".tmp")
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(data, f)
            os.replace(tmp, self.state_file)
        except BaseException:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise

    def _parse_lease(self, text: str) -> Optional[int]:
        """Extracts hours from the prompt if a lease is requested."""
        for pattern in self.lease_patterns:
            match = re.search(pattern, text.lower())
            if match:
                return int(match.group(1))
        return None

    def _is_lease_active(self) -> bool:
        """Checks if a previously granted time-lease is still valid.

        Re-read from the file every time: the API, the terminal and the jobs
        each hold a LogicGate, and the state was read only in the constructor
        — "desliga a internet" in the terminal left the API searching until
        the lease's original expiry (s88, Codex lote 10, reproduced)."""
        self.session_expiry = self._load_state()
        if self.session_expiry and datetime.now(timezone.utc) < self.session_expiry:
            return True
        if self.session_expiry:
            self.revoke_lease()
        return False

    # Public alias — orchestrator and external callers should use this.
    is_lease_active = _is_lease_active

    @staticmethod
    def _clamp_hours(hours: int) -> int:
        """THE lease clamp (M6-A7): floor 1, cap 24. Single definition so
        grant_lease and the classify 1b path can't drift again (1b used to
        cap without flooring — 'connect for 0 hours' silently granted
        nothing while grant_lease(0) granted 1h)."""
        return max(1, min(int(hours), 24))

    def grant_lease(self, hours: int) -> datetime:
        """Programmatically grant a time-lease (used by the orchestrator after approval)."""
        clamped = self._clamp_hours(hours)
        self.session_expiry = datetime.now(timezone.utc) + timedelta(hours=clamped)
        self._save_state()
        return self.session_expiry

    def revoke_lease(self):
        """Manually closes the gate."""
        self.session_expiry = None
        self._save_state()

    def classify_gate(self, user_input: str) -> AccessStatus:
        """PURE classification: what does this input mean to the gate?

        Never writes state (M1-A2 / M6-A3). Any caller that only wants the
        routing decision — `_is_pure_chat`, the eval simulator, future
        surfaces — can call this freely; the mutation lives in
        `apply_gate`, and `evaluate_gate` composes the two.
        """
        cleaned_input = user_input.strip().lower()

        # 1a. Explicit revoke ("disconnect" / "desligue a internet")
        if any(re.search(p, cleaned_input) for p in self.revoke_patterns):
            was_active = self._is_lease_active()
            return AccessStatus(
                is_authorised=False,
                mode="LEASE_REVOKED" if was_active else "ALREADY_OFFLINE",
            )

        # 1b. Grant lease ("connect por 4 horas" / "go online for 2h")
        hours = self._parse_lease(user_input)
        # Bare grant ("liga a internet", "fica online") — no duration in the
        # phrase, so the default window applies. The revoke check above runs
        # FIRST, so "desliga..." can never reach this branch. `is None` (not
        # falsy — M6-A7): an explicit "0 horas" is a duration, not a bare
        # grant; it clamps to the 1h floor instead of silently widening to
        # the default window.
        if hours is None and any(
            re.search(p, cleaned_input) for p in self.grant_patterns
        ):
            hours = self.default_lease_hours
        if hours is not None:
            # Same clamp as grant_lease (M6-A7) — the paths can't diverge.
            clamped_hours = self._clamp_hours(hours)
            expires_at = datetime.now(timezone.utc) + timedelta(hours=clamped_hours)
            return AccessStatus(is_authorised=True, mode="LEASE_GRANTED", expires_at=expires_at)

        # 2. Intent Parsing: Does the user explicitly want to search?
        is_search_intent = any(re.search(p, cleaned_input) for p in self.search_triggers)

        if is_search_intent:
            if self._is_lease_active():
                return AccessStatus(is_authorised=True, mode="ONLINE_REQUEST", expires_at=self.session_expiry)
            else:
                return AccessStatus(is_authorised=False, mode="GATE_CLOSED")

        # Default: Standard conversation routed to local engine.
        # The orchestrator now classifies intent via LLM and may upgrade this
        # to AUTO_SEARCH (lease active) or AWAITING_APPROVAL (no lease).
        return AccessStatus(is_authorised=self._is_lease_active(), mode="OFFLINE")

    def apply_gate(self, status: AccessStatus) -> None:
        """Apply the state change a classification implies. The ONLY writer
        on this path: LEASE_REVOKED revokes, LEASE_GRANTED persists the
        expiry the classification computed. Every other mode is read-only.
        """
        if status.mode == "LEASE_REVOKED":
            self.revoke_lease()
        elif status.mode == "LEASE_GRANTED":
            self.session_expiry = status.expires_at
            self._save_state()

    def evaluate_gate(self, user_input: str, peek: bool = False) -> AccessStatus:
        """Classify the input against the gate, applying state changes.

        `peek=True` classifies WITHOUT mutating — equivalent to calling
        `classify_gate` directly. Kept for the `_is_pure_chat` call site
        (streaming entry point), which only wants the routing decision:
        with the default it would revoke the lease there and
        `_process_core` would re-evaluate to ALREADY_OFFLINE, replying
        "já estamos offline" to the very command that just turned it off
        (M6-A1 / M1-A2). The classify/apply split (M6-A3) makes the
        mutation explicit and single-homed in `apply_gate`.
        """
        status = self.classify_gate(user_input)
        if not peek:
            self.apply_gate(status)
        return status