"""
LogicGate unit tests — revoke/grant phrasings + the peek contract (M6-A1).

The revoke patterns existed since day one but had no unit coverage; the
M6-A1 fix found that "desliga"/"corta" (the forms Pedro actually says)
never matched — only "desligue"/"corte" did. These tests pin the full
phrase set, the boundary with app-quit verbs, and `peek=True` (classify
without mutating), which the streaming `_is_pure_chat` probe relies on.
"""
import pytest

from src.core.gatekeeper import LogicGate, NET_WORDS_RE


@pytest.fixture
def gate(tmp_path, monkeypatch):
    monkeypatch.delenv("CORTEX_LEASE_HOURS", raising=False)
    return LogicGate(state_file=str(tmp_path / "gate_state.json"))


class TestRevokePhrasings:
    @pytest.mark.parametrize("text", [
        # The user's real phrasings (s63)
        "desconecta a net",
        "desliga a net",
        "desliga a internet",
        "fica offline",
        # close conjugations / variations
        "desliga a internet aí",
        "desligar a internet",
        "corta a internet",
        "corta a net",
        "desliga o wi-fi",
        "desliga a rede",
        "ficar offline",
        "fique offline",
        # pre-existing forms that must keep working
        "desligue a internet",
        "corte a internet",
        "disconnect",
        "go offline",
        "offline now",
    ])
    def test_revoke_with_active_lease(self, gate, text):
        gate.grant_lease(2)
        status = gate.evaluate_gate(text)
        assert status.mode == "LEASE_REVOKED", f"{text!r} → {status.mode}"
        assert not gate.is_lease_active()

    def test_revoke_without_lease_is_already_offline(self, gate):
        status = gate.evaluate_gate("desliga a internet")
        assert status.mode == "ALREADY_OFFLINE"

    @pytest.mark.parametrize("text", [
        # quit verbs with a real app: the gate must NOT treat these as
        # revoke (they belong to MAC_QUIT_APP_RE upstream).
        "desliga o spotify",
        "desliga o notion",
        "desliga a música",
        "corta o barulho",
        # "netflix" must not be eaten by the `net` word boundary
        "desliga a netflix",
    ])
    def test_app_phrasings_do_not_revoke(self, gate, text):
        gate.grant_lease(2)
        status = gate.evaluate_gate(text)
        assert status.mode == "OFFLINE", f"{text!r} → {status.mode}"
        assert gate.is_lease_active(), f"{text!r} must not revoke"


class TestBareGrantPhrasings:
    """The grant mirror of the revoke verbs (s63 follow-up). Exposed live:
    'Liga a internet!' fell through to plain chat — the only grant shape
    was '<verb> por N horas', so there was no spoken way to reopen the
    gate after a revoke."""

    @pytest.mark.parametrize("text", [
        "liga a internet",
        "liga a internet!",
        "ligue a internet",
        "ligar a internet",
        "liga a net",
        "conecta a internet",
        "conecta na internet",
        "conecte-me à rede",
        "fica online",
        "volta online",
        "go online",
    ])
    def test_bare_grant_opens_default_window(self, gate, text):
        status = gate.evaluate_gate(text)
        assert status.mode == "LEASE_GRANTED", f"{text!r} → {status.mode}"
        assert gate.is_lease_active()
        # Default window (CORTEX_LEASE_HOURS unset in tests → 4h).
        from datetime import datetime, timedelta, timezone
        delta = status.expires_at - datetime.now(timezone.utc)
        assert timedelta(hours=3, minutes=59) < delta <= timedelta(hours=4)

    @pytest.mark.parametrize("text, hours", [
        ("liga a internet por 2 horas", 2),
        ("conecta a internet por 2 horas", 2),
        ("fica online por 1 hora", 1),
        ("conecte por 4 horas", 4),   # pre-existing shape keeps working
        ("connect for 4 h", 4),
    ])
    def test_explicit_duration_wins_over_default(self, gate, text, hours):
        from datetime import datetime, timedelta, timezone
        status = gate.evaluate_gate(text)
        assert status.mode == "LEASE_GRANTED", f"{text!r} → {status.mode}"
        delta = status.expires_at - datetime.now(timezone.utc)
        assert timedelta(hours=hours, minutes=-1) < delta <= timedelta(hours=hours)

    @pytest.mark.parametrize("text", [
        # "liga" alone is generic PT ("liga pra ele" = phone him,
        # "liga o som" = turn on the speakers) — net noun required.
        "liga pra ele",
        "liga o som",
        "liga a netflix",
        "liga a televisão",
    ])
    def test_generic_liga_does_not_open_the_gate(self, gate, text):
        status = gate.evaluate_gate(text)
        assert status.mode == "OFFLINE", f"{text!r} → {status.mode}"
        assert not gate.is_lease_active()

    @pytest.mark.parametrize("text", [
        "conecta a internet por 0 horas",
        "connect for 0 h",
    ])
    def test_zero_hours_clamps_to_floor_like_grant_lease(self, gate, text):
        # M6-A7: '0 horas' used to be swallowed by `if hours:` (falsy) and
        # fall through to the bare-grant default (4h) — while grant_lease(0)
        # clamps to 1h. Same input, same clamp, whatever the path.
        from datetime import datetime, timedelta, timezone
        status = gate.evaluate_gate(text)
        assert status.mode == "LEASE_GRANTED", f"{text!r} → {status.mode}"
        delta = status.expires_at - datetime.now(timezone.utc)
        assert timedelta(minutes=59) < delta <= timedelta(hours=1)

    def test_inline_clamp_matches_grant_lease_cap(self, gate):
        # Cap side of the shared clamp: 99h → 24h on both paths.
        from datetime import datetime, timedelta, timezone
        status = gate.evaluate_gate("connect for 99 h")
        assert status.mode == "LEASE_GRANTED"
        delta = status.expires_at - datetime.now(timezone.utc)
        assert timedelta(hours=23, minutes=59) < delta <= timedelta(hours=24)
        gate.revoke_lease()
        expiry = gate.grant_lease(99)
        delta2 = expiry - datetime.now(timezone.utc)
        assert timedelta(hours=23, minutes=59) < delta2 <= timedelta(hours=24)

    def test_save_state_is_atomic_no_tmp_left_behind(self, gate, tmp_path):
        # M6-A8: tmp+rename — the state file is always old or new JSON,
        # never a truncated write; and the tmp doesn't linger.
        import json
        gate.grant_lease(2)
        state_file = tmp_path / "gate_state.json"
        assert state_file.exists()
        data = json.loads(state_file.read_text())  # valid JSON
        assert data["expires_at"]
        assert not (tmp_path / "gate_state.json.tmp").exists()

    def test_revoke_then_grant_round_trip(self, gate):
        gate.evaluate_gate("liga a internet")
        assert gate.is_lease_active()
        gate.evaluate_gate("desliga a internet")
        assert not gate.is_lease_active()
        gate.evaluate_gate("liga a internet")
        assert gate.is_lease_active()

    def test_peek_bare_grant_does_not_grant(self, gate):
        status = gate.evaluate_gate("liga a internet", peek=True)
        assert status.mode == "LEASE_GRANTED"
        assert not gate.is_lease_active()


class TestPeek:
    """peek=True classifies without mutating — the streaming
    `_is_pure_chat` probe depends on this (M6-A1 / M1-A2 side-effect)."""

    def test_peek_revoke_does_not_revoke(self, gate):
        gate.grant_lease(2)
        status = gate.evaluate_gate("desliga a internet", peek=True)
        assert status.mode == "LEASE_REVOKED"
        assert gate.is_lease_active(), "peek must not mutate the lease"

    def test_peek_grant_does_not_grant(self, gate):
        status = gate.evaluate_gate("conecte por 2 horas", peek=True)
        assert status.mode == "LEASE_GRANTED"
        assert status.expires_at is not None
        assert not gate.is_lease_active(), "peek must not grant"

    def test_default_still_mutates(self, gate):
        gate.evaluate_gate("conecte por 2 horas")
        assert gate.is_lease_active()
        status = gate.evaluate_gate("desliga a internet")
        assert status.mode == "LEASE_REVOKED"
        assert not gate.is_lease_active()


class TestNetWordsSharedWithDispatch:
    def test_mac_quit_lookahead_uses_the_same_words(self):
        # NET_WORDS_RE is the single source of truth: the dispatch layer
        # excludes exactly these words from the quit-app group. Drift
        # between the two would resurrect M6-A1 for the drifted word.
        from src.core.dispatch import macos
        assert NET_WORDS_RE in macos.MAC_QUIT_APP_RE.pattern


class TestClassifyApplySplit:
    """M1-A2 / M6-A3: classification and mutation are separate methods now.
    `classify_gate` is PURE (any caller may probe freely); `apply_gate` is
    the single writer; `evaluate_gate` composes the two (peek=True ≡
    classify only). The s63 peek flag fixed the symptom; this pins the
    contract."""

    def test_classify_never_writes_on_revoke(self, gate, tmp_path):
        gate.grant_lease(2)
        before = gate.session_expiry
        status = gate.classify_gate("desliga a internet")
        assert status.mode == "LEASE_REVOKED"
        assert gate.session_expiry == before, "classify_gate must not revoke"
        assert gate.is_lease_active()

    def test_classify_never_writes_on_grant(self, gate):
        status = gate.classify_gate("liga a internet")
        assert status.mode == "LEASE_GRANTED"
        assert status.expires_at is not None
        assert not gate.is_lease_active(), "classify_gate must not grant"

    def test_apply_revoked_status_revokes(self, gate):
        gate.grant_lease(2)
        status = gate.classify_gate("desliga a internet")
        gate.apply_gate(status)
        assert not gate.is_lease_active()

    def test_apply_granted_status_persists_classified_expiry(self, gate):
        status = gate.classify_gate("conecta a internet por 2 horas")
        gate.apply_gate(status)
        assert gate.is_lease_active()
        # The expiry persisted is the EXACT one classification computed —
        # no second clock read, no drift between the two halves.
        assert gate.session_expiry == status.expires_at

    def test_apply_readonly_modes_touch_nothing(self, gate):
        gate.grant_lease(2)
        before = gate.session_expiry
        for text in ("bom dia", "pesquisa o tempo em londres"):
            gate.apply_gate(gate.classify_gate(text))
        assert gate.session_expiry == before

    def test_evaluate_is_classify_plus_apply(self, gate):
        # Parity: the composed path ends in the same state as the split.
        status = gate.evaluate_gate("liga a internet")
        assert status.mode == "LEASE_GRANTED"
        assert gate.session_expiry == status.expires_at

    def test_peek_equals_pure_classify(self, gate):
        gate.grant_lease(2)
        peeked = gate.evaluate_gate("desliga a internet", peek=True)
        classified = gate.classify_gate("desliga a internet")
        assert peeked.mode == classified.mode == "LEASE_REVOKED"
        assert gate.is_lease_active()


class TestTheGateIsSharedBetweenProcesses:
    """Each process holds its own LogicGate over gate_state.json; the state
    was read only in the constructor (s88, Codex lote 10)."""

    def test_a_revoke_elsewhere_closes_it_here(self, tmp_path):
        from src.core.gatekeeper import LogicGate
        f = str(tmp_path / "gate.json")
        api = LogicGate(f)
        api.grant_lease(4)
        LogicGate(f).revoke_lease()       # "desliga a internet" in the terminal
        assert api.is_lease_active() is False

    def test_a_grant_elsewhere_opens_it_here(self, tmp_path):
        from src.core.gatekeeper import LogicGate
        f = str(tmp_path / "gate.json")
        api = LogicGate(f)
        LogicGate(f).grant_lease(2)
        assert api.is_lease_active() is True


def test_two_writers_never_share_a_tmp(tmp_path, monkeypatch):
    """One fixed gate_state.json.tmp: two writers interleaved before the
    rename and the second got FileNotFoundError (s88, Codex lote 10)."""
    import os
    import threading
    from src.core import gatekeeper as gk
    f = str(tmp_path / "gate.json")
    a, b = gk.LogicGate(f), gk.LogicGate(f)
    real, barrier = os.replace, threading.Barrier(2, timeout=2)

    def replace_after_both_wrote(src, dst):
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass
        return real(src, dst)
    monkeypatch.setattr(gk.os, "replace", replace_after_both_wrote)
    errors = []

    def grant(g, h):
        try:
            g.grant_lease(h)
        except Exception as e:  # noqa: BLE001
            errors.append(e)
    threads = [threading.Thread(target=grant, args=(a, 1)),
               threading.Thread(target=grant, args=(b, 2))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert not [p for p in os.listdir(tmp_path) if p.endswith(".tmp")]
