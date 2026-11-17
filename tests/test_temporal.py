"""
Temporal retrieval helpers — pure logic tests.

Covers the three pieces of src/memory/temporal.py:
  - detect_temporal_intent: pt-BR + EN regex sweep
  - compute_age_days: ISO timestamp parsing + clock-skew safety
  - lambda_for / decay_factor: per-tipo decay constants
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from src.memory.temporal import (
    TemporalIntent,
    compute_age_days,
    decay_factor,
    detect_temporal_intent,
    lambda_for,
)


# ----------------------------------------------------------------------
# detect_temporal_intent
# ----------------------------------------------------------------------


class TestDetectTemporalIntent:

    def test_empty_query_returns_default(self):
        intent = detect_temporal_intent("")
        assert intent == TemporalIntent()
        assert intent.decay_off is False
        assert intent.window_days is None

    def test_none_query_returns_default(self):
        assert detect_temporal_intent(None) == TemporalIntent()  # type: ignore[arg-type]

    def test_unremarkable_query_returns_default(self):
        intent = detect_temporal_intent("como funciona attention em transformers?")
        assert intent.window_days is None
        assert intent.decay_off is False

    @pytest.mark.parametrize("query,expected_days,expected_label", [
        ("o que aconteceu ontem?",                       2,  "yesterday"),
        ("revisa o que falamos hoje",                    1,  "today"),
        ("e na semana passada, o que decidi?",           8,  "last_week"),
        ("Procurando algo da última semana",             8,  "last_week"),
        ("estava bom no mês passado",                    31, "last_month"),
        ("último mês foi corrido",                       31, "last_month"),
        ("what happened yesterday in our chat?",         2,  "yesterday"),
        ("look at last week's briefing",                 8,  "last_week"),
        ("last month results",                           31, "last_month"),
        ("today's plan",                                 1,  "today"),
    ])
    def test_window_patterns(self, query, expected_days, expected_label):
        intent = detect_temporal_intent(query)
        assert intent.window_days == expected_days
        assert intent.label == expected_label
        assert intent.decay_off is False

    @pytest.mark.parametrize("query", [
        "sempre prefiro respostas curtas",
        "geralmente me chamam de Pedro",
        "prefiro café preto",
        "costumo correr de manhã",
        "minha preferência é silêncio",
        "I always pick the third option",
        "usually I work late",
        "I prefer dark mode",
        "my preference is markdown",
    ])
    def test_decay_off_cues(self, query):
        intent = detect_temporal_intent(query)
        assert intent.decay_off is True
        assert intent.window_days is None
        assert intent.label == "decay_off"

    def test_window_wins_over_decay_off(self):
        """If both signals fire, the harder constraint (window) should win.
        'lembra ontem o que eu sempre falo' is asking about yesterday."""
        intent = detect_temporal_intent("lembra ontem o que eu sempre falo")
        assert intent.window_days == 2
        assert intent.decay_off is False

    def test_case_insensitive(self):
        assert detect_temporal_intent("ONTEM").window_days == 2
        assert detect_temporal_intent("Sempre").decay_off is True

    def test_query_stripped_removes_window_cue(self):
        intent = detect_temporal_intent("ontem falamos sobre cats")
        assert intent.window_days == 2
        assert intent.query_stripped == "falamos sobre cats"

    def test_query_stripped_removes_decay_off_cue(self):
        intent = detect_temporal_intent("sempre prefiro cats")
        assert intent.decay_off is True
        # both cues "sempre" and "prefiro" match — only the FIRST is stripped
        # (single sub call). That's fine — what remains is still cleaner.
        assert "cats" in intent.query_stripped
        assert "sempre" not in intent.query_stripped.lower()

    def test_effective_query_falls_back_when_no_cue(self):
        intent = detect_temporal_intent("plain content query")
        assert intent.query_stripped is None
        assert intent.effective_query("plain content query") == "plain content query"

    def test_effective_query_returns_stripped_when_cue_fired(self):
        intent = detect_temporal_intent("ontem cats")
        assert intent.effective_query("ontem cats") == "cats"

    def test_word_boundary_does_not_overmatch(self):
        # 'hoje' should NOT match inside a longer token, e.g. a hash or word fragment
        assert detect_temporal_intent("não falamos sobre nada").window_days is None
        # 'sou' alone shouldn't fire (the pattern requires "sou o/um/uma"):
        # but a sentence with "sou um dev" should be classified as profile-like
        assert detect_temporal_intent("eu sou um desenvolvedor").decay_off is True
        assert detect_temporal_intent("isso e seu carro").decay_off is False


# ----------------------------------------------------------------------
# compute_age_days
# ----------------------------------------------------------------------


class TestComputeAgeDays:

    def test_zero_for_none(self):
        assert compute_age_days(None) == 0.0
        assert compute_age_days("") == 0.0

    def test_zero_for_unparseable(self):
        assert compute_age_days("not a date") == 0.0
        assert compute_age_days("2024-13-99") == 0.0

    def test_basic_age_in_days(self):
        now = datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc)
        ts = "2026-05-15T12:00:00+00:00"  # exactly 7 days earlier
        assert compute_age_days(ts, now) == pytest.approx(7.0, abs=0.001)

    def test_fractional_day(self):
        now = datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc)
        ts = "2026-05-22T00:00:00+00:00"  # 12 hours earlier
        assert compute_age_days(ts, now) == pytest.approx(0.5, abs=0.001)

    def test_missing_tz_treated_as_utc(self):
        now = datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc)
        # No timezone in stored timestamp — Palace's bug-safe convention
        # says treat it as UTC, not local.
        age_naive = compute_age_days("2026-05-15T12:00:00", now)
        age_with_tz = compute_age_days("2026-05-15T12:00:00+00:00", now)
        assert age_naive == pytest.approx(age_with_tz, abs=0.001)

    def test_future_timestamps_clamped_to_zero(self):
        """Clock skew between Cortex instances should not flip the exp(-λage)
        modulator into a boost for 'newer than now' documents."""
        now = datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc)
        future = "2026-05-23T00:00:00+00:00"
        assert compute_age_days(future, now) == 0.0

    def test_default_now_is_current_utc(self):
        # If now is None, function picks current UTC. Insert a fresh ts;
        # age should be near 0.
        fresh = datetime.now(timezone.utc).isoformat()
        assert compute_age_days(fresh) == pytest.approx(0.0, abs=0.01)


# ----------------------------------------------------------------------
# lambda_for / decay_factor
# ----------------------------------------------------------------------


class TestDecayLambdas:

    def test_known_tipos(self):
        assert lambda_for("conversation") == 0.10
        assert lambda_for("knowledge") == 0.05
        assert lambda_for("profile") == 0.005

    def test_unknown_tipo_falls_back_to_default(self):
        assert lambda_for("totally_made_up_tipo") == 0.05

    def test_none_tipo_falls_back_to_default(self):
        assert lambda_for(None) == 0.05

    def test_decay_factor_at_zero_age(self):
        now = datetime(2026, 5, 22, tzinfo=timezone.utc)
        fresh = "2026-05-22T00:00:00+00:00"
        # exp(-λ × 0) == 1 regardless of tipo
        assert decay_factor("conversation", fresh, now) == pytest.approx(1.0, abs=0.001)
        assert decay_factor("knowledge", fresh, now) == pytest.approx(1.0, abs=0.001)

    def test_decay_factor_at_half_life(self):
        """Conversation has λ=0.10. At 7 days (≈ ln(2)/0.10), decay ≈ 0.5."""
        now = datetime(2026, 5, 22, tzinfo=timezone.utc)
        ts = (now - timedelta(days=math.log(2) / 0.10)).isoformat()
        assert decay_factor("conversation", ts, now) == pytest.approx(0.5, abs=0.01)

    def test_knowledge_decays_slower_than_conversation(self):
        """Same age, but knowledge tipo should retain more signal."""
        now = datetime(2026, 5, 22, tzinfo=timezone.utc)
        ts = (now - timedelta(days=14)).isoformat()
        d_conv = decay_factor("conversation", ts, now)
        d_know = decay_factor("knowledge", ts, now)
        assert d_know > d_conv

    def test_recency_weight_is_bounded_by_the_age_share(self):
        """s83: age takes away at most RECENCY_WEIGHT of a semantic score;
        a fresh doc keeps all of it, and the order by age is preserved."""
        from src.memory.temporal import RECENCY_WEIGHT, recency_weight
        now = datetime(2026, 10, 2, tzinfo=timezone.utc)
        fresh = now.isoformat()
        week = (now - timedelta(days=7)).isoformat()
        ancient = (now - timedelta(days=365)).isoformat()
        assert recency_weight("conversation", fresh, now) == pytest.approx(1.0, abs=0.001)
        assert (recency_weight("conversation", fresh, now)
                > recency_weight("conversation", week, now)
                > recency_weight("conversation", ancient, now))
        assert recency_weight("conversation", ancient, now) == pytest.approx(
            1.0 - RECENCY_WEIGHT, abs=0.001)

    def test_profile_barely_decays(self):
        now = datetime(2026, 5, 22, tzinfo=timezone.utc)
        ts = (now - timedelta(days=180)).isoformat()  # 6 months old
        d_prof = decay_factor("profile", ts, now)
        # λ=0.005, age=180 → exp(-0.9) ≈ 0.407 (still substantial)
        assert d_prof > 0.40
        assert d_prof < 0.42
