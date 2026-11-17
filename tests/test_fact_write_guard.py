"""
fact_write_allowed — the unified no-downgrade guard (M3-A1, review Fable 5).

`add_fact` closes the active triple unconditionally, so the write decision
lives with the callers. The rule existed only in the passive consolidator;
the two nightly persisters skipped just the identical-objeto case, so a
0.55 studier inference could close a fact Pedro stated himself
("mora_em → Londres", user_explicit/1.0). One helper now serves all three
paths; the headline test is "inferred never closes explicit".
"""
import pytest

from src.memory.palace import Palace, fact_write_allowed
from src.memory.consolidator import FactTriple, persist_facts
from src.memory.preferences import PreferenceTriple, persist_preferences


@pytest.fixture
def palace():
    return Palace(db_path=":memory:")


class TestGuardRules:
    def test_no_existing_allows(self):
        assert fact_write_allowed(None, "Londres", "studier_inferred", 0.55)

    def test_same_objeto_skips_even_when_stronger(self):
        existing = {"objeto": "Londres", "source": "studier_inferred",
                    "confidence": 0.55}
        assert not fact_write_allowed(existing, "Londres", "user_explicit", 1.0)

    def test_inferred_never_closes_explicit(self):
        existing = {"objeto": "Londres", "source": "user_explicit",
                    "confidence": 1.0}
        assert not fact_write_allowed(
            existing, "Paris", "studier_inferred", 0.55
        )

    def test_inferred_never_closes_explicit_even_at_full_confidence(self):
        existing = {"objeto": "Londres", "source": "user_explicit",
                    "confidence": 1.0}
        assert not fact_write_allowed(existing, "Paris", "studier_inferred", 1.0)

    def test_explicit_may_close_explicit(self):
        # The user restating a fact wins — newest explicit statement rules.
        existing = {"objeto": "Londres", "source": "user_explicit",
                    "confidence": 1.0}
        assert fact_write_allowed(existing, "Paris", "user_explicit", 1.0)

    def test_weaker_never_closes_stronger(self):
        existing = {"objeto": "Londres", "source": "studier_inferred",
                    "confidence": 0.9}
        assert not fact_write_allowed(existing, "Paris", "digest_inferred", 0.55)

    def test_equal_confidence_overwrites(self):
        # Ties go to the newer write — matches the original passive guard
        # (strict > on the existing side).
        existing = {"objeto": "Londres", "source": "studier_inferred",
                    "confidence": 0.55}
        assert fact_write_allowed(existing, "Paris", "studier_inferred", 0.55)


class TestNightlyPersistersHonourTheGuard:
    """The two persisters that used to skip ONLY the identical-objeto case."""

    def test_persist_facts_cannot_downgrade_explicit(self, palace):
        palace.add_fact("user", "mora_em", "Londres",
                        source="user_explicit", confidence=1.0)
        written = persist_facts(palace, [
            FactTriple(sujeito="user", predicado="mora_em",
                       objeto="Paris", confidence=0.55),
        ], source="studier_inferred")
        assert written == 0
        active = palace.get_fact("user", "mora_em")
        assert active["objeto"] == "Londres"
        assert active["source"] == "user_explicit"

    def test_persist_preferences_cannot_downgrade_explicit(self, palace):
        palace.add_fact("user", "prefere_musica", "MPB",
                        source="user_explicit", confidence=1.0)
        written = persist_preferences(palace, [
            PreferenceTriple(predicado="prefere_musica",
                             objeto="techno", confidence=0.6),
        ])
        assert written == 0
        active = palace.get_fact("user", "prefere_musica")
        assert active["objeto"] == "MPB"

    def test_persist_facts_still_writes_fresh_and_upgrades(self, palace):
        written = persist_facts(palace, [
            FactTriple(sujeito="user", predicado="time_do_coracao",
                       objeto="Flamengo", confidence=0.6),
        ], source="studier_inferred")
        assert written == 1
        # A stronger inference may replace a weaker one.
        written = persist_facts(palace, [
            FactTriple(sujeito="user", predicado="time_do_coracao",
                       objeto="Botafogo", confidence=0.9),
        ], source="studier_inferred")
        assert written == 1
        assert palace.get_fact("user", "time_do_coracao")["objeto"] == "Botafogo"

    def test_persist_preferences_noop_skip_preserved(self, palace):
        palace.add_fact("user", "prefere_musica", "MPB",
                        source="studier_inferred", confidence=0.6)
        written = persist_preferences(palace, [
            PreferenceTriple(predicado="prefere_musica",
                             objeto="MPB", confidence=0.9),
        ])
        assert written == 0, "same objeto must not roll the bitemporal log"
