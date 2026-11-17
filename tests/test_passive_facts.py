"""
Integration tests for passive profile-fact persistence + surfacing in
`CortexMemory` (the store_interaction chokepoint side of design-reference ch. 16).

The pure extraction logic is covered in test_fact_extractor.py; here we
test the Palace wiring: the no-downgrade guard, the kill-switch, and that
captured facts surface in the system-prompt profile block.
"""
import pytest

from src.core.memory import CortexMemory
from src.memory.palace import Palace


@pytest.fixture
def mem():
    # In-memory Palace; facts are triplas, so no embed_fn needed.
    return CortexMemory(palace=Palace(db_path=":memory:"))


# --- persistence ----------------------------------------------------------

def test_passive_fact_written(mem):
    n = mem._consolidate_passive_facts("meu nome é Pedro")
    assert n == 1
    row = mem._palace.get_fact("user", "nome")
    assert row["objeto"] == "Pedro"
    assert row["source"] == "passive_regex"
    assert row["confidence"] == pytest.approx(0.6)


def test_no_fact_no_write(mem):
    assert mem._consolidate_passive_facts("bom dia, tudo bem?") == 0
    assert mem._palace.list_facts("user") == []


def test_multiple_facts_in_one_turn(mem):
    n = mem._consolidate_passive_facts("me chamo Pedro e trabalho como engenheiro")
    assert n == 2
    assert mem._palace.get_fact("user", "nome")["objeto"] == "Pedro"
    assert mem._palace.get_fact("user", "profissao")["objeto"] == "engenheiro"


# --- no-downgrade guard ---------------------------------------------------

def test_does_not_clobber_explicit_fact(mem):
    mem._palace.add_fact("user", "nome", "Pedro Guerra",
                         source="user_explicit", confidence=1.0)
    assert mem._consolidate_passive_facts("meu nome é Pedrinho") == 0
    assert mem._palace.get_fact("user", "nome")["objeto"] == "Pedro Guerra"


def test_does_not_clobber_higher_confidence(mem):
    mem._palace.add_fact("user", "mora_em", "São Paulo",
                         source="studier_inferred", confidence=0.9)
    assert mem._consolidate_passive_facts("moro no Rio") == 0
    assert mem._palace.get_fact("user", "mora_em")["objeto"] == "São Paulo"


def test_same_value_no_churn(mem):
    assert mem._consolidate_passive_facts("meu nome é Pedro") == 1
    # Re-stating the same fact must not write a second (churning) row.
    assert mem._consolidate_passive_facts("meu nome é Pedro") == 0
    active = [r for r in mem._palace.list_facts("user") if r["predicado"] == "nome"]
    assert len(active) == 1


def test_passive_updates_passive(mem):
    # A later passive hit may replace an earlier passive value (bitemporal:
    # the old row is closed, not deleted).
    assert mem._consolidate_passive_facts("moro em São Paulo") == 1
    assert mem._consolidate_passive_facts("moro no Rio de Janeiro") == 1
    assert mem._palace.get_fact("user", "mora_em")["objeto"] == "Rio de Janeiro"


# --- kill-switch ----------------------------------------------------------

def test_kill_switch_disables(mem, monkeypatch):
    monkeypatch.setenv("CORTEX_PASSIVE_FACTS", "0")
    assert mem._consolidate_passive_facts("meu nome é Pedro") == 0
    assert mem._palace.list_facts("user") == []


def test_enabled_by_default(mem, monkeypatch):
    monkeypatch.delenv("CORTEX_PASSIVE_FACTS", raising=False)
    assert mem._consolidate_passive_facts("meu nome é Pedro") == 1


# --- surfacing into the system-prompt profile block -----------------------

def test_profile_fact_surfaces(mem):
    mem._consolidate_passive_facts("meu nome é Pedro")
    profile = mem.get_user_profile()
    assert "nome → Pedro" in profile


def test_non_whitelisted_fact_not_surfaced(mem):
    # 'idade' is stored (queryable) but must not hit the prompt — it's a
    # volatile attr deliberately kept off PROFILE_INJECT_KEYS (s31 cont.
    # widened the curated whitelist, but bounding still holds).
    mem._consolidate_passive_facts("minha idade é 33")
    assert mem._palace.get_fact("user", "idade")["objeto"] == "33"
    assert "idade" not in mem.get_user_profile()


def test_profile_and_preferences_coexist(mem):
    mem._palace.add_fact("user", "prefere_idioma", "pt-br",
                         source="studier_inferred", confidence=0.95)
    mem._consolidate_passive_facts("moro em São Paulo")
    profile = mem.get_user_profile()
    assert "prefere_idioma → pt-br" in profile
    assert "mora_em → São Paulo" in profile


def test_empty_profile_when_nothing_known(mem):
    assert mem.get_user_profile() == ""


class TestSomeoneElsesSentence:
    """Text Pedro hands over to be translated, revised or shown as an example
    is not Pedro speaking. Through store_interaction, a translation and a
    revision put "nome → Ana, mora_em → Porto" in the profile every prompt
    carries (s88, Codex lote 6, reproduced)."""

    @staticmethod
    def _memory(tmp_path):
        import numpy as np
        from src.core.memory import CortexMemory
        from src.memory.palace import Palace
        palace = Palace(db_path=str(tmp_path / "p.db"),
                        embed_fn=lambda t: np.ones(8, dtype=np.float32))
        return CortexMemory(palace=palace), palace

    @pytest.mark.parametrize("said", [
        'Traduza para inglês: "moro em Paris".',
        "traduz pra mim: moro em Paris e trabalho como médico",
        'Revisa esse texto: "Meu nome é Ana e moro em Porto."',
        'Corrige a gramática dessa frase: "eu moro em Lisboa desde 2019"',
        "Exemplo de frase em português: moro em Madrid.",
        'o João disse "moro em Roma" ontem',
    ])
    def test_it_never_becomes_a_fact(self, tmp_path, said):
        mem, palace = self._memory(tmp_path)
        mem.store_interaction(said, "ok")
        assert palace.get_fact("user", "mora_em") is None
        assert palace.get_fact("user", "nome") is None
        assert "Paris" not in mem.get_user_profile()

    def test_pedro_s_own_sentence_still_does(self, tmp_path):
        mem, palace = self._memory(tmp_path)
        mem.store_interaction("eu moro em Londres e trabalho com dados", "legal")
        assert palace.get_fact("user", "mora_em")["objeto"] == "Londres"
