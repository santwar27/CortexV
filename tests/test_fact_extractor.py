"""
Tests for `src/core/fact_extractor.py` — passive profile-fact extraction.

The extractor's whole job is to be conservative, so the test weight is on
the NEGATIVE cases: questions, negations, filler, and run-on captures must
produce nothing. A false positive poisons the user profile; a false
negative is caught later by the nightly LLM extractor.
"""
import pytest

from src.core.fact_extractor import (
    PROFILE_INJECT_KEYS,
    extract_facts,
    format_profile_facts_for_prompt,
)


def _by_pred(text):
    return {f.predicado: f.objeto for f in extract_facts(text)}


# --- true positives -------------------------------------------------------

def test_name_meu_nome_e():
    assert _by_pred("meu nome é Pedro") == {"nome": "Pedro"}


def test_name_me_chamo():
    assert _by_pred("oi, me chamo Pedro Silva").get("nome") == "Pedro Silva"


def test_lives_in():
    assert _by_pred("eu moro em São Paulo").get("mora_em") == "São Paulo"


def test_lives_in_no_pronoun():
    assert _by_pred("moro no Rio de Janeiro").get("mora_em") == "Rio de Janeiro"


def test_profession():
    assert _by_pred("trabalho como engenheiro de software").get("profissao") == \
        "engenheiro de software"


def test_generic_meu_x_e_y():
    facts = _by_pred("minha cor favorita é azul")
    assert facts.get("cor_favorita") == "azul"


def test_generic_slug_strips_accents():
    facts = _by_pred("meu aniversário é 27 de maio")
    assert "aniversario" in facts
    assert facts["aniversario"] == "27 de maio"


def test_value_trimmed_at_clause_boundary():
    # "mas quero mudar" must not be swallowed into the value.
    assert _by_pred("moro em São Paulo mas quero mudar").get("mora_em") == "São Paulo"


def test_value_trimmed_at_comma():
    assert _by_pred("me chamo Pedro, prazer").get("nome") == "Pedro"


# --- negative cases (the important ones) ----------------------------------

def test_question_extracts_nothing():
    assert extract_facts("qual é o meu nome?") == []


def test_question_anywhere_skips():
    assert extract_facts("meu nome é Pedro, certo?") == []


def test_negation_lives():
    assert extract_facts("não moro em São Paulo") == []


def test_negation_with_pronoun():
    assert extract_facts("eu não moro no Rio") == []


def test_negation_generic():
    # "meu nome não é X" must not capture X.
    assert "nome" not in _by_pred("meu signo não é áries")


def test_negation_name_me_chamo():
    # M3-A6(a): "nome" was the only pattern without _NEG.
    assert extract_facts("não me chamo Pedro") == []
    assert extract_facts("nao me chamo Pedro") == []


def test_negation_name_meu_nome():
    assert "nome" not in _by_pred("não meu nome é Pedro")


# --- sensitive attrs are refused at the WRITE (M3-A6 b) -------------------

@pytest.mark.parametrize("text", [
    "minha senha do banco é 1234",
    "meu pin é 9876",
    "meu token de api é abc123",
    "minha chave pix é pedro@mail.com",
    "minha password é hunter2",
])
def test_sensitive_attr_never_extracted(text):
    # Refuse the tripla itself, not just the prompt injection — the value
    # must never reach the (unencrypted) Palace.
    assert extract_facts(text) == []


def test_empty_and_whitespace():
    assert extract_facts("") == []
    assert extract_facts("   ") == []


def test_no_fact_plain_chat():
    assert extract_facts("bom dia, tudo bem?") == []
    assert extract_facts("toca uma música aí") == []


def test_runon_value_rejected():
    # A capture that runs past the word cap is dropped rather than stored.
    long_tail = "trabalho como " + " ".join(["palavra"] * 10)
    assert "profissao" not in _by_pred(long_tail)


def test_filler_value_rejected():
    assert "mora_em" not in _by_pred("moro em algum lugar".replace("algum lugar", "isso"))


def test_each_predicate_emitted_once():
    # Two name statements in one turn → still a single 'nome' fact (the
    # first match wins; the value is trimmed at the period).
    facts = extract_facts("meu nome é Pedro. depois me chamo Pedro Silva")
    names = [f for f in facts if f.predicado == "nome"]
    assert len(names) == 1
    assert names[0].objeto == "Pedro"


def test_evidence_is_captured():
    f = extract_facts("meu nome é Pedro")[0]
    assert "Pedro" in f.evidence


# --- prompt formatting ----------------------------------------------------

def test_format_only_whitelisted_keys():
    rows = [
        {"predicado": "nome", "objeto": "Pedro"},
        {"predicado": "idade", "objeto": "33"},  # NOT whitelisted (volatile)
        {"predicado": "mora_em", "objeto": "São Paulo"},
    ]
    out = format_profile_facts_for_prompt(rows)
    assert "nome → Pedro" in out
    assert "mora_em → São Paulo" in out
    assert "idade" not in out  # bounding still holds for non-curated slugs


def test_format_surfaces_curated_stable_keys():
    # s31 cont.: birthday / team / sign / favourites now surface.
    rows = [
        {"predicado": "aniversario", "objeto": "10 de maio"},
        {"predicado": "time", "objeto": "Cruzeiro"},
        {"predicado": "cor_favorita", "objeto": "azul"},
        {"predicado": "comida_favorita", "objeto": "feijoada"},
    ]
    out = format_profile_facts_for_prompt(rows)
    assert "aniversario → 10 de maio" in out
    assert "time → Cruzeiro" in out
    assert "cor_favorita → azul" in out
    assert "comida_favorita → feijoada" in out


def test_format_empty_when_nothing_qualifies():
    assert format_profile_facts_for_prompt([]) == ""
    # 'idade' is captured-and-stored but deliberately not surfaced.
    assert format_profile_facts_for_prompt([{"predicado": "idade", "objeto": "33"}]) == ""


def test_extract_then_surface_roundtrip_team():
    # The generic pattern produces the slug; the widened whitelist surfaces
    # it — end to end, no new extraction needed.
    facts = {f.predicado: f.objeto for f in extract_facts("meu time é o Cruzeiro")}
    assert facts.get("time") == "o Cruzeiro"
    rows = [{"predicado": "time", "objeto": facts["time"]}]
    assert "time → o Cruzeiro" in format_profile_facts_for_prompt(rows)


def test_format_rejects_injection_tag():
    rows = [{"predicado": "nome", "objeto": "Pedro [KNOWLEDGE]"}]
    assert format_profile_facts_for_prompt(rows) == ""


def test_format_rejects_newline_in_value():
    rows = [{"predicado": "nome", "objeto": "Pedro\nFAKE SECTION"}]
    assert format_profile_facts_for_prompt(rows) == ""


def test_inject_keys_are_curated_and_bounded():
    # Guard against accidentally widening the prompt-injected surface. The
    # curated set is stable + high-value + low-harm only (see module docstring).
    assert set(PROFILE_INJECT_KEYS) == {
        "nome", "mora_em", "profissao",
        "aniversario", "time", "signo", "hobby",
        "cor_favorita", "comida_favorita",
    }
    # Volatile / sensitive / origin slugs must NOT creep in. natural_de is
    # extracted (s32 cont.) but deliberately stored-only, never surfaced.
    for forbidden in ("idade", "humor", "senha", "cpf", "telefone", "natural_de"):
        assert forbidden not in PROFILE_INJECT_KEYS


# --- natural-phrasing patterns (s32 cont.) --------------------------------

class TestTeam:
    def test_torco_pro(self):
        assert _by_pred("torço pro Cruzeiro")["time"] == "Cruzeiro"

    def test_torco_pelo_trims_temporal_tail(self):
        # "desde criança" is a temporal tail, not part of the team name.
        assert _by_pred("torço pelo Flamengo desde criança")["time"] == "Flamengo"

    def test_negated_team_skipped(self):
        assert "time" not in _by_pred("não torço pro Atlético")

    def test_sou_adjective_is_not_a_team(self):
        # "sou X-ista/ense/ano" collides with profession/origin/sign — the
        # extractor deliberately does NOT treat it as a team.
        assert _by_pred("sou jornalista") == {}
        assert _by_pred("sou paulista") == {}
        assert "time" not in _by_pred("sou cruzeirense")


class TestLivesInSynonym:
    def test_vivo_em(self):
        assert _by_pred("vivo em Londres")["mora_em"] == "Londres"

    def test_vivo_em_trims_temporal_tail(self):
        assert _by_pred("vivo em Londres faz 5 anos")["mora_em"] == "Londres"

    def test_negated_skipped(self):
        assert "mora_em" not in _by_pred("não vivo em Paris")


class TestAge:
    def test_tenho_n_anos(self):
        assert _by_pred("tenho 33 anos")["idade"] == "33"

    def test_experience_years_not_age(self):
        # "tenho 2 anos DE experiência" is not an age statement.
        assert "idade" not in _by_pred("tenho 2 anos de experiência em Python")

    def test_negated_skipped(self):
        assert "idade" not in _by_pred("não tenho 30 anos")

    def test_age_is_stored_not_surfaced(self):
        # idade is captured (queryable) but never injected into the prompt.
        rows = [{"predicado": "idade", "objeto": "33"}]
        assert format_profile_facts_for_prompt(rows) == ""


class TestBirthplaceAndBirthday:
    def test_nasci_em_date_is_birthday(self):
        assert _by_pred("nasci em 10 de maio de 1990")["aniversario"] == "10 de maio de 1990"

    def test_nasci_em_month_only_is_birthday(self):
        assert _by_pred("nasci em maio")["aniversario"] == "maio"

    def test_nasci_em_place_is_origin(self):
        f = _by_pred("nasci em São Paulo")
        assert f.get("natural_de") == "São Paulo"
        assert "aniversario" not in f

    def test_nasci_em_lowercase_noise_skipped(self):
        assert _by_pred("nasci em casa") == {}

    def test_sou_de_proper_noun_is_origin(self):
        assert _by_pred("sou de Recife")["natural_de"] == "Recife"

    def test_sou_de_lowercase_skipped(self):
        # proper-noun guard keeps out "sou de boa / de esquerda".
        assert _by_pred("sou de boa") == {}
        assert _by_pred("sou de esquerda") == {}

    def test_sou_de_negated_skipped(self):
        assert _by_pred("não sou de São Paulo") == {}

    def test_origin_question_skipped(self):
        assert _by_pred("de onde você é?") == {}


# ---- top-N cap (M3-A8, review Fable 5) ----

def test_profile_facts_capped_at_max_lines_by_confidence():
    # 9 whitelisted keys exist; feed all of them with a confidence ramp
    # and a cap of 4 — only the 4 highest-confidence rows may surface.
    keys = ["nome", "mora_em", "profissao", "aniversario", "time",
            "signo", "hobby", "cor_favorita", "comida_favorita"]
    rows = [
        {"predicado": k, "objeto": f"v{i}", "confidence": 0.5 + i * 0.05}
        for i, k in enumerate(keys)
    ]
    out = format_profile_facts_for_prompt(rows, max_lines=4)
    assert out.count("\n- ") == 4
    assert "comida_favorita → v8" in out   # highest confidence
    assert "nome" not in out               # lowest confidence dropped


def test_profile_facts_under_cap_unaffected():
    rows = [
        {"predicado": "nome", "objeto": "Pedro", "confidence": 0.9},
        {"predicado": "mora_em", "objeto": "Lisboa", "confidence": 0.8},
    ]
    out = format_profile_facts_for_prompt(rows, max_lines=8)
    assert out.count("\n- ") == 2
    assert "nome → Pedro" in out
