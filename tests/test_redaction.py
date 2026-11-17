"""
Tests for src/core/redaction.py — the single redaction module applied
at the border of every plaintext sink (M6-A6).

The module is pure (no I/O); sink wiring is tested in each sink's own
test file (audit/gate_queue/action_ledger/telemetry/fact_extractor).
"""
import pytest

from src.core.redaction import (
    REDACTED,
    is_sensitive_attr,
    redact_payload,
    redact_text,
)


class TestIsSensitiveAttr:

    @pytest.mark.parametrize("attr", [
        "senha", "password", "pin", "token", "chave", "key",
        "secret", "credencial",
        # composed / slugified forms
        "senha_do_banco", "api_key", "API_KEY", "auth-token",
        "minha chave pix", "Senha",
    ])
    def test_sensitive(self, attr):
        assert is_sensitive_attr(attr) is True

    @pytest.mark.parametrize("attr", [
        # substrings must NOT trigger — token matching only
        "monkey", "happiness", "pintura", "chaveiro", "tokenizer",
        # ordinary attrs
        "nome", "mora_em", "to", "body", "cor_favorita", "",
        None,
    ])
    def test_not_sensitive(self, attr):
        assert is_sensitive_attr(attr) is False


class TestRedactText:

    def test_masks_stated_secret_pt(self):
        out = redact_text("minha senha do banco é 1234")
        assert "1234" not in out
        assert REDACTED in out
        assert out.startswith("minha senha do banco é")

    def test_masks_to_end_of_clause(self):
        out = redact_text("o token é abc def ghi, e o resto fica")
        assert "abc" not in out and "ghi" not in out
        assert "e o resto fica" in out

    def test_masks_colon_and_equals(self):
        assert "hunter2" not in redact_text("password: hunter2")
        assert "hunter2" not in redact_text("API key = hunter2")

    def test_plain_text_untouched(self):
        text = "me chamo Pedro e moro em Londres"
        assert redact_text(text) == text

    def test_sensitive_word_without_value_untouched(self):
        # No copula/separator in the clause → nothing to mask.
        text = "esqueci a senha de novo."
        assert redact_text(text) == text

    def test_non_string_passthrough(self):
        assert redact_text(None) is None
        assert redact_text(42) == 42
        assert redact_text("") == ""


class TestRedactPayload:

    def test_sensitive_key_masked_wholesale(self):
        out = redact_payload({"user": "pedro", "password": "hunter2"})
        assert out == {"user": "pedro", "password": REDACTED}

    def test_nested_containers(self):
        payload = {
            "args": {"api_key": "sk-123", "to": "a@b.c"},
            "items": [{"token": "t1"}, "plain"],
        }
        out = redact_payload(payload)
        assert out["args"] == {"api_key": REDACTED, "to": "a@b.c"}
        assert out["items"][0] == {"token": REDACTED}
        assert out["items"][1] == "plain"

    def test_free_text_values_scrubbed(self):
        out = redact_payload({"text": "minha senha é 1234"})
        assert "1234" not in out["text"]

    def test_non_string_sensitive_value_masked(self):
        assert redact_payload({"pin": 1234}) == {"pin": REDACTED}

    def test_input_not_mutated(self):
        payload = {"senha": "x", "inner": {"token": "y"}}
        redact_payload(payload)
        assert payload == {"senha": "x", "inner": {"token": "y"}}

    def test_tuple_stays_tuple(self):
        out = redact_payload(("a", {"key": "k"}))
        assert out == ("a", {"key": REDACTED})
        assert isinstance(out, tuple)

    def test_scalars_passthrough(self):
        assert redact_payload(7) == 7
        assert redact_payload(None) is None


class TestWhatUsedToLeak:
    """Measured (s88, Codex lote 10): the mask stopped at the first "." of a
    password, and a CPF, a recovery code and a bare API key had no rule."""

    @pytest.mark.parametrize("text,secret", [
        ("minha senha é abc.DEF987", "DEF987"),
        ("my password is abc.DEF987", "DEF987"),
        ("meu CPF: 123.456.789-09", "456.789"),
        ("o código de recuperação é ABCD-EFGH", "ABCD-EFGH"),
        ("my recovery codes are 1111-2222", "1111-2222"),
        ("usa a chave sk-proj-AbCdEf1234567890abcdef1234", "AbCdEf1234567890"),
        ("token ghp_abcdefghijklmnopqrstuvwxyz0123", "ghp_abc"),
    ])
    def test_the_whole_value_is_masked(self, text, secret):
        out = redact_text(text)
        assert secret not in out and REDACTED in out, out

    def test_a_clause_after_the_secret_survives(self):
        assert redact_text("minha senha é 1234, e o resto fica") == \
            "minha senha é [REDACTED], e o resto fica"

    def test_the_sinks_get_it_too(self, tmp_path):
        from src.core.audit import AuditLogger
        log = AuditLogger(str(tmp_path / "a.jsonl"))
        log.log("t", "voice", "transcript", payload={"text": "minha senha é abc.DEF987"})
        assert "DEF987" not in (tmp_path / "a.jsonl").read_text()
