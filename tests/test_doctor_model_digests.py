"""
Tests for the model-digest TOFU check (src/doctor.py, plan item 4).

The digest source (_ollama_model_digests) is monkeypatched so nothing touches
a real Ollama; the baseline path is redirected to a tmp file via env.
"""
import json

import pytest

from src import doctor


@pytest.fixture
def baseline(tmp_path, monkeypatch):
    p = tmp_path / "model_digests.json"
    monkeypatch.setenv("CORTEX_MODEL_DIGEST_FILE", str(p))
    # A stable configured set → known pulled digests.
    monkeypatch.setenv("CORTEX_MODEL_ANALYTICAL", "cortex-lite")
    monkeypatch.setenv("CORTEX_EMBED_MODEL", "bge-m3")
    return p


def _fake_digests(mapping):
    return lambda: dict(mapping)


class TestDigestCheck:
    def test_first_run_records_baseline(self, baseline, monkeypatch):
        monkeypatch.setattr(doctor, "_ollama_model_digests",
                            _fake_digests({"cortex-lite:latest": "sha-A",
                                           "bge-m3:latest": "sha-B"}))
        r = doctor.check_model_digests()
        assert r.status == doctor.OK and "first run" in r.detail
        saved = json.loads(baseline.read_text())
        assert saved == {"cortex-lite": "sha-A", "bge-m3": "sha-B"}

    def test_unchanged_matches(self, baseline, monkeypatch):
        baseline.write_text(json.dumps({"cortex-lite": "sha-A", "bge-m3": "sha-B"}))
        monkeypatch.setattr(doctor, "_ollama_model_digests",
                            _fake_digests({"cortex-lite:latest": "sha-A",
                                           "bge-m3:latest": "sha-B"}))
        r = doctor.check_model_digests()
        assert r.status == doctor.OK and "match" in r.detail

    def test_changed_digest_warns(self, baseline, monkeypatch):
        baseline.write_text(json.dumps({"cortex-lite": "sha-A", "bge-m3": "sha-B"}))
        monkeypatch.setattr(doctor, "_ollama_model_digests",
                            _fake_digests({"cortex-lite:latest": "sha-TAMPERED",
                                           "bge-m3:latest": "sha-B"}))
        r = doctor.check_model_digests()
        assert r.status == doctor.WARN
        assert "changed" in r.detail and "cortex-lite" in r.detail

    def test_missing_model_warns(self, baseline, monkeypatch):
        baseline.write_text(json.dumps({"cortex-lite": "sha-A", "bge-m3": "sha-B"}))
        monkeypatch.setattr(doctor, "_ollama_model_digests",
                            _fake_digests({"cortex-lite:latest": "sha-A"}))
        r = doctor.check_model_digests()
        assert r.status == doctor.WARN and "no longer pulled" in r.detail
        assert "bge-m3" in r.detail

    def test_ollama_unreachable_is_soft_warn(self, baseline, monkeypatch):
        def _boom():
            raise RuntimeError("connection refused")

        monkeypatch.setattr(doctor, "_ollama_model_digests", _boom)
        r = doctor.check_model_digests()
        assert r.status == doctor.WARN and "unreachable" in r.detail

    def test_registered_in_checks(self):
        assert doctor.check_model_digests in doctor.CHECKS


class TestTheConfiguredTagIsTheOneWatched:
    """By base name alone the first tag listed won: with cortex-lite:seguro
    configured and :latest listed first, :seguro could change with no
    warning, and an unpulled tag counted as present (s88, Codex lote 10)."""

    STATE = {"cortex-lite:latest": "AAA", "cortex-lite:seguro": "S1",
             "cortex-criativo:latest": "C", "llava:latest": "L",
             "nomic-embed-text:latest": "N"}

    def test_a_change_to_the_configured_tag_warns(self, tmp_path, monkeypatch):
        import src.doctor as d
        monkeypatch.setenv("CORTEX_MODEL_DIGEST_FILE", str(tmp_path / "d.json"))
        monkeypatch.setenv("CORTEX_MODEL_ANALYTICAL", "cortex-lite:seguro")
        state = dict(self.STATE)
        monkeypatch.setattr(d, "_ollama_model_digests", lambda: dict(state))
        d.check_model_digests()
        state["cortex-lite:seguro"] = "S2"
        r = d.check_model_digests()
        assert r.status == d.WARN and "cortex-lite:seguro" in r.detail

    def test_an_old_baseline_by_base_name_still_matches(self, tmp_path, monkeypatch):
        import json
        import src.doctor as d
        f = tmp_path / "d.json"
        f.write_text(json.dumps({"cortex-lite": "AAA", "cortex-criativo": "C",
                                 "llava": "L", "nomic-embed-text": "N"}))
        monkeypatch.setenv("CORTEX_MODEL_DIGEST_FILE", str(f))
        for k in ("CORTEX_MODEL_ANALYTICAL", "CORTEX_MODEL_CREATIVE",
                  "CORTEX_MODEL_VISION", "CORTEX_EMBED_MODEL"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setattr(d, "_ollama_model_digests", lambda: dict(self.STATE))
        assert d.check_model_digests().status == d.OK

    def test_an_unpulled_tag_is_not_present(self):
        from src.doctor import _has_model
        assert not _has_model(["cortex-lite:latest"], "cortex-lite:seguro")
        assert _has_model(["cortex-lite:latest"], "cortex-lite")
