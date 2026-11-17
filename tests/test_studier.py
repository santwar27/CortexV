"""
NightlyStudier: ML-context detection, query injection, blocklist filter.

Pure logic only — no Ollama / Tavily / DDG calls.
"""
from pathlib import Path

import pytest

from src.core.studier import NightlyStudier


class TestMLContextDetection:
    """is_ml_topic flags topics that need disambiguation."""

    @pytest.fixture
    def studier(self):
        return NightlyStudier.__new__(NightlyStudier)

    @pytest.mark.parametrize("topic", [
        "Quantização: Q4 vs Q5 vs Q8 trade-offs práticos",
        "KV cache em transformers",
        "GQA vs MHA performance comparativa",
        "Qwen 2.5 Coder vs DeepSeek Coder",
        "attention mechanism em transformers",
        "RAG vs fine-tuning quando usar cada um",
        "Llama 3.1 8B quantization",
        "tokenization in transformer models",
    ])
    def test_detects_ml_topic(self, studier, topic):
        assert studier.is_ml_topic(topic) is True

    @pytest.mark.parametrize("topic", [
        "OAuth 2.0 Authorization Code Flow com PKCE",
        "história do Brasil colonial",
        "cidadania italiana via discendenza",
        "regras do xadrez",
    ])
    def test_excludes_non_ml(self, studier, topic):
        assert studier.is_ml_topic(topic) is False


class TestQueryInjection:
    """_inject_ml_context adds 'LLM' to queries missing ML context."""

    def test_adds_llm_when_missing(self):
        assert NightlyStudier._inject_ml_context(
            "Q4 vs Q5 comparison"
        ) == "Q4 vs Q5 comparison LLM"

    def test_skips_when_attention_present(self):
        q = "attention mechanism explanation"
        assert NightlyStudier._inject_ml_context(q) == q

    def test_skips_when_transformer_present(self):
        q = "transformer architecture"
        assert NightlyStudier._inject_ml_context(q) == q

    def test_skips_when_llm_present(self):
        q = "LLM quantization"
        assert NightlyStudier._inject_ml_context(q) == q

    def test_adds_when_only_acronym(self):
        result = NightlyStudier._inject_ml_context("KV cache benchmarks")
        assert "LLM" in result

    def test_skips_when_neural_present(self):
        q = "neural network pruning"
        assert NightlyStudier._inject_ml_context(q) == q


class TestBlocklistFilter:
    """_filter_blocklist drops snippets from known off-topic domains."""

    def test_drops_audi(self):
        snippets = (
            "Source 1 (Real ML):\nContent.\n[https://geeksforgeeks.org/x]\n\n"
            "Source 2 (Audi):\nQ4 review.\n[https://www.audiusa.com/q4]"
        )
        filtered, dropped = NightlyStudier._filter_blocklist(snippets, include_ml=True)
        assert dropped == 1
        assert "audiusa" not in filtered.lower()
        assert "geeksforgeeks" in filtered.lower()

    def test_drops_rick_roll(self):
        snippets = (
            "Source 1 (Decoy):\nMusic video.\n"
            "[https://youtube.com/watch?v=dQw4w9WgXcQ]\n\n"
            "Source 2 (Real):\nArxiv paper.\n[https://arxiv.org/pdf/x]"
        )
        filtered, dropped = NightlyStudier._filter_blocklist(snippets, include_ml=True)
        assert dropped == 1
        assert "dqw4w9" not in filtered.lower()
        assert "arxiv" in filtered.lower()

    def test_drops_q8_gas_station(self):
        snippets = (
            "Source 1 (Gas):\nQ8 stations in Italy.\n[https://www.q8.it/]"
        )
        filtered, dropped = NightlyStudier._filter_blocklist(snippets, include_ml=True)
        assert dropped == 1
        assert filtered.strip() == ""

    def test_drops_fiscal_quarter(self):
        snippets = (
            "Source 1 (Fiscal):\nWhat is fiscal Q4.\n"
            "[https://www.investopedia.com/fiscal-quarter-q1-q2-q3-q4-what-is]"
        )
        filtered, dropped = NightlyStudier._filter_blocklist(snippets, include_ml=True)
        assert dropped == 1

    def test_drops_glass_qualifications_authority(self):
        snippets = (
            "Source 1 (Wrong GQA):\nGlass tests.\n[https://gqa.co.uk/]"
        )
        filtered, dropped = NightlyStudier._filter_blocklist(snippets, include_ml=True)
        assert dropped == 1

    def test_keeps_arxiv(self):
        snippets = (
            "Source 1 (Real ML):\nAttention paper.\n"
            "[https://arxiv.org/pdf/1706.03762.pdf]"
        )
        filtered, dropped = NightlyStudier._filter_blocklist(snippets, include_ml=True)
        assert dropped == 0
        assert "arxiv" in filtered.lower()

    def test_no_change_when_all_clean(self):
        snippets = (
            "Source 1 (A):\nX.\n[https://huggingface.co/y]\n\n"
            "Source 2 (B):\nY.\n[https://github.com/z]"
        )
        filtered, dropped = NightlyStudier._filter_blocklist(snippets, include_ml=True)
        assert dropped == 0
        assert filtered == snippets


class TestGeneralBlocklist:
    """The general junk list (crypto explorers, mining) is dropped for EVERY
    topic — include_ml defaults to False, so it applies even to non-ML
    studies. Observed live 2026-06-03: a 'constrained decoding' study cited
    etherscan/iqmining/xprnetwork."""

    @pytest.mark.parametrize("junk_url", [
        "https://etherscan.io/tx/0xabc",
        "https://iqmining.com/en/mining-aeon",
        "https://explorer.xprnetwork.org/",
        "https://www.bscscan.com/address/0x1",
    ])
    def test_drops_crypto_junk_for_any_topic(self, junk_url):
        snippets = (
            "Source 1 (Real):\nConstrained decoding.\n[https://ollama.com/blog/x]\n\n"
            f"Source 2 (Junk):\nCrypto noise.\n[{junk_url}]"
        )
        # Default call (include_ml=False) — proves it fires for non-ML topics.
        filtered, dropped = NightlyStudier._filter_blocklist(snippets)
        assert dropped == 1
        assert "ollama.com" in filtered.lower()

    def test_general_list_does_not_need_ml_flag(self):
        snippets = "Source 1 (Junk):\nX.\n[https://etherscan.io/]"
        _, dropped_default = NightlyStudier._filter_blocklist(snippets)
        _, dropped_ml = NightlyStudier._filter_blocklist(snippets, include_ml=True)
        assert dropped_default == 1 and dropped_ml == 1

    def test_ml_noise_needs_the_flag(self):
        # An ML-specific junk domain (Audi) is NOT dropped without include_ml.
        snippets = "Source 1 (Audi):\nQ4.\n[https://www.audiusa.com/q4]"
        _, dropped_default = NightlyStudier._filter_blocklist(snippets)
        _, dropped_ml = NightlyStudier._filter_blocklist(snippets, include_ml=True)
        assert dropped_default == 0
        assert dropped_ml == 1


class TestSynthesisPromptRules:
    """Prompt-level guards added 2026-06-03 after a live review found the 8B
    translating technical terms ('constrained decoding' → 'Construído
    decodificação') and drifting to generic background. Asserted on the
    prompt text the studier actually sends."""

    def _capture_prompt(self, monkeypatch, topic="MLX vs llama.cpp", snippets="s"):
        import src.core.studier as studier_mod
        captured = {}

        def _fake_generate(model=None, prompt=None, options=None, **kw):
            captured["prompt"] = prompt
            return {"response": "## Resumo executivo\nok\n\n## Confiança\nbaixa — x"}

        monkeypatch.setattr(studier_mod.ollama, "generate", _fake_generate)
        s = NightlyStudier.__new__(NightlyStudier)

        class _Eng:
            model_name = "cortex-lite"

        s.engine = _Eng()
        s._summarise(topic, snippets)
        return captured["prompt"]

    def test_prompt_keeps_technical_terms_in_english(self, monkeypatch):
        p = self._capture_prompt(monkeypatch)
        assert "do NOT translate" in p
        assert "constrained decoding" in p  # the worked example term

    def test_prompt_anchors_on_the_specific_question(self, monkeypatch):
        p = self._capture_prompt(monkeypatch)
        assert "ANSWER THE SPECIFIC QUESTION" in p


class TestPalaceBackup:
    """`run()` snapshots data/palace.db before any work via SQLite's hot-backup
    API. Errors don't propagate — backup is best-effort."""

    def _bare_studier(self, palace_path: str, backup_dir: str) -> NightlyStudier:
        s = NightlyStudier.__new__(NightlyStudier)
        s.PALACE_DB_PATH = palace_path
        s.BACKUP_DIR = backup_dir
        return s

    @staticmethod
    def _seed_palace(path, rows):
        """Create a real Palace at `path` containing the given (tipo, texto) rows."""
        from src.memory.palace import Palace
        p = Palace(db_path=str(path))
        for tipo, texto in rows:
            p.add_document(tipo, texto)
        p.close()

    @staticmethod
    def _read_palace(path):
        """Open a backup file as a Palace and return its (tipo, texto) docs."""
        from src.memory.palace import Palace
        p = Palace(db_path=str(path))
        out = [(d["tipo"], d["texto"]) for d in p.list_documents()]
        p.close()
        return out

    def test_copies_existing_palace_db(self, tmp_path):
        palace = tmp_path / "palace.db"
        self._seed_palace(palace, [("note", "row before backup")])
        backup_dir = tmp_path / "backups"
        s = self._bare_studier(str(palace), str(backup_dir))

        result = s._backup_palace()

        assert result is not None
        assert Path(result).exists()
        assert Path(result).parent == backup_dir
        assert Path(result).suffix == ".db"
        # backup must be a valid SQLite DB with the source's row visible
        assert self._read_palace(result) == [("note", "row before backup")]

    def test_skips_when_palace_missing(self, tmp_path):
        s = self._bare_studier(
            str(tmp_path / "nope.db"), str(tmp_path / "backups")
        )
        assert s._backup_palace() is None
        # backup dir should not be created on no-op
        assert not (tmp_path / "backups").exists()

    def test_second_run_same_day_reflects_latest_source(self, tmp_path):
        palace = tmp_path / "palace.db"
        self._seed_palace(palace, [("note", "v1")])
        backup_dir = tmp_path / "backups"
        s = self._bare_studier(str(palace), str(backup_dir))

        first = s._backup_palace()
        # Append a new row to the source after the first backup ran.
        self._seed_palace(palace, [("note", "v2 added")])
        second = s._backup_palace()

        # Same calendar day → same destination filename → single file.
        assert first == second
        assert len(list(backup_dir.iterdir())) == 1
        # Backup reflects the latest source, including the row added after run #1.
        rows = self._read_palace(second)
        assert ("note", "v1") in rows
        assert ("note", "v2 added") in rows

    def test_sqlite_error_is_swallowed(self, tmp_path, monkeypatch):
        """Backup failure must not block the nightly run."""
        palace = tmp_path / "palace.db"
        self._seed_palace(palace, [("note", "data")])
        s = self._bare_studier(str(palace), str(tmp_path / "backups"))

        import sqlite3 as _sqlite3

        def _boom(*_a, **_kw):
            raise _sqlite3.Error("simulated backup failure")

        monkeypatch.setattr("src.core.studier.sqlite3.connect", _boom)
        # must not raise — backup failure cannot block the nightly run
        assert s._backup_palace() is None


class TestRequeueFailed:
    """A topic that produces nothing gets ONE retry, then is dropped.

    Regression guard: live 2026-07-10, the web lease expired mid-run and
    4 of 6 topics hit no_snippets — consumed but never re-enqueued, so
    they were silently lost. _requeue_failed re-adds them once."""

    def _studier(self, tmp_path):
        from src.core.study_queue import StudyQueue

        s = NightlyStudier.__new__(NightlyStudier)
        s.queue = StudyQueue(path=str(tmp_path / "q.jsonl"))
        return s

    def test_first_failure_requeues_with_force(self, tmp_path):
        s = self._studier(tmp_path)
        # The topic already exists as consumed → without '!!' the dup guard
        # would veto the re-add. _requeue_failed must force it through.
        s.queue.add("SQLite WAL internals")
        s.queue.consume_all()
        s._requeue_failed({"topic": "SQLite WAL internals"}, None)
        pending = s.queue.list_pending()
        assert len(pending) == 1
        assert pending[0]["topic"] == "SQLite WAL internals"
        assert pending[0]["retries"] == 1
        assert pending[0]["requeued"] == "failed_run"

    def test_second_failure_is_dropped(self, tmp_path):
        s = self._studier(tmp_path)
        # Entry already carries retries=1 → exhausted, no re-add.
        s._requeue_failed({"topic": "FastAPI DI", "retries": 1}, None)
        assert s.queue.list_pending() == []

    def test_empty_topic_noop(self, tmp_path):
        s = self._studier(tmp_path)
        s._requeue_failed({"topic": "   "}, None)
        assert s.queue.list_pending() == []
