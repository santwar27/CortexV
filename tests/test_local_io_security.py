"""
Path traversal hardening for local_reader and local_writer.

Covers:
- Classic ../ traversal
- Symlink escape (a symlink inside the vault pointing outside)
- Null byte + control chars
- Slashes / backslashes in filename
- Empty / None filename
- Happy path still works
"""
import os
import pytest

from src.tools.local_reader import LocalFileReader
from src.tools.local_writer import LocalFileWriter


@pytest.fixture
def vault(tmp_path):
    v = tmp_path / "documents"
    v.mkdir()
    return v


@pytest.fixture
def reader(vault):
    return LocalFileReader(base_dir=str(vault))


@pytest.fixture
def writer(vault):
    return LocalFileWriter(base_dir=str(vault))


class TestReaderHardening:

    def test_happy_path(self, reader, vault):
        (vault / "notes.txt").write_text("hello world", encoding="utf-8")
        out = reader.read_document("notes.txt")
        assert "hello world" in out

    def test_markdown_read_as_text(self, reader, vault):
        # M4-A12: .md is the vault's native format and the PROJECT_MAP
        # advertises it — it must read, not bounce off "unsupported".
        (vault / "readme.md").write_text("# Title\n\nbody text", encoding="utf-8")
        out = reader.read_document("readme.md")
        assert "body text" in out
        assert "Unsupported" not in out

    def test_unsupported_extension_still_rejected(self, reader, vault):
        # The guard still rejects formats the reader genuinely can't parse.
        (vault / "data.csv").write_text("a,b,c", encoding="utf-8")
        out = reader.read_document("data.csv")
        assert "Unsupported file format" in out

    def test_traversal_rejected(self, reader):
        out = reader.read_document("../../etc/passwd")
        assert "Invalid filename" in out or "Path traversal" in out

    def test_null_byte_rejected(self, reader):
        out = reader.read_document("notes\x00.txt")
        assert "Invalid filename" in out

    def test_slash_in_filename_rejected(self, reader):
        out = reader.read_document("sub/notes.txt")
        assert "Invalid filename" in out

    def test_backslash_in_filename_rejected(self, reader):
        out = reader.read_document("sub\\notes.txt")
        assert "Invalid filename" in out

    def test_empty_rejected(self, reader):
        out = reader.read_document("")
        assert "Invalid filename" in out

    def test_exactly_15000_chars_not_flagged_truncated(self, reader, vault):
        # M4-A14c: a file of EXACTLY 15000 chars used to get the truncation
        # notice (len(read(15000)) == 15000 is also true for a full read).
        (vault / "exact.txt").write_text("x" * 15000, encoding="utf-8")
        out = reader.read_document("exact.txt")
        assert "truncated" not in out.lower()
        assert out == "x" * 15000

    def test_over_15000_chars_is_truncated_with_notice(self, reader, vault):
        (vault / "big.txt").write_text("x" * 15001, encoding="utf-8")
        out = reader.read_document("big.txt")
        assert "truncated" in out.lower()
        assert out.startswith("x" * 100)
        assert len(out) < 15200  # capped + notice, not the whole file

    def test_symlink_escape_blocked(self, reader, vault, tmp_path):
        """A symlink inside the vault pointing outside it must be refused."""
        secret = tmp_path / "secret.txt"
        secret.write_text("top secret", encoding="utf-8")
        link = vault / "innocent.txt"
        os.symlink(str(secret), str(link))

        out = reader.read_document("innocent.txt")
        assert "Path traversal" in out
        assert "top secret" not in out


class TestReadPath:
    """`read_path` extracts from an ARBITRARY absolute path with no vault
    guard — the screen-context "doc aberto" path (inovação #3) reads the file
    the user has open in Preview, which lives outside the vault. The vault
    guard stays on `read_document` (the LLM tool); these two contracts are
    intentionally different."""

    def test_reads_file_outside_vault(self, reader, tmp_path):
        # The whole point: a path the vault-guarded read_document would reject.
        outside = tmp_path / "elsewhere" / "open_doc.txt"
        outside.parent.mkdir()
        outside.write_text("conteúdo do documento aberto", encoding="utf-8")
        out = reader.read_path(str(outside))
        assert "conteúdo do documento aberto" in out

    def test_markdown_outside_vault(self, reader, tmp_path):
        f = tmp_path / "notes.md"
        f.write_text("# H\n\ncorpo do md", encoding="utf-8")
        assert "corpo do md" in reader.read_path(str(f))

    def test_unsupported_extension(self, reader, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b,c", encoding="utf-8")
        assert "Unsupported" in reader.read_path(str(f))

    def test_parse_failure_returns_error_marker(self, reader, tmp_path):
        # A .pdf that isn't a real PDF → the try/except returns the marker,
        # not a raised exception (the screen-context caller keys off it).
        f = tmp_path / "broken.pdf"
        f.write_text("not a pdf", encoding="utf-8")
        assert reader.read_path(str(f)).startswith("[SYSTEM ERROR]")

    def test_read_document_still_vault_guarded(self, reader):
        # Sanity: the refactor didn't loosen read_document — an absolute path
        # is still rejected by the filename whitelist.
        out = reader.read_document("/etc/passwd")
        assert "[SYSTEM ERROR]" in out


class TestWriterHardening:

    def test_happy_path(self, writer, vault):
        result = writer.write_document("note", "first line")
        assert "successfully saved" in result
        # M4-A14b: the reply says the real semantics — append, not overwrite.
        assert "appended" in result
        assert (vault / "note.txt").read_text(encoding="utf-8") == "first line\n\n"

    def test_repeated_write_appends_documented_contract(self, writer, vault):
        # M4-A14b: two identical calls = two copies. Documented behaviour
        # (nothing is lost, nothing is deduplicated) — pinned so a future
        # "fix" to overwrite mode is a conscious decision, not an accident.
        writer.write_document("note", "same line")
        writer.write_document("note", "same line")
        content = (vault / "note.txt").read_text(encoding="utf-8")
        assert content.count("same line") == 2

    def test_traversal_rejected(self, writer):
        out = writer.write_document("../../tmp/evil", "x")
        assert "Invalid filename" in out

    def test_null_byte_rejected(self, writer):
        out = writer.write_document("note\x00name", "x")
        assert "Invalid filename" in out

    def test_slash_in_filename_rejected(self, writer):
        out = writer.write_document("sub/note", "x")
        assert "Invalid filename" in out

    def test_empty_rejected(self, writer):
        out = writer.write_document("", "x")
        assert "Invalid filename" in out

    def test_symlink_escape_blocked(self, writer, vault, tmp_path):
        """Append into a symlink that points outside the vault must be refused."""
        outside = tmp_path / "outside.txt"
        outside.write_text("original\n", encoding="utf-8")
        link = vault / "trap.txt"
        os.symlink(str(outside), str(link))

        out = writer.write_document("trap.txt", "injected payload")
        assert "Path traversal" in out
        # File outside the vault must NOT have been appended to.
        assert outside.read_text(encoding="utf-8") == "original\n"

    def test_engine_error_marker_refused(self, writer):
        out = writer.write_document("note", "result was [SYSTEM ERROR] something")
        assert "Refused to persist" in out
