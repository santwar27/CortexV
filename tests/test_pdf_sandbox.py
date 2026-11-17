"""Pure boundary checks; opt-in native probes only ever use the complete profile."""
from dataclasses import FrozenInstanceError, replace
import os
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import Mock

import pytest

from src.core import pdf_sandbox as sandbox

NAMES = ("pdfinfo", "pdftotext", "pdftoppm", "pdfimages", "pdfseparate", "pdfunite")


@pytest.fixture
def brew(tmp_path, monkeypatch):
    root = tmp_path.resolve() / "homebrew"
    bins, cellar = root / "bin", root / "Cellar/poppler"
    bins.mkdir(parents=True)
    for name in NAMES:
        target = cellar / "26.04.0/bin" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fake executable, never run")
        target.chmod(0o700)
        (bins / name).symlink_to(target)
    monkeypatch.setattr(sandbox, "HOMEBREW_BIN", str(bins))
    monkeypatch.setattr(sandbox, "POPPLER_CELLAR", str(cellar))
    return bins, cellar


@pytest.fixture
def dirs(tmp_path):
    paths = [tmp_path.resolve() / name for name in ('input"\\dir', "output")]
    for path in paths:
        path.mkdir()
    return tuple(map(str, paths))


def test_discover_fixed_and_frozen(brew, monkeypatch):
    monkeypatch.setenv("PATH", "/attacker")
    monkeypatch.setenv("POPPLER_CELLAR", "/attacker")
    record = sandbox.discover()
    assert sandbox.TOOLS == NAMES
    assert record.version == "26.04.0"
    assert dict(record.paths) == {n: str(brew[1] / "26.04.0/bin" / n) for n in NAMES}
    with pytest.raises(FrozenInstanceError):
        record.version = "evil"
    with pytest.raises(TypeError):
        record.paths["pdfinfo"] = "/bin/cat"
    mutable = dict(record.paths)
    copied = replace(record, paths=mutable)
    mutable["pdfinfo"] = "/bin/cat"
    assert copied.paths == record.paths


@pytest.mark.parametrize("fault", ["missing", "outside", "mixed", "directory",
                                      "not_executable", "nested", "wrong_name", "control"])
def test_discover_refuses_bad_tree(brew, tmp_path, fault):
    link, target = brew[0] / "pdfinfo", brew[1] / "26.04.0/bin/pdfinfo"
    if fault == "not_executable":
        target.chmod(0o600)
    elif fault == "directory":
        target.unlink()
        target.mkdir()
    else:
        link.unlink()
        if fault != "missing":
            alternate = {
                "outside": tmp_path / "elsewhere/pdfinfo",
                "mixed": brew[1] / "25.01.0/bin/pdfinfo",
                "nested": brew[1] / "26.04.0/bin/nested/pdfinfo",
                "wrong_name": brew[1] / "26.04.0/bin/cat",
                "control": brew[1] / "bad\nversion/bin/pdfinfo",
            }[fault]
            alternate.parent.mkdir(parents=True, exist_ok=True)
            alternate.write_bytes(b"fake")
            alternate.chmod(0o700)
            link.symlink_to(alternate)
    assert sandbox.discover() is None


def test_profile_exact_rules(brew, dirs):
    record = sandbox.discover()
    text = sandbox.profile_text(record, *dirs)
    quoted = dirs[0].replace('\\', '\\\\').replace('"', '\\"')
    expected_exec = "(allow process-exec " + " ".join(
        f'(literal "{record.paths[n]}")' for n in NAMES) + ")"
    assert text.splitlines()[:3] == ["(version 1)", "(deny default)", expected_exec]
    assert text.count("(allow process-exec") == 1
    assert '(allow file-read-metadata)' in text
    assert '(literal "/")' in text  # Required by dyld; never execute a profile missing it.
    read_rules = text.split("(allow file-read*", 1)[1].split("(allow file-write*", 1)[0]
    for path in ("/System", "/usr/lib", "/usr/share", "/private/var/db/dyld",
                 "/Library/Apple", "/Library/Fonts", "/opt/homebrew/Cellar",
                 "/opt/homebrew/opt", "/opt/homebrew/lib", "/opt/homebrew/share",
                 "/opt/homebrew/etc/fonts", quoted, dirs[1]):
        assert f'(subpath "{path}")' in read_rules
    for path in ("/dev/null", "/dev/urandom", "/dev/random"):
        assert f'(literal "{path}")' in read_rules
    assert [line for line in text.splitlines() if "file-write" in line] == [
        f'(allow file-write* (subpath "{dirs[1]}") (literal "/dev/null"))']
    assert '(allow sysctl-read)' in text
    assert '(allow mach-lookup (global-name "com.apple.system.opendirectoryd.libinfo"))' in text
    for forbidden in ("network", "process-fork", "(allow default)", '(subpath "/opt/homebrew")'):
        assert forbidden not in text
    assert text.count("(allow ") == 6


def test_directory_validation(brew, dirs, tmp_path):
    record = sandbox.discover()
    link = tmp_path / "alias"
    link.symlink_to(dirs[0], target_is_directory=True)
    regular = tmp_path / "file"
    regular.touch()
    child = Path(dirs[0]) / "child"
    child.mkdir()
    invalid = ["relative", str(link), str(regular), str(tmp_path / "missing"),
               dirs[0] + "/.", dirs[0] + "/", b"/bytes"]
    for path in invalid:
        for pair in ((path, dirs[1]), (dirs[0], path)):
            with pytest.raises(ValueError):
                sandbox.profile_text(record, *pair)
    for pair in ((dirs[0], dirs[0]), (dirs[0], str(child)), (str(child), dirs[0])):
        with pytest.raises(ValueError):
            sandbox.profile_text(record, *pair)
    sibling = Path(dirs[0] + "-sibling")
    sibling.mkdir()
    sandbox.profile_text(record, dirs[0], str(sibling))


@pytest.mark.parametrize("char", ["\0", "\n", "\r", "\t", "\x1f", "\x7f", "\x85"])
def test_control_characters_refused(brew, dirs, tmp_path, char):
    path = tmp_path.resolve() / ("bad" + char)
    if char != "\0":
        path.mkdir()
    for pair in ((str(path), dirs[1]), (dirs[0], str(path))):
        with pytest.raises(ValueError):
            sandbox.profile_text(sandbox.discover(), *pair)


def test_record_cannot_widen_exec(brew, dirs):
    record = sandbox.discover()
    for paths in ({**record.paths, "pdfinfo": "/bin/cat"},
                  {**record.paths, "cat": "/bin/cat"},
                  {k: v for k, v in record.paths.items() if k != "pdfinfo"},
                  {**record.paths, "pdfinfo": "/bad\npath"}):
        with pytest.raises(ValueError):
            sandbox.profile_text(replace(record, paths=paths), *dirs)
        with pytest.raises(ValueError):
            sandbox.contained_command(replace(record, paths=paths), "pdfinfo", [], *dirs)
    with pytest.raises(ValueError):
        sandbox.profile_text(replace(record, version="wrong"), *dirs)
    (brew[0] / "pdfinfo").unlink()
    with pytest.raises(ValueError):
        sandbox.profile_text(record, *dirs)


@pytest.mark.parametrize("fontconfig", [False, True])
def test_command_exact_env_and_argv(brew, dirs, tmp_path, monkeypatch, fontconfig):
    config = tmp_path / "fonts.conf"
    monkeypatch.setattr(sandbox, "FONTCONFIG_FILE", str(config))
    if fontconfig:
        config.touch()
    for name in ("HOME", "TMPDIR", "XDG_CACHE_HOME", "PATH", "FONTCONFIG_FILE",
                 "DYLD_INSERT_LIBRARIES", "PYTHONPATH", "LANG"):
        monkeypatch.setenv(name, "/attacker")
    record = sandbox.discover()
    args = ["-layout", "a file; $(touch nope).pdf", "-"]
    argv, env = sandbox.contained_command(record, "pdftotext", args, *dirs)
    assert argv == ["/usr/bin/sandbox-exec", "-p", sandbox.profile_text(record, *dirs),
                    record.paths["pdftotext"], *args]
    assert env == {"PATH": "/usr/bin:/bin", "HOME": dirs[1], "TMPDIR": dirs[1],
                   "XDG_CACHE_HOME": dirs[1], "LANG": "en_US.UTF-8",
                   **({"FONTCONFIG_FILE": str(config)} if fontconfig else {})}
    assert args == ["-layout", "a file; $(touch nope).pdf", "-"]


@pytest.mark.parametrize("tool,args", [("cat", []), ("/bin/cat", []), (None, []),
    ("pdfinfo", ["bad\0arg"]), ("pdfinfo", [b"bytes"]), ("pdfinfo", [1]),
    ("pdfinfo", [None]), ("pdfinfo", "-v")])
def test_command_refusals(brew, dirs, tool, args):
    with pytest.raises(ValueError):
        sandbox.contained_command(sandbox.discover(), tool, args, *dirs)


@pytest.mark.parametrize("outcome", [0, 1, OSError("denied"),
    subprocess.TimeoutExpired("probe", 10), RuntimeError("broken")])
def test_available_probe_and_cleanup(brew, tmp_path, monkeypatch, outcome):
    binary = tmp_path / "sandbox-exec"
    binary.touch(mode=0o700)
    monkeypatch.setattr(sandbox, "SANDBOX_EXEC", str(binary))
    seen = []

    def run(argv, **kwargs):
        env = kwargs["env"]
        output = Path(env["HOME"])
        root = output.parent
        seen.append(root)
        inputs = [p for p in root.iterdir() if p != output]
        assert len(inputs) == 1 and inputs[0].is_dir() and output.is_dir()
        assert str(root) == os.path.realpath(root)
        expected = sandbox.contained_command(sandbox.discover(), "pdfinfo", ["-v"],
                                             str(inputs[0]), str(output))
        assert (argv, env) == expected
        assert kwargs == dict(env=env, stdin=subprocess.DEVNULL, capture_output=True, timeout=10)
        if isinstance(outcome, Exception):
            raise outcome
        return subprocess.CompletedProcess(argv, outcome)

    probe = Mock(side_effect=run)
    monkeypatch.setattr(sandbox.subprocess, "run", probe)
    assert sandbox.available() is (outcome == 0)
    probe.assert_called_once()
    assert seen and all(not path.exists() for path in seen)


def test_available_missing_or_setup_failure(brew, tmp_path, monkeypatch):
    binary = tmp_path / "absent"
    monkeypatch.setattr(sandbox, "SANDBOX_EXEC", str(binary))
    probe = Mock()
    monkeypatch.setattr(sandbox.subprocess, "run", probe)
    assert sandbox.available() is False
    binary.touch(mode=0o700)
    with monkeypatch.context() as patch:
        patch.setattr(sandbox.tempfile, "TemporaryDirectory", Mock(side_effect=OSError("denied")))
        assert sandbox.available() is False
    (brew[0] / "pdfinfo").unlink()
    assert sandbox.available() is False
    monkeypatch.setattr(sandbox, "discover", Mock(side_effect=RuntimeError("broken")))
    assert sandbox.available() is False
    probe.assert_not_called()


def test_available_partial_setup_cleanup(brew, tmp_path, monkeypatch):
    binary = tmp_path / "sandbox-exec"
    binary.touch(mode=0o700)
    monkeypatch.setattr(sandbox, "SANDBOX_EXEC", str(binary))
    original_mkdir, seen = os.mkdir, []

    def mkdir(path, *args, **kwargs):
        if Path(path).name == "output":
            seen.append(Path(path).parent)
            assert (seen[-1] / "input").is_dir()
            raise OSError("output creation failed")
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(sandbox.os, "mkdir", mkdir)
    probe = Mock()
    monkeypatch.setattr(sandbox.subprocess, "run", probe)
    assert sandbox.available() is False
    assert seen and all(not path.exists() for path in seen)
    probe.assert_not_called()


REAL = pytest.mark.skipif(
    os.environ.get("CORTEX_REAL_SANDBOX_TESTS") != "1"
    or not Path("/usr/bin/sandbox-exec").exists()
    or not Path("/opt/homebrew/bin/pdfinfo").exists(),
    reason="requires real sandbox-exec, Homebrew pdfinfo and CORTEX_REAL_SANDBOX_TESTS=1")


def tiny_pdf():
    stream = b"BT /F1 12 Tf 20 100 Td (Cortex fictitious PDF) Tj ET\n"
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"endstream"]
    pdf, offsets = b"%PDF-1.4\n", [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"
    start = len(pdf)
    pdf += b"xref\n0 6\n0000000000 65535 f \n"
    pdf += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:])
    return pdf + f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode()


@pytest.fixture
def real_case():
    with tempfile.TemporaryDirectory(prefix="cortex-pdf-test-") as raw:
        root = Path(raw).resolve()
        inp, out = root / "input", root / "output"
        inp.mkdir()
        out.mkdir()
        (inp / "one.pdf").write_bytes(tiny_pdf())
        (root / "outside.pdf").write_bytes(tiny_pdf())
        record = sandbox.discover()
        assert record is not None

        def run(tool, args):
            argv, env = sandbox.contained_command(record, tool, list(map(str, args)), str(inp), str(out))
            return subprocess.run(argv, env=env, stdin=subprocess.DEVNULL, capture_output=True, timeout=10)

        yield record, inp, out, run


@REAL
@pytest.mark.parametrize("tool", ["pdfinfo", "pdftotext", "pdfunite"])
def test_real_success(real_case, tool):
    _, inp, out, run = real_case
    args = {"pdfinfo": [inp / "one.pdf"], "pdftotext": ["-layout", inp / "one.pdf", "-"],
            "pdfunite": [inp / "one.pdf", inp / "one.pdf", out / "merged.pdf"]}[tool]
    result = run(tool, args)
    assert result.returncode == 0, result.stderr
    if tool == "pdfinfo":
        assert any(line.split() == [b"Pages:", b"1"] for line in result.stdout.splitlines())
    elif tool == "pdftotext":
        assert b"Cortex fictitious PDF" in result.stdout
    else:
        assert (out / "merged.pdf").read_bytes().startswith(b"%PDF-")


@REAL
@pytest.mark.parametrize("attempt", ["read", "write", "exec"])
def test_real_denials(real_case, attempt):
    record, inp, out, run = real_case
    if attempt == "read":
        result = run("pdftotext", [inp.parent / "outside.pdf", "-"])
        assert b"Cortex fictitious PDF" not in result.stdout
    elif attempt == "write":
        result = run("pdftotext", [inp / "one.pdf", inp.parent / "escaped.txt"])
        assert not (inp.parent / "escaped.txt").exists()
    else:
        argv, env = sandbox.contained_command(record, "pdfinfo", [], str(inp), str(out))
        # Keep the complete proven profile; only substitute the forbidden executable.
        result = subprocess.run([*argv[:3], "/bin/cat", str(inp / "one.pdf")], env=env,
                                stdin=subprocess.DEVNULL, capture_output=True, timeout=10)
        assert not result.stdout
    assert result.returncode > 0, result.stderr  # Signals are never an acceptable denial.


@REAL
def test_real_available():
    assert sandbox.available() is True
