"""Disposable copies and the dormant Seatbelt boundary, including hostile probes."""
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from unittest.mock import Mock

import pytest

from src.core import code_commands, code_roots, code_sandbox as sandbox


@pytest.fixture
def folder(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CORTEX_CODE_ROOTS_FILE", raising=False)
    root = tmp_path / "cortex_sandbox"
    root.mkdir()
    assert code_roots.add(str(root))[0]
    return root


@pytest.fixture
def workspace(folder):
    copy = sandbox.prepare_workspace(str(folder))
    try:
        yield copy
    finally:
        sandbox.discard_workspace(copy)


def test_profile_text(tmp_path, monkeypatch):
    workspace = tmp_path / 'quoted"\\workspace'
    workspace.mkdir()
    text = sandbox.profile_text(str(workspace))
    quoted = str(workspace).replace('\\', '\\\\').replace('"', '\\"')
    assert text.startswith('(version 1)\n(deny default)\n')
    assert f'(allow file-write* (subpath "{quoted}") (literal "/dev/null"))' in text
    assert 'network' not in text and '(allow default)' not in text
    assert 'process-fork' not in text
    link = tmp_path / 'link'
    link.symlink_to(workspace, target_is_directory=True)
    for path in ('relative', str(link), str(tmp_path / 'missing')):
        with pytest.raises(ValueError):
            sandbox.profile_text(path)
    for char in ('\n', '\x1f', '\x7f'):
        bad = tmp_path / ('bad' + char)
        bad.mkdir()
        with pytest.raises(ValueError):
            sandbox.profile_text(str(bad))
    for attr in ('executable', 'prefix', 'base_prefix'):
        with monkeypatch.context() as patch:
            patch.setattr(sys, attr, '/bad\npath')
            with pytest.raises(ValueError):
                sandbox.profile_text(str(workspace))


def test_contained_command(folder, workspace, tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox, 'available', lambda: True)
    profile = code_commands.tests_profile(str(folder))
    argv = list(profile['argv'])
    for flag in ('-s', '-t'):
        argv[argv.index(flag) + 1] = workspace
    command = sandbox.contained_command(profile, workspace)
    assert command == dict(argv=[sandbox.SANDBOX_EXEC, '-p', sandbox.profile_text(workspace), *argv],
        cwd=workspace, env=profile['env'], timeout_s=30, max_output_bytes=65536)
    assert command['env'] is not profile['env']
    assert profile == code_commands.tests_profile(str(folder))
    for bad in ({**profile, 'argv': profile['argv'] + ['extra']},
                {**profile, 'argv': ['/bin/sh', *profile['argv'][1:]]},
                code_commands.tests_profile(str(tmp_path))):
        with pytest.raises(ValueError, match=code_commands.validate(bad)[1]):
            sandbox.contained_command(bad, workspace)
    with pytest.raises(ValueError):
        sandbox.contained_command(profile, str(tmp_path))
    monkeypatch.setattr(sandbox, 'available', lambda: False)
    with pytest.raises(sandbox.SandboxUnavailable):
        sandbox.contained_command(profile, workspace)


def test_available(tmp_path, monkeypatch):
    binary = tmp_path / 'sandbox-exec'
    monkeypatch.setattr(sandbox, 'SANDBOX_EXEC', str(binary))
    probe = Mock(return_value=subprocess.CompletedProcess([], 0))
    monkeypatch.setattr(subprocess, 'run', probe)
    assert sandbox.available() is False
    binary.touch(mode=0o600)
    assert sandbox.available() is False
    probe.assert_not_called()
    binary.chmod(0o700)
    assert sandbox.available() is True
    probe.assert_called_once_with([str(binary), '-p', '(version 1)(allow default)', '/usr/bin/true'],
        stdin=subprocess.DEVNULL, capture_output=True, env={'PATH': '/usr/bin:/bin'}, timeout=5)
    for error in (OSError('denied'), subprocess.TimeoutExpired('probe', 5), RuntimeError('broken')):
        probe.side_effect = error
        assert sandbox.available() is False
    probe.side_effect = None
    probe.return_value.returncode = 1
    assert sandbox.available() is False


def test_copy_and_discard(folder, tmp_path):
    (folder / 'nested').mkdir()
    (folder / 'nested/data').write_bytes(b'\x00fictitious\xff')
    secret = tmp_path / 'secret.env'
    secret.write_bytes(b'fictitious secret')
    (folder / 'link').symlink_to(secret)
    os.link(secret, folder / 'hard')
    for name in ('__pycache__', '.git'):
        (folder / 'nested' / name).mkdir()
        (folder / 'nested' / name / 'ignored').touch()
    workspace = sandbox.prepare_workspace(str(folder))
    copy = Path(workspace)
    alias = tmp_path / 'alias'
    alias.symlink_to(copy, target_is_directory=True)
    elsewhere = tmp_path / 'cortex-code-run-elsewhere'
    elsewhere.mkdir()
    try:
        assert workspace == os.path.realpath(workspace)
        assert (copy / 'nested/data').read_bytes() == b'\x00fictitious\xff'
        assert (copy / 'link').is_symlink() and os.readlink(copy / 'link') == str(secret)
        assert (copy / 'hard').stat().st_ino != secret.stat().st_ino
        (copy / 'hard').write_bytes(b'changed')
        assert secret.read_bytes() == b'fictitious secret'
        assert not list(copy.rglob('__pycache__')) and not list(copy.rglob('.git'))
        for invalid in (tmp_path, elsewhere, alias):
            with pytest.raises(ValueError):
                sandbox.discard_workspace(str(invalid))
            assert invalid.exists()
    finally:
        sandbox.discard_workspace(workspace)
    assert not copy.exists()


def test_discard_refuses_test_session_directory():
    unrelated = Path(tempfile.mkdtemp(prefix="cortex-tests-"))
    try:
        with pytest.raises(ValueError):
            sandbox.discard_workspace(os.path.realpath(unrelated))
        assert unrelated.is_dir()
    finally:
        if unrelated.exists():
            unrelated.rmdir()


@pytest.mark.parametrize('failure', ['MAX_FILES', 'MAX_BYTES', 'copy'])
def test_copy_failure_cleans_up(folder, monkeypatch, failure):
    (folder / 'data').write_bytes(b'ab')
    before = set(Path(tempfile.gettempdir()).glob('cortex-code-run-*'))
    if failure == 'copy':
        monkeypatch.setattr(sandbox.shutil, 'copytree', Mock(side_effect=OSError('copy failed')))
    else:
        monkeypatch.setattr(sandbox, failure, 0 if failure == 'MAX_FILES' else 1)
    with pytest.raises(OSError if failure == 'copy' else ValueError):
        sandbox.prepare_workspace(str(folder))
    assert set(Path(tempfile.gettempdir()).glob('cortex-code-run-*')) == before


def test_dormant():
    core = Path(__file__).resolve().parents[1] / 'src/core'
    for name in ('staging_mixin.py', 'orchestrator.py'):
        assert 'code_sandbox' not in (core / name).read_text()


def snapshot(folder):
    return {str(p.relative_to(folder)): os.readlink(p) if p.is_symlink() else p.read_bytes()
            for p in folder.rglob('*') if p.is_symlink() or p.is_file()}


def run_project(folder, count):
    before = snapshot(folder)
    workspace = sandbox.prepare_workspace(str(folder))
    try:
        command = sandbox.contained_command(code_commands.tests_profile(str(folder)), workspace)
        result = subprocess.run(command['argv'], cwd=command['cwd'], env=command['env'],
            stdin=subprocess.DEVNULL, capture_output=True, timeout=60)
        assert result.returncode == 0, (result.stdout, result.stderr)
        assert b'OK' in result.stderr and f'Ran {count} test'.encode() in result.stderr
        assert snapshot(folder) == before and not list(folder.rglob('__pycache__'))
    finally:
        sandbox.discard_workspace(workspace)


def test_real_passing_project(folder):
    if not sandbox.available():
        pytest.skip('sandbox-exec unavailable here')
    (folder / 'calc.py').write_text('def add(a, b): return a + b\n')
    (folder / 'test_calc.py').write_text('import unittest\nfrom calc import add\n'
        'class Calc(unittest.TestCase):\n    def test_add(self): self.assertEqual(add(2, 3), 5)\n')
    run_project(folder, 1)


def test_real_hostile_project(folder, tmp_path):
    if not sandbox.available():
        pytest.skip('sandbox-exec unavailable here')
    secret = tmp_path / 'secret.env'
    secret.write_bytes(b'fictitious secret')
    (folder / 'escape').symlink_to(tmp_path, target_is_directory=True)
    markers = [tmp_path / 'outside-marker', Path.home() / 'home-marker',
               folder / 'original-marker', tmp_path / 'symlink-marker', tmp_path / 'child-marker']
    with socket.socket() as listener:
        try:
            listener.bind(('127.0.0.1', 0))
        except PermissionError:
            pytest.skip('local sockets are blocked in this sandbox')
        listener.listen(1)
        listener.setblocking(False)
        sleeper = subprocess.Popen(['/bin/sleep', '60'])
        try:
            denied = [f'Path({str(p)!r}).write_text("escape")' for p in markers[:3]] + [
                'Path("escape/symlink-marker").write_text("escape")',
                f'os.link({str(secret)!r}, "hl")', f'Path({str(secret)!r}).read_bytes()',
                f'os.listdir({str(tmp_path)!r})',
                'socket.create_connection(("1.1.1.1", 443), timeout=2)',
                f'socket.create_connection({listener.getsockname()!r}, timeout=2)',
                'socket.socket().bind(("127.0.0.1", 0))',
                'socket.socket(socket.AF_UNIX).connect("/var/run/mDNSResponder")',
                'subprocess.run(["/bin/sh", "-c", "true"])', f'os.kill({sleeper.pid}, 15)', 'os.fork()']
            source = 'import os, socket, subprocess, sys, unittest\nfrom pathlib import Path\nclass Escape(unittest.TestCase):\n'
            for i, statement in enumerate(denied):
                source += f'    def test_denied_{i}(self):\n        with self.assertRaises(PermissionError):\n            {statement}\n'
            source += ('    def test_dns(self):\n        with self.assertRaises(socket.gaierror):\n'
                       '            socket.getaddrinfo("example.com", 443)\n')
            children = [f'from pathlib import Path; Path({str(markers[-1])!r}).write_text("escape")',
                        'import socket; socket.create_connection(("1.1.1.1", 443), timeout=2)']
            for i, child in enumerate(children):
                source += (f'    def test_child_{i}(self):\n'
                    '        with self.assertRaises(PermissionError):\n'
                    f'            subprocess.run([sys.executable, "-I", "-c", {child!r}], capture_output=True, timeout=5)\n')
            source += ('    def test_write_inside(self):\n        Path("inside").write_text("ok")\n'
                       '        self.assertEqual(Path("inside").read_text(), "ok")\n')
            (folder / 'test_escape.py').write_text(source)
            run_project(folder, len(denied) + 4)
            assert sleeper.poll() is None
            with pytest.raises(BlockingIOError):
                listener.accept()
            assert not any(p.exists() for p in markers)
            assert secret.read_bytes() == b'fictitious secret'
        finally:
            sleeper.terminate()
            sleeper.wait(timeout=5)
