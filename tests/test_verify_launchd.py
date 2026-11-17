"""
Tests for scripts/verify_launchd.py — the launchd drift check (M11-A2, s66).

Two halves:
  1. The drift engine (`check`) against fixture dirs — equal plists are OK
     regardless of key order/format, a differing env var is DRIFT with the
     key path named, missing live copy is informational, orphan is a warning.
  2. The reconciliation itself: the repo backup plist MUST carry
     CORTEX_ICLOUD_SYNC=1 (the encrypted offsite layer). This is the pin
     that stops the drift from coming back — if someone strips the key,
     this test is the alarm.
"""
import importlib.util
import plistlib
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "verify_launchd", _ROOT / "scripts" / "verify_launchd.py"
)
vl = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(vl)


def _write_plist(path: Path, data: dict, fmt=plistlib.FMT_XML) -> None:
    with path.open("wb") as fh:
        plistlib.dump(data, fh, fmt=fmt)


_BASE = {
    "Label": "com.cortex.fake",
    "ProgramArguments": ["/bin/bash", "x.sh"],
    "EnvironmentVariables": {"PATH": "/usr/bin"},
}


class TestCheck:
    def test_identical_content_is_ok_even_in_binary_format(self, tmp_path):
        repo, live = tmp_path / "repo", tmp_path / "live"
        repo.mkdir(), live.mkdir()
        _write_plist(repo / "com.cortex.fake.plist", _BASE)
        # live copies get rewritten by plutil/defaults — binary format and
        # different key order must still compare equal
        _write_plist(
            live / "com.cortex.fake.plist", _BASE, fmt=plistlib.FMT_BINARY
        )
        lines, drift = vl.check(repo, live)
        assert not drift
        assert any("com.cortex.fake.plist — ok" in l for l in lines)

    def test_env_var_only_in_live_is_drift_with_key_path(self, tmp_path):
        repo, live = tmp_path / "repo", tmp_path / "live"
        repo.mkdir(), live.mkdir()
        _write_plist(repo / "com.cortex.fake.plist", _BASE)
        live_data = dict(_BASE)
        live_data["EnvironmentVariables"] = {
            "PATH": "/usr/bin",
            "CORTEX_ICLOUD_SYNC": "1",
        }
        _write_plist(live / "com.cortex.fake.plist", live_data)
        lines, drift = vl.check(repo, live)
        assert drift
        joined = "\n".join(lines)
        assert "DRIFT" in joined
        # the differing key is named with its dotted path, both sides shown
        assert "EnvironmentVariables.CORTEX_ICLOUD_SYNC" in joined
        assert "repo=<missing>" in joined

    def test_scalar_drift_is_reported(self, tmp_path):
        repo, live = tmp_path / "repo", tmp_path / "live"
        repo.mkdir(), live.mkdir()
        _write_plist(repo / "com.cortex.fake.plist", _BASE)
        live_data = dict(_BASE)
        live_data["ProgramArguments"] = ["/bin/bash", "y.sh"]
        _write_plist(live / "com.cortex.fake.plist", live_data)
        lines, drift = vl.check(repo, live)
        assert drift
        assert any("ProgramArguments" in l for l in lines)

    def test_not_installed_is_informational_not_drift(self, tmp_path):
        # dream is dormant by decision; a missing live copy must not fail
        repo, live = tmp_path / "repo", tmp_path / "live"
        repo.mkdir(), live.mkdir()
        _write_plist(repo / "com.cortex.dream.plist", _BASE)
        lines, drift = vl.check(repo, live)
        assert not drift
        assert any("not installed" in l for l in lines)

    def test_orphan_live_plist_is_warned_not_drift(self, tmp_path):
        repo, live = tmp_path / "repo", tmp_path / "live"
        repo.mkdir(), live.mkdir()
        _write_plist(repo / "com.cortex.fake.plist", _BASE)
        _write_plist(live / "com.cortex.fake.plist", _BASE)
        _write_plist(live / "com.cortex.mystery.plist", _BASE)
        lines, drift = vl.check(repo, live)
        assert not drift
        assert any("orphan" in l and "mystery" in l for l in lines)

    def test_double_hyphen_in_xml_comment_parses_via_plutil(self, tmp_path):
        # `--` inside an XML comment is illegal for expat but fine for
        # CoreFoundation — the live heartbeat plist documents its own
        # `--notify` flag this way. The plutil fallback must absorb it.
        repo, live = tmp_path / "repo", tmp_path / "live"
        repo.mkdir(), live.mkdir()
        ugly = (
            b'<?xml version="1.0" encoding="UTF-8"?>\n'
            b'<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
            b'"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
            b"<!-- runs with --notify every morning -->\n"
            b'<plist version="1.0"><dict>'
            b"<key>Label</key><string>com.cortex.fake</string>"
            b"</dict></plist>\n"
        )
        (repo / "com.cortex.fake.plist").write_bytes(ugly)
        (live / "com.cortex.fake.plist").write_bytes(ugly)
        lines, drift = vl.check(repo, live)
        assert not drift
        assert any("com.cortex.fake.plist — ok" in l for l in lines)

    def test_unreadable_live_plist_is_drift(self, tmp_path):
        repo, live = tmp_path / "repo", tmp_path / "live"
        repo.mkdir(), live.mkdir()
        _write_plist(repo / "com.cortex.fake.plist", _BASE)
        (live / "com.cortex.fake.plist").write_bytes(b"not a plist")
        lines, drift = vl.check(repo, live)
        assert drift
        assert any("unreadable" in l for l in lines)

    def test_non_cortex_plists_are_ignored(self, tmp_path):
        repo, live = tmp_path / "repo", tmp_path / "live"
        repo.mkdir(), live.mkdir()
        _write_plist(repo / "com.cortex.fake.plist", _BASE)
        _write_plist(live / "com.cortex.fake.plist", _BASE)
        _write_plist(live / "com.apple.something.plist", {"Label": "x"})
        lines, drift = vl.check(repo, live)
        assert not drift
        assert not any("com.apple" in l for l in lines)


