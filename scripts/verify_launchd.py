#!/usr/bin/env python3
"""
Launchd drift check (M11-A2, s66).

The repo's `launchd/` is the documented source of truth for the LaunchAgents
(PROJECT_MAP §7), but nothing ever compared it against what the machine is
actually running. That bit us: the LIVE backup plist gained
`CORTEX_ICLOUD_SYNC=1` by hand and the repo copy didn't — a reinstall from
the repo (e.g. the Mac Studio migration, runbook §0) would have silently
dropped the encrypted offsite backup.

`make verify-launchd` diffs every `launchd/com.cortex.*.plist` against its
counterpart in `~/Library/LaunchAgents/`:

  - OK            — parsed content identical (key order / comments ignored)
  - DRIFT         — same plist, different content; the differing key paths
                    are printed (this is the failure mode that matters)
  - not installed — repo plist with no live copy. Informational, NOT a
                    failure: dream is dormant by decision, and a fresh
                    machine starts with nothing loaded.
  - orphan        — live com.cortex.* plist with no repo source. Warned,
                    NOT a failure (but it means the repo is not the whole
                    story — consider adopting or removing it).

Exit code: 1 if any DRIFT, else 0. Deliberately NOT part of `make check` or
the git hooks — it reads machine state (~/Library), so it belongs to the
operational world (`make doctor` family), run manually and from runbook §0.

Test hooks: the two roots are injectable via CORTEX_LAUNCHD_REPO_DIR and
CORTEX_LAUNCHD_LIVE_DIR so the suite can exercise this against fixtures.
"""
from __future__ import annotations

import os
import plistlib
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_ROOT = Path(__file__).resolve().parent.parent

_PREFIX = "com.cortex."


def _repo_dir() -> Path:
    return Path(os.environ.get("CORTEX_LAUNCHD_REPO_DIR", _ROOT / "launchd"))


def _live_dir() -> Path:
    return Path(
        os.environ.get(
            "CORTEX_LAUNCHD_LIVE_DIR",
            Path.home() / "Library" / "LaunchAgents",
        )
    )


def _load(path: Path) -> Dict[str, Any]:
    # plistlib reads both XML and binary plists; XML comments are ignored
    # by the parser, so a hand-`plutil`-rewritten live plist compares equal
    # to the commented repo source when the *content* is the same.
    try:
        with path.open("rb") as fh:
            return plistlib.load(fh)
    except Exception:
        # CoreFoundation is laxer than expat: e.g. `--` inside an XML
        # comment (the heartbeat plist documents its own `--notify` flag)
        # is illegal XML but fine for launchd. Let plutil re-emit clean
        # XML; if even plutil refuses, the file really is broken.
        out = subprocess.run(
            ["plutil", "-convert", "xml1", "-o", "-", str(path)],
            capture_output=True,
            check=True,
        ).stdout
        return plistlib.loads(out)


def _diff_paths(repo: Any, live: Any, prefix: str = "") -> List[str]:
    """Flat list of dotted key paths where the two parsed plists differ."""
    if isinstance(repo, dict) and isinstance(live, dict):
        lines: List[str] = []
        for key in sorted(set(repo) | set(live)):
            sub = f"{prefix}.{key}" if prefix else str(key)
            if key not in repo:
                lines.append(f"{sub}: repo=<missing> live={live[key]!r}")
            elif key not in live:
                lines.append(f"{sub}: repo={repo[key]!r} live=<missing>")
            else:
                lines.extend(_diff_paths(repo[key], live[key], sub))
        return lines
    if repo != live:
        return [f"{prefix or '<root>'}: repo={repo!r} live={live!r}"]
    return []


def check(repo_dir: Path, live_dir: Path) -> Tuple[List[str], bool]:
    """Return (report lines, drift?) for every com.cortex.* plist."""
    lines: List[str] = []
    drift = False

    repo_plists = sorted(repo_dir.glob(f"{_PREFIX}*.plist"))
    if not repo_plists:
        return [f"no {_PREFIX}*.plist found in {repo_dir}"], False

    for repo_path in repo_plists:
        live_path = live_dir / repo_path.name
        if not live_path.exists():
            lines.append(f"·  {repo_path.name} — not installed (no live copy)")
            continue
        try:
            repo_data, live_data = _load(repo_path), _load(live_path)
        except Exception as exc:  # unparseable file IS drift-worthy noise
            drift = True
            lines.append(f"✗  {repo_path.name} — unreadable: {exc}")
            continue
        diffs = _diff_paths(repo_data, live_data)
        if diffs:
            drift = True
            lines.append(f"✗  {repo_path.name} — DRIFT:")
            lines.extend(f"     {d}" for d in diffs)
        else:
            lines.append(f"✓  {repo_path.name} — ok")

    repo_names = {p.name for p in repo_plists}
    for live_path in sorted(live_dir.glob(f"{_PREFIX}*.plist")):
        if live_path.name not in repo_names:
            lines.append(
                f"⚠  {live_path.name} — orphan (live but no repo source)"
            )

    return lines, drift


def main() -> int:
    lines, drift = check(_repo_dir(), _live_dir())
    print("launchd drift check (repo launchd/ vs ~/Library/LaunchAgents)")
    for line in lines:
        print(line)
    if drift:
        print(
            "\nDRIFT: repo and live disagree. Reconcile on purpose — either "
            "edit launchd/ (repo wins) or reinstall the live copy "
            "(cp + launchctl unload/load)."
        )
        return 1
    print("\nno drift.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
