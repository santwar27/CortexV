"""Collecting API tests must not open the real HOME's effect record."""
import os
from pathlib import Path
import subprocess
import sys


def test_collection_does_not_create_effects_db_in_home(tmp_path):
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)
    env.pop("CORTEX_EFFECTS_DB", None)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "--co", "-q",
         "tests/test_api_error_hygiene.py"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert not (tmp_path / ".cortex" / "effects.db").exists()
