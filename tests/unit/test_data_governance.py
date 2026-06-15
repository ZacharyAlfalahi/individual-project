"""Guard test: licensed/raw data (FISD + TRACE) must never be tracked by git.

Mirrors the runtime governance precedent in tests/conftest.py (block_holdout_reads),
but for the commit boundary. Shells out to the SAME script the git hooks and CI use,
so the forbidden-path patterns are defined exactly once
(see scripts/check_no_licensed_data.sh). A red bar here means licensed data has
entered the index.
"""
import pathlib
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_no_licensed_data.sh"


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_no_licensed_data_tracked():
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "Licensed/raw data (FISD/TRACE) is tracked by git:\n"
        + result.stdout
        + result.stderr
    )
