"""The third-party import surface stays inside what CI installs.

WHY THIS EXISTS. A module-level import of a package that is present in a local working
environment but absent from `.github/workflows/tests.yml` passes every local run and fails CI at
collection — the whole suite, not one test. That happened: two modules landed importing `scipy`,
which no requirements file lists, and the push went red at collection while the local run was
green. This test closes the gap on the local side, before the push.

It also pins a stated contract. `agents/auditor/checks/stats.py`, `evaluation/harness/stats.py`,
`agents/quant/library/characteristic_sort.py` and `agents/quant/library/ipca.py` each tell the
reader, in prose, that the project carries no scipy/statsmodels and hand-roll their primitives
for that reason. Those sentences are claims about the repository; this is the check that keeps
them true.

Tripwire: reinstate `from scipy import stats` anywhere under the tracked tree and this fails.
"""
from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Packages deliberately NOT part of the dependency surface. The value is the reason, quoted
#: back in the failure so whoever trips this reads the decision, not just the rule.
FORBIDDEN_IMPORTS = {
    "scipy": ("the stats contract is deliberately small: the primitives live in "
              "agents/auditor/checks/stats.py in pure numpy/math, and CI installs no scipy"),
    "statsmodels": ("same contract — the regression/HAC machinery the project needs is "
                    "implemented against numpy, not delegated"),
}


def _tracked_python_files() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "ls-files", "-z", "*.py"],
            capture_output=True, text=True, check=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git is unavailable, so the tracked file list cannot be resolved")
    return [_REPO_ROOT / name for name in out.split("\0") if name]


def _imported_roots(tree: ast.AST) -> set[str]:
    """Top-level package name of every import in the module, wherever it sits.

    Deliberately NOT restricted to module-level imports: a lazily imported optional package is
    a legitimate pattern (the vendor SDKs use it), but scipy is not optional-by-design here —
    it is absent, so an import of it inside a function is a runtime failure rather than a
    collection failure, which is worse, not better.
    """
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_no_tracked_module_imports_a_forbidden_package():
    offenders: list[str] = []
    for path in _tracked_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for package in sorted(_imported_roots(tree) & FORBIDDEN_IMPORTS.keys()):
            rel = path.relative_to(_REPO_ROOT)
            offenders.append(f"{rel} imports {package!r} — {FORBIDDEN_IMPORTS[package]}")
    assert not offenders, (
        "these modules import a package outside the dependency surface, so a clean clone "
        "(and CI) cannot import them:\n  " + "\n  ".join(offenders))


def test_the_scanner_actually_sees_imports(tmp_path):
    """The guard above is only worth its line count if it detects what it claims to."""
    sample = tmp_path / "sample.py"
    sample.write_text(
        "import os\nfrom scipy import stats\n\ndef f():\n    import statsmodels.api as sm\n"
        "    return sm, stats\n", encoding="utf-8")
    roots = _imported_roots(ast.parse(sample.read_text(encoding="utf-8")))
    assert {"scipy", "statsmodels"} <= roots
    assert "os" in roots


def test_relative_imports_are_not_mistaken_for_packages(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text("from . import scipy\nfrom .scipy import thing\n", encoding="utf-8")
    assert not _imported_roots(ast.parse(sample.read_text(encoding="utf-8")))
