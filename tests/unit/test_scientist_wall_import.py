"""
test_scientist_wall_import.py — the import-level wall (R2 / INVARIANT 1).

No module under `agents/scientist/` may import a magnitude-bearing Auditor module. The seam in
`shared/handoff/` is the sole exception and is OUT of scope for this scan. This is a STATIC
(AST) scan, not a runtime import, so it catches a forbidden import even on a code path that
never executes.

A wall test that has never failed asserts nothing. `test_wall_scanner_catches_violation` is the
durable red proof: it runs the SAME scanner against a fresh fixture file containing a known
violation and asserts the scanner flags it. (Additionally verified by hand on 2026-07-27:
dropping a probe file that imports `agents.auditor.checks.economic` under `agents/scientist/`
turned `test_wall_import_invariant` RED; deleting the probe turned it green. See the build
report for the captured failing output.)
"""

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

SCIENTIST_ROOT = REPO_ROOT / "agents" / "scientist"

# The magnitude-bearing Auditor modules the wall forbids (R2). `agents.auditor.schemas.toggle`
# is NOT here — it carries toggle identities only (no magnitudes) and is a permitted, required
# reuse (deliverable A).
FORBIDDEN_MODULES = frozenset(
    {
        "agents.auditor.schemas.audit_report",
        "agents.auditor.schemas.audit_core",
        "agents.auditor.schemas.decomposition",
        "agents.auditor.checks.economic",
        "agents.auditor.checks.inference",
        "agents.auditor.checks.bayes",
    }
)


def _imports_of(path: Path) -> set[str]:
    """Every absolute module name a file imports, resolving `from . import x` and
    `from ..pkg import y` against the file's own package path (relative to REPO_ROOT). Both the
    `from a.b.c import X` form (base = a.b.c) and the `from a.b import c` form
    (base.c = a.b.c) are captured, so a forbidden module cannot hide behind either spelling."""
    tree = ast.parse(path.read_text(), filename=str(path))
    try:
        rel = path.resolve().relative_to(REPO_ROOT)
        pkg_parts = list(rel.with_suffix("").parts)[:-1]  # the file's package path
    except ValueError:
        pkg_parts = []  # file outside the repo (a tmp fixture with absolute imports only)

    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = node.module or ""
            else:
                # Resolve a relative import against the file's package. level=1 -> same package.
                keep = len(pkg_parts) - (node.level - 1)
                up = pkg_parts[:keep] if keep > 0 else []
                base = ".".join(up + ([node.module] if node.module else []))
            if base:
                found.add(base)
            for alias in node.names:  # `from pkg import submodule`
                found.add(f"{base}.{alias.name}" if base else alias.name)
    return found


def scan_for_violations(root: Path) -> list[tuple[str, str]]:
    """Return (relative_file, forbidden_module) pairs for every forbidden import under `root`.
    Accepts an arbitrary root so the same scanner proves both the real invariant and the
    durable red fixture (deliverable D)."""
    violations: list[tuple[str, str]] = []
    for py in sorted(root.rglob("*.py")):
        for mod in _imports_of(py):
            if mod in FORBIDDEN_MODULES:
                try:
                    label = str(py.relative_to(REPO_ROOT))
                except ValueError:
                    label = str(py)
                violations.append((label, mod))
    return violations


# --- the real invariant ------------------------------------------------------------------

def test_wall_import_invariant():
    """No file under agents/scientist/ imports a magnitude-bearing Auditor module."""
    violations = scan_for_violations(SCIENTIST_ROOT)
    assert violations == [], (
        "WALL BREACH — agents/scientist/ must not import magnitude-bearing Auditor schemas "
        f"(R2 / INVARIANT 1). Offending (file, module): {violations}"
    )


# --- the durable red proof: the scanner MUST catch a real violation ----------------------

_VIOLATING_ABSOLUTE = (
    '"""A deliberately-violating fixture — must be flagged by the wall scanner."""\n'
    "from agents.auditor.checks.economic import EconomicResult  # forbidden magnitude module\n"
    "_ = EconomicResult\n"
)

_VIOLATING_RELATIVE = (
    '"""A deliberately-violating fixture using the sneaky relative spelling."""\n'
    "from ...auditor.schemas.decomposition import SaturatedBasis  # forbidden, via relative\n"
    "_ = SaturatedBasis\n"
)


def test_wall_scanner_catches_violation(tmp_path):
    """Prove the scanner can go red: a file importing a forbidden module (absolute spelling) is
    flagged. Without this, `test_wall_import_invariant` passing would assert nothing."""
    fixture_dir = tmp_path / "agents" / "scientist"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "bad_module.py").write_text(_VIOLATING_ABSOLUTE)

    violations = scan_for_violations(fixture_dir)
    assert violations, "scanner failed to flag a known forbidden import — the wall is toothless"
    assert any(m == "agents.auditor.checks.economic" for _, m in violations)


def test_wall_scanner_catches_relative_spelling():
    """The `from ...auditor.schemas.decomposition import X` spelling (level=3 relative from an
    agents/scientist/schemas file) must also be resolved and flagged — relative resolution is
    tied to REPO_ROOT, so place the fixture under the real tree, scan, then remove it."""
    fixture = SCIENTIST_ROOT / "schemas" / "_wall_relative_probe.py"
    fixture.write_text(_VIOLATING_RELATIVE)
    try:
        violations = scan_for_violations(SCIENTIST_ROOT)
        assert any(
            m == "agents.auditor.schemas.decomposition" for _, m in violations
        ), f"relative-import wall breach not detected; got {violations}"
    finally:
        fixture.unlink()

    # And with the probe gone, the invariant is clean again.
    assert scan_for_violations(SCIENTIST_ROOT) == []
