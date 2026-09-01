"""one-shot holdout firewall (spec §1.4): no lattice/attribution inference on the holdout.

The evaluator reaches statistics ONLY through the sanctioned ``shared.stats`` surface (which
itself wraps the pre-registered Auditor primitives, "wrap don't move"); no one-shot holdout module reaches
into ``agents.auditor`` directly, and the evaluator names none of the 2^k attribution functions.
This pins the property at the source level — the meaningful firewall, since the shared wrap's
transitive import of the Auditor's block-bootstrap module is the sanctioned reuse, not lattice
inference that one-shot holdout performs.
"""

from __future__ import annotations

from pathlib import Path

ONESHOT_HOLDOUT_DIR = Path(__file__).resolve().parents[2] / "agents" / "scientist" / "experimentalist" / "oneshot_holdout"

# The attribution machinery the evaluator must never run on the holdout.
LATTICE_NAMES = (
    "doe_effects", "harsanyi_dividends", "walsh_coefficients", "shapley_values",
    "cell_runner", "ipca_differential", "lattice_types",
)


def _oneshot_holdout_sources():
    return [p for p in ONESHOT_HOLDOUT_DIR.glob("*.py")]


def test_no_oneshot_holdout_module_imports_agents_auditor_directly():
    offenders = []
    for py in _oneshot_holdout_sources():
        for line in py.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")) and "agents.auditor" in stripped:
                offenders.append(f"{py.name}: {stripped}")
    assert not offenders, f"one-shot holdout must reach stats via shared.stats, not agents.auditor: {offenders}"


def test_evaluator_names_no_lattice_attribution_function():
    src = (ONESHOT_HOLDOUT_DIR / "stage2_evaluate.py").read_text()
    hits = [name for name in LATTICE_NAMES if name in src]
    assert not hits, f"evaluator references lattice/attribution machinery: {hits}"


def test_evaluator_goes_through_shared_stats():
    # Positive assertion: the evaluator DOES use the sanctioned shared surface.
    src = (ONESHOT_HOLDOUT_DIR / "stage2_evaluate.py").read_text()
    assert "from shared.stats import" in src
    assert "holdout_inference" in src
