"""ADR bias_class_taxonomy — the Harsanyi 3-component partition (§5.2, §5.3, §9.2)
and the dummy_reason derivation (§5.4).

The partition splits the endpoint gap Y(N)-Y(∅) into a methodological-construction
component, a data-quality component (meas_err), and a cross-class modulation term.
The three buckets partition every non-empty coalition, so they reconcile exactly to
the endpoint gap (§9.2). A component is None — never 0 — when its bucket of present
coalitions is empty (§5.4 / CF-2).
"""

from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path

import pytest

from agents.auditor.checks.algebra import bias_class_partition, harsanyi_dividends
from agents.auditor.checks.shapley import shapley_result
from agents.auditor.schemas.decomposition import label_to_subset
from agents.auditor.schemas.toggle import (
    TOGGLE_IDS,
    construction_toggles,
    data_quality_toggles,
)

_FIXture_DIR = Path("results/auditor/drf")
_ANCHORS = ("drf", "mom6", "str")


def _all_subsets(toggles):
    out = []
    for r in range(len(toggles) + 1):
        out.extend(frozenset(c) for c in combinations(toggles, r))
    return out


# --------------------------------------------------------------------------
# Bucket membership — exhaustive & mutually exclusive over all coalitions
# --------------------------------------------------------------------------

def test_buckets_partition_every_nonempty_coalition():
    toggles = list(TOGGLE_IDS)
    m = set(construction_toggles())
    d = set(data_quality_toggles())
    method, dq, cross = set(), set(), set()
    for S in _all_subsets(toggles):
        if not S:
            continue
        tm, td = bool(S & m), bool(S & d)
        if tm and td:
            cross.add(S)
        elif td:
            dq.add(S)
        else:
            method.add(S)
    # 15 / 1 / 15 = 31 non-empty coalitions (ADR §5.2).
    assert (len(method), len(dq), len(cross)) == (15, 1, 15)
    assert method.isdisjoint(dq) and method.isdisjoint(cross) and dq.isdisjoint(cross)
    assert method | dq | cross == {S for S in _all_subsets(toggles) if S}
    assert dq == {frozenset({"meas_err"})}


# --------------------------------------------------------------------------
# Reconciliation to the endpoint gap (§9.2) — synthetic + committed fixtures
# --------------------------------------------------------------------------

def _reconciles(part, gap):
    total = (
        (part.methodological_construction_component or 0.0)
        + (part.data_quality_component or 0.0)
        + (part.cross_class_modulation or 0.0)
    )
    # str's magnitudes are ~770, so use a relative tolerance (CF-4).
    return math.isclose(total, gap, rel_tol=1e-9, abs_tol=1e-12)


def test_partition_reconciles_on_synthetic_lattice():
    toggles = list(TOGGLE_IDS)
    Y = {s: (1.3 * len(s) - 0.4 * sum(hash(t) % 9 for t in s)) for s in _all_subsets(toggles)}
    h = harsanyi_dividends(Y, toggles)
    part = bias_class_partition(h, toggles)
    gap = Y[frozenset(toggles)] - Y[frozenset()]
    assert _reconciles(part, gap)
    assert math.isclose(part.endpoint_gap, gap, rel_tol=1e-9, abs_tol=1e-12)
    assert abs(part.reconciliation_residual) < 1e-9
    # Also reconciles to the independently-computed Shapley endpoint gap.
    sh = shapley_result(Y, toggles, percentage_denominator_min=0.0)
    assert math.isclose(part.endpoint_gap, sh.endpoint_gap, rel_tol=1e-9, abs_tol=1e-12)


@pytest.mark.parametrize("anchor", _ANCHORS)
def test_partition_reconciles_on_committed_fixture(anchor):
    path = _FIXture_DIR / f"{anchor}_core.json"
    if not path.exists():
        pytest.skip(f"fixture {path} not present")
    core = json.loads(path.read_text())
    hd = core["saturated_bases"]["harsanyi_dividends"]
    harsanyi = {label_to_subset(k): float(v) for k, v in hd.items()}
    runnable = tuple(core["runnable_toggles"])
    part = bias_class_partition(harsanyi, runnable)
    gap = float(core["shapley"]["registered_endpoint_gap"])
    assert _reconciles(part, gap), (anchor, part.to_dict(), gap)
    assert math.isclose(part.endpoint_gap, gap, rel_tol=1e-9, abs_tol=1e-12)
    # All five toggles runnable in these fixtures => all three components present.
    assert part.data_quality_present
    assert part.data_quality_component is not None
    assert part.methodological_construction_component is not None
    assert part.cross_class_modulation is not None


# --------------------------------------------------------------------------
# Structural absence is None, never 0 (§5.4 / CF-2), both sides
# --------------------------------------------------------------------------

def test_data_quality_and_cross_are_none_when_meas_err_absent():
    # A reduced lattice over the four construction toggles only.
    toggles = list(construction_toggles())
    Y = {s: (2.0 * len(s) + 0.5) for s in _all_subsets(toggles)}
    part = bias_class_partition(harsanyi_dividends(Y, toggles), toggles)
    assert not part.data_quality_present
    assert part.data_quality_component is None       # NOT 0.0
    assert part.cross_class_modulation is None        # NOT 0.0
    assert part.methodological_construction_component is not None
    gap = Y[frozenset(toggles)] - Y[frozenset()]
    assert math.isclose(part.methodological_construction_component, gap,
                        rel_tol=1e-9, abs_tol=1e-12)
    # to_dict preserves None (never coerces to 0.0).
    d = part.to_dict()
    assert d["data_quality_component"] is None
    assert d["cross_class_modulation"] is None


def test_method_component_is_none_on_a_meas_err_only_lattice():
    # A lattice reduced to only meas_err: the method bucket is EMPTY -> None,
    # not sum([]) == 0.0 (CF-2, the symmetric leak).
    toggles = ["meas_err"]
    Y = {frozenset(): 1.0, frozenset({"meas_err"}): 4.0}
    part = bias_class_partition(harsanyi_dividends(Y, toggles), toggles)
    assert part.methodological_construction_component is None   # NOT 0.0
    assert part.cross_class_modulation is None                  # no mixed coalition
    assert part.data_quality_component is not None
    assert math.isclose(part.data_quality_component, 3.0, abs_tol=1e-12)


def test_partition_ignores_coalitions_outside_the_reduced_lattice():
    # Defensive: pass the FULL 2^5 harsanyi but a reduced toggle set; coalitions
    # touching excluded toggles must be dropped, and the result must match a
    # partition computed from the reduced harsanyi directly.
    full = list(TOGGLE_IDS)
    Y = {s: (0.7 * len(s) - 0.2 * sum(hash(t) % 4 for t in s)) for s in _all_subsets(full)}
    full_h = harsanyi_dividends(Y, full)
    reduced = list(construction_toggles())          # drop meas_err
    part = bias_class_partition(full_h, reduced)
    assert part.data_quality_component is None
    assert part.cross_class_modulation is None
    # Σ h(S) over non-empty S ⊆ reduced == Y(reduced-full) - Y(∅).
    gap_reduced = Y[frozenset(reduced)] - Y[frozenset()]
    assert math.isclose(part.methodological_construction_component, gap_reduced,
                        rel_tol=1e-9, abs_tol=1e-12)
