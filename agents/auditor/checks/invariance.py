"""
invariance.py — the no-op invariance gate (§3.5).

Where a toggle is declared `expect_no_op: true`, ALL cells still run, and the
prediction is verified by execution — never asserted pre-flight (a pre-flight
no-op flag would prune exactly the cells that could falsify it, §3.3).

The claim proved is *different config, identical behaviour*, and BOTH halves are
required (D-A6):

    assert  RunConfig hashes       DIFFER      ← the intervention WAS applied
    assert  return-series hashes   identical
    assert  membership hashes      identical   ← and was behaviourally inert
    assert  metric hashes          identical

The first assertion is the correction v1.3 got wrong: two runs with different
toggle settings MUST have different run hashes — identical run hashes would prove
the hash is not encoding the configuration (a hashing bug), not a no-op.

The test is run over EVERY parallel edge in the toggle's direction (the toggle
flipped with the other runnable toggles held at each of their 2^{k-1}
combinations), so `is_no_op` means "inert everywhere in the lattice", not merely
at one vertex. `is_no_op: true` earns the toggle its place as a known-null fixture
(§10.3-10.4); any output-hash difference => `is_no_op: false`, report the measured
effect and investigate (§3.5).

Membership/weight hashing is limited to the observable envelope (the monthly
return series, the per-month bond count, and the metrics): the frozen engine does
not expose per-bond holdings, so this is the strongest invariance the public
surface supports, and the n_bonds series stands in as the membership proxy.
"""

from __future__ import annotations

from itertools import combinations
from typing import Mapping, Sequence

from ..schemas.audit_core import InvarianceResult
from ..schemas.lattice_types import LatticeResult
from ..schemas.toggle import ToggleFacts, ToggleId


def _other_subsets(others: Sequence[ToggleId]):
    for r in range(len(others) + 1):
        for c in combinations(others, r):
            yield frozenset(c)


def invariance_test(lattice: LatticeResult, toggle_id: ToggleId) -> InvarianceResult:
    """Verify the §3.5 invariance for one runnable toggle over every parallel edge.

    Raises ValueError if the toggle is not runnable (a held toggle has no ON/OFF
    pair in the lattice, so its inertness is not testable here)."""
    if toggle_id not in lattice.runnable_toggles:
        raise ValueError(
            f"toggle {toggle_id!r} is not runnable in this lattice "
            f"({lattice.runnable_toggles}); its invariance is not testable"
        )
    others = tuple(t for t in lattice.runnable_toggles if t != toggle_id)

    all_config_differ = True
    all_returns_same = True
    all_n_bonds_same = True
    all_metrics_same = True
    membership_available = True
    first_violation = ""

    for S in _other_subsets(others):
        off = lattice.cell_for(S)
        on = lattice.cell_for(S | {toggle_id})
        config_differs = off.run_config_hash != on.run_config_hash
        returns_same = off.return_hash == on.return_hash
        n_bonds_same = off.n_bonds_hash == on.n_bonds_hash
        metrics_same = off.metric_hash == on.metric_hash

        all_config_differ = all_config_differ and config_differs
        all_returns_same = all_returns_same and returns_same
        all_n_bonds_same = all_n_bonds_same and n_bonds_same
        all_metrics_same = all_metrics_same and metrics_same

        # The membership proxy is unavailable when a cell has returns but no
        # per-month bond count (e.g. a multi-leg envelope that dropped n_bonds):
        # then n_bonds_same is trivially True (empty==empty) and proves nothing.
        for cell in (off, on):
            if len(cell.returns) > 0 and len(cell.n_bonds) == 0:
                membership_available = False

        if not first_violation:
            if not config_differs:
                first_violation = (
                    f"config hashes identical at others={sorted(S)} — the hash is "
                    "not encoding the toggle (a hashing bug, not a no-op)"
                )
            elif not returns_same:
                first_violation = f"return series differ at others={sorted(S)}"
            elif not n_bonds_same:
                first_violation = f"bond counts differ at others={sorted(S)}"
            elif not metrics_same:
                first_violation = f"metrics differ at others={sorted(S)}"

    # membership_verified requires the proxy to be present AND identical.
    membership_verified = membership_available and all_n_bonds_same
    is_no_op = (
        all_config_differ and all_returns_same and all_metrics_same and membership_verified
    )
    if is_no_op:
        note = "verified inert across all parallel edges"
    elif not membership_available and all_returns_same and all_metrics_same and all_config_differ:
        note = (
            "NOT certified a no-op: membership proxy (n_bonds) unavailable, so an "
            "inert membership cannot be verified (conservative)"
        )
    else:
        note = f"NOT a no-op: {first_violation}"
    return InvarianceResult(
        toggle_id=toggle_id,
        config_hashes_differ=all_config_differ,
        returns_identical=all_returns_same,
        n_bonds_identical=all_n_bonds_same,
        metrics_identical=all_metrics_same,
        is_no_op=is_no_op,
        membership_verified=membership_verified,
        note=note,
    )


def run_invariance_tests(
    lattice: LatticeResult, facts: Mapping[ToggleId, ToggleFacts]
) -> tuple[InvarianceResult, ...]:
    """Run the invariance test for every runnable toggle declared expect_no_op.
    Toggles without the prediction are not tested here (they are not claimed
    inert); the lattice still ran them, so a later stage can inspect their
    measured effect."""
    results = []
    for tid in lattice.runnable_toggles:
        fact = facts.get(tid)
        if fact is not None and fact.expect_no_op:
            results.append(invariance_test(lattice, tid))
    return tuple(results)
