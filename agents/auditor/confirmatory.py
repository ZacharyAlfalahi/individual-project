"""
confirmatory.py — the pre-registered confirmatory coordinate set (§7.2, §7.3.1).

The confirmatory family is the five first-order DOE effects plus the THREE
end-to-end-validated interaction pairs (§10.2) — nothing else. The seven other
pairs and all higher orders are computed and reported (§5.2) but are exploratory,
never confirmatory (D-A28).

On a reduced lattice a coordinate exists only when ALL its constituent toggles are
runnable — the selection matrix A_s of §7.3.1. `confirmatory_coordinates` applies
that selection, so the family (and the Bayesian θ vector) has variable dimension.
"""

from __future__ import annotations

from typing import Sequence

from .schemas.toggle import TOGGLE_IDS, ToggleId

# The three pre-registered, end-to-end-validated interaction pairs (§7.2 / §10.2),
# spanning panel-layer, construction-layer and cross-layer.
CONFIRMATORY_INTERACTIONS: tuple[frozenset, ...] = (
    frozenset({"meas_err", "stale_price"}),
    frozenset({"lib_gap", "lab_trim"}),
    frozenset({"survivorship", "meas_err"}),
)


def confirmatory_coordinates(
    runnable_toggles: Sequence[ToggleId],
) -> list[frozenset]:
    """The confirmatory coordinates present on this (possibly reduced) lattice:
    the runnable first-order effects, then the interaction pairs whose BOTH
    toggles are runnable (A_s, §7.3.1). Canonical order: singletons first (in
    TOGGLE_IDS order), then the registered pairs in their fixed order."""
    runnable = set(runnable_toggles)
    coords: list[frozenset] = [frozenset({t}) for t in TOGGLE_IDS if t in runnable]
    for pair in CONFIRMATORY_INTERACTIONS:
        if pair <= runnable:
            coords.append(pair)
    return coords
