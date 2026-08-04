"""
decomposition.py — the frozen result types for the three lattice readings (§5).

All three are exact linear functionals of the same 2^k common-support metric
vector; nothing is fitted (§5.2). Each result carries its basis name explicitly —
the generic name `factorial_coefficients` is banned (§5.1), every coefficient
carries its basis.

Subsets T ⊆ N (toggle sets) are the natural index of the saturated bases. They
are stored as `frozenset[ToggleId]` internally and rendered to a stable string
label (canonical-order, "×"-joined; "∅" for the empty set) for serialisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .toggle import TOGGLE_IDS, ToggleId


def subset_label(subset: frozenset[ToggleId]) -> str:
    """A stable, canonical-order string label for a toggle subset."""
    if not subset:
        return "∅"
    return "×".join(t for t in TOGGLE_IDS if t in subset)


def label_to_subset(label: str) -> frozenset[ToggleId]:
    """Inverse of `subset_label`: parse a "×"-joined (or "∅") label back to a
    frozenset of ToggleIds. Round-trips every label `subset_label` emits."""
    if label == "∅":
        return frozenset()
    parts = label.split("×")
    unknown = [p for p in parts if p not in TOGGLE_IDS]
    if unknown:
        raise ValueError(f"label_to_subset: unknown toggle(s) {unknown} in {label!r}")
    return frozenset(parts)  # type: ignore[arg-type]


def _subset_map_to_dict(m: Mapping[frozenset, float]) -> dict:
    """Render a subset-keyed map to a label-keyed dict (YAML/JSON-safe)."""
    return {subset_label(frozenset(k)): float(v) for k, v in m.items()}


@dataclass(frozen=True)
class CornerMarginals:
    """Effects at named vertices (§5.1a) — the objects DRR report, the external
    triangulation artefact.

      add_in[i]     = Y({i}) - Y(∅)          (add-one-in at the bottom vertex)
      leave_out[i]  = Y(N)   - Y(N \\ {i})   (leave-one-out at the top vertex)
      bracket[i]    = add_in[i] - leave_out[i]  (the interaction bracket)
    """

    add_in: Mapping[ToggleId, float]
    leave_out: Mapping[ToggleId, float]
    bracket: Mapping[ToggleId, float]

    def to_dict(self) -> dict:
        return {
            "add_in": {k: float(v) for k, v in self.add_in.items()},
            "leave_out": {k: float(v) for k, v in self.leave_out.items()},
            "interaction_bracket": {k: float(v) for k, v in self.bracket.items()},
        }


@dataclass(frozen=True)
class SaturatedBasis:
    """The three saturated representations over every subset T ⊆ N (§5.1b,c).

      harsanyi[T]  = Möbius/Harsanyi dividend h(T)   (0/1 coding; exact, NOT orthogonal)
      walsh[T]     = Walsh coefficient γ_T           (±1 coding; orthogonal)
      doe[T]       = DOE effect E_T = 2·γ_T          (average ON-minus-OFF difference; REPORTED)
    """

    harsanyi: Mapping[frozenset, float]
    walsh: Mapping[frozenset, float]
    doe: Mapping[frozenset, float]

    def to_dict(self) -> dict:
        return {
            "harsanyi_dividends": _subset_map_to_dict(self.harsanyi),
            "walsh_coefficients": _subset_map_to_dict(self.walsh),
            "doe_effects": _subset_map_to_dict(self.doe),
        }


@dataclass(frozen=True)
class ShapleyResult:
    """Additive attribution via Shapley (§5.3). `values` are φ_i in metric units
    (always well defined); `shares` are φ_i / Δ_correction, each None when the
    percentage-denominator guard trips (|Δ| below the pre-registered minimum,
    §5.3). `efficiency_residual` = Σφ_i - (Y(N) - Y(∅)); asserted to tolerance."""

    values: Mapping[ToggleId, float]
    shares: Mapping[ToggleId, float | None]
    endpoint_gap: float                 # Δ_correction = Y(N) - Y(∅)
    efficiency_residual: float
    denominator_ok: bool                # False => shares are all None (guard tripped)

    def to_dict(self) -> dict:
        return {
            "shapley_values": {k: float(v) for k, v in self.values.items()},
            "shapley_share_of_registered_endpoint_gap": {
                k: (None if v is None else float(v)) for k, v in self.shares.items()
            },
            "registered_endpoint_gap": float(self.endpoint_gap),
            "efficiency_residual": float(self.efficiency_residual),
            "shares_reported": self.denominator_ok,
        }


@dataclass(frozen=True)
class BiasClassPartition:
    """The endpoint gap split by `bias_class` (ADR bias_class_taxonomy §5.2).

    The substantive headline is THREE components, never one "total bias". They are
    computed from the Harsanyi dividends over the coalitions present in the
    (possibly reduced) runnable lattice:

      methodological_construction_component = Σ h(S), non-empty S ⊆ construction toggles
      data_quality_component                = Σ h(S), non-empty S ⊆ data-quality toggles
      cross_class_modulation                = Σ h(S), S mixing both classes (|S|>1)

    These three buckets partition every non-empty coalition, so
    (method + data_quality + cross_class) == endpoint_gap == Y(N)-Y(∅) exactly
    (ADR §9.2; `reconciliation_residual` exposes the float error).

    A component is **None — NOT 0** — when its bucket of present coalitions is empty
    (e.g. `meas_err` not runnable => the data-quality and cross-class components have
    no estimand). A zero in a component column is an OBSERVATION; structural absence
    is not (ADR §5.4). Cross-class modulation is a RESULT in its own right — how data
    quality modulates a construction bias — belonging to neither headline (ADR §5.3).
    """

    methodological_construction_component: float | None
    data_quality_component: float | None
    cross_class_modulation: float | None
    endpoint_gap: float
    reconciliation_residual: float
    data_quality_present: bool          # any data-quality toggle in the lattice (today: meas_err)

    def to_dict(self) -> dict:
        def _f(v: float | None) -> float | None:
            return None if v is None else float(v)

        return {
            "methodological_construction_component": _f(
                self.methodological_construction_component
            ),
            "data_quality_component": _f(self.data_quality_component),
            "cross_class_modulation": _f(self.cross_class_modulation),
            "endpoint_gap": float(self.endpoint_gap),
            "reconciliation_residual": float(self.reconciliation_residual),
            "data_quality_present": bool(self.data_quality_present),
        }
