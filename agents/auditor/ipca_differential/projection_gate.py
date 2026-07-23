"""
projection_gate.py — the projection identifiability gate (spec §6.4, D-A53).

"After survivorship reintroduction, stale-price masking, missing characteristics or
cross-sectional thinning, (B_tᵀB_t)⁻¹ may not exist or may be ill-conditioned; a silently
applied pseudoinverse with an arbitrary tolerance is an undocumented researcher degree of freedom
deciding a reported number."

The gate is applied **literally on B_t = Z_t Γ** — the (N_m × K) per-month loading matrix built
from the evaluation panel's characteristics Z_t and the frozen Γβ — not merely on the Γ'WΓ solve
(B_tᵀB_t = N·Γ'WΓ, the same object, but checked on B_t directly). Per month it requires a minimum
cross-section, full column rank K, and a bounded condition number; the pseudoinverse policy is
registered (default: forbidden). A failed month is either excluded from the cell's valid months or
triggers cell refusal, capped by the maximum permitted fraction of failed months.

**Common support is formed only AFTER this per-cell validity filter** (the ``valid_periods`` this
module returns): a four-cell calendar can look complete while one cell's factor recovery is
numerically defined but statistically unidentified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from agents.auditor.thresholds import IPCAProjectionGate
from agents.quant.library.ipca_feed import IPCAFeed


@dataclass(frozen=True)
class ProjectionDiagnostics:
    """The mandatory per-cell §6.4 report block."""

    min_cross_section_n: int
    rank_fail_months: int
    max_condition_number: float
    pseudoinverse_used: bool
    pseudoinverse_tolerance: float | None
    months_removed: tuple[int, ...]        # return-month ordinals excluded from the cell
    cell_refused: bool
    n_valid_months: int

    def to_dict(self) -> dict:
        return {
            "min_cross_section_n": self.min_cross_section_n,
            "rank_fail_months": self.rank_fail_months,
            "max_condition_number": self.max_condition_number,
            "pseudoinverse_used": self.pseudoinverse_used,
            "pseudoinverse_tolerance": self.pseudoinverse_tolerance,
            "months_removed": list(self.months_removed),
            "cell_refused": self.cell_refused,
            "n_valid_months": self.n_valid_months,
        }


@dataclass(frozen=True)
class GatedProjection:
    """The gate's verdict for one cell: which return-month periods survived, and the diagnostics."""

    valid_periods: tuple[pd.Period, ...]   # return months that passed the gate (ascending)
    valid_positions: tuple[int, ...]       # their positions in feed.months
    diagnostics: ProjectionDiagnostics


def _month_condition(B: np.ndarray) -> tuple[int, float]:
    """Return (rank, condition number) of the (N_m × K) loading matrix B_t via its singular
    values. Full column rank ⟺ all K singular values > 0; cond = s_max / s_min."""
    s = np.linalg.svd(B, compute_uv=False)
    smax = float(s[0]) if s.size else 0.0
    smin = float(s[-1]) if s.size else 0.0
    # numpy's default rank tolerance: s_max * max(shape) * eps.
    tol = smax * max(B.shape) * np.finfo(float).eps
    rank = int((s > tol).sum())
    cond = smax / smin if smin > 0 else float("inf")
    return rank, cond


def gate_cell(feed: IPCAFeed, gamma_beta: np.ndarray, gate: IPCAProjectionGate) -> GatedProjection:
    """Apply the §6.4 gate to every month of a cell's evaluation feed against the frozen Γβ.

    A month passes iff N_m > min_cross_section_n, rank(B_t) == require_rank (= K), and
    cond(B_t) <= max_condition_number. Since the pseudoinverse is (by default) forbidden, a failed
    month is dropped from the valid set. If the failed fraction exceeds max_failed_month_fraction —
    or failed_month_handling is 'cell_refusal' and any month fails — the whole cell is refused."""
    K = gamma_beta.shape[1]
    if gate.require_rank != K:
        raise ValueError(
            f"projection gate require_rank={gate.require_rank} != K={K} (gamma_beta columns)"
        )

    valid_periods: list[pd.Period] = []
    valid_positions: list[int] = []
    removed: list[int] = []
    rank_fails = 0
    max_cond = 0.0
    n_below_min = 0

    for i, month in enumerate(feed.months):
        Z_t = feed.Z[i]
        n = Z_t.shape[0]
        B = Z_t @ gamma_beta                     # (N_m, K) — the literal §5.1/§6.4 object
        rank, cond = _month_condition(B)
        if np.isfinite(cond):
            max_cond = max(max_cond, cond)
        ok_n = n > gate.min_cross_section_n
        ok_rank = rank == K
        ok_cond = cond <= gate.max_condition_number
        if ok_n and ok_rank and ok_cond:
            valid_periods.append(pd.Period(ordinal=int(month), freq="M"))
            valid_positions.append(i)
        else:
            removed.append(int(month))
            if not ok_rank:
                rank_fails += 1
            if not ok_n:
                n_below_min += 1

    total = len(feed.months)
    failed_fraction = len(removed) / total if total else 0.0
    cell_refused = (
        (gate.failed_month_handling == "cell_refusal" and len(removed) > 0)
        or (failed_fraction > gate.max_failed_month_fraction)
    )

    diagnostics = ProjectionDiagnostics(
        min_cross_section_n=gate.min_cross_section_n,
        rank_fail_months=rank_fails,
        max_condition_number=max_cond,
        pseudoinverse_used=False,                # forbidden by default; never silently applied
        pseudoinverse_tolerance=gate.pseudoinverse_tolerance if gate.pseudoinverse_permitted else None,
        months_removed=tuple(removed),
        cell_refused=cell_refused,
        n_valid_months=0 if cell_refused else len(valid_positions),
    )
    if cell_refused:
        return GatedProjection(valid_periods=(), valid_positions=(), diagnostics=diagnostics)
    return GatedProjection(
        valid_periods=tuple(valid_periods),
        valid_positions=tuple(valid_positions),
        diagnostics=diagnostics,
    )
