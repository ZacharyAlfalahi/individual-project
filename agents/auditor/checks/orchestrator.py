"""
orchestrator.py — run_audit: sequence the instrument core into an AuditCore.

The deterministic analytical path (§2): pre-flight -> lattice -> common support ->
the three lattice readings -> invariance. No language model appears anywhere here
(§11) — that is a structural property of the design, not a policy.

Pre-registration constants are read fail-loud from thresholds.yaml (§13.1); every
constant may also be passed explicitly (so tests never need the not-yet-registered
numbers). A REFUSED audit has no endpoint (§3.4) and raises `AuditRefused` — the
caller records it for the refusal accounting (§11) rather than fabricating a gap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from ..schemas.audit_core import AuditCore
from ..schemas.lattice_types import LatticeResult
from ..schemas.toggle import ToggleFacts
from ..thresholds import (
    SupportGate,
    load_primary_metric,
    load_shapley_pct_denominator_min,
    load_support_gate,
)
from .algebra import corner_marginals_result, saturated_basis
from .invariance import run_invariance_tests
from .lattice import run_lattice
from .preflight import PreflightResult, derive_scope
from .shapley import shapley_result
from .support import common_support, primary_metric_vector, support_info


class AuditRefused(RuntimeError):
    """The strategy audit is REFUSED — a non-runnable toggle has no defensible
    fixed_state, so the strategy has no audit endpoint (§3.4). Carries the
    pre-flight result for the refusal accounting."""

    def __init__(self, preflight: PreflightResult) -> None:
        self.preflight = preflight
        super().__init__(
            f"strategy {preflight.strategy_label!r} REFUSED: no defensible "
            f"fixed_state for {list(preflight.refused_toggles)}"
        )


def audit_spine(
    strategy,
    maximal_panel: pd.DataFrame,
    facts: Sequence[ToggleFacts],
    *,
    signals: pd.DataFrame | None = None,
    primary_metric: str,
    percentage_denominator_min: float,
    support_gate: SupportGate,
    lib_gap_lags: tuple[int, int] = (0, 1),
    months_per_year: int = 12,
    nw_lags: int | None = None,
    pre_registration_tag: str | None = None,
) -> tuple[PreflightResult, LatticeResult, AuditCore]:
    """Run the deterministic analytical spine (steps 2-7) and return the pre-flight
    result, the executed lattice, and the AuditCore. Shared by `run_audit` (which
    returns just the core) and `run_full_audit` (which needs the lattice for the
    bootstrap-based inference layers). Raises `AuditRefused` for a REFUSED scope."""
    pf = derive_scope(getattr(strategy, "strategy_label", "?"), facts)
    if pf.audit_scope == "REFUSED":
        raise AuditRefused(pf)

    lattice = run_lattice(
        strategy, pf.runnable_toggles, pf.fixed_states, maximal_panel,
        signals=signals, lib_gap_lags=lib_gap_lags,
    )

    info = support_info(lattice.cells, support_gate)
    common = common_support(lattice.cells)
    Y = primary_metric_vector(
        lattice.cells, common, primary_metric,
        months_per_year=months_per_year, nw_lags=nw_lags,
    )
    core = AuditCore(
        strategy_label=pf.strategy_label,
        audit_scope=pf.audit_scope,
        runnable_toggles=pf.runnable_toggles,
        conditioning_signature=pf.conditioning_signature,
        conditioning_statement=pf.conditioning_statement,
        primary_metric=primary_metric,
        support=info,
        corner_marginals=corner_marginals_result(Y, pf.runnable_toggles),
        saturated=saturated_basis(Y, pf.runnable_toggles),
        shapley=shapley_result(
            Y, pf.runnable_toggles,
            percentage_denominator_min=percentage_denominator_min,
        ),
        invariance=run_invariance_tests(lattice, pf.facts),
        pre_registration_tag=pre_registration_tag,
        not_applicable_toggles=pf.not_applicable_toggles,
    )
    return pf, lattice, core


def run_audit(
    strategy,
    maximal_panel: pd.DataFrame,
    facts: Sequence[ToggleFacts],
    *,
    signals: pd.DataFrame | None = None,
    primary_metric: str | None = None,
    percentage_denominator_min: float | None = None,
    support_gate: SupportGate | None = None,
    lib_gap_lags: tuple[int, int] = (0, 1),
    months_per_year: int = 12,
    nw_lags: int | None = None,
    pre_registration_tag: str | None = None,
    thresholds_path: str | Path | None = None,
) -> AuditCore:
    """Audit one strategy's analytical spine -> AuditCore.

    Raises `AuditRefused` if the derived audit scope is REFUSED."""
    if primary_metric is None:
        primary_metric = load_primary_metric(thresholds_path)
    if percentage_denominator_min is None:
        percentage_denominator_min = load_shapley_pct_denominator_min(thresholds_path)
    if support_gate is None:
        support_gate = load_support_gate(thresholds_path)

    _, _, core = audit_spine(
        strategy, maximal_panel, facts,
        signals=signals, primary_metric=primary_metric,
        percentage_denominator_min=percentage_denominator_min,
        support_gate=support_gate, lib_gap_lags=lib_gap_lags,
        months_per_year=months_per_year, nw_lags=nw_lags,
        pre_registration_tag=pre_registration_tag,
    )
    return core
