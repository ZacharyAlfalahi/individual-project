"""Corpus prevalence π̃_i(ϑ) over the audited strategies — the registered estimand, computed
from the audit reports.

``agents/auditor/checks/hierarchical.py`` implements the two-level model and the prevalence
denominator rule. This module
assembles that model's input from the audit reports and nothing else.

Three rules from the registration (§8.2.3, D-A31) that decide the arithmetic:

  * the denominator is **all runnable strategies within COMPLETE audits** — not the
    susceptible ones;
  * a proven **no-op contributes 0 to the numerator and 1 to the denominator** (a degenerate
    posterior at zero), which is an honest contribution and never an exclusion;
  * a **structurally non-applicable** coordinate (``str``'s ``meas_err``, which has no
    estimand) is ABSENT from that strategy's coordinates and is never counted either way.

Measurement variance comes from each coordinate's HAC interval in the report. The reports do
not persist the k×k bootstrap covariance, so this is the registered DIAGONAL-measurement
path (``fit_coordinate_hierarchy``), i.e. the sensitivity arm of D-A32 rather than its joint
primary — recorded in the output so the two are never confused.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from agents.auditor.checks.hierarchical import StrategyEffect, fit_hierarchy

#: z for a two-sided 95% interval — the reports' HAC intervals are 95%.
_Z95 = 1.959963984540054

#: The measurement path this module can supply from the persisted artefacts.
MEASUREMENT_PATH = "diagonal (fit_coordinate_hierarchy) — the registered D-A32 SENSITIVITY; the joint path needs the k×k bootstrap covariance, which the reports do not persist"


class PrevalenceInputError(ValueError):
    """An audit report cannot supply what the hierarchy needs — raised, never defaulted."""


@dataclass(frozen=True)
class CoordinateInput:
    """One (coordinate, strategy) cell on its way into the hierarchy."""

    coordinate: str
    strategy: str
    estimate: float
    variance: float
    is_no_op: bool

    def to_dict(self) -> dict:
        return {"coordinate": self.coordinate, "strategy": self.strategy,
                "estimate": self.estimate, "variance": self.variance,
                "is_no_op": self.is_no_op}


def variance_from_interval(ci_low: float, ci_high: float) -> float:
    """Measurement variance implied by a 95% interval. A degenerate (zero-width) interval —
    an inert coordinate whose contrast is identically zero — yields variance 0.0, which the
    hierarchy floors; it is the no-op path, not a missing value."""
    if ci_high < ci_low:
        raise PrevalenceInputError(f"interval is inverted: [{ci_low}, {ci_high}]")
    se = (ci_high - ci_low) / (2.0 * _Z95)
    return float(se * se)


def first_order_coordinates(report: dict) -> tuple[str, ...]:
    """The strategy's first-order correction coordinates: its runnable toggles, in report
    order. Interaction cells are excluded — the registered hierarchy is over corrections."""
    return tuple(report.get("runnable_toggles") or ())


def no_op_coordinates(report: dict) -> frozenset[str]:
    """Coordinates the invariance gate PROVED inert for this strategy (ON ≡ OFF)."""
    return frozenset(
        str(row["toggle_id"]) for row in (report.get("invariance") or [])
        if row.get("is_no_op")
    )


def collect_inputs(reports: dict[str, dict]) -> tuple[CoordinateInput, ...]:
    """Every (coordinate, strategy) cell the hierarchy will see, from ``{strategy: report}``.

    A report whose ``audit_scope`` is not COMPLETE is refused: the registered denominator is
    "runnable strategies within COMPLETE audits", so silently folding an incomplete audit in
    would change the estimand."""
    out: list[CoordinateInput] = []
    for strategy, report in sorted(reports.items()):
        scope = report.get("audit_scope")
        if scope != "COMPLETE":
            raise PrevalenceInputError(
                f"{strategy}: audit_scope is {scope!r}, not 'COMPLETE' — the prevalence "
                "denominator is defined over complete audits only"
            )
        inference = report.get("inference") or {}
        no_ops = no_op_coordinates(report)
        for coordinate in first_order_coordinates(report):
            cell = inference.get(coordinate)
            if cell is None:
                raise PrevalenceInputError(
                    f"{strategy}: runnable toggle {coordinate!r} has no inference cell"
                )
            out.append(CoordinateInput(
                coordinate=coordinate,
                strategy=strategy,
                estimate=float(cell["point"]),
                variance=variance_from_interval(float(cell["ci_low"]), float(cell["ci_high"])),
                is_no_op=bool(coordinate in no_ops or cell.get("inert")),
            ))
    return tuple(out)


def corpus_from_inputs(inputs: tuple[CoordinateInput, ...]) -> dict[str, list[StrategyEffect]]:
    """``coordinate -> [StrategyEffect]``, the hierarchy's input shape. A coordinate absent
    from a strategy (structurally non-applicable) simply has no entry for it."""
    corpus: dict[str, list[StrategyEffect]] = {}
    for cell in inputs:
        corpus.setdefault(cell.coordinate, []).append(StrategyEffect(
            strategy_label=cell.strategy,
            estimate=cell.estimate,
            variance=cell.variance,
            is_no_op=cell.is_no_op,
        ))
    return corpus


def compute_prevalence(
    reports: dict[str, dict],
    *,
    vartheta: float,
    vartheta_grid: tuple[float, ...] = (),
    n_iter: int = 3000,
    burn: int = 1000,
    seed: int = 0,
) -> dict:
    """The registered prevalence block over ``{strategy: report}``.

    Returns one entry per coordinate with π̃(ϑ), its sweep, μ and τ, and the two denominators
    kept apart (μ/τ over susceptible strategies; π̃ over all runnable)."""
    inputs = collect_inputs(reports)
    corpus = corpus_from_inputs(inputs)
    fitted = fit_hierarchy(corpus, vartheta=vartheta, vartheta_grid=vartheta_grid,
                           n_iter=n_iter, burn=burn, seed=seed)

    coordinates = {}
    for coordinate, hierarchy in fitted.items():
        block = hierarchy.to_dict()
        block["strategies_present"] = sorted(
            c.strategy for c in inputs if c.coordinate == coordinate)
        block["no_op_strategies"] = sorted(
            c.strategy for c in inputs if c.coordinate == coordinate and c.is_no_op)
        coordinates[coordinate] = block

    return {
        "estimand": "pi-tilde_i(vartheta) = (1/n_i) sum_s P(|E_{i,s}| > vartheta | data)",
        "denominator_rule": ("all runnable strategies within COMPLETE audits; a proven no-op "
                             "contributes 0 to the numerator and 1 to the denominator; a "
                             "structurally non-applicable coordinate is absent from both"),
        "measurement_path": MEASUREMENT_PATH,
        "vartheta": vartheta,
        "vartheta_grid": list(vartheta_grid),
        "sampler": {"n_iter": n_iter, "burn": burn, "seed": seed},
        "n_strategies": len(reports),
        "strategies": sorted(reports),
        "coordinates": coordinates,
        "inputs": [c.to_dict() for c in inputs],
    }


def fmt_percent(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100.0 * x:.1f}%"
