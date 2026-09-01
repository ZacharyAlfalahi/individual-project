"""
audit_report.py — the full per-strategy AuditReport (§12).

Superset of the AuditCore: it carries the spine (scope, support, the
three lattice readings, invariance) PLUS the inference layers computed
on the bootstrap — routed inference, BH-FDR over the confirmatory family, the
Bayesian normal-approximation posteriors, and the compression-adequacy statistic —
and the economic-significance block.

The cross-strategy hierarchical prevalence (§8.2) is a CORPUS quantity, not a
per-strategy field, so it lives outside this report (see checks.hierarchical). The
recovery sweep and anchor triangulation are separate validation gates.

`to_dict()` exposes every numeric field so the numeric verifier (§11) can confirm
that no number in the rendered prose originates outside the typed report.
"""

from __future__ import annotations

from dataclasses import dataclass

from .audit_core import AuditCore
from .decomposition import subset_label


@dataclass(frozen=True)
class BootstrapMeta:
    n_replicates: int
    block_length: int
    effective_blocks: int
    t_common: int
    # Pre-registered stationary-bootstrap sensitivity (D-A29), reported BESIDE the
    # normative fixed-block bootstrap — never a decision rule. None when not computed;
    # `stationary_block_length` is the EXPECTED (mean) geometric block length.
    stationary_gap_ci_low: float | None = None
    stationary_gap_ci_high: float | None = None
    stationary_block_length: int | None = None

    def to_dict(self) -> dict:
        d = {
            "n_replicates": self.n_replicates,
            "block_length": self.block_length,
            "effective_blocks": self.effective_blocks,
            "t_common": self.t_common,
        }
        if None not in (self.stationary_gap_ci_low, self.stationary_gap_ci_high,
                        self.stationary_block_length):
            d["stationary_gap_ci_low"] = self.stationary_gap_ci_low
            d["stationary_gap_ci_high"] = self.stationary_gap_ci_high
            d["stationary_block_length"] = self.stationary_block_length
        return d


@dataclass(frozen=True, eq=False)
class AuditReport:
    core: AuditCore
    bootstrap: BootstrapMeta
    inference: dict            # frozenset -> InferenceResult
    fdr: object               # FdrReport
    bayesian: object          # BayesianResult
    compression: object       # CompressionResult
    economic: object          # EconomicResult

    def to_dict(self) -> dict:
        d = self.core.to_dict()
        d["bootstrap"] = self.bootstrap.to_dict()
        d["inference"] = {
            subset_label(T): {
                "point": r.point, "method": r.method, "p_value": r.p_value,
                "ci_low": r.ci_low, "ci_high": r.ci_high, "t_stat": r.t_stat,
            }
            for T, r in self.inference.items()
        }
        d["fdr"] = self.fdr.to_dict()
        d["bayesian"] = self.bayesian.to_dict()
        d["compression"] = self.compression.to_dict()
        d["economic"] = self.economic.to_dict()
        return d
