"""
report.py — assemble the full per-strategy AuditReport (§12).

`run_full_audit` runs the spine once (via `orchestrator.audit_spine`), then the
bootstrap-based inference layers on the SAME lattice: routed inference (§7.1),
BH-FDR over the confirmatory family (§7.2), the Bayesian normal approximation
(§7.3), the compression-adequacy statistic (§8.1) and economic significance (§9).

`AuditorConfig` gathers every constant. `AuditorConfig.from_thresholds` reads them
fail-loud from docs/thresholds.yaml (§13.1) — a missing constant raises, never
defaults. Tests build the config explicitly, so they never need the not-yet-
pre-registered numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..confirmatory import confirmatory_coordinates
from ..schemas.audit_report import AuditReport, BootstrapMeta
from ..schemas.lattice_types import METRIC_NAMES, MetricSet
from ..schemas.toggle import ToggleFacts
from ..thresholds import (
    BayesParams,
    EconomicGapBands,
    SupportGate,
    load_bayes_params,
    load_bootstrap_config,
    load_compression_dmax,
    load_economic_gap_bands,
    load_fdr_q,
    load_primary_metric,
    load_shapley_pct_denominator_min,
    load_support_gate,
    load_vartheta,
)
from .algebra import saturated_basis
from .bayes import assemble_theta, run_bayes
from .bootstrap import run_bootstrap
from .compression import run_compression
from .economic import EconomicBands, run_economic
from .fdr import run_fdr
from .inference import infer_doe_effects
from .metrics import metric_set_on
from .orchestrator import audit_spine
from .support import common_support, primary_metric_vector


@dataclass(frozen=True)
class AuditorConfig:
    primary_metric: str
    percentage_denominator_min: float
    support_gate: SupportGate
    n_replicates: int
    block_length_months: int
    min_effective_blocks: int
    vartheta: float
    d_max: float
    fdr_q: float
    bayes: BayesParams
    gap_bands: EconomicGapBands
    holding_period: int = 1
    months_per_year: int = 12
    lib_gap_lags: tuple[int, int] = (0, 1)
    # WS-5 (D12): the meas_err OFF arm for the FULL-audit/report path. Defaults "raw" so
    # run_full_audit is byte-identical unless a per-strategy baseline is threaded in; the
    # lattice/orchestrator already default OFF="raw". Activation is a logged, pre-registered
    # re-run, never a bare default flip.
    meas_err_off_family: str = "raw"
    alpha: float = 0.05

    @classmethod
    def from_thresholds(cls, path: str | Path | None = None) -> "AuditorConfig":
        """Read every constant fail-loud from thresholds.yaml (§13.1)."""
        boot = load_bootstrap_config(path)
        return cls(
            primary_metric=load_primary_metric(path),
            percentage_denominator_min=load_shapley_pct_denominator_min(path),
            support_gate=load_support_gate(path),
            n_replicates=boot.n_replicates,
            block_length_months=boot.block_length_months,
            min_effective_blocks=boot.min_effective_blocks,
            vartheta=load_vartheta(path),
            d_max=load_compression_dmax(path),
            fdr_q=load_fdr_q(path),
            bayes=load_bayes_params(path),
            gap_bands=load_economic_gap_bands(path),
        )


def run_full_audit(
    strategy,
    maximal_panel: pd.DataFrame,
    facts: list[ToggleFacts],
    config: AuditorConfig,
    *,
    signals: pd.DataFrame | None = None,
    n_trials: int,
    sr_std: float,
    seed: int = 0,
    pre_registration_tag: str | None = None,
    meas_err_off_family: str | None = None,
    expost_trim_off=None,
) -> AuditReport:
    """Full per-strategy audit: spine + bootstrap + inference/FDR/Bayes/compression
    /economic. `n_trials` and `sr_std` are the strategy-level deflated-Sharpe inputs
    (O-A4: the discovery count is a strategy property, identical across cells).
    `sr_std` must be the PER-PERIOD cross-trial Sharpe SD (see run_economic).

    `meas_err_off_family` (per-paper baseline profile, spec v4 D1) and `expost_trim_off`
    (mom6's re-injected published lab_trim, spec E) are the PER-ANCHOR Part-D/E inputs.
    Both default to the config-scalar / None (the default raw-arm behaviour), so a run that
    passes neither is byte-identical; the confirmatory driver threads them per anchor."""
    off_family = meas_err_off_family if meas_err_off_family is not None else config.meas_err_off_family
    pf, lattice, core = audit_spine(
        strategy, maximal_panel, facts,
        signals=signals,
        primary_metric=config.primary_metric,
        percentage_denominator_min=config.percentage_denominator_min,
        support_gate=config.support_gate,
        lib_gap_lags=config.lib_gap_lags,
        meas_err_off_family=off_family,
        expost_trim_off=expost_trim_off,
        months_per_year=config.months_per_year,
        pre_registration_tag=pre_registration_tag,
    )
    toggles = pf.runnable_toggles
    common = common_support(lattice.cells)

    bootstrap = run_bootstrap(
        lattice.cells, common, toggles,
        n_replicates=config.n_replicates,
        data_driven_block_months=config.block_length_months,
        min_effective_blocks=config.min_effective_blocks,
        holding_period=config.holding_period,
        metric=config.primary_metric,
        months_per_year=config.months_per_year,
        seed=seed,
    )

    coords = confirmatory_coordinates(toggles)
    inference = infer_doe_effects(
        lattice.cells, common, toggles, bootstrap,
        metric=config.primary_metric, coordinates=coords,
        months_per_year=config.months_per_year, alpha=config.alpha,
    )
    # WITHIN-STRATEGY diagnostic multiplicity control — NOT the §7.2 confirmatory
    # verdict. The pre-registered confirmatory family is the UNION across the locked
    # anchors (mom6, drf); assemble it with fdr.corpus_confirmatory_fdr over multiple
    # strategies' coordinate p-values (D-A9). The `scope` field marks this as
    # within_strategy so the Reporter cannot mistake it for the confirmatory result.
    fdr = run_fdr({T: inference[T].p_value for T in coords}, config.fdr_q)

    Y = primary_metric_vector(
        lattice.cells, common, config.primary_metric,
        months_per_year=config.months_per_year,
    )
    doe = saturated_basis(Y, toggles).doe
    theta_hat, draws = assemble_theta(bootstrap, coords, doe)
    bayesian = run_bayes(
        theta_hat, draws, coords,
        prior_scale=config.bayes.prior_scale, vartheta=config.vartheta,
        epsilon=config.bayes.epsilon,
        conditioning_signature=pf.conditioning_signature,
    )
    compression = run_compression(doe, bootstrap, d_max=config.d_max, alpha=config.alpha)

    # Economic endpoints on common support: all-OFF vs all-ON over the runnable set.
    uncorrected = metric_set_on(
        lattice.cell_for(frozenset()).returns, common,
        months_per_year=config.months_per_year,
    )
    corrected = metric_set_on(
        lattice.cell_for(frozenset(toggles)).returns, common,
        months_per_year=config.months_per_year,
    )
    economic = run_economic(
        uncorrected, corrected,
        gap_bands=EconomicBands(
            small=config.gap_bands.small,
            moderate=config.gap_bands.moderate,
            large=config.gap_bands.large,
        ),
        n_trials=n_trials, sr_std=sr_std,
    )

    return AuditReport(
        core=core,
        bootstrap=BootstrapMeta(
            n_replicates=bootstrap.n_replicates,
            block_length=bootstrap.block_length,
            effective_blocks=bootstrap.effective_blocks,
            t_common=bootstrap.t_common,
        ),
        inference=inference,
        fdr=fdr,
        bayesian=bayesian,
        compression=compression,
        economic=economic,
    )


# helper re-exported for the renderer/tests
__all__ = ["AuditorConfig", "run_full_audit", "MetricSet", "METRIC_NAMES"]
