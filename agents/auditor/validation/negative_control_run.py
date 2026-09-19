"""Negative-control specificity evaluation path (§10.3).

Connects the registered, unit-verified ``evaluate_negative_control`` gate to development data.

The ``traded_liquidity`` negative control maps to the existing LRF (gamma-illiquidity long-short).
This module computes LRF's CORRECTED-vs-UNCORRECTED monthly-premium differential and its
pre-registered confirmatory bootstrap CI, then grades it with the separated-mode gate.

**Pre-registered estimand (run once, report as-is):**
  * differential = LRF premium under the ``meas_err`` / DRR price-cleaning correction
    (``price_family`` raw→corr, ALL other toggles held) — the universal data-layer correction,
    directly comparable to the ``drf`` anchor (whose dominant bias IS meas_err). NOT the full
    multi-toggle envelope.
  * panel = the maximal (clean-price) dev panel (the basis the RQ3 audit anchors use).
  * interval = the pre-registered confirmatory block bootstrap ``auditor.bootstrap`` via
    ``run_bootstrap`` → ``gap_ci(0.05)`` (gap = corrected − uncorrected).
  * gate = ``evaluate_negative_control`` — PASS iff the whole CI is strictly within ±vartheta.

A CI outside ±vartheta is a legitimate FINDING (the correction moves the clean factor / specificity
not certified with precision), NEVER something to re-run or re-spec toward a pass. Deterministic
(seeded), dev-only (``load_dev_inputs`` never reads the holdout), $0.
"""

from __future__ import annotations

import hashlib

import pandas as pd

from agents.auditor.checks.bootstrap import run_bootstrap
from agents.auditor.checks.report import AuditorConfig
from agents.auditor.checks.support import common_support
from agents.auditor.schemas.lattice_types import CellReturns, MetricSet
from agents.auditor.validation.hypothesis_registry import (
    FactorHypothesis,
    NegativeControlVerdict,
    evaluate_negative_control,
    load_hypothesis_registry,
)
from agents.quant.library.bbw_factors import run_bbw_factor
from agents.quant.library.characteristic_sort import summarize_returns
from agents.quant.library.run_config import (
    ConstructionConfig,
    EvaluationConfig,
    PanelViewConfig,
    RunConfig,
)
from agents.quant.library.views import view

#: The price-cleaning correction toggle (``price_family`` raw→corr) this control is measured against.
MEAS_ERR = "meas_err"
NEG_CONTROL_FACTOR = "traded_liquidity"


class NegativeControlRunError(RuntimeError):
    """A negative-control run input is malformed or the factor is not the registered control —
    surfaced loudly, never a silent default."""


def _series_hash(s: pd.Series) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(s, index=True).values.tobytes()).hexdigest()


def _cell(on_set: frozenset, returns: pd.Series, n_bonds: pd.Series | None = None) -> CellReturns:
    """Wrap a monthly long-short series as a ``CellReturns`` for the bootstrap. The bootstrap
    consumes only ``on_set`` + ``returns`` (via ``return_matrix``/``common_support``); the metrics,
    ``n_bonds`` and content hashes are computed honestly for completeness (never fabricated)."""
    ret = returns.sort_index()
    nb = (n_bonds.sort_index() if n_bonds is not None
          else pd.Series(0, index=ret.index, dtype="int64"))
    metrics = MetricSet.from_summary(summarize_returns(ret, None, 12))
    rh = _series_hash(ret)
    metric_hash = hashlib.sha256(
        repr(sorted(metrics.as_metric_dict().items())).encode("utf-8")).hexdigest()
    return CellReturns(
        on_set=on_set, run_config=None, returns=ret, n_bonds=nb, metrics_native=metrics,
        run_config_hash=("lrf_corr" if on_set else "lrf_raw"), panel_view_hash="maximal_clean",
        return_hash=rh, n_bonds_hash=_series_hash(nb.astype(float)), metric_hash=metric_hash)


def compute_negative_control(
    uncorrected: pd.Series,
    corrected: pd.Series,
    *,
    hyp: FactorHypothesis,
    vartheta: float,
    n_replicates: int,
    block_length_months: int,
    min_effective_blocks: int,
    seed: int = 0,
    n_bonds_off: pd.Series | None = None,
    n_bonds_on: pd.Series | None = None,
) -> tuple[NegativeControlVerdict, dict]:
    """Bootstrap the corrected-minus-uncorrected mean-gap CI (pre-registered confirmatory block
    bootstrap) and grade it with the separated-mode gate. ``uncorrected``/``corrected`` are
    date-indexed monthly LRF premium series. Returns ``(verdict, meta)``."""
    off = _cell(frozenset(), uncorrected, n_bonds_off)
    on = _cell(frozenset({MEAS_ERR}), corrected, n_bonds_on)
    common = common_support([off, on])
    boot = run_bootstrap(
        [off, on], common, [MEAS_ERR],
        n_replicates=n_replicates, data_driven_block_months=block_length_months,
        min_effective_blocks=min_effective_blocks, holding_period=1, metric="average", seed=seed)
    ci_low, ci_high = boot.gap_ci(0.05)
    verdict = evaluate_negative_control(hyp, ci_low, ci_high, vartheta)
    native_max = max(int(uncorrected.dropna().shape[0]), int(corrected.dropna().shape[0]))
    meta = {
        "n_months_common": int(len(common)),
        "native_max_months": native_max,          # raw↔corr month overlap visible at a glance
        "block_length_months": int(boot.block_length),
        "effective_blocks": int(boot.effective_blocks),
        "n_replicates": int(boot.n_replicates),
        "seed": int(seed),
        "point_gap_mean": float(corrected.reindex(common).mean() - uncorrected.reindex(common).mean()),
    }
    return verdict, meta


def lrf_family_returns(maximal: pd.DataFrame, signals: pd.DataFrame, price_family: str) -> pd.DataFrame:
    """Run LRF on one price family (``raw``|``corr``) of the maximal dev panel — mirrors
    ``scripts/build_bbw_factors.run_family`` (flips ONLY ``price_family``: the meas_err correction)."""
    cfg = RunConfig(
        panel_view=PanelViewConfig(price_family=price_family, stale_mask=False,
                                   include_terminal_rows=False),
        construction=ConstructionConfig(signal_lag=0, expost_trim="none"),
        evaluation=EvaluationConfig())
    panel = view(maximal, cfg, signals=signals).drop_duplicates(
        subset=["cusip", "date"]).reset_index(drop=True)
    return run_bbw_factor(panel, "lrf")["monthly_returns"]


def _series_of(monthly_returns: pd.DataFrame, col: str) -> pd.Series:
    return pd.Series(monthly_returns[col].to_numpy(),
                     index=pd.DatetimeIndex(monthly_returns["date"])).sort_index()


def run_negative_control_from_dev(
    *,
    seed: int = 0,
    thresholds_path=None,
    maximal: pd.DataFrame | None = None,
    signals: pd.DataFrame | None = None,
    return_basis: str | None = None,
) -> dict:
    """Orchestrate the dev-data negative-control run: LRF raw vs corr on the maximal panel →
    pre-registered bootstrap CI → separated-mode gate. Deterministic, dev-only, $0.

    ``maximal``/``signals`` inject a basis-specific dev panel and its signals (both or neither);
    omitted, ``load_dev_inputs()`` supplies the clean maximal dev panel. ``return_basis`` labels the
    injected panel's return basis in the output; omitted, the output carries no return_basis key and
    estimand.panel is maximal_clean_dev."""
    from agents.auditor.ipca_differential.runner import load_dev_inputs

    if (maximal is None) != (signals is None):
        raise NegativeControlRunError(
            "inject maximal AND signals together, or neither (then load_dev_inputs supplies both)")
    hyp = load_hypothesis_registry()[NEG_CONTROL_FACTOR]
    if hyp.magnitude_mode != "separated":
        raise NegativeControlRunError(
            f"{NEG_CONTROL_FACTOR} is registered {hyp.magnitude_mode!r}, not 'separated' — "
            "the specificity gate requires the falsifiable separated mode")
    cfg = AuditorConfig.from_thresholds(thresholds_path)
    vartheta = cfg.vartheta          # the canonical auditor.practical_significance.vartheta (one source)

    if maximal is None:
        maximal, signals, _registry = load_dev_inputs()
    # LRF sorts on score "gamma" (the illiquidity measure). load_dev_inputs names it
    # "gamma_illiq_<family>"; LRF's rulebook + view expect "gamma". Pass ONLY the gamma raw/corr
    # columns (renamed), so view() resolves gamma_<family>->gamma cleanly — passing the other
    # signals' bbw_2019/jostova variants would trip view's A9 cross-family consistency guard.
    lrf_signals = signals[["cusip", "date", "gamma_illiq_raw", "gamma_illiq_corr"]].rename(
        columns={"gamma_illiq_raw": "gamma_raw", "gamma_illiq_corr": "gamma_corr"})
    raw_mr = lrf_family_returns(maximal, lrf_signals, "raw")
    corr_mr = lrf_family_returns(maximal, lrf_signals, "corr")

    verdict, meta = compute_negative_control(
        _series_of(raw_mr, "strategy_ret"), _series_of(corr_mr, "strategy_ret"),
        hyp=hyp, vartheta=vartheta,
        n_replicates=cfg.n_replicates, block_length_months=cfg.block_length_months,
        min_effective_blocks=cfg.min_effective_blocks, seed=seed,
        n_bonds_off=_series_of(raw_mr, "n_bonds"), n_bonds_on=_series_of(corr_mr, "n_bonds"))

    result = {
        "control": "negative_control_specificity",
        "factor": NEG_CONTROL_FACTOR,
        "basis": "real_dev_data",
        "status": "fired",
        "estimand": {
            "differential": "LRF corrected-minus-uncorrected premium under the meas_err / DRR "
                            "price-cleaning correction (price_family raw->corr; all other toggles held)",
            "scope_note": "the meas_err/price-cleaning correction ONLY, NOT the full bias envelope — "
                          "comparable to the drf anchor's dominant bias",
            "panel": f"maximal_{return_basis or 'clean'}_dev",
            "interval": "pre-registered confirmatory block bootstrap (auditor.bootstrap), 95% gap CI",
            "run_once": True,
        },
        "gate": {"mode": hyp.magnitude_mode, "vartheta": vartheta,
                 "pass_rule": "the corrected-vs-uncorrected CI lies strictly within (-vartheta, +vartheta)"},
        "ci_low": verdict.ci_low,
        "ci_high": verdict.ci_high,
        "vartheta": verdict.vartheta,
        "passed": verdict.passed,
        "absolute_gap": verdict.absolute_gap,
        "point_gap_mean": meta["point_gap_mean"],
        "bootstrap": {k: meta[k] for k in
                      ("n_months_common", "native_max_months", "block_length_months",
                       "effective_blocks", "n_replicates", "seed")},
    }
    if return_basis is not None:
        result["return_basis"] = return_basis
    return result
