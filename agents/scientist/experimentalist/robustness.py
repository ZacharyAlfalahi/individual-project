"""G4 — robustness (spec §9/§10). Runs POST-BH-FDR (on G3 survivors only). Reuses the wrapped
stats + the Layer-1 crowding diagnostic; the audited engine / library are untouched.

  * CPCV (shared/stats) over the candidate's returns — purge = the per-candidate information span,
    embargo = holding_period months (F8/A4). The gate boolean `cpcv_qualified` is the registered
    G4 criterion: a positive MEDIAN OOS-fold Sharpe (the informative statistic for a fit-free
    candidate — see cpcv.py).
  * Deflated Sharpe (n_trials = the strategy's m) and the Layer-1 crowding diagnostic (BBW-4 +
    str + mom6) are MEASUREMENTS reported beside the gate, never gating.

Populates the measurements block; sets `cpcv_qualified`.
"""

from __future__ import annotations

import math

from pathlib import Path

import pandas as pd
import yaml

from agents.quant.library.characteristic_sort import summarize_returns

from shared.evaluation.crowding import crowding_diagnostic
from shared.evaluation.crowding_l2 import ipca_crowding_diagnostic
from shared.evaluation.thresholds import load_crowding_config
from shared.stats import cpcv_evaluate, deflated_sharpe_ratio

from ..schemas.evaluation import Measurements
from .gate_result import GateOutcome

_THRESHOLDS = Path(__file__).resolve().parents[3] / "docs" / "thresholds.yaml"


def load_registered_sr_std(strategy: str | None = None, path: Path | None = None) -> float:
    """The REGISTERED per-period (monthly) cross-trial Sharpe SD from
    ``auditor.dsr.<strategy>.sr_std`` — the same calibration the Auditor deflates against.

    Read fail-loud: a deflation benchmark built from an unregistered constant is a number
    nobody chose, and at these sample sizes it decides whether the statistic is degenerate.
    When the strategy is unknown, every registered strategy must agree on the value (they do,
    by construction) — otherwise the caller has to say which one it means."""
    doc = yaml.safe_load((path or _THRESHOLDS).read_text(encoding="utf-8"))
    try:
        block = doc["auditor"]["dsr"]
    except (KeyError, TypeError) as exc:
        raise KeyError("thresholds.yaml has no auditor.dsr block — the deflated-Sharpe "
                       "benchmark refuses an invented cross-trial dispersion") from exc
    if strategy is not None:
        if strategy not in block:
            raise KeyError(f"auditor.dsr has no entry for strategy {strategy!r}")
        return float(block[strategy]["sr_std"])
    values = {float(v["sr_std"]) for v in block.values()}
    if len(values) != 1:
        raise KeyError(f"auditor.dsr registers differing sr_std values {sorted(values)}; "
                       "the caller must name its strategy")
    return values.pop()


def robustness_g4(
    candidate_returns: pd.Series,
    gross,
    *,
    n_trials: int,
    information_span: int,
    holding_period: int,
    sr_std: float,
    months_per_year: int = 12,
    nw_lags: int | None = None,
    n_groups: int = 8,
    test_groups: int = 2,
    direction: int = 1,
    crowding_config=None,
    crowding_factors=None,
    ipca_factors=None,
):
    embargo = max(1, holding_period)
    # A month-filter extension can shrink the series below n_groups; CPCV cannot partition it.
    # That is a clean non-qualification (DEVELOPMENT_SURVIVOR_NOT_ADVANCED), NOT a crash — and it
    # must never propagate out and take down the whole batch.
    n_ret = int(len(candidate_returns.dropna()))
    if n_ret < n_groups:
        cpcv_qualified = False
        cpcv_summary = {"n_groups": n_groups, "test_groups": test_groups, "n_folds": 0,
                        "n_paths": 0, "median_sharpe": float("nan"), "frac_positive": float("nan"),
                        "insufficient_months": float(n_ret)}
    else:
        cpcv = cpcv_evaluate(candidate_returns, n_groups=n_groups, test_groups=test_groups,
                             purge=information_span, embargo=embargo, months_per_year=months_per_year)
        # SC-SCI-8 / D-Q17 — DIRECTIONAL: a positive median OOS-fold Sharpe IN THE STRATEGY'S
        # GATING DIRECTION. `direction` is DERIVED from the realised parent premium sign (not the
        # paper's claim): a genuinely negative-premium parent has negative Sharpes, so a
        # sign-agnostic `median > 0` would silently fail its extensions. The gating sign is always the
        # realised development sign, never a paper's published claim (e.g. DRR's reversal).
        cpcv_qualified = (math.isfinite(cpcv.median_sharpe)
                          and direction * cpcv.median_sharpe > 0.0)
        cpcv_summary = cpcv.to_dict()

    # Deflated Sharpe — per-period Sharpe deflated against the m-trial max (measurement only).
    # `sr_std` MUST be the per-period (monthly) cross-trial Sharpe SD, because `sr_periodic`
    # below is monthly: a cross-trial SD in annual units inflates the deflation benchmark by
    # ~sqrt(12) and drives the probability to 0 for every candidate; the value therefore comes
    # from the registered calibration (load_registered_sr_std), never a default.
    summ = summarize_returns(candidate_returns, nw_lags, months_per_year)
    sr = summ["sharpe"]
    if math.isfinite(sr) and summ["n_months"] >= 2:
        dsr = deflated_sharpe_ratio(sr / math.sqrt(months_per_year), summ["n_months"],
                                    n_trials, sr_std=sr_std)
    else:
        dsr = float("nan")

    # Layer-1 crowding (measurement only). Real run loads the factor bundle from the config;
    # tests inject `crowding_factors`.
    config = crowding_config if crowding_config is not None else load_crowding_config()
    crowd = dict(crowding_diagnostic(candidate_returns, config=config, factors=crowding_factors))
    # Layer 2 (recursive-OOS IPCA spanning) — a SECOND lens, merged in beside Layer 1 when the
    # recursive-OOS IPCA factor frame is supplied (measurement only, never gating).
    if ipca_factors is not None:
        crowd.update(ipca_crowding_diagnostic(candidate_returns, ipca_factors, nw_lags=nw_lags))

    measurements = Measurements(gross=gross, cpcv=cpcv_summary, deflated_sharpe=dsr, crowding=crowd)
    return (GateOutcome("G4", passed=cpcv_qualified, booleans={"cpcv_qualified": cpcv_qualified}),
            measurements)
