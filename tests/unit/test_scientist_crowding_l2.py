"""Layer-2 crowding — recursive-OOS IPCA factor build (no look-ahead), the spanning lens
(known-answer), and the G4 two-lens wiring. Uses a small synthetic IPCA panel with known truth."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.quant.library.ipca import rank_transform  # noqa: E402
from agents.scientist.experimentalist.robustness import robustness_g4  # noqa: E402
from agents.scientist.schemas.evaluation import GrossMeasurements  # noqa: E402
from shared.evaluation.crowding_l2 import ipca_crowding_diagnostic, recursive_ipca_factors  # noqa: E402
from shared.evaluation.thresholds import CrowdingConfig  # noqa: E402

FACTORS = ("mktb", "drf", "crf", "lrf", "str", "mom6")


def _make_ipca_panel(rng, *, L=6, K=2, T=60, n=50, snr=15.0):
    n_inst = L - 1
    gamma_beta = np.linalg.qr(rng.standard_normal((L, K)))[0][:, :K]
    var_f = np.linspace(K, 1.0, K)
    mu_f = 0.3 * np.sqrt(var_f)
    Z_list, R_list = [], []
    for _ in range(T):
        raw = rng.standard_normal((n, n_inst))
        Z = np.column_stack([*(rank_transform(raw[:, j]) for j in range(n_inst)), np.ones(n)])
        f = mu_f + np.sqrt(var_f) * rng.standard_normal(K)
        signal = Z @ gamma_beta @ f
        eps_sd = (float(signal.std()) or 1.0) / np.sqrt(snr)
        Z_list.append(Z)
        R_list.append(signal + eps_sd * rng.standard_normal(n))
    return Z_list, R_list, np.arange(1, T + 1)


def _cfg():
    return CrowdingConfig(factor_set=FACTORS, hac_lag_rule="newey_west_auto", min_obs=10,
                          bundles={f: (f"{f}.parquet", f"{f}_corr") for f in FACTORS})


def test_recursive_ipca_factors_are_oos_shaped():
    Z, R, months = _make_ipca_panel(np.random.default_rng(0), T=60)
    frame = recursive_ipca_factors(Z, R, months, K=2, burn_in=36)
    assert list(frame.columns) == ["date", "ipca_1", "ipca_2"]
    assert len(frame) == 60 - 36                        # one OOS row per t in {burn_in..T-1}


def test_ipca_spanning_is_a_known_answer():
    Z, R, months = _make_ipca_panel(np.random.default_rng(1), T=60)
    frame = recursive_ipca_factors(Z, R, months, K=2, burn_in=36)
    dates = frame["date"]
    spanned = pd.Series(0.5 * frame["ipca_1"].values + 0.3 * frame["ipca_2"].values, index=dates)
    assert abs(ipca_crowding_diagnostic(spanned, frame, nw_lags=0)["ipca_alpha"]) < 1e-6  # absorbed
    with_alpha = spanned + 0.01
    res = ipca_crowding_diagnostic(with_alpha, frame, nw_lags=0)
    assert res["ipca_alpha"] == __import__("pytest").approx(0.01, abs=1e-6)   # alpha survives


def test_g4_wires_both_crowding_lenses():
    rng = np.random.default_rng(2)
    Z, R, months = _make_ipca_panel(rng, T=120)
    dates = pd.date_range("2004-08-31", periods=120, freq="ME")   # real month-end dates
    month_to_date = {int(m): d for m, d in zip(months, dates)}
    frame = recursive_ipca_factors(Z, R, months, K=2, burn_in=60, month_to_date=month_to_date)
    J = len(frame)
    cand = pd.Series(0.02 + 0.005 * np.sin(np.arange(J)), index=frame["date"])
    layer1 = pd.DataFrame({"date": frame["date"],
                           **{f: rng.normal(scale=0.02, size=J) for f in FACTORS}})
    out, meas = robustness_g4(cand, GrossMeasurements(alpha_bbw4=0.02), n_trials=6,
                              information_span=2, holding_period=1, sr_std=0.5,
                              crowding_config=_cfg(), crowding_factors=layer1, ipca_factors=frame,
                              nw_lags=0)
    assert "alpha" in meas.crowding and "ipca_alpha" in meas.crowding   # both lenses present
    assert "ipca_alpha_t" in meas.crowding
