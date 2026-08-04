"""
_shared_eval_fixtures.py — synthetic builders for the shared-evaluation diagnostics
tests (shared/evaluation). The SAME objects production uses (D-E18): a wide corrected
factor frame, planted candidates, macro-spread series, and turnover series. Deterministic
(seeded numpy). Mirrors the tests/unit/_*_fixtures.py convention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shared.evaluation.thresholds import CrowdingConfig

FACTORS = ("mktb", "drf", "crf", "lrf", "str", "mom6")

# extension_1's frozen evaluation median (docs/extension_1_config.yaml).
FROZEN_MEDIAN = 0.935


def factor_frame(T: int = 200, seed: int = 1) -> pd.DataFrame:
    """Wide factor frame: `date` + the six logical factors, i.i.d. month-end draws."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-31", periods=T, freq="ME")
    data = {"date": dates}
    for f in FACTORS:
        data[f] = rng.normal(scale=0.02, size=T)
    return pd.DataFrame(data)


def candidate(
    frame: pd.DataFrame,
    a0: float = 0.002,
    betas: dict[str, float] | None = None,
    noise: float = 0.0005,
    seed: int = 2,
) -> pd.Series:
    """A candidate return series = a0 + sum beta_f * factor_f + noise, indexed by date.
    An explicit `betas={}` means NO factor loadings (not the default) — `is None` is the
    only trigger for the default, so the empty dict is honoured."""
    betas = {"mktb": 0.5, "drf": 0.3} if betas is None else betas
    rng = np.random.default_rng(seed)
    T = len(frame)
    y = np.full(T, a0) + rng.normal(scale=noise, size=T)
    for f, b in betas.items():
        y = y + b * frame[f].to_numpy()
    return pd.Series(y, index=frame["date"])


def crowding_config(hac: str = "floor_t_pow_0.25", min_obs: int = 60) -> CrowdingConfig:
    """A CrowdingConfig for tests that pass `factors=` explicitly (bundles unused)."""
    return CrowdingConfig(
        factor_set=FACTORS,
        hac_lag_rule=hac,
        min_obs=min_obs,
        bundles={f: (f"{f}.parquet", f"{f}_corr") for f in FACTORS},
    )


def spread_series(T: int = 200, lo: float = 0.5, hi: float = 1.4, seed: int | None = None) -> pd.Series:
    """A BAA-AAA-like macro spread ramp that crosses the frozen 0.935 median, month-end
    aligned to `factor_frame(T)`."""
    dates = pd.date_range("2000-01-31", periods=T, freq="ME")
    base = np.linspace(lo, hi, T)
    if seed is not None:
        base = base + np.random.default_rng(seed).normal(scale=0.05, size=T)
    return pd.Series(base, index=dates)


def turnover_series(
    frame: pd.DataFrame,
    base: float = 0.2,
    cov_factor: str = "mktb",
    cov: float = 0.5,
    noise: float = 0.01,
    seed: int = 11,
) -> pd.Series:
    """A monthly turnover series (fraction traded) that COVARIES with a control — the
    D-E13 setup where the control-adjusted break-even differs from the naive one."""
    rng = np.random.default_rng(seed)
    T = len(frame)
    vals = base + cov * frame[cov_factor].to_numpy() + rng.normal(scale=noise, size=T)
    return pd.Series(np.abs(vals), index=frame["date"])
