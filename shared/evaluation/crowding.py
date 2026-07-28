"""
crowding.py — Layer-1 crowding survivor diagnostic (shared/evaluation).

Supersedes the KPP-5 factor bundle (docs/archive/kpp5_spec.md): KPP's characteristics are
unavailable on the TRACE/FISD panel (only bond volatility exists; duration is a
maturity proxy; credit spread and spread-to-distance-to-default have no data).
Layer 1 regresses a candidate strategy's monthly returns on a FIXED 6-factor set
— BBW-4 (mktb, drf, crf, lrf) plus the project's own corrected short-term-reversal
(str) and 6-month-momentum (mom6) factors — identical for every candidate, and
reports the surviving alpha + NW-HAC t-stat.

The set is unconditional (NO leave-one-parent-out): excluding a candidate's own
parent factor would strip the most informative regressor and, when the parent is a
BBW-4 constituent, mutate the pre-registered primary benchmark.

This is thin orchestration over
`agents.quant.library.characteristic_sort.regress_on_benchmark` — no new
statistics. It takes explicit arguments and imports no Scientist gate, so it is
complete and testable before the G4 robustness gate that will call it exists. G4
wires it POST-BH-FDR in one line:

    measurements = dataclasses.replace(
        measurements, crowding=crowding_diagnostic(candidate_returns))

Layer 2 (recursive-OOS IPCA factor spanning) is deferred — see
docs/backlog/remaining_work.md.
"""

from __future__ import annotations

import functools
import math

import pandas as pd

from agents.quant.library.characteristic_sort import regress_on_benchmark

from .thresholds import CrowdingConfig, load_crowding_config


def load_crowding_factor_bundle(config: CrowdingConfig) -> pd.DataFrame:
    """Assemble the wide crowding factor frame from the corrected bundles.

    For each logical factor in `config.factor_set`, read exactly its `_corr` column
    from the parquet named in `config.bundles`, rename it to the logical name,
    normalise `date` to datetime64[ns] (the mktb bundle stores `us`), and inner-join
    all factors on `date`. Returns `['date', *config.factor_set]`.

    Corrected lattice ONLY: only the corrected column named in the config is ever
    read (never a `*_raw` / `n_bonds` column), and a column that is not a corrected
    (`_corr`) column is refused here — the benchmark must sit at the corrected
    lattice point, the same point every candidate is regressed against.
    """
    frames: list[pd.DataFrame] = []
    for name in config.factor_set:
        path, column = config.bundles[name]
        if not column.endswith("_corr"):
            raise ValueError(
                f"crowding bundle '{name}' names column {column!r}, not a corrected "
                f"(_corr) column; the crowding benchmark must sit at the corrected "
                f"lattice point"
            )
        df = pd.read_parquet(path, columns=["date", column])
        df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
        frames.append(df.rename(columns={column: name}))
    merged = functools.reduce(lambda a, b: a.merge(b, on="date", how="inner"), frames)
    return merged[["date", *config.factor_set]].sort_values("date").reset_index(drop=True)


def _resolve_nw_lags(rule: str, n_obs: int) -> int | None:
    """Map the pre-registered HAC lag rule + realised sample T to a lag count.

    `floor_t_pow_0.25` -> floor(T**0.25) (the same rule as the primary alpha test);
    `newey_west_auto`  -> None (the characteristic-sort engine's native auto lag).
    `regress_on_benchmark` caps the returned int into [0, T-1]."""
    if rule == "newey_west_auto":
        return None
    if rule == "floor_t_pow_0.25":
        return int(math.floor(n_obs ** 0.25)) if n_obs > 0 else 0
    raise ValueError(f"unknown hac_lag_rule {rule!r}")  # config loader already guards this


def _overlap_months(candidate_returns: pd.Series, factors: pd.DataFrame) -> int:
    """Count the complete-case overlap `regress_on_benchmark` will fit on — the T the
    `floor_t_pow_0.25` rule needs before the regression runs. Replicates the engine's
    own inner-join + dropna so the T here is exactly the T the engine caps against."""
    factor_cols = [c for c in factors.columns if c != "date"]
    y_df = pd.DataFrame({"date": candidate_returns.index, "_y": candidate_returns.to_numpy()})
    merged = y_df.merge(factors, on="date", how="inner").dropna(subset=["_y", *factor_cols])
    return len(merged)


def crowding_diagnostic(
    candidate_returns: pd.Series,
    *,
    config: CrowdingConfig | None = None,
    factors: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Layer-1 crowding survivor diagnostic for one candidate.

    `candidate_returns` is a monthly (month-end) excess-return Series indexed by
    date. Regresses it on the fixed corrected factor set via `regress_on_benchmark`
    and returns an all-float mapping for `EvaluationRecord.measurements.crowding`:
    `{alpha, alpha_t, n_obs, nw_lags_used, below_min_obs, beta_<f>..., beta_t_<f>...}`.
    The excluded-nothing set means every factor gets a `beta_` key. Never raises on
    data shape — the degenerate cases (no overlap, T<=k) delegate to the engine and
    surface as NaN.

    `config` / `factors` are loaded from the pre-registered defaults when omitted;
    passing `factors` (a pre-assembled bundle) avoids re-reading the parquets.
    """
    if not isinstance(candidate_returns, pd.Series):
        raise TypeError("candidate_returns must be a pandas Series indexed by date")
    if config is None:
        config = load_crowding_config()
    if factors is None:
        factors = load_crowding_factor_bundle(config)

    # Normalise the candidate index to datetime64[ns] so it joins the (ns-normalised)
    # factor dates regardless of the series' original datetime resolution.
    idx = pd.to_datetime(candidate_returns.index).astype("datetime64[ns]")
    candidate_returns = pd.Series(candidate_returns.to_numpy(), index=idx)

    n_obs = _overlap_months(candidate_returns, factors)
    nw_lags = _resolve_nw_lags(config.hac_lag_rule, n_obs)
    reg = regress_on_benchmark(candidate_returns, factors, nw_lags=nw_lags)

    out: dict[str, float] = {
        "alpha": float(reg["alpha"]),
        "alpha_t": float(reg["alpha_t"]),
        "n_obs": float(reg["n_obs"]),
        "nw_lags_used": float(reg["nw_lags_used"]),
        "below_min_obs": 1.0 if reg["n_obs"] < config.min_obs else 0.0,
    }
    for name, beta in reg["betas"].items():
        out[f"beta_{name}"] = float(beta)
    for name, t in reg["beta_t"].items():
        out[f"beta_t_{name}"] = float(t)
    return out
