"""
Characteristic-sort engine.

A general, signal-agnostic engine that measures what a "rank bonds by a
property, buy the top group, sell the bottom group" strategy earned over
history. One run measures one factor. The engine never computes the property
being ranked on -- it sorts on a column that is already present in the panel.

Spec: docs/characteristic_sort_engine_spec.md.

Design conventions, deliberately fixed:

- NW automatic-lag rule: floor(4 * (T/100)^(2/9)), Newey-West (1987),
  clamped to [0, T-1].
- Tie-breaking: stable sort by (score asc, bond_id asc), rank(method='first'),
  then group = ((rank-1) * n_groups) // n. Truly fixed and repeatable.
- Gap policy: calendar-strict. next-month return uses the row at exactly
  t + MonthEnd(1). signal_lag uses the row at exactly t - MonthEnd(signal_lag).
  Gaps -> NaN -> the row is filtered at eligibility.
- Empty-stripe rule: skip a control stripe if EITHER the long or the short
  leg has zero bonds.
- Result shape: plain dict with keys monthly_returns / summary /
  relationship_to_benchmark / settings_used / bookkeeping.
- Rulebook: plain dict argument; no thresholds.yaml dependency on the
  initial build.
- Benchmark regression SEs: hand-rolled OLS + Bartlett-kernel NW HAC using
  the SAME nw_lags choice as the strategy luck-check.
- Stats dependency: numpy + pandas only. No statsmodels, no scipy.
- monthly_returns.date label = realisation month t+1 (NOT formation month t --
  this diverges from spec section 4.3 by design).

The NW HAC convention used here:
  beta_hat = (X'X)^-1 X'y, residuals u = y - X beta_hat.
  g_t = u_t * X_t (a k-vector per t).
  Bartlett weights w_l = 1 - l/(L+1) for l = 1..L.
  S = (1/T) * [ sum_t g_t g_t' + sum_{l=1..L} w_l * sum_{t=l+1..T} (g_t g_{t-l}' + g_{t-l} g_t') ]
  Var(beta_hat) = T * (X'X)^-1 * S * (X'X)^-1.
  No small-sample df correction (matches statsmodels' use_correction=False).
  At L=0 this reduces exactly to White HC0.
"""

from __future__ import annotations

import math
from copy import deepcopy

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Defaults and validation
# ---------------------------------------------------------------------------

def _apply_defaults(rulebook: dict) -> dict:
    """
    Return a settings dict with all defaults filled in.

    Required key: 'score' (the name of the ranking column).
    Defaults:
      groups          = 5
      weighting       = "by_size"
      signal_lag      = 0
      min_bonds       = groups
      control         = None
      control_groups  = groups
      long_group      = groups - 1   (highest-numbered group)
      short_group     = 0            (lowest-numbered group)
      nw_lags         = None  (None means "use automatic")
      months_per_year = 12
      benchmark_cols  = None (not part of rulebook; benchmark is a separate arg)

    Does not mutate the input.
    """
    if not isinstance(rulebook, dict):
        raise TypeError("rulebook must be a dict")
    if "score" not in rulebook or not isinstance(rulebook["score"], str):
        raise ValueError("rulebook must include a 'score' column name (str)")

    settings: dict = deepcopy(rulebook)
    settings.setdefault("groups", 5)
    settings.setdefault("weighting", "by_size")
    settings.setdefault("signal_lag", 0)
    settings.setdefault("control", None)
    settings.setdefault("nw_lags", None)
    settings.setdefault("months_per_year", 12)

    if not isinstance(settings["groups"], int) or settings["groups"] < 2:
        raise ValueError("'groups' must be an integer >= 2")
    if settings["weighting"] not in ("by_size", "equal"):
        raise ValueError("'weighting' must be 'by_size' or 'equal'")
    if not isinstance(settings["signal_lag"], int) or settings["signal_lag"] < 0:
        raise ValueError("'signal_lag' must be a non-negative integer")
    if not isinstance(settings["months_per_year"], int) or settings["months_per_year"] <= 0:
        raise ValueError("'months_per_year' must be a positive integer")
    if settings["nw_lags"] is not None and (
        not isinstance(settings["nw_lags"], int) or settings["nw_lags"] < 0
    ):
        raise ValueError("'nw_lags' must be None or a non-negative integer")

    settings.setdefault("min_bonds", settings["groups"])
    if not isinstance(settings["min_bonds"], int) or settings["min_bonds"] < 1:
        raise ValueError("'min_bonds' must be a positive integer")

    settings.setdefault("control_groups", settings["groups"])
    if settings["control"] is not None and (
        not isinstance(settings["control_groups"], int) or settings["control_groups"] < 2
    ):
        raise ValueError("'control_groups' must be an integer >= 2 when 'control' is set")

    settings.setdefault("long_group", settings["groups"] - 1)
    settings.setdefault("short_group", 0)
    if not (0 <= settings["long_group"] < settings["groups"]):
        raise ValueError("'long_group' must be in [0, groups)")
    if not (0 <= settings["short_group"] < settings["groups"]):
        raise ValueError("'short_group' must be in [0, groups)")
    if settings["long_group"] == settings["short_group"]:
        raise ValueError("'long_group' and 'short_group' must differ")

    return settings


_RESERVED_COLUMNS = (
    "_realisation_date",
    "_score_obs_date",
    "_score_group",
    "_control_group",
    "ranking_score",
    "next_ret",
)


def _validate_panel(panel: pd.DataFrame, settings: dict) -> None:
    """
    Raise on malformed inputs. Checks required columns, dtypes, that date is
    month-end-normalized, tz-naive, that (bond_id, date) is unique, and that
    the panel does not collide with any of the engine's internal column names.

    Does not mutate the input.
    """
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame")

    required = ["bond_id", "date", "ret", "size", settings["score"]]
    if settings["control"] is not None:
        required.append(settings["control"])
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise ValueError(f"panel missing required columns: {missing}")

    collisions = [c for c in _RESERVED_COLUMNS if c in panel.columns]
    if collisions:
        raise ValueError(
            f"panel uses reserved engine-internal column names: {collisions}. "
            "Rename these columns before passing to the engine."
        )

    if not pd.api.types.is_datetime64_any_dtype(panel["date"]):
        raise TypeError("panel['date'] must be datetime64")
    if getattr(panel["date"].dt, "tz", None) is not None:
        raise ValueError("panel['date'] must be tz-naive")
    month_end = panel["date"] + pd.offsets.MonthEnd(0)
    if not (panel["date"] == month_end).all():
        raise ValueError("panel['date'] must be month-end-normalized")

    if panel.duplicated(subset=["bond_id", "date"]).any():
        raise ValueError("panel has duplicate (bond_id, date) rows")


# ---------------------------------------------------------------------------
# Calendar-strict lagged-panel builder
# ---------------------------------------------------------------------------

def _build_lagged_panel(
    panel: pd.DataFrame, score: str, signal_lag: int
) -> pd.DataFrame:
    """
    Add `ranking_score` and `next_ret` to the panel via calendar-strict joins:

      ranking_score = `score` observed at  date - MonthEnd(signal_lag)  for this bond
      next_ret      = `ret`   observed at  date + MonthEnd(1)           for this bond

    Bonds with a calendar gap at either edge get NaN in the affected column,
    which the eligibility filter in `_month_step` will then drop.

    Always uses pd.offsets.MonthEnd (never DateOffset(months=...)), so day-of-
    month preservation cannot misalign a comparison.
    """
    base = panel.copy()

    # Next-month return: row at t + MonthEnd(1).
    right_next = panel[["bond_id", "date", "ret"]].copy()
    right_next = right_next.rename(columns={"date": "_realisation_date", "ret": "next_ret"})
    base["_realisation_date"] = base["date"] + pd.offsets.MonthEnd(1)
    base = base.merge(right_next, on=["bond_id", "_realisation_date"], how="left")
    base = base.drop(columns=["_realisation_date"])

    # Ranking score: row at t - MonthEnd(signal_lag).
    right_score = panel[["bond_id", "date", score]].copy()
    right_score = right_score.rename(
        columns={"date": "_score_obs_date", score: "ranking_score"}
    )
    base["_score_obs_date"] = base["date"] - pd.offsets.MonthEnd(signal_lag)
    base = base.merge(right_score, on=["bond_id", "_score_obs_date"], how="left")
    base = base.drop(columns=["_score_obs_date"])

    return base


# ---------------------------------------------------------------------------
# Group assignment with deterministic tie-breaking
# ---------------------------------------------------------------------------

def _assign_groups(
    df: pd.DataFrame, value_col: str, id_col: str, n_groups: int
) -> pd.Series:
    """
    Assign each row to one of `n_groups` integer groups (0..n_groups-1) by
    `value_col`, breaking ties deterministically on `id_col`.

    Algorithm:
      1. Stable sort by (value asc, id asc).
      2. Assign rank 1..n in that order.
      3. Group = ((rank - 1) * n_groups) // n.

    Heavy ties may produce slightly uneven group sizes (spec section 5
    accepts this). Two runs on the same input always produce the same
    assignment regardless of original row order.
    """
    n = len(df)
    if n == 0:
        return pd.Series([], dtype=int, index=df.index)
    sorted_idx = df.sort_values([value_col, id_col], kind="mergesort").index
    ranks = np.arange(1, n + 1, dtype=np.int64)
    groups = ((ranks - 1) * n_groups) // n
    result = pd.Series(groups.astype(int), index=sorted_idx)
    return result.reindex(df.index)


# ---------------------------------------------------------------------------
# Leg formation
# ---------------------------------------------------------------------------

def _form_legs(
    stripe_df: pd.DataFrame,
    long_group: int,
    short_group: int,
    weighting: str,
) -> tuple[float, float, float, int]:
    """
    Compute (spread, long_ret, short_ret, n_bonds) for one control stripe.

    Returns (NaN, NaN, NaN, 0) if EITHER the long or the short cell is empty
    -- caller treats this as "skip this stripe".

    weighting:
      "equal"   -> each bond weighted 1/count within its leg
      "by_size" -> each bond weighted size_t / sum(size_t) within its leg
                   (size is the formation-month size; next_ret is the only
                   forward-looking field on the row)

    n_bonds is the long-leg count + short-leg count for this stripe.
    """
    long_df = stripe_df[stripe_df["_score_group"] == long_group]
    short_df = stripe_df[stripe_df["_score_group"] == short_group]
    if len(long_df) == 0 or len(short_df) == 0:
        return (float("nan"), float("nan"), float("nan"), 0)

    if weighting == "equal":
        long_ret = float(long_df["next_ret"].mean())
        short_ret = float(short_df["next_ret"].mean())
    elif weighting == "by_size":
        long_w_sum = float(long_df["size"].sum())
        short_w_sum = float(short_df["size"].sum())
        if long_w_sum <= 0 or short_w_sum <= 0:
            return (float("nan"), float("nan"), float("nan"), 0)
        long_ret = float((long_df["size"] * long_df["next_ret"]).sum() / long_w_sum)
        short_ret = float((short_df["size"] * short_df["next_ret"]).sum() / short_w_sum)
    else:
        raise ValueError(f"Unknown weighting: {weighting!r}")

    return (long_ret - short_ret, long_ret, short_ret, len(long_df) + len(short_df))


# ---------------------------------------------------------------------------
# Per-month step
# ---------------------------------------------------------------------------

def _month_step(
    month_df: pd.DataFrame, settings: dict, bookkeeping: dict
) -> dict | None:
    """
    Run the engine's per-month logic on one formation-month frame.

    Returns the monthly_returns row dict on success, or None to signal "skip
    this month entirely" (too few eligible bonds, or all stripes unusable).

    Mutates `bookkeeping` in place:
      bond_months_dropped_no_next_ret -- counts rows that had ranking_score
                                          and size but no next_ret
      stripes_skipped_by_month        -- dict {formation_date: int}
    """
    has_score = month_df["ranking_score"].notna()
    has_size = month_df["size"].notna()
    has_next = month_df["next_ret"].notna()

    bookkeeping["bond_months_dropped_no_next_ret"] += int(
        ((has_score & has_size) & ~has_next).sum()
    )

    eligibility = has_score & has_size & has_next
    if settings["control"] is not None:
        # A bond with a missing control value cannot be assigned to a stripe;
        # drop it from eligibility rather than silently lumping it into the
        # lowest-numbered stripe via NaN-sort behaviour.
        eligibility = eligibility & month_df[settings["control"]].notna()

    eligible = month_df[eligibility].copy()
    if len(eligible) < settings["min_bonds"]:
        return None

    eligible["_score_group"] = _assign_groups(
        eligible, "ranking_score", "bond_id", settings["groups"]
    )

    if settings["control"] is not None:
        eligible["_control_group"] = _assign_groups(
            eligible, settings["control"], "bond_id", settings["control_groups"]
        )
        stripe_keys = list(range(settings["control_groups"]))
    else:
        eligible["_control_group"] = 0
        stripe_keys = [0]

    stripe_spreads: list[float] = []
    stripe_longs: list[float] = []
    stripe_shorts: list[float] = []
    stripe_nbonds: list[int] = []
    n_stripes_skipped = 0

    for s in stripe_keys:
        stripe = eligible[eligible["_control_group"] == s]
        spread, long_ret, short_ret, nb = _form_legs(
            stripe,
            settings["long_group"],
            settings["short_group"],
            settings["weighting"],
        )
        if math.isnan(spread):
            n_stripes_skipped += 1
            continue
        stripe_spreads.append(spread)
        stripe_longs.append(long_ret)
        stripe_shorts.append(short_ret)
        stripe_nbonds.append(nb)

    if n_stripes_skipped > 0:
        formation_date = month_df["date"].iloc[0]
        bookkeeping["stripes_skipped_by_month"][formation_date] = n_stripes_skipped

    if not stripe_spreads:
        return None

    return {
        "strategy_ret": float(np.mean(stripe_spreads)),
        "long_ret": float(np.mean(stripe_longs)),
        "short_ret": float(np.mean(stripe_shorts)),
        "n_bonds": int(sum(stripe_nbonds)),
    }


# ---------------------------------------------------------------------------
# Newey-West auto-lag + HAC covariance
# ---------------------------------------------------------------------------

def _nw_auto_lags(T: int) -> int:
    """
    Newey-West (1987) automatic lag: floor(4 * (T/100)^(2/9)),
    clamped to [0, max(0, T-1)].
    """
    if T < 2:
        return 0
    raw = math.floor(4.0 * (T / 100.0) ** (2.0 / 9.0))
    return int(max(0, min(raw, T - 1)))


def _nw_hac_variance(
    residuals: np.ndarray, X: np.ndarray, lags: int
) -> np.ndarray:
    """
    Bartlett-kernel HAC covariance for OLS, no small-sample df correction.

      L = min(lags, T - 1)                   (clamped at entry)
      g_t = residuals_t * X_t
      S   = (1/T) [ sum_t g_t g_t' + sum_{l=1..L} w_l (Gamma_l + Gamma_l') ]
            where Gamma_l = (1/T) sum_{t=l+1..T} g_t g_{t-l}'
                  w_l     = 1 - l/(L+1)
      Var(beta_hat) = T * (X'X)^-1 S (X'X)^-1.

    Returns an all-NaN matrix when T <= k or T < 2 (not enough information).

    Clamping is at the function boundary so the kernel denominator (L+1)
    always matches the effective bandwidth -- a caller asking for more lags
    than the sample supports gets the same answer as if they had asked for
    `T-1` directly, rather than an inconsistent truncated-kernel form.

    At lags=0 this reduces exactly to White HC0.
    """
    T, k = X.shape
    if T < 2 or T <= k:
        return np.full((k, k), np.nan)

    L = max(0, min(int(lags), T - 1))
    u = residuals.reshape(-1)
    g = X * u[:, None]  # (T, k)

    S = (g.T @ g) / T
    for ell in range(1, L + 1):
        w = 1.0 - ell / (L + 1)
        Gamma = (g[ell:].T @ g[:-ell]) / T
        S = S + w * (Gamma + Gamma.T)

    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return np.full((k, k), np.nan)
    return T * XtX_inv @ S @ XtX_inv


# ---------------------------------------------------------------------------
# Public: summary statistics
# ---------------------------------------------------------------------------

def summarize_returns(
    returns: pd.Series, nw_lags: int | None, months_per_year: int
) -> dict:
    """
    Standalone summary stats for a monthly long-short return series.

    No risk-free subtraction (long-short already nets it out -- spec section 6).

    Returns:
      average, annualised_average, bumpiness (sample sd, ddof=1),
      sharpe = (mean/sd) * sqrt(months_per_year),
      t_stat, nw_lags_used,
      n_months, first_date, last_date, months_per_year.

    Sharpe / t_stat return NaN when sd == 0 or T < 2 (never raises).

    Callable independently of the sorting machinery (spec section 6 emphasis).
    """
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    s = returns.dropna()
    T = len(s)

    # first_date/last_date are reported only when the caller's index is a
    # DatetimeIndex; for a plain RangeIndex (the spec section 6 "standalone
    # statistics function" use case) we return None rather than leaking an
    # integer that would be silently mis-typed.
    is_dt = isinstance(s.index, pd.DatetimeIndex)
    out: dict = {
        "n_months": T,
        "first_date": s.index.min() if (T > 0 and is_dt) else None,
        "last_date": s.index.max() if (T > 0 and is_dt) else None,
        "months_per_year": int(months_per_year),
    }

    if T == 0:
        out.update(
            average=float("nan"),
            annualised_average=float("nan"),
            bumpiness=float("nan"),
            sharpe=float("nan"),
            t_stat=float("nan"),
            nw_lags_used=0,
        )
        return out

    avg = float(s.mean())
    out["average"] = avg
    out["annualised_average"] = avg * months_per_year

    if T < 2:
        out.update(
            bumpiness=float("nan"),
            sharpe=float("nan"),
            t_stat=float("nan"),
            nw_lags_used=0,
        )
        return out

    sd = float(s.std(ddof=1))
    out["bumpiness"] = sd
    if sd == 0 or math.isnan(sd):
        out["sharpe"] = float("nan")
    else:
        out["sharpe"] = (avg / sd) * math.sqrt(months_per_year)

    requested_L = _nw_auto_lags(T) if nw_lags is None else int(nw_lags)
    # Reflect what the kernel actually used after T-1 clamping inside
    # _nw_hac_variance, so the reported value is honest.
    L = max(0, min(requested_L, T - 1))
    out["nw_lags_used"] = L

    y = s.values.astype(float)
    X = np.ones((T, 1))
    residuals = y - avg
    var = _nw_hac_variance(residuals, X, L)
    v00 = var[0, 0]
    if math.isnan(v00) or v00 <= 0:
        out["t_stat"] = float("nan")
    else:
        out["t_stat"] = avg / math.sqrt(v00)

    return out


# ---------------------------------------------------------------------------
# Public: benchmark regression
# ---------------------------------------------------------------------------

def regress_on_benchmark(
    y: pd.Series, factors: pd.DataFrame, nw_lags: int | None
) -> dict:
    """
    Time-series regression of strategy returns on benchmark factors.

      strategy_ret = alpha + sum_k beta_k * factor_k + residual

    Inner-joins `y` (Series indexed by date) to `factors` (wide DataFrame
    with `date` + one column per factor) on date, drops rows with any NaN,
    fits OLS, reports alpha + betas + NW-HAC t-stats per coefficient using
    the SAME `nw_lags` setting as the strategy luck-check.

    Returns alphas/betas where computable and NaN SEs/t-stats when too few
    overlapping months. Never raises on data shape.
    """
    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series indexed by date")
    if not isinstance(factors, pd.DataFrame):
        raise TypeError("factors must be a pandas DataFrame")
    if "date" not in factors.columns:
        raise ValueError("factors must include a 'date' column")

    factor_cols = [c for c in factors.columns if c != "date"]
    if not factor_cols:
        raise ValueError("factors must include at least one factor column")

    y_df = pd.DataFrame({"date": y.index, "_y": y.values})
    merged = y_df.merge(factors, on="date", how="inner").dropna(
        subset=["_y"] + factor_cols
    )
    T = len(merged)
    k = len(factor_cols) + 1

    nan_result = {
        "alpha": float("nan"),
        "alpha_t": float("nan"),
        "betas": {c: float("nan") for c in factor_cols},
        "beta_t": {c: float("nan") for c in factor_cols},
        "nw_lags_used": 0,
        "n_obs": T,
    }

    if T == 0:
        return nan_result

    requested_L = _nw_auto_lags(T) if nw_lags is None else int(nw_lags)
    L = max(0, min(requested_L, T - 1)) if T >= 2 else 0

    y_arr = merged["_y"].values.astype(float)
    X = np.column_stack(
        [np.ones(T)] + [merged[c].values.astype(float) for c in factor_cols]
    )

    if T <= k:
        # Cannot fit OLS at all.
        return {**nan_result, "nw_lags_used": L}

    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return {**nan_result, "nw_lags_used": L}
    beta_hat = XtX_inv @ X.T @ y_arr
    residuals = y_arr - X @ beta_hat

    out: dict = {
        "alpha": float(beta_hat[0]),
        "betas": {c: float(beta_hat[i + 1]) for i, c in enumerate(factor_cols)},
        "nw_lags_used": L,
        "n_obs": T,
    }

    # HAC standard errors are only meaningful with at least two residual
    # degrees of freedom. With T - k <= 1 the OLS fit is essentially
    # interpolating the data; residuals are numerical noise, not signal.
    # Return NaN SEs rather than t-stats that look meaningful but aren't.
    if T - k <= 1:
        out["alpha_t"] = float("nan")
        out["beta_t"] = {c: float("nan") for c in factor_cols}
        return out

    var = _nw_hac_variance(residuals, X, L)
    diag = np.diag(var)
    ses = np.where(diag > 0, np.sqrt(np.abs(diag)), np.nan)

    out["alpha_t"] = (
        float(beta_hat[0] / ses[0]) if not math.isnan(ses[0]) else float("nan")
    )
    out["beta_t"] = {
        c: (
            float(beta_hat[i + 1] / ses[i + 1])
            if not math.isnan(ses[i + 1])
            else float("nan")
        )
        for i, c in enumerate(factor_cols)
    }
    return out


# ---------------------------------------------------------------------------
# Public: top-level engine
# ---------------------------------------------------------------------------

def run_characteristic_sort(
    panel: pd.DataFrame,
    rulebook: dict,
    safe_rate: pd.DataFrame | None = None,
    benchmark: pd.DataFrame | None = None,
) -> dict:
    """
    Characteristic-sort engine.

    Inputs:
      panel    : DataFrame with bond_id, date (month-end, tz-naive), ret, size,
                 the score column named in rulebook['score'], and -- if the
                 rulebook specifies one -- a control column.
      rulebook : settings dict. See `_apply_defaults` for the supported keys
                 and defaults. Required key: 'score'.
      safe_rate: optional DataFrame with date + rf. Accepted but unused in
                 this build (long-short nets out the safe rate).
      benchmark: optional wide DataFrame with date + one column per factor.

    Returns a dict:
      monthly_returns : DataFrame[date, strategy_ret, long_ret, short_ret,
                                  n_bonds]
                        `date` is the realisation month t+1 (diverges from
                        spec section 4.3 by design).
      summary         : dict from `summarize_returns` + avg_bonds_per_month.
      relationship_to_benchmark : dict from `regress_on_benchmark`; empty if
                                  no benchmark provided.
      settings_used   : the full rulebook with defaults filled in.
      bookkeeping     : months_skipped (list of Timestamps),
                        bond_months_dropped_no_next_ret (int),
                        stripes_skipped_by_month (dict[Timestamp, int]).

    Does not mutate the input.
    """
    settings = _apply_defaults(rulebook)
    _validate_panel(panel, settings)

    if safe_rate is not None:
        # This build produces only long-short strategies; the safe rate cancels
        # out of the spread (spec section 2.2 / section 6). The argument is
        # accepted on the signature so callers can already supply it ahead
        # of a future long-only summary code path, but its shape is
        # validated now so that future code path can rely on the invariant.
        if not isinstance(safe_rate, pd.DataFrame):
            raise TypeError("safe_rate must be a pandas DataFrame when provided")
        missing_rf = {"date", "rf"} - set(safe_rate.columns)
        if missing_rf:
            raise ValueError(
                f"safe_rate must include columns 'date' and 'rf'; "
                f"missing: {sorted(missing_rf)}"
            )

    # Normalise datetime resolution so the calendar-strict self-merge in
    # _build_lagged_panel does not silently miss matches when the input is
    # datetime64[us] / [ms] (pandas 3.0 supports multiple resolutions and
    # MonthEnd arithmetic preserves the source resolution).
    panel = panel.copy()
    panel["date"] = panel["date"].astype("datetime64[ns]")

    work = _build_lagged_panel(panel, settings["score"], settings["signal_lag"])

    bookkeeping: dict = {
        "months_skipped": [],
        "bond_months_dropped_no_next_ret": 0,
        "stripes_skipped_by_month": {},
    }

    rows: list[dict] = []
    for formation_date, month_df in work.groupby("date", sort=True):
        result = _month_step(month_df, settings, bookkeeping)
        if result is None:
            bookkeeping["months_skipped"].append(formation_date)
            continue
        # NOTE: monthly_returns.date carries the realisation month t+1, not
        # the formation month t (diverges from spec section 4.3 by design).
        realisation_date = formation_date + pd.offsets.MonthEnd(1)
        rows.append(
            {
                "date": realisation_date,
                "strategy_ret": result["strategy_ret"],
                "long_ret": result["long_ret"],
                "short_ret": result["short_ret"],
                "n_bonds": result["n_bonds"],
            }
        )

    if rows:
        monthly_returns = pd.DataFrame(rows)
        monthly_returns = monthly_returns.sort_values("date").reset_index(drop=True)
    else:
        # Explicit dtypes so an empty result frame still has datetime64[ns]
        # on `date` and float on the return columns -- downstream consumers
        # can rely on dtype invariants even at zero rows.
        monthly_returns = pd.DataFrame(
            {
                "date": pd.Series(dtype="datetime64[ns]"),
                "strategy_ret": pd.Series(dtype=float),
                "long_ret": pd.Series(dtype=float),
                "short_ret": pd.Series(dtype=float),
                "n_bonds": pd.Series(dtype=int),
            }
        )

    if len(monthly_returns) > 0:
        ret_series = pd.Series(
            monthly_returns["strategy_ret"].values,
            index=pd.DatetimeIndex(monthly_returns["date"].values),
        )
        summary = summarize_returns(
            ret_series, settings["nw_lags"], settings["months_per_year"]
        )
        summary["avg_bonds_per_month"] = float(monthly_returns["n_bonds"].mean())
    else:
        summary = {
            "n_months": 0,
            "first_date": None,
            "last_date": None,
            "months_per_year": settings["months_per_year"],
            "average": float("nan"),
            "annualised_average": float("nan"),
            "bumpiness": float("nan"),
            "sharpe": float("nan"),
            "t_stat": float("nan"),
            "nw_lags_used": 0,
            "avg_bonds_per_month": float("nan"),
        }

    if benchmark is not None and len(monthly_returns) > 0:
        ret_series_for_reg = pd.Series(
            monthly_returns["strategy_ret"].values,
            index=pd.DatetimeIndex(monthly_returns["date"].values),
        )
        relationship = regress_on_benchmark(
            ret_series_for_reg, benchmark, settings["nw_lags"]
        )
    else:
        relationship = {}

    return {
        "monthly_returns": monthly_returns,
        "summary": summary,
        "relationship_to_benchmark": relationship,
        "settings_used": settings,
        "bookkeeping": bookkeeping,
    }
