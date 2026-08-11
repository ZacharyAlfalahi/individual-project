"""
ipca_feed.py — the importable, in-memory IPCA characteristic-panel feed builder.

Extracted (verbatim) from ``scripts/build_ipca_panel.py`` so a feed can be built from ANY
already-materialised panel state — the corrected panel P_N, a leave-one-bias-out panel
P_{N\\b}, or a synthetic fixture — without file I/O. The script becomes thin file-I/O glue
over these same pure functions; the on-disk parquet is unchanged (a bit-for-bit regression
is pinned in tests). Spec: docs/quant/specs/characteristic_registry_spec.md; family policy
(A9): price-derived instruments carry _raw/_corr, rating & maturity are agnostic.

Pipeline (all pure, deterministic, no I/O):
  1. ``align_scale``   — next-return alignment (instruments at m-1, return at m, adjacent
                         only), VOL-scaling R = xret_m / max(bond_vol_{m-1}, floor), and
                         complete-case selection on the 7 instruments + R.
  2. ``rank_and_emit`` — per return-month cross-sectional ``ipca.rank_transform`` of each
                         instrument; drop months with N_m <= L; emit the long z_* frame.
  3. ``validate_feed`` — round-trip the emitted feed through ``ipca.validate_panel`` + wall.
  4. ``feed_matrices`` — stack the long frame into the per-month (Z, R, months, asof, vol,
                         cusips) matrices the estimator consumes (constant appended LAST).

``build_ipca_feed`` composes 1-3; ``feed_matrices`` does 4. All category-2 "deterministic
per-cell recomputation" (spec §4.1) — closed-form, parameter-free, point-in-time.

This is a NEW module (not a modification of the audited runner ``ipca.py``, which is
imported, never touched).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from .ipca import assert_within_wall, rank_transform, validate_panel

# Instrument order is fixed (constant is appended LAST at matrix-build time, so L = 8).
INSTRUMENTS: list[str] = [
    "str_reversal",
    "mom6",
    "var_5pct",
    "gamma_illiq",
    "rating",
    "time_to_maturity",
    "bond_vol",
]
L: int = len(INSTRUMENTS) + 1


# ---------------------------------------------------------------------------
# Pure transforms (moved verbatim from scripts/build_ipca_panel.py)
# ---------------------------------------------------------------------------


def align_scale(df: pd.DataFrame, reg: dict) -> tuple[pd.DataFrame, dict]:
    """Next-return alignment (instruments at m-1, return at m, adjacent only), VOL-scaling,
    and complete-case selection. ``df`` must carry cusip, date, the 7 instrument columns
    (str_reversal = instrument-time excess return), and bond_vol. Unit-testable, no I/O."""
    floor = float(reg["scaler"]["floor"])
    counts = {"rows_merged": int(len(df))}

    df = df.sort_values(["cusip", "date"], kind="mergesort").reset_index(drop=True)
    df["period"] = df["date"].dt.to_period("M")
    g = df.groupby("cusip", sort=False)
    df["next_xret"] = g["str_reversal"].shift(-1)   # str_reversal IS xret at instrument-time t
    df["next_period"] = g["period"].shift(-1)
    df["next_date"] = g["date"].shift(-1)

    # next-return convention: keep only adjacent month pairs (return at t+1, instruments at t)
    adj = df["next_period"] == (df["period"] + 1)
    df = df[adj].copy()
    counts["rows_adjacent"] = int(len(df))

    # VOLScaled010: exclude bond_vol<=0, then scale next-month return by the instrument-time vol
    df = df[df["bond_vol"] > 0].copy()
    counts["rows_vol_positive"] = int(len(df))
    df["below_floor"] = df["bond_vol"] <= floor
    df["R"] = df["next_xret"] / np.maximum(df["bond_vol"].to_numpy(), floor)

    df["month"] = df["period"].apply(lambda p: p.ordinal + 1)   # return-month ordinal
    df["asof"] = df["period"].apply(lambda p: p.ordinal)        # instrument month = month - 1
    df["ret_date"] = df["next_date"]

    # complete-case on the 7 instruments + R
    df = df.dropna(subset=[*INSTRUMENTS, "R"]).reset_index(drop=True)
    counts["rows_complete_case"] = int(len(df))
    counts["below_floor_share"] = float(df["below_floor"].mean()) if len(df) else float("nan")
    return df, counts


def rank_and_emit(df: pd.DataFrame, counts: dict) -> pd.DataFrame:
    """Per return-month: rank_transform each instrument, drop months with N_m <= L, emit long
    frame. ``counts`` is updated in place with the month/row bookkeeping."""
    blocks: list[pd.DataFrame] = []
    dropped_small: list[int] = []
    n_per_month: dict[int, int] = {}
    for month, grp in df.groupby("month", sort=True):
        if len(grp) <= L:
            dropped_small.append(int(month))
            continue
        block = pd.DataFrame({
            "cusip": grp["cusip"].to_numpy(),
            "month": grp["month"].to_numpy(),
            "asof": grp["asof"].to_numpy(),
            "ret_date": grp["ret_date"].to_numpy(),
            "R": grp["R"].to_numpy(dtype=float),
            "vol_scaler": grp["bond_vol"].to_numpy(dtype=float),   # raw scaler (VOL-lane diagnostics)
        })
        for col in INSTRUMENTS:
            block[f"z_{col}"] = rank_transform(grp[col].to_numpy(dtype=float))
        blocks.append(block)
        n_per_month[int(month)] = int(len(grp))
    if not blocks:
        raise RuntimeError("no months survived the N_m > L filter — check inputs")
    out = pd.concat(blocks, ignore_index=True)
    counts["months_kept"] = len(n_per_month)
    counts["months_dropped_small"] = len(dropped_small)
    counts["N_m_min"] = int(min(n_per_month.values()))
    counts["N_m_median"] = int(np.median(list(n_per_month.values())))
    counts["N_m_max"] = int(max(n_per_month.values()))
    counts["rows_emitted"] = int(len(out))
    return out


def validate_feed(out: pd.DataFrame, reg: dict, family: str) -> None:
    """Round-trip: the emitted Z must pass the module's own receipt check + wall.

    The wall train_end is the ordinal of the last in-window RETURN month (inclusive); ``month``
    is the return-month period ordinal, so this rejects any return beyond reg.meta.train_end.
    Raises ``ipca.ContractViolation`` on any violation."""
    z_cols = [f"z_{c}" for c in INSTRUMENTS]
    Z: list[np.ndarray] = []
    R: list[np.ndarray] = []
    months: list[int] = []
    asof: list[int] = []
    for month, grp in out.groupby("month", sort=True):
        zmat = grp[z_cols].to_numpy(dtype=float)
        zmat = np.column_stack([zmat, np.ones(len(grp))])   # constant appended last
        Z.append(zmat)
        R.append(grp["R"].to_numpy(dtype=float))
        months.append(int(month))
        asof.append(int(grp["asof"].iloc[0]))
    validate_panel(Z, R, np.asarray(months), L=L, family=family,
                   characteristic_asof=np.asarray(asof))
    train_end = pd.Period(reg["meta"]["train_end"], "M").ordinal   # inclusive last return month
    assert_within_wall(np.asarray(months), train_end=train_end)


# ---------------------------------------------------------------------------
# Composition + matrix stacking (the importable entry points)
# ---------------------------------------------------------------------------


class IPCAFeed(NamedTuple):
    """Per-month estimator inputs stacked from the emitted long frame. Months are ascending.
    The constant column is appended LAST in each Z[m] (so L = 7 instruments + 1)."""

    Z: list[np.ndarray]            # per-month (N_m, L), rank-transformed, constant LAST
    R: list[np.ndarray]            # per-month (N_m,), VOL-scaled excess returns
    months: np.ndarray             # (T,) return-month period ordinals, ascending
    asof: np.ndarray               # (T,) = months - 1 (instrument month)
    vol_scaler: list[np.ndarray]   # per-month (N_m,) raw bond_vol (VOL-lane diagnostics)
    cusips: list[np.ndarray]       # per-month (N_m,) cusip ids (membership diagnostics)


def build_ipca_feed(
    merged: pd.DataFrame, reg: dict, family: str, *, validate: bool = True
) -> tuple[pd.DataFrame, dict]:
    """Build the emitted long feed frame from an already-merged, in-memory panel state.

    ``merged`` carries cusip, date, and the 7 canonical instrument columns (str_reversal =
    instrument-time excess return, plus mom6, var_5pct, gamma_illiq, rating, time_to_maturity,
    bond_vol). Returns ``(out_frame, counts)``. When ``validate`` (default), the emitted feed is
    round-tripped through ``validate_feed``. No file I/O; deterministic."""
    df, counts = align_scale(merged, reg)
    out = rank_and_emit(df, counts)
    if validate:
        validate_feed(out, reg, family)
    return out, counts


def feed_matrices(out: pd.DataFrame, *, n_instruments_plus_const: int = L) -> IPCAFeed:
    """Stack the emitted long frame into per-month (Z, R, months, asof, vol, cusips) matrices.

    The constant column (≡ 1) is appended LAST in each Z[m]. This mirrors the shakedown's
    ``load_feed`` so the estimator sees byte-identical inputs whether the feed came from disk or
    from an in-memory panel state."""
    z_cols = [f"z_{c}" for c in INSTRUMENTS]
    Z: list[np.ndarray] = []
    R: list[np.ndarray] = []
    months: list[int] = []
    asof: list[int] = []
    vol: list[np.ndarray] = []
    cusips: list[np.ndarray] = []
    for month, grp in out.groupby("month", sort=True):
        zmat = grp[z_cols].to_numpy(dtype=float)
        zmat = np.column_stack([zmat, np.ones(len(grp))])
        Z.append(zmat)
        R.append(grp["R"].to_numpy(dtype=float))
        months.append(int(month))
        asof.append(int(grp["asof"].iloc[0]))
        vol.append(grp["vol_scaler"].to_numpy(dtype=float))
        cusips.append(grp["cusip"].to_numpy())
    return IPCAFeed(
        Z=Z, R=R, months=np.asarray(months), asof=np.asarray(asof),
        vol_scaler=vol, cusips=cusips,
    )


def load_feed(path):
    """Read a materialised IPCA feed parquet from disk and stack it into per-month
    (Z, R, months, asof, vol) matrices — the thin on-disk loader the shakedown and the one-shot holdout
    dev-pseudo builder share (lifted verbatim from scripts/run_ipca_shakedown.py). The pure
    in-memory equivalent is ``feed_matrices``; this is the ONLY I/O in this module. The constant
    column (≡ 1) is appended LAST in each Z[m], byte-identical to ``feed_matrices``."""
    d = pd.read_parquet(path)
    z_cols = [f"z_{c}" for c in INSTRUMENTS]
    Z, R, months, asof, vol = [], [], [], [], []
    for month, grp in d.groupby("month", sort=True):
        zmat = np.column_stack([grp[z_cols].to_numpy(dtype=float), np.ones(len(grp))])
        Z.append(zmat)
        R.append(grp["R"].to_numpy(dtype=float))
        months.append(int(month))
        asof.append(int(grp["asof"].iloc[0]))
        vol.append(grp["vol_scaler"].to_numpy(dtype=float))
    return Z, R, np.asarray(months), np.asarray(asof), np.concatenate(vol)
