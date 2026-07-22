"""
The importable in-memory feed builder ``agents/quant/library/ipca_feed.py``.

Two jobs: (1) the pure composition ``build_ipca_feed`` + the matrix stacker ``feed_matrices``
behave correctly on a small in-memory frame; (2) the refactor is transparent — rebuilding the
real corrected feed through the relocated functions reproduces the committed
``ipca_panel_corr.parquet`` bit-for-bit (skipped when the dev inputs are absent).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agents.quant.library.ipca import ContractViolation
from agents.quant.library.ipca_feed import (
    INSTRUMENTS,
    L,
    build_ipca_feed,
    feed_matrices,
)

REG = {"meta": {"default_family": "corr", "train_end": "2010-12"}, "scaler": {"floor": 0.01}}


def _merged(rng: np.random.Generator, n_bonds: int = 14, n_months: int = 5) -> pd.DataFrame:
    dates = pd.date_range("2010-01-31", periods=n_months, freq="ME")
    rows = []
    for c in range(n_bonds):
        for d in dates:
            rows.append({
                "cusip": f"B{c:02d}", "date": d,
                "str_reversal": float(rng.normal() * 0.02),
                "mom6": float(rng.normal()),
                "var_5pct": float(rng.normal()),
                "gamma_illiq": float(rng.normal()),
                "rating": float(rng.integers(1, 22)),
                "time_to_maturity": float(rng.uniform(1.0, 20.0)),
                "bond_vol": float(rng.uniform(0.005, 0.05)),
            })
    return pd.DataFrame(rows)


def test_build_ipca_feed_emits_valid_frame():
    rng = np.random.default_rng(7)
    out, counts = build_ipca_feed(_merged(rng), REG, "corr")   # validate=True must not raise
    assert (out["asof"] == out["month"] - 1).all()
    assert set(f"z_{c}" for c in INSTRUMENTS).issubset(out.columns)
    assert counts["rows_emitted"] == len(out)
    assert counts["months_kept"] >= 1


def test_build_ipca_feed_validate_flag_defers_receipt_check():
    """validate=False returns the frame without the wall check; an out-of-wall feed only raises
    when validated."""
    rng = np.random.default_rng(8)
    early = {"meta": {"default_family": "corr", "train_end": "2009-12"}, "scaler": {"floor": 0.01}}
    out, _ = build_ipca_feed(_merged(rng), early, "corr", validate=False)   # no raise
    assert len(out) > 0
    with pytest.raises(ContractViolation):
        build_ipca_feed(_merged(rng), early, "corr", validate=True)


def test_feed_matrices_shapes_and_constant_last():
    rng = np.random.default_rng(9)
    out, _ = build_ipca_feed(_merged(rng), REG, "corr")
    feed = feed_matrices(out)
    T = len(feed.Z)
    assert T == len(feed.R) == len(feed.months) == len(feed.asof) == len(feed.cusips) == len(feed.vol_scaler)
    assert (feed.asof == feed.months - 1).all()
    assert list(feed.months) == sorted(feed.months)      # ascending
    for m in range(T):
        n = feed.Z[m].shape[0]
        assert feed.Z[m].shape == (n, L)
        np.testing.assert_array_equal(feed.Z[m][:, -1], np.ones(n))   # constant appended LAST
        assert feed.R[m].shape == (n,)
        assert feed.cusips[m].shape == (n,)
        # ranked instrument columns live in [-0.5, +0.5], per-month max exactly +0.5
        cols = feed.Z[m][:, :-1]
        assert cols.min() >= -0.5 - 1e-9 and cols.max() <= 0.5 + 1e-9


def test_feed_matrices_matches_disk_loader_convention():
    """feed_matrices stacks exactly as scripts/run_ipca_shakedown.load_feed does (constant last,
    R and asof from the same groups), so the estimator sees identical inputs from disk or memory."""
    rng = np.random.default_rng(10)
    out, _ = build_ipca_feed(_merged(rng), REG, "corr")
    feed = feed_matrices(out)
    z_cols = [f"z_{c}" for c in INSTRUMENTS]
    for i, (month, grp) in enumerate(out.groupby("month", sort=True)):
        expected_Z = np.column_stack([grp[z_cols].to_numpy(float), np.ones(len(grp))])
        np.testing.assert_array_equal(feed.Z[i], expected_Z)
        np.testing.assert_array_equal(feed.R[i], grp["R"].to_numpy(float))
        assert feed.months[i] == int(month)


# ---------------------------------------------------------------------------
# Bit-for-bit regression: the relocated functions reproduce the committed parquet.
# Skipped when the dev inputs are absent (e.g. CI without the 153MB maximal panel).
# ---------------------------------------------------------------------------

def test_refactor_reproduces_committed_parquet_bitforbit():
    import importlib

    bip = importlib.import_module("build_ipca_panel")
    committed = bip.DEV / "ipca_panel_corr.parquet"
    inputs = [
        bip.PANEL_FILE,
        bip.DEV / "signals" / "mom6.parquet",
        bip.DEV / "signals" / "var_5pct.parquet",
        bip.DEV / "signals" / "gamma_illiq.parquet",
        bip.DEV / "signals" / "bond_vol.parquet",
    ]
    if not committed.exists() or not all(p.exists() for p in inputs):
        pytest.skip("dev inputs / committed ipca_panel_corr.parquet not present")

    reg = bip.load_registry()
    merged = bip.assemble("corr", reg)
    rebuilt, _ = build_ipca_feed(merged, reg, "corr")
    expected = pd.read_parquet(committed)
    # Column order + dtypes + values must match exactly (the parquet is the frame `out`).
    pd.testing.assert_frame_equal(
        rebuilt.reset_index(drop=True), expected.reset_index(drop=True),
        check_like=False, check_dtype=True,
    )
