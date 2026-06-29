"""
Unit tests for scripts/build_ipca_panel.py (Workstream B IPCA panel feed).

Exercise the pure transform (`align_scale`, `rank_and_emit`, `validate_emitted`) on synthetic
merged input: next-return lag alignment (asof == month-1), VOL-scaling + vol<=0 exclusion,
complete-case selection, adjacency (gaps dropped), small-month drop (N_m <= L), the emitted feed
passing ipca.validate_panel, and the train_end wall. Plus a smoke check on the real artefact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agents.quant.library.ipca import ContractViolation
from build_ipca_panel import DEV, align_scale, rank_and_emit, validate_emitted

INSTR = ["str_reversal", "mom6", "var_5pct", "gamma_illiq", "rating", "time_to_maturity", "bond_vol"]
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


def test_align_scale_asof_and_scaling() -> None:
    rng = np.random.default_rng(0)
    df = _merged(rng)
    out, counts = align_scale(df, REG)
    assert (out["asof"] == out["month"] - 1).all()              # double-lag guard satisfied
    assert (out["bond_vol"] > 0).all()
    # R = next-month excess return / max(instrument-time vol, floor)
    src = df.sort_values(["cusip", "date"]).reset_index(drop=True)
    src["period"] = src["date"].dt.to_period("M")
    g = src.groupby("cusip", sort=False)
    src["next_xret"] = g["str_reversal"].shift(-1)
    src["expR"] = src["next_xret"] / np.maximum(src["bond_vol"].to_numpy(), 0.01)
    merged = out.merge(src[["cusip", "period", "expR"]],
                       left_on=["cusip", "asof"],
                       right_on=["cusip", src["period"].apply(lambda p: p.ordinal)], how="left")
    np.testing.assert_allclose(merged["R"].to_numpy(), merged["expR"].to_numpy(), rtol=0, atol=1e-12)


def test_vol_exclusion_and_complete_case() -> None:
    rng = np.random.default_rng(1)
    df = _merged(rng)
    # vol<=0 for B00 at month index 1 -> that instrument-row excluded.
    df.loc[(df.cusip == "B00") & (df.date == df.date.unique()[1]), "bond_vol"] = -0.01
    # NaN instrument for B01 at month index 2 -> excluded.
    df.loc[(df.cusip == "B01") & (df.date == df.date.unique()[2]), "mom6"] = np.nan
    out, _ = align_scale(df, REG)
    asof1 = pd.Period("2010-02", "M").ordinal
    asof2 = pd.Period("2010-03", "M").ordinal
    assert not ((out.cusip == "B00") & (out.asof == asof1)).any()
    assert not ((out.cusip == "B01") & (out.asof == asof2)).any()


def test_adjacency_gap_dropped() -> None:
    rng = np.random.default_rng(2)
    df = _merged(rng)
    # Drop B00's middle month -> the pair straddling the gap is non-adjacent -> dropped.
    df = df[~((df.cusip == "B00") & (df.date == df.date.unique()[2]))]
    out, _ = align_scale(df, REG)
    b00 = out[out.cusip == "B00"]
    other = out[out.cusip == "B05"]
    assert len(b00) < len(other)   # B00 lost the pairs around the gap


def test_emitted_passes_validate_and_ranges() -> None:
    rng = np.random.default_rng(3)
    out, counts = align_scale(_merged(rng), REG)
    emitted = rank_and_emit(out, counts)
    validate_emitted(emitted, REG, "corr")   # must not raise
    for col in [f"z_{c}" for c in INSTR]:
        assert emitted[col].between(-0.5 - 1e-9, 0.5 + 1e-9).all()
    # per-month max is exactly +0.5 (faithful rank map)
    for _, grp in emitted.groupby("month"):
        for col in [f"z_{c}" for c in INSTR]:
            assert grp[col].max() == pytest.approx(0.5, abs=1e-9)


def test_small_month_dropped() -> None:
    rng = np.random.default_rng(4)
    df = _merged(rng, n_bonds=14)
    # Thin the last return-month to <= L bonds by removing most bonds' final observation.
    last = df.date.unique()[-1]
    keep = [f"B{c:02d}" for c in range(6)]   # only 6 bonds keep the last month -> N_m=6 <= L=8
    df = df[~((df.date == last) & (~df.cusip.isin(keep)))]
    out, counts = align_scale(df, REG)
    rank_and_emit(out, counts)
    assert counts["months_dropped_small"] >= 1


def test_wall_enforced_inclusive_boundary() -> None:
    rng = np.random.default_rng(5)
    out, counts = align_scale(_merged(rng), REG)
    emitted = rank_and_emit(out, counts)
    max_month = int(emitted["month"].max())
    # far before the data -> raises
    early = {"meta": {"default_family": "corr", "train_end": "2009-12"}, "scaler": {"floor": 0.01}}
    with pytest.raises(ContractViolation):
        validate_emitted(emitted, early, "corr")
    # inclusive boundary: train_end == last return month passes; one month earlier raises.
    # (Catches the off-by-one — a +1 in train_end would let max_month slip through.)
    at = {"meta": {"default_family": "corr", "train_end": str(pd.Period(ordinal=max_month, freq="M"))},
          "scaler": {"floor": 0.01}}
    validate_emitted(emitted, at, "corr")
    before = {"meta": {"default_family": "corr", "train_end": str(pd.Period(ordinal=max_month - 1, freq="M"))},
              "scaler": {"floor": 0.01}}
    with pytest.raises(ContractViolation):
        validate_emitted(emitted, before, "corr")


def test_real_artefact_smoke() -> None:
    f = DEV / "ipca_panel_corr.parquet"
    if not f.exists():
        pytest.skip("ipca_panel_corr.parquet not built")
    d = pd.read_parquet(f)
    assert (d["asof"] == d["month"] - 1).all()
    z_cols = [f"z_{c}" for c in INSTR]
    assert not d[z_cols + ["R"]].isna().any().any()
    assert len(d.groupby("month")) >= 1
    assert d["month"].max() <= pd.Period("2021-12", "M").ordinal   # within the wall (inclusive)
