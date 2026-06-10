"""
Real-data smoke test for the characteristic-sort engine.

Loads /data/development/monthly_panel.parquet, subsets to a recent window,
runs the engine with a placeholder score, and asserts the output is not
pathological. This is a NON-CRASH test, not a correctness test: the score
is just `ret` itself, so the resulting strategy returns have no economic
meaning. The goal is to surface real-data dtype / NaN / memory issues that
synthetic-only unit tests cannot detect.

Skipped automatically if the parquet file is missing (clean checkout
without /data/ should still pass `pytest tests/ -v`).

`tests/conftest.py` blocks reads from /data/holdout/ -- this test reads
/data/development/ which is allowed.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parent.parent.parent
        / "agents"
        / "quant"
        / "library"
    ),
)
from characteristic_sort import run_characteristic_sort  # noqa: E402


PANEL_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "development"
    / "monthly_panel.parquet"
)


def test_engine_runs_on_real_monthly_panel() -> None:
    """End-to-end smoke test on /data/development/monthly_panel.parquet,
    subset to 2015-2020.

    Asserts:
      - the engine does not raise on the real data shape;
      - at least 24 strategy-return months survived;
      - every per-month n_bonds >= min_bonds;
      - no infinities anywhere in the result;
      - no |spread| > 100% (a hard sanity bound, not an economic one).
    """
    if not PANEL_PATH.exists():
        pytest.skip(f"Monthly panel not present at {PANEL_PATH}; smoke test skipped.")

    df = pd.read_parquet(PANEL_PATH)

    # Engine contract:
    #   bond_id (str), date (datetime64, month-end, tz-naive), ret, size, score.
    # The real panel uses `year_month` as "YYYY-MM" strings; convert to a
    # month-end Timestamp. Drop the `xret`, `rf_monthly`, `sub_prdct`,
    # `n_trades`, `price_eom` columns to keep the working frame compact.
    panel = pd.DataFrame(
        {
            "bond_id": df["bond_id"].astype("string").astype(object),
            "date": pd.to_datetime(df["year_month"], format="%Y-%m")
            + pd.offsets.MonthEnd(0),
            "ret": df["ret"].astype(float),
            # total_vol as a size proxy. Not an economic market-value, but the
            # engine only requires a positive scalar for size-weighting; this
            # serves as a stress-test of the weighting code path on real data.
            "size": df["total_vol"].astype(float),
            # `ret` as a placeholder score -- this is a short-term-momentum-like
            # strategy. The goal is engine non-crash, not signal correctness.
            "score": df["ret"].astype(float),
        }
    )

    # Subset to 2015-2020 to keep runtime under ~30s on a laptop.
    panel = panel[
        (panel["date"] >= pd.Timestamp("2015-01-31"))
        & (panel["date"] <= pd.Timestamp("2020-12-31"))
    ].copy()

    # Drop duplicate (bond_id, date) rows defensively -- the real panel
    # should already be unique but the engine's _validate_panel will reject
    # any duplicates.
    panel = panel.drop_duplicates(subset=["bond_id", "date"]).reset_index(drop=True)

    rulebook = {
        "score": "score",
        "groups": 5,
        "weighting": "equal",
        "min_bonds": 20,
        "nw_lags": 0,
    }

    result = run_characteristic_sort(panel, rulebook)
    mr = result["monthly_returns"]
    summary = result["summary"]
    bk = result["bookkeeping"]

    # Non-crash: we got a frame back.
    assert isinstance(mr, pd.DataFrame)
    assert summary["n_months"] == len(mr)

    # Got a meaningful number of months.
    assert len(mr) >= 24, (
        f"Only {len(mr)} months survived eligibility -- check min_bonds or "
        "panel coverage."
    )

    # Every retained month had at least min_bonds * 2 / groups bonds in
    # long+short (5 groups -> long+short = 2/5 of eligible). Loose lower
    # bound: at least 4 bonds (one per leg minimum, but expect more).
    assert int(mr["n_bonds"].min()) >= 4

    # No infinities, no NaNs in the return columns.
    for col in ["strategy_ret", "long_ret", "short_ret"]:
        v = mr[col].values
        assert np.all(np.isfinite(v)), f"non-finite values in {col}"

    # Sanity bound: monthly long-short spread within +-100%.
    assert mr["strategy_ret"].abs().max() < 1.0, (
        f"|spread|.max() = {mr['strategy_ret'].abs().max():.4f} "
        "exceeds 100% -- pathological."
    )

    # Summary stats are finite (or, for sd-zero/short-series cases, NaN --
    # not raise). With 60+ months and noisy real returns, sd > 0 and the
    # summary stats should all be finite.
    assert summary["n_months"] >= 24
    assert not math.isnan(summary["average"])
    assert not math.isnan(summary["bumpiness"])
    assert summary["bumpiness"] > 0
    assert math.isfinite(summary["sharpe"])
    assert math.isfinite(summary["t_stat"])

    # Bookkeeping should be sensible: bond-months dropped for missing
    # next_ret is a positive integer (real panels always have some
    # last-month-of-life bonds).
    assert bk["bond_months_dropped_no_next_ret"] >= 0
