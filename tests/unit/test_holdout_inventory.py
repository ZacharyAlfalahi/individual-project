"""Unit tests for the seeded holdout inventory builder's GATE and format (scripts/holdout_inventory).

These are fast, holdout-free tests of the safety gate and the runner's gated real path. The heavy
correctness proof (exact reproduction of the recorded dev maximal + 4 signals) is the
``--selfcheck`` integration run, not a unit test — same split as the descriptive runner's rehearsal.
NONE of these tests reads or lists /data/holdout/.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import holdout_inventory as H
import run_holdout_oos_descriptive as R


def _tiny_daily(dates):
    n = len(dates)
    return pd.DataFrame({"cusip_id": ["A"] * n, "trd_exctn_dt": pd.to_datetime(dates),
                         "price_vwap": [100.0] * n, "total_vol": [1000] * n, "n_trades": [1] * n})


def _close_gate(monkeypatch):
    monkeypatch.delenv(H._GATE_ENV, raising=False)


def test_gate_closed_by_default(monkeypatch):
    _close_gate(monkeypatch)
    with pytest.raises(H.HoldoutGateError, match="REFUSED"):
        H.load_holdout_inputs(holdout_daily_dir=Path("/nonexistent/holdout"), gate=None)


def test_gate_wrong_token_refuses(monkeypatch):
    monkeypatch.setenv(H._GATE_ENV, "1")   # even with the env set, a wrong/absent token refuses
    with pytest.raises(H.HoldoutGateError):
        H.load_holdout_inputs(holdout_daily_dir=Path("/nonexistent/holdout"), gate="not-the-token")


def test_gate_requires_env_even_with_token(monkeypatch):
    _close_gate(monkeypatch)                # correct token but env unset -> still refuses
    with pytest.raises(H.HoldoutGateError):
        H.load_holdout_inputs(holdout_daily_dir=Path("/nonexistent/holdout"), gate=H._GATE_TOKEN)


def test_require_gate_passes_only_with_both(monkeypatch):
    # Pure gate logic — reads nothing. Both locks required.
    _close_gate(monkeypatch)
    with pytest.raises(H.HoldoutGateError):
        H._require_gate(H._GATE_TOKEN)
    monkeypatch.setenv(H._GATE_ENV, "1")
    H._require_gate(H._GATE_TOKEN)          # token + env -> no raise
    with pytest.raises(H.HoldoutGateError):
        H._require_gate(None)


def test_gamma_rename_matches_runner_format():
    # The inventory must present gamma as the canonical gamma_illiq_* so the runner's negative
    # control (which renames back to gamma_*) resolves — mirrors load_dev_signals.
    assert H._GAMMA_RENAME == {"gamma_raw": "gamma_illiq_raw", "gamma_corr": "gamma_illiq_corr"}


def test_runner_real_refuses_without_gate(monkeypatch, capsys):
    # main(--real) must refuse (exit 2) and read no holdout when the gate env is unset.
    monkeypatch.delenv("HOLDOUT_OOS_GATE", raising=False)
    monkeypatch.delenv(H._GATE_ENV, raising=False)
    rc = R.main(["--real"])
    assert rc == 2
    assert "REFUSED" in capsys.readouterr().err


def test_profile_families_supported_and_wired():
    # drf/mom6 as-published are buildable: the flag is True and the
    # assembly functions exist, so run_real's pre-open guard does not block (the gate is the only
    # barrier). Reproduction of the profile artefacts is proven by --selfcheck.
    assert H.PROFILE_FAMILIES_SUPPORTED is True
    assert H.PROFILE_IDS == ("bbw_2019", "jostova_2013")
    assert H._PROFILE_DAILY_NAMES == {
        "bbw_2019": "trace_daily_bbw_2019__dedup_on.parquet",
        "jostova_2013": "trace_daily_jostova_2013__dedup_on.parquet",
    }
    assert callable(H.assemble_profile_monthly) and callable(H.assemble_profile_signals)


def test_fisd_holdout_layer_supported_and_wired():
    # FISD is built OVER THE PANEL GRID from the shared data/fisd/ source (holdout months get
    # their as-of rating + static size), so drf/lrf resolve on the holdout. Flag True + the grid
    # merge function exists. Exact dev reproduction of size/rating/universe/exit_reason is proven
    # by --selfcheck.
    assert H.FISD_HOLDOUT_LAYER_SUPPORTED is True
    assert callable(H._merge_fisd_grid)


def test_concat_daily_seam_assert(tmp_path, monkeypatch):
    monkeypatch.setenv(H._GATE_ENV, "1")   # not a holdout read — synthetic parquets under tmp_path
    devp, holdp = tmp_path / "dev.parquet", tmp_path / "hold.parquet"
    _tiny_daily(["2021-11-15", "2021-12-20"]).to_parquet(devp)
    _tiny_daily(["2022-01-10", "2022-02-11"]).to_parquet(holdp)
    # clean split: dev < 2022-01, holdout >= 2022-01 -> OK
    H._concat_daily(devp, holdp, tmp_path / "ok.parquet")
    # mis-split A: a 2022 trade leaked into the dev daily -> raises before any concat
    _tiny_daily(["2021-12-20", "2022-01-05"]).to_parquet(devp)
    with pytest.raises(RuntimeError, match="mis-split"):
        H._concat_daily(devp, holdp, tmp_path / "bad1.parquet")
    # mis-split B: a 2021 trade leaked into the holdout daily -> raises
    _tiny_daily(["2021-11-15", "2021-12-20"]).to_parquet(devp)
    _tiny_daily(["2021-12-30", "2022-01-10"]).to_parquet(holdp)
    with pytest.raises(RuntimeError, match="mis-split"):
        H._concat_daily(devp, holdp, tmp_path / "bad2.parquet")


def test_pinned_as_published_baselines_are_frozen():
    # Change-detector for the frozen dev as-published / negative-control baselines (the values
    # the seeded --real run self-verifies against). A change here must be deliberate; the
    # corrected baselines get a stronger check in test_pinned_baselines_match_recorded_artifacts.
    assert R.PINNED_AS_PUBLISHED_IN_SAMPLE == {
        "str": 0.007949651424605457,
        "drf": 0.0034386749221172516,
        "mom6": 0.006478265466729551,
    }
    assert R.PINNED_NEG_CONTROL_IN_SAMPLE == 7.76033181823128e-05
