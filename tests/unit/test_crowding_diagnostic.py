"""
test_crowding_diagnostic.py — unit + known-answer tests for the Layer-1 crowding
diagnostic (shared/evaluation/crowding.py + thresholds.py).

Mirrors the project convention (synthetic known-answer + fail-loud config +
integration round-trip). The spanning statistics themselves are the audited
`regress_on_benchmark` (tested in tests/unit/test_characteristic_sort.py); these
tests prove the crowding orchestration delegates to it faithfully, assembles only
corrected columns, resolves the pre-registered HAC rule, flags short samples, and
slots into EvaluationRecord.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from agents.quant.library.characteristic_sort import regress_on_benchmark
from shared.evaluation.crowding import (
    _resolve_nw_lags,
    crowding_diagnostic,
    load_crowding_factor_bundle,
)
from shared.evaluation.thresholds import (
    CrowdingConfig,
    CrowdingThresholdError,
    load_crowding_config,
)

FACTORS = ("mktb", "drf", "crf", "lrf", "str", "mom6")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _cfg(hac: str = "newey_west_auto", min_obs: int = 10) -> CrowdingConfig:
    """A config for tests that pass `factors=` explicitly (bundles unused)."""
    return CrowdingConfig(
        factor_set=FACTORS,
        hac_lag_rule=hac,
        min_obs=min_obs,
        bundles={f: (f"{f}.parquet", f"{f}_corr") for f in FACTORS},
    )


def _synth_frame(T: int, seed: int) -> pd.DataFrame:
    """Wide factor frame: date + the six logical factors, i.i.d. draws."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-31", periods=T, freq="ME")
    data = {"date": dates}
    for f in FACTORS:
        data[f] = rng.normal(scale=0.02, size=T)
    return pd.DataFrame(data)


def _planted_candidate(
    frame: pd.DataFrame, a0: float, betas: dict[str, float], noise: float, seed: int
) -> pd.Series:
    rng = np.random.default_rng(seed)
    T = len(frame)
    y = np.full(T, a0) + rng.normal(scale=noise, size=T)
    for f, b in betas.items():
        y = y + b * frame[f].to_numpy()
    return pd.Series(y, index=frame["date"])


# ---------------------------------------------------------------------------
# (a) known-answer spanning regression + faithful delegation
# ---------------------------------------------------------------------------

def test_known_answer_and_delegates_to_engine() -> None:
    frame = _synth_frame(T=200, seed=1)
    betas = {"mktb": 0.5, "drf": 0.3, "crf": 0.0, "lrf": 0.0, "str": 0.0, "mom6": 0.0}
    y = _planted_candidate(frame, a0=0.002, betas=betas, noise=0.0005, seed=2)

    res = crowding_diagnostic(y, config=_cfg("newey_west_auto"), factors=frame)

    # Recovers the planted coefficients.
    assert res["alpha"] == pytest.approx(0.002, abs=5e-3)
    assert res["beta_mktb"] == pytest.approx(0.5, abs=5e-3)
    assert res["beta_drf"] == pytest.approx(0.3, abs=5e-3)
    assert res["n_obs"] == 200.0
    # Every factor gets a beta_ and beta_t_ key (fixed set, nothing excluded).
    for f in FACTORS:
        assert f"beta_{f}" in res and f"beta_t_{f}" in res

    # Delegation proof: identical to a direct engine call at the same lag.
    eng = regress_on_benchmark(y, frame, nw_lags=None)
    assert res["alpha"] == pytest.approx(eng["alpha"], rel=1e-12, abs=1e-15)
    assert res["alpha_t"] == pytest.approx(eng["alpha_t"], rel=1e-12, abs=1e-15)
    for f in FACTORS:
        assert res[f"beta_{f}"] == pytest.approx(eng["betas"][f], rel=1e-12, abs=1e-15)


def test_zero_noise_alpha_is_exact() -> None:
    frame = _synth_frame(T=120, seed=5)
    betas = {"mktb": 0.4, "drf": -0.2, "crf": 0.1, "lrf": 0.0, "str": 0.0, "mom6": 0.0}
    y = _planted_candidate(frame, a0=0.003, betas=betas, noise=0.0, seed=6)
    res = crowding_diagnostic(y, config=_cfg("newey_west_auto"), factors=frame)
    assert res["alpha"] == pytest.approx(0.003, abs=1e-10)
    assert res["beta_mktb"] == pytest.approx(0.4, abs=1e-10)
    assert res["beta_drf"] == pytest.approx(-0.2, abs=1e-10)


# ---------------------------------------------------------------------------
# (b) factor assembly: only _corr columns, ns date, inner-join
# ---------------------------------------------------------------------------

def test_assembly_loads_only_corr_and_joins_on_date(tmp_path) -> None:
    # mktb spans more months than the others AND stores date as datetime64[us];
    # includes a *_raw + n_bonds decoy that must never be read.
    m_dates = pd.date_range("2002-01-31", periods=10, freq="ME")
    pd.DataFrame(
        {
            "date": pd.Series(m_dates).astype("datetime64[us]"),
            "mktb_corr": np.arange(10, dtype=float),
            "mktb_raw": np.arange(100, 110, dtype=float),   # decoy
            "n_bonds_corr": np.arange(10),                  # decoy
        }
    ).to_parquet(tmp_path / "mktb.parquet")

    # The other five start two months later -> inner-join drops the first two.
    o_dates = pd.date_range("2002-03-31", periods=8, freq="ME")
    bbw = {"date": o_dates}
    for f in ("drf", "crf", "lrf"):
        bbw[f"{f}_corr"] = np.arange(8, dtype=float)
    bbw["drf_raw"] = np.arange(8, dtype=float)              # decoy
    pd.DataFrame(bbw).to_parquet(tmp_path / "bbw.parquet")
    pd.DataFrame({"date": o_dates, "str_corr": np.arange(8, dtype=float)}).to_parquet(
        tmp_path / "str.parquet"
    )
    pd.DataFrame({"date": o_dates, "mom6_corr": np.arange(8, dtype=float)}).to_parquet(
        tmp_path / "mom6.parquet"
    )

    cfg = CrowdingConfig(
        factor_set=FACTORS,
        hac_lag_rule="floor_t_pow_0.25",
        min_obs=1,
        bundles={
            "mktb": (str(tmp_path / "mktb.parquet"), "mktb_corr"),
            "drf": (str(tmp_path / "bbw.parquet"), "drf_corr"),
            "crf": (str(tmp_path / "bbw.parquet"), "crf_corr"),
            "lrf": (str(tmp_path / "bbw.parquet"), "lrf_corr"),
            "str": (str(tmp_path / "str.parquet"), "str_corr"),
            "mom6": (str(tmp_path / "mom6.parquet"), "mom6_corr"),
        },
    )
    bundle = load_crowding_factor_bundle(cfg)

    assert list(bundle.columns) == ["date", *FACTORS]           # no _raw / n_bonds
    assert str(bundle["date"].dtype) == "datetime64[ns]"        # normalised from us
    assert len(bundle) == 8                                     # inner-join intersection
    assert bundle["date"].min() == o_dates[0]


# ---------------------------------------------------------------------------
# (c) corrected-lattice invariant: a *_raw column is refused
# ---------------------------------------------------------------------------

def test_raw_column_is_rejected() -> None:
    cfg = CrowdingConfig(
        factor_set=("mktb",),
        hac_lag_rule="newey_west_auto",
        min_obs=1,
        bundles={"mktb": ("data/development/factors/mktb.parquet", "mktb_raw")},
    )
    with pytest.raises(ValueError, match="corrected"):
        load_crowding_factor_bundle(cfg)


# ---------------------------------------------------------------------------
# (d) config loader fail-loud + HAC rule resolution
# ---------------------------------------------------------------------------

def _write_yaml(tmp_path, body: str):
    p = tmp_path / "thresholds.yaml"
    p.write_text(body)
    return p


def test_missing_crowding_block_raises(tmp_path) -> None:
    p = _write_yaml(tmp_path, "auditor:\n  primary_metric: sharpe\n")
    with pytest.raises(CrowdingThresholdError):
        load_crowding_config(p)


def test_unknown_hac_rule_raises(tmp_path) -> None:
    body = (
        "crowding:\n"
        "  factor_set: [mktb, drf, crf, lrf, str, mom6]\n"
        "  hac_lag_rule: floor_T_pow_0.25\n"           # the never-implemented spelling
        "  min_obs: 60\n"
        "  bundles:\n"
        + "".join(f"    {f}: {{path: x.parquet, column: {f}_corr}}\n" for f in FACTORS)
    )
    with pytest.raises(CrowdingThresholdError, match="hac_lag_rule"):
        load_crowding_config(_write_yaml(tmp_path, body))


def test_valid_config_loads(tmp_path) -> None:
    body = (
        "crowding:\n"
        "  factor_set: [mktb, drf, crf, lrf, str, mom6]\n"
        "  hac_lag_rule: floor_t_pow_0.25\n"
        "  min_obs: 60\n"
        "  bundles:\n"
        + "".join(f"    {f}: {{path: {f}.parquet, column: {f}_corr}}\n" for f in FACTORS)
    )
    cfg = load_crowding_config(_write_yaml(tmp_path, body))
    assert cfg.hac_lag_rule == "floor_t_pow_0.25"
    assert cfg.min_obs == 60
    assert cfg.factor_set == FACTORS
    assert cfg.bundles["mktb"] == ("mktb.parquet", "mktb_corr")


def test_bundle_missing_factor_source_raises(tmp_path) -> None:
    body = (
        "crowding:\n"
        "  factor_set: [mktb, drf, crf, lrf, str, mom6]\n"
        "  hac_lag_rule: newey_west_auto\n"
        "  min_obs: 60\n"
        "  bundles:\n"
        "    mktb: {path: mktb.parquet, column: mktb_corr}\n"   # only one source
    )
    with pytest.raises(CrowdingThresholdError, match="missing sources"):
        load_crowding_config(_write_yaml(tmp_path, body))


def test_resolve_nw_lags() -> None:
    assert _resolve_nw_lags("newey_west_auto", 209) is None
    assert _resolve_nw_lags("floor_t_pow_0.25", 209) == 3       # floor(209**0.25)=3
    assert _resolve_nw_lags("floor_t_pow_0.25", 16) == 2        # floor(16**0.25)=2
    assert _resolve_nw_lags("floor_t_pow_0.25", 0) == 0


def test_floor_rule_sets_expected_lag() -> None:
    frame = _synth_frame(T=209, seed=9)
    y = _planted_candidate(frame, 0.001, {f: 0.0 for f in FACTORS}, 0.001, seed=10)
    res = crowding_diagnostic(y, config=_cfg("floor_t_pow_0.25"), factors=frame)
    assert res["nw_lags_used"] == 3.0                          # floor(209**0.25)


# ---------------------------------------------------------------------------
# (e) min_obs flag
# ---------------------------------------------------------------------------

def test_below_min_obs_flag() -> None:
    frame = _synth_frame(T=100, seed=11)
    y = _planted_candidate(frame, 0.001, {f: 0.0 for f in FACTORS}, 0.001, seed=12)

    flagged = crowding_diagnostic(y, config=_cfg("newey_west_auto", min_obs=500), factors=frame)
    assert flagged["below_min_obs"] == 1.0
    assert flagged["n_obs"] == 100.0
    assert not math.isnan(flagged["alpha"])                    # still returns, no raise

    ok = crowding_diagnostic(y, config=_cfg("newey_west_auto", min_obs=50), factors=frame)
    assert ok["below_min_obs"] == 0.0


# ---------------------------------------------------------------------------
# (f) integration: mapping slots into EvaluationRecord and round-trips
# ---------------------------------------------------------------------------

def test_slots_into_evaluation_record() -> None:
    # The crowding module has no dependency on the Scientist package; this
    # integration case is skipped if the (currently in-progress) schema is absent,
    # so the committed test tree never hard-requires it.
    pytest.importorskip("agents.scientist.schemas.evaluation")
    from agents.scientist.schemas.evaluation import (
        Booleans,
        EvaluationRecord,
        GrossMeasurements,
        Measurements,
    )

    frame = _synth_frame(T=150, seed=13)
    y = _planted_candidate(frame, 0.002, {"mktb": 0.5}, 0.001, seed=14)
    result = crowding_diagnostic(y, config=_cfg("newey_west_auto"), factors=frame)

    booleans = Booleans(**{name: False for name in Booleans.__dataclass_fields__})
    measurements = Measurements(
        gross=GrossMeasurements(alpha_bbw4=0.004), crowding=result
    )
    record = EvaluationRecord(proposal_id="p1", booleans=booleans, measurements=measurements)

    d = record.to_dict()
    assert d["measurements"]["crowding"] == result             # round-trips intact
    # alpha_bbw4 (the PRIMARY alpha) is a separate field, untouched by crowding.
    assert d["measurements"]["gross"]["alpha_bbw4"] == 0.004
    assert "alpha" in d["measurements"]["crowding"]
    # Whole record is JSON-serialisable (all-float mapping).
    json.dumps(d)


# ---------------------------------------------------------------------------
# (g) degenerate shape passes through as NaN without raising
# ---------------------------------------------------------------------------

def test_degenerate_shape_returns_nan() -> None:
    frame = _synth_frame(T=4, seed=15)                         # T <= k (k = 7)
    y = pd.Series([0.01, 0.02, -0.01, 0.0], index=frame["date"])
    res = crowding_diagnostic(y, config=_cfg("newey_west_auto"), factors=frame)
    assert math.isnan(res["alpha"])
    assert res["n_obs"] == 4.0


def test_no_overlap_returns_nan() -> None:
    frame = _synth_frame(T=50, seed=16)
    # candidate months disjoint from the factor months
    y = pd.Series(
        np.arange(5, dtype=float),
        index=pd.date_range("2050-01-31", periods=5, freq="ME"),
    )
    res = crowding_diagnostic(y, config=_cfg("newey_west_auto"), factors=frame)
    assert res["n_obs"] == 0.0
    assert math.isnan(res["alpha"])
