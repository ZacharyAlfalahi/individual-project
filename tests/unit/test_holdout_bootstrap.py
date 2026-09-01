"""SC-SCI-13 holdout block-bootstrap diagnostic — behaviour.

All synthetic data (a wide factor frame + planted return series); nothing here reads
`/data/holdout/`. Pins the pre-registered disclosure numbers (effective-block counts),
CI coverage, the lag-2 pin, determinism, the never-raise contract, and never-confirmatory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agents.quant.library.characteristic_sort import regress_on_benchmark
from shared.stats.holdout_bootstrap import (
    holdout_inference,
    own_alpha_bootstrap,
    paired_difference_bootstrap,
)

FACTORS = ("mktb", "drf", "crf", "lrf")   # BBW-4


def _dates(T: int, start: str = "2022-01-31") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=T, freq="ME")


def _frame(T: int, *, seed: int = 1, start: str = "2022-01-31") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data: dict = {"date": _dates(T, start)}
    for f in FACTORS:
        data[f] = rng.normal(scale=0.02, size=T)
    return pd.DataFrame(data)


def _series(frame: pd.DataFrame, a0: float, betas: dict, *, noise: float = 0.0005, seed: int = 2) -> pd.Series:
    rng = np.random.default_rng(seed)
    T = len(frame)
    y = np.full(T, a0) + rng.normal(scale=noise, size=T)
    for f, b in betas.items():
        y = y + b * frame[f].to_numpy()
    return pd.Series(y, index=frame["date"])


def test_effective_block_counts_exact():
    # The numbers SC-SCI-13 pre-registers on the registered 45-month window:
    # 45/{3,6}=15/7, labelled vs the 10-floor.
    for T, expected in [
        (45, {3: (15, "floor_met"), 6: (7, "below_floor")}),
    ]:
        diff = pd.Series(np.linspace(-0.01, 0.01, T), index=_dates(T))
        cis = paired_difference_bootstrap(
            diff, block_lengths=(3, 6), n_replicates=200, min_effective_blocks=10, seed=7
        )
        by_ell = {c.block_length_months: c for c in cis}
        for ell, (eff, label) in expected.items():
            assert by_ell[ell].effective_blocks == eff, (T, ell)
            assert by_ell[ell].floor_label == label, (T, ell)


def test_paired_ci_covers_known_mean():
    T = 45
    rng = np.random.default_rng(3)
    diff = pd.Series(0.004 + 0.0002 * rng.standard_normal(T), index=_dates(T))
    (ci,) = paired_difference_bootstrap(
        diff, block_lengths=(3,), n_replicates=1000, min_effective_blocks=10, seed=11
    )
    assert ci.point == pytest.approx(float(diff.mean()))
    assert ci.ci_low < 0.004 < ci.ci_high


def test_own_alpha_ci_covers_known_alpha():
    fr = _frame(45, seed=5)
    y = _series(fr, 0.003, {"mktb": 0.5, "drf": 0.3}, noise=0.0005, seed=6)
    (ci,) = own_alpha_bootstrap(
        y, fr, statistic_label="survivor_alpha", block_lengths=(3,),
        n_replicates=1000, min_effective_blocks=10, seed=13,
    )
    assert ci.point == pytest.approx(0.003, abs=5e-4)              # OLS alpha ~ planted a0
    assert ci.point == pytest.approx(regress_on_benchmark(y, fr, 2)["alpha"], abs=1e-9)
    assert ci.ci_low < 0.003 < ci.ci_high


def test_primary_uses_pinned_lag_2():
    fr = _frame(45, seed=8)
    surv = _series(fr, 0.003, {"mktb": 0.5}, seed=9)
    par = _series(fr, 0.001, {"mktb": 0.5}, seed=10)
    full = holdout_inference(
        surv, par, fr,
        n_replicates=100, min_effective_blocks=10, seed=1,
    )
    assert full.nw_lags_used == 2
    # The auto rule would NOT pick 2 at T=45 — proves the pin is doing work, not coinciding.
    assert regress_on_benchmark(surv, fr, None)["nw_lags_used"] != 2


def test_determinism_same_seed():
    fr = _frame(45, seed=8)
    surv = _series(fr, 0.003, {"mktb": 0.5}, seed=9)
    par = _series(fr, 0.001, {"mktb": 0.5}, seed=10)
    kw = dict(n_replicates=300, min_effective_blocks=10)
    a = holdout_inference(surv, par, fr, seed=42, **kw)
    b = holdout_inference(surv, par, fr, seed=42, **kw)
    c = holdout_inference(surv, par, fr, seed=43, **kw)
    assert a.to_dict() == b.to_dict()
    assert a.to_dict() != c.to_dict()


def test_never_raises_on_degenerate():
    for s in (pd.Series(dtype=float), pd.Series([0.01], index=_dates(1)), pd.Series([np.nan] * 10, index=_dates(10))):
        cis = paired_difference_bootstrap(
            s, block_lengths=(3, 6), n_replicates=50, min_effective_blocks=10, seed=1
        )
        assert all(c.ci_low is None for c in cis)                 # T<2 -> no interval, no raise
        assert all(c.floor_label in ("below_floor", "floor_met") for c in cis)
    fr1 = _frame(1, seed=1)
    (ci,) = own_alpha_bootstrap(
        pd.Series([0.01], index=fr1["date"]), fr1, statistic_label="survivor_alpha",
        block_lengths=(3,), n_replicates=50, min_effective_blocks=10, seed=1,
    )
    assert ci.ci_low is None and ci.point is None


def test_is_confirmatory_always_false():
    fr = _frame(45, seed=8)
    surv = _series(fr, 0.003, {"mktb": 0.5}, seed=9)
    par = _series(fr, 0.001, {"mktb": 0.5}, seed=10)
    full = holdout_inference(
        surv, par, fr,
        n_replicates=50, min_effective_blocks=10, seed=1,
    )
    assert all(c.is_confirmatory is False for c in full.bootstrap_cis)


def test_single_registered_window():
    fr = _frame(45, seed=8, start="2022-01-31")   # synthetic 2022-01 .. 2025-09
    surv = _series(fr, 0.003, {"mktb": 0.5}, seed=9)
    par = _series(fr, 0.001, {"mktb": 0.5}, seed=10)
    full = holdout_inference(
        surv, par, fr,
        n_replicates=50, min_effective_blocks=10, seed=1,
    )
    assert full.window_label == "full_45m" and full.n_obs == 45
    assert len(full.bootstrap_cis) == 6                          # 3 statistics x {3, 6}


def test_diagnostic_labels_are_licensed():
    # Every label the diagnostic can surface must avoid the A7 forbidden vocabulary
    # (esp. "statistically significant") — the diagnostic is never a verdict.
    from shared.evaluation.reporting_rules import FORBIDDEN_STRINGS

    vocab = {
        "paired_mean_difference", "survivor_alpha", "parent_alpha",
        "floor_met", "below_floor", "full_45m",
    }
    for label in vocab:
        low = label.lower()
        for bad in FORBIDDEN_STRINGS:
            assert bad not in low, (label, bad)
