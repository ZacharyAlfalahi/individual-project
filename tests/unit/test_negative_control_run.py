"""Negative-control RUN path (§10.3) — the corrected-vs-uncorrected differential → bootstrap CI →
separated-mode gate. Offline/synthetic for the compute; the dev-data end-to-end is skip-gated.

The point of a negative control is that it CAN fail: a clean differential passes (CI within ±ϑ), a
planted divergence fails. Both are pinned here so the "cannot fail" defect stays fixed at the run
level too."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.auditor.checks.bootstrap import BootstrapError  # noqa: E402
from agents.auditor.validation.hypothesis_registry import (  # noqa: E402
    FactorHypothesis,
    HypothesisRegistryError,
)
from agents.auditor.validation.negative_control_run import (  # noqa: E402
    NegativeControlRunError,
    compute_negative_control,
    run_negative_control_from_dev,
)

VARTHETA = 0.001
_BOOT = dict(n_replicates=400, block_length_months=6, min_effective_blocks=10)


def _hyp(mode: str = "separated") -> FactorHypothesis:
    return FactorHypothesis(
        factor_id="traded_liquidity", dominant_bias="none", expected_sign=0,
        magnitude_mode=mode, expected_magnitude_range=None, is_locked=True,
        status="negative_control", source="s", registered_commit="abc")


def _series(vals, start="2005-01-31"):
    return pd.Series(vals, index=pd.date_range(start, periods=len(vals), freq="ME"))


# --- the specificity result CAN pass and CAN fail --------------------------------------------

def test_clean_differential_passes_within_band():
    rng = np.random.default_rng(0)
    base = _series(rng.normal(0.001, 0.008, 132))
    corrected = base + rng.normal(0, 0.0001, 132)          # near-identical → tiny gap
    v, meta = compute_negative_control(base, corrected, hyp=_hyp(), vartheta=VARTHETA, seed=0, **_BOOT)
    assert v.passed is True
    assert -VARTHETA < v.ci_low <= v.ci_high < VARTHETA
    assert meta["n_months_common"] == 132 and meta["effective_blocks"] >= 10


def test_planted_divergence_fails():
    rng = np.random.default_rng(1)
    base = _series(rng.normal(0.001, 0.008, 132))
    corrected = base + 0.003                                # +30 bp/mo, far beyond ±ϑ
    v, _ = compute_negative_control(base, corrected, hyp=_hyp(), vartheta=VARTHETA, seed=0, **_BOOT)
    assert v.passed is False and v.absolute_gap > VARTHETA


def test_gap_direction_is_corrected_minus_uncorrected():
    # tripwire: gap = corrected - uncorrected. A +30bp planted shift on the CORRECTED arm must
    # read POSITIVE; if the arguments were ever swapped, |gap| would be unchanged but the SIGN
    # (and point_gap_mean, and the reported CI) would flip silently.
    rng = np.random.default_rng(3)
    base = _series(rng.normal(0.0, 0.006, 132))
    v, meta = compute_negative_control(base, base + 0.003, hyp=_hyp(), vartheta=VARTHETA, seed=0, **_BOOT)
    assert meta["point_gap_mean"] > 0 and v.ci_low > 0        # corrected raised → positive gap
    v2, meta2 = compute_negative_control(base + 0.003, base, hyp=_hyp(), vartheta=VARTHETA, seed=0, **_BOOT)
    assert meta2["point_gap_mean"] < 0 and v2.ci_high < 0     # swapped → negative gap


def test_deterministic_same_seed():
    rng = np.random.default_rng(2)
    base = _series(rng.normal(0.001, 0.008, 132))
    corrected = base + rng.normal(0, 0.0002, 132)
    a, _ = compute_negative_control(base, corrected, hyp=_hyp(), vartheta=VARTHETA, seed=5, **_BOOT)
    b, _ = compute_negative_control(base, corrected, hyp=_hyp(), vartheta=VARTHETA, seed=5, **_BOOT)
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)


# --- fail-loud guards ------------------------------------------------------------------------

def test_short_series_triggers_bootstrap_refusal():
    base = _series(np.zeros(30))                            # 30 months, block 6 → 5 blocks < 10
    with pytest.raises(BootstrapError):
        compute_negative_control(base, base + 0.0001, hyp=_hyp(), vartheta=VARTHETA, seed=0, **_BOOT)


def test_non_separated_hyp_is_rejected_by_the_gate():
    base = _series(np.linspace(-0.01, 0.01, 132))
    with pytest.raises(HypothesisRegistryError, match="separated"):
        compute_negative_control(base, base, hyp=_hyp(mode="near_zero"), vartheta=VARTHETA,
                                 seed=0, **_BOOT)


# --- dev-data end-to-end (skip if the panel is absent) ---------------------------------------

def _dev_present() -> bool:
    return (REPO_ROOT / "data" / "development" / "monthly_panel_maximal.parquet").is_file() or \
        (REPO_ROOT / "data" / "development" / "monthly_panel_total_return.parquet").is_file()


@pytest.mark.skipif(not _dev_present(), reason="dev panel not present")
def test_run_from_dev_produces_a_real_verdict():
    r = run_negative_control_from_dev(seed=0)
    assert r["basis"] == "real_dev_data" and r["status"] == "fired"
    assert isinstance(r["passed"], bool)
    assert r["ci_low"] <= r["ci_high"]
    assert r["bootstrap"]["n_months_common"] > 60
    assert r["estimand"]["run_once"] is True
    # never a NegativeControlRunError leaking for the registered control
    assert r["factor"] == "traded_liquidity"


def test_run_from_dev_error_type_exists():
    assert issubclass(NegativeControlRunError, RuntimeError)
