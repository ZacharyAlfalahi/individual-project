"""Panel-transform executor (F8, incl. the expanding-window-not-full-sample look-ahead test),
G1b execution-fidelity (runs the real audited engine on the transformed panel), and G3 inference
(alpha vs BBW-4 + BH-FDR) — all on a synthetic panel with a known answer."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.scientist.experimentalist.compiler import CompiledExtension, PanelTransform  # noqa: E402
from agents.scientist.experimentalist.execution_verifier import execute_g1b  # noqa: E402
from agents.scientist.experimentalist.inference import inference_g3, two_sided_p  # noqa: E402
from agents.scientist.experimentalist.panel_transform import (  # noqa: E402
    apply_row_filter,
    expanding_regime_mask,
)
from agents.scientist.schemas.outcomes import RefusalCode  # noqa: E402

MONTHS = pd.date_range("2010-01-31", periods=24, freq="ME")


def _panel(n_bonds=20):
    rows = []
    for b in range(n_bonds):
        for i, d in enumerate(MONTHS):
            spread = 0.001 + 0.0003 * (i % 5)                    # time-varying -> sd(strat)>0
            rows.append(dict(cusip=f"B{b:02d}", date=d, ret=spread * (b - (n_bonds - 1) / 2),
                             size=100.0, score=float(b), investment_grade=(b >= n_bonds // 2),
                             rating=float((b * 7) % n_bonds),   # permutation -> independent of score
                             gamma_illiq=float(n_bonds - b)))
    return pd.DataFrame(rows)


_BASE_RULEBOOK = {"score": "score", "groups": 5, "weighting": "equal", "min_bonds": 4,
                  "signal_lag": 0, "nw_lags": 0}


def _ext(mode, *, control=None, transform=None):
    return CompiledExtension("p1", "str", mode, "expanding_window_past_only", "qc://c",
                             control=control, control_groups=2 if control else None,
                             panel_transform=transform)


# ---- panel-transform executor (F8) --------------------------------------------------------

def test_expanding_median_is_not_full_sample_median():
    # A rising regime: full-sample "above median" flags only the late half; the EXPANDING
    # (past-only) median flags a month as soon as it exceeds its own history — no look-ahead.
    macro = pd.Series(range(24), index=MONTHS, dtype=float)
    expanding = expanding_regime_mask(macro, lag=1, high=True, min_history=3)
    full_sample = (macro.shift(1) > macro.median()).fillna(False)     # the WRONG, look-ahead way
    d8 = MONTHS[8]                                                     # macro=8, lagged=7
    assert expanding.loc[d8] and not full_sample.loc[d8]              # expanding keeps it; FS drops it
    assert expanding.sum() > full_sample.sum()                       # differ materially


def test_row_filter_investment_grade_keeps_only_ig():
    ig = apply_row_filter(_panel(), variable="rating", form="restrict_investment_grade")
    assert ig["investment_grade"].all() and len(ig) == 10 * 24        # top half of 20 bonds


def test_row_filter_excludes_unrated_nan_rows_from_both_segments():
    # Tripwire: the REAL dev panel carries NaN investment_grade (unrated bonds); .astype(bool) used
    # to raise on them. An unrated bond is unclassifiable ex-ante -> excluded from BOTH IG and HY.
    panel = _panel()
    panel["investment_grade"] = panel["investment_grade"].astype(float)
    panel.loc[panel["cusip"] == "B00", "investment_grade"] = np.nan   # one bond becomes unrated
    ig = apply_row_filter(panel, variable="rating", form="restrict_investment_grade")
    hy = apply_row_filter(panel, variable="rating", form="restrict_high_yield")
    assert not ig["investment_grade"].isna().any()                    # no NaN survives either filter
    assert not hy["investment_grade"].isna().any()
    assert "B00" not in set(ig["cusip"]) and "B00" not in set(hy["cusip"])   # unrated in NEITHER
    assert (ig["investment_grade"] == 1).all() and (hy["investment_grade"] == 0).all()


def test_row_filter_liquidity_tercile_is_cross_sectional():
    top = apply_row_filter(_panel(), variable="gamma_illiq", form="restrict_top_liquidity_tercile")
    # each formation month keeps ~top third by gamma_illiq (ex-ante cross-sectional rank).
    per_month = top.groupby("date").size()
    assert per_month.min() >= 1 and per_month.max() <= 8


# ---- G1b execution fidelity ---------------------------------------------------------------

def test_g1b_month_filter_runs_engine_on_subset():
    macro = pd.Series([1.0] * 12 + [5.0] * 12, index=MONTHS)
    ext = _ext("month_filter", transform=PanelTransform("month_filter", "baa_aaa_spread", 1,
                                                        "binary_above_historical_median"))
    out, res = execute_g1b(ext, _panel(), _BASE_RULEBOOK, macro=macro, min_history=6)
    assert out.passed and out.booleans["execution_verified"]
    assert len(res.candidate_returns) > 0 and res.diff["execution_fidelity"] == "PASS"


def test_g1b_row_filter_runs_and_respects_segment():
    ext = _ext("row_filter", transform=PanelTransform("row_filter", "rating", 1,
                                                      "restrict_investment_grade"))
    out, res = execute_g1b(ext, _panel(), _BASE_RULEBOOK)
    assert out.passed and len(res.candidate_returns) > 0


def test_g1b_empty_transform_is_execution_mismatch():
    macro = pd.Series(range(24), index=MONTHS, dtype=float)
    ext = _ext("month_filter", transform=PanelTransform("month_filter", "baa_aaa_spread", 1,
                                                        "binary_above_historical_median"))
    out, res = execute_g1b(ext, _panel(), _BASE_RULEBOOK, macro=macro, min_history=100)  # no months
    assert not out.passed and out.refusal_code is RefusalCode.EXECUTION_MISMATCH and res is None


def test_g1b_month_filter_without_macro_is_missing_input():
    # A month_filter proposal with NO conditioning series is a harness INPUT gap (MISSING_INPUT),
    # not an economic EXECUTION_MISMATCH. The RQ4 funnel wiring bug (2026-09-05) left `macro=None`
    # for all 10 regime-timing proposals, and the old code mislabelled that as EXECUTION_MISMATCH
    # -- masking a wiring omission as a refusal. The transform never ran, so there is no
    # realised-vs-declared claim: MISSING_INPUT is the honest code.
    ext = _ext("month_filter", transform=PanelTransform("month_filter", "baa_aaa_spread", 1,
                                                        "binary_above_historical_median"))
    out, res = execute_g1b(ext, _panel(), _BASE_RULEBOOK)   # macro omitted
    assert not out.passed and out.refusal_code is RefusalCode.MISSING_INPUT and res is None


def test_g1b_double_sort_adds_control_and_runs():
    ext = _ext("double_sort", control="rating")
    rb = {**_BASE_RULEBOOK, "groups": 2}
    out, res = execute_g1b(ext, _panel(), rb)
    assert out.passed and res.diff["declared_changes"] == {"control": "rating"}


# ---- G3 inference -------------------------------------------------------------------------

def _factors_and_y(alpha):
    # four LINEARLY INDEPENDENT factors (a singular/collinear design gives a NaN alpha).
    rng = np.random.default_rng(0)
    F = rng.normal(0.0, 0.01, size=(24, 4))
    resid = rng.normal(0.0, 0.001, size=24)                  # small residual -> finite HAC t
    factors = pd.DataFrame({"date": MONTHS, "mktb": F[:, 0], "drf": F[:, 1],
                            "crf": F[:, 2], "lrf": F[:, 3]})
    y = pd.Series(alpha + 0.4 * F[:, 0] + resid, index=MONTHS)   # loads on mktb only
    return factors, y


def test_g3_strong_alpha_survives_bh():
    factors, y = _factors_and_y(alpha=0.008)
    out, gross = inference_g3(y, factors, proposal_id="p1", q=0.10, nw_lags=0)
    assert out.booleans["bh_survived"]
    assert gross.alpha_bbw4 == pytest.approx(0.008, abs=1e-3)
    assert gross.p_raw < 0.10 and gross.p_bh <= 0.10


def test_g3_zero_alpha_does_not_survive():
    factors, y = _factors_and_y(alpha=0.0)
    out, gross = inference_g3(y, factors, proposal_id="p1", q=0.10, nw_lags=0)
    assert not out.booleans["bh_survived"]
    assert gross.p_raw > 0.10


def test_g3_bh_fdr_is_joint_across_the_family():
    # A marginal proposal with other strong survivors in the family -> BH is joint (all counted).
    factors, y = _factors_and_y(alpha=0.004)
    out, gross = inference_g3(y, factors, proposal_id="p1",
                              family_pvalues={"q1": 0.001, "q2": 0.002}, q=0.10, nw_lags=0)
    assert gross.p_bh >= gross.p_raw                          # adjusted p never below raw


def test_g3_sign_aware_survivor_for_negative_premium_strategy():
    # SC-SCI-8: for a negative-premium strategy (direction=-1, e.g. str winners-losers), a
    # significant NEGATIVE alpha is a survivor; a significant POSITIVE alpha is REJECTED but not a
    # survivor (it moved the wrong way).
    factors, y_neg = _factors_and_y(alpha=-0.008)
    out, gross = inference_g3(y_neg, factors, proposal_id="p1", q=0.10, nw_lags=0, direction=-1)
    assert gross.bh_rejected and out.booleans["bh_survived"]
    _, y_pos = _factors_and_y(alpha=0.008)
    out2, gross2 = inference_g3(y_pos, factors, proposal_id="p1", q=0.10, nw_lags=0, direction=-1)
    assert gross2.bh_rejected and not out2.booleans["bh_survived"]   # rejected, wrong sign


def test_two_sided_p_matches_normal():
    assert two_sided_p(1.96) == pytest.approx(0.05, abs=2e-3)
    assert two_sided_p(0.0) == pytest.approx(1.0)
