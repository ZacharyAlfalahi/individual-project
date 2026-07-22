"""
Output-schema discipline (spec §2.2, §2.3, §5.3, amendment).

No naked effects: an EffectEstimate cannot exist without an interval-status label AND a conditioning
statement, and must carry the _corr suffix + correction orientation. The bracket and the DOE effect
never share a name. The algebraic identity is asserted at result construction. "primary" never
appears in a field name.
"""

from __future__ import annotations

import pytest

from agents.auditor.ipca_differential.schemas import (
    INTERVAL_COMPUTED,
    IPCADifferentialCell,
    IPCADifferentialResult,
    IPCASchemaError,
    EffectEstimate,
    deferred_effect,
)


def _cell(label, value):
    return IPCADifferentialCell(
        panel_state="P_N", fitted_state="Theta_N", label=label, value=value,
        frozen_state_hash="deadbeef", projection_diagnostics={},
    )


def _result(y_nn, y_nb, y_bn, y_bb, *, bracket=None, doe=None):
    """Construct a result from the four cell values with (by default) consistent margins."""
    d_data_tn = y_nn - y_bn
    d_data_tb = y_nb - y_bb
    d_est_pn = y_nn - y_nb
    d_est_pb = y_bn - y_bb
    d_total = y_nn - y_bb
    bkt = y_nn - y_bn - y_nb + y_bb
    return IPCADifferentialResult(
        bias="meas_err", anchor="str", is_focal=False,
        cells=(_cell("Y_NN", y_nn), _cell("Y_Nb", y_nb), _cell("Y_bN", y_bn), _cell("Y_bb", y_bb)),
        data_margin_theta_n=deferred_effect("data_margin_theta_n_corr", d_data_tn),
        data_margin_theta_b=deferred_effect("data_margin_theta_b_corr", d_data_tb),
        interaction_bracket_raw=deferred_effect("interaction_bracket_raw_corr", bracket if bracket is not None else bkt),
        doe_interaction_effect=deferred_effect("doe_interaction_effect_corr", doe if doe is not None else bkt / 2),
        est_margin_p_n=deferred_effect("est_margin_p_n_corr", d_est_pn),
        est_margin_p_b=deferred_effect("est_margin_p_b_corr", d_est_pb),
        total=deferred_effect("total_corr", d_total),
        secondary_endtoend=None, common_support_n_months=42,
    )


# ---- EffectEstimate: no naked effects -------------------------------------

def test_effect_requires_conditioning():
    with pytest.raises(IPCASchemaError, match="conditioning"):
        EffectEstimate("x_corr", 0.0, "corr", "deferred_inc3", None, "")


def test_effect_requires_corr_suffix():
    with pytest.raises(IPCASchemaError, match="_corr"):
        EffectEstimate("x", 0.0, "corr", "deferred_inc3", None, "cond")


def test_effect_requires_correction_orientation():
    with pytest.raises(IPCASchemaError, match="orientation"):
        EffectEstimate("x_corr", 0.0, "biasintro", "deferred_inc3", None, "cond")


def test_computed_status_requires_interval():
    with pytest.raises(IPCASchemaError, match="no interval"):
        EffectEstimate("x_corr", 0.0, "corr", INTERVAL_COMPUTED, None, "cond")


def test_deferred_effect_is_wellformed():
    e = deferred_effect("x_corr", 1.5)
    assert e.interval is None and e.interval_status == "deferred_inc3"
    assert e.conditioning.strip() and e.to_dict()["interval"] is None


# ---- IPCADifferentialResult: algebra + naming -----------------------------

def test_valid_result_serializes():
    res = _result(1.0, 0.3, 0.4, 0.2)
    d = res.to_dict()
    assert d["interaction_bracket_raw"]["name"] == "interaction_bracket_raw_corr"
    assert d["doe_interaction_effect"]["name"] == "doe_interaction_effect_corr"
    assert d["interaction_bracket_raw"]["name"] != d["doe_interaction_effect"]["name"]


def test_result_rejects_broken_bracket():
    with pytest.raises(IPCASchemaError, match="algebraic identity"):
        _result(1.0, 0.3, 0.4, 0.2, bracket=5.0)


def test_result_rejects_broken_doe():
    with pytest.raises(IPCASchemaError, match="algebraic identity"):
        _result(1.0, 0.3, 0.4, 0.2, doe=99.0)


def test_no_primary_token_in_serialized_fields():
    d = _result(1.0, 0.3, 0.4, 0.2).to_dict()
    blob = str(d).lower()
    assert "primary" not in blob   # the word is banned inside the extension (§3.2)
