"""
G3 gold<->run field pairing for the FIRST multi-leg anchor (CRF, D34).

The single-leg pairing is exercised end-to-end by test_g3_scoring on the real
drf/mom6/str runs. This file guards the multi-leg join specifically:

  * ``match_legs`` is order-invariant AND breaks ties on control_axis -- CRF's
    three legs all sort on credit_rating, so the sort-signal term is constant and
    the DISTINGUISHING axis is the control. Without the tiebreak every permutation
    ties and the arbitrary identity pairing mis-pairs the legs (the landmine).
  * ``pair_fields`` routes a multi-leg spec through ``match_legs`` and looks each
    leg field up under the MATCHED run leg's ``legs[{run_j}].{field}`` record, so
    two gold legs never collapse onto one trace record.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.gold_specs.gold_loader import load_gold_spec  # noqa: E402
from evaluation.harness.field_pairing import match_legs, pair_fields  # noqa: E402


def _leg(ctrl: str, signal: str = "credit_rating"):
    return SimpleNamespace(
        sort_signal=SimpleNamespace(concept_id=SimpleNamespace(value=signal)),
        control_axis=SimpleNamespace(concept_id=SimpleNamespace(value=ctrl)),
    )


def test_match_legs_breaks_ties_on_control_axis():
    """The landmine: CRF's three legs share sort_signal=credit_rating, so a
    sort-signal-only score ties on every permutation and returns the arbitrary
    identity -- pairing gold's VaR leg to a run's ILLIQ leg. The control_axis
    tiebreak recovers the correct order-invariant match."""
    gold = [_leg("var_5pct"), _leg("gamma"), _leg("xret")]
    run = [_leg("gamma"), _leg("xret"), _leg("var_5pct")]      # permuted
    assert dict(match_legs(gold, run)) == {0: 2, 1: 0, 2: 1}
    # identity fast-path preserved for single-leg
    assert match_legs([_leg("var_5pct")], [_leg("var_5pct")]) == [(0, 0)]


def _rf(val):
    """A minimal RunField stand-in (only the attrs the pairing/matching read)."""
    return SimpleNamespace(normalised_a=val, normalised_b=val,
                           a_answered=val is not None, b_answered=val is not None,
                           models_agree=True)


def _crf_run_artefacts(control_order):
    """Synthetic run artefacts with legs authored in a given control-axis order
    (``legs[{j}].{field}`` keys, the multi-leg run convention)."""
    fields = {}
    for j, ctrl in enumerate(control_order):
        fields[f"legs[{j}].sort_signal"] = _rf("credit_rating")
        fields[f"legs[{j}].control_axis"] = _rf(ctrl)
        fields[f"legs[{j}].n_groups"] = _rf(5)
        fields[f"legs[{j}].long_leg"] = _rf("highest_signal")
    return SimpleNamespace(fields=fields)


def test_crf_pair_fields_order_invariant_by_control_axis():
    """pair_fields on the real 3-leg CRF gold + a run whose legs are in a DIFFERENT
    order: each gold leg's control_axis must pair to the run record with the SAME
    control concept (no PairingError, no leg collapse)."""
    spec = load_gold_spec("crf")   # gold legs (concepts): var_5pct, bpw_gamma, prior_1m_excess_return
    # run authored in a DIFFERENT order (concept ids, as the trace records them)
    art = _crf_run_artefacts(["bpw_gamma", "prior_1m_excess_return", "var_5pct"])
    paired, _excluded = pair_fields(spec, art)

    ctrl = {pf.key.leg_index: pf for pf in paired
            if pf.key.name == "control_axis" and pf.key.leg_index is not None}
    assert set(ctrl) == {0, 1, 2}                      # three distinct leg keys, no collapse
    gold_ctrl = {i: leg.control_axis.concept_id.value
                 for i, leg in enumerate(spec.part2.legs)}
    for i, pf in ctrl.items():
        assert pf.run is not None
        assert pf.run.normalised_a == gold_ctrl[i]     # matched by control axis, order-invariant

    # sort_signal is also present per leg, three distinct keys
    sig_keys = {pf.key.leg_index for pf in paired
                if pf.key.name == "sort_signal" and pf.key.leg_index is not None}
    assert sig_keys == {0, 1, 2}


def test_crf_pair_fields_non_contiguous_run_indices():
    """Run leg indices need not be 0..n-1: a match POSITION must translate back to
    the ACTUAL legs[{idx}] record. With naive position-as-index this silently pairs
    gold's REV leg against the run's gamma leg (the reviewer's landmine)."""
    spec = load_gold_spec("crf")   # gold: var_5pct, bpw_gamma, prior_1m_excess_return
    fields = {}
    for j, ctrl in {0: "bpw_gamma", 2: "prior_1m_excess_return", 5: "var_5pct"}.items():
        fields[f"legs[{j}].sort_signal"] = _rf("credit_rating")
        fields[f"legs[{j}].control_axis"] = _rf(ctrl)
    paired, _ = pair_fields(spec, SimpleNamespace(fields=fields))
    ctrl = {pf.key.leg_index: pf for pf in paired
            if pf.key.name == "control_axis" and pf.key.leg_index is not None}
    gold_ctrl = {i: leg.control_axis.concept_id.value for i, leg in enumerate(spec.part2.legs)}
    for i, pf in ctrl.items():
        assert pf.run is not None
        assert pf.run.normalised_a == gold_ctrl[i]     # matched to the run leg with the SAME control


def test_crf_pair_fields_unmatched_run_leg_is_none_not_crash():
    """If the run emitted fewer legs than the gold, the unmatched gold leg's fields
    pair to None (coverage denominator) rather than raising."""
    spec = load_gold_spec("crf")
    art = _crf_run_artefacts(["var_5pct", "bpw_gamma"])    # only two run legs (concept ids)
    paired, _ = pair_fields(spec, art)
    ctrl = {pf.key.leg_index: pf for pf in paired
            if pf.key.name == "control_axis" and pf.key.leg_index is not None}
    matched = [i for i, pf in ctrl.items() if pf.run is not None]
    assert len(matched) == 2                            # two legs matched, one None
