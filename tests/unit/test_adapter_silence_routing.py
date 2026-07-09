"""
Silence routing (D26): the three branches + the D23 refuse-on-conflict overrides,
driven against the real silence-policy table.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _librarian_fixtures import stated, unknown  # noqa: E402
from agents.librarian.adapter.silence_routing import (  # noqa: E402
    FlagField,
    OmitField,
    Proceed,
    RefuseField,
    route_field,
)
from agents.librarian.registries.silence_policy import load_silence_policy_table  # noqa: E402
from agents.quant.config import RefusalCode  # noqa: E402

SILENCE = load_silence_policy_table()


def _pol(block, field):
    return SILENCE.policy_for(block, field)


def test_silent_tag_and_proceed_omits():
    # bucketing_method: tag_and_proceed -> silence omits (factory default).
    out = route_field(_pol("sort_block", "bucketing_method"), unknown(), control_present=False)
    assert isinstance(out, OmitField)


def test_silent_refuse_policy_refuses_on_silence():
    # long_leg: refuse -> silence is a hard REFUSED_ON_SILENCE.
    out = route_field(_pol("sort_block", "long_leg"), unknown(), control_present=False)
    assert isinstance(out, RefuseField) and out.code is RefusalCode.REFUSED_ON_SILENCE


def test_review_reasons_route_to_review():
    for reason in ("disagreement", "quote_match_failure", "single_response"):
        out = route_field(
            _pol("common", "signal_lag"), unknown(reason=reason), control_present=False
        )
        assert isinstance(out, RefuseField) and out.code is RefusalCode.REVIEW_REQUIRED


def test_stated_value_proceeds():
    out = route_field(_pol("common", "weighting_scheme"), stated("value"), control_present=False)
    assert isinstance(out, Proceed)


def test_refuse_on_stated_is_assumption_mismatch():
    # strategy_side long_only: a STATED non-representable value (LIMIT conflict).
    out = route_field(_pol("common", "strategy_side"), stated("long_only"), control_present=False)
    assert isinstance(out, RefuseField) and out.code is RefusalCode.ASSUMPTION_MISMATCH


def test_flag_on_stated_is_soft_flag():
    out = route_field(
        _pol("common", "transaction_cost_convention"), stated("net_of_costs"), control_present=False
    )
    assert isinstance(out, FlagField)


def test_sort_kind_conditional_control_present_refuses_on_silence():
    out = route_field(_pol("sort_block", "sort_kind"), unknown(), control_present=True)
    assert isinstance(out, RefuseField) and out.code is RefusalCode.REFUSED_ON_SILENCE


def test_sort_kind_conditional_no_control_omits_on_silence():
    out = route_field(_pol("sort_block", "sort_kind"), unknown(), control_present=False)
    assert isinstance(out, OmitField)
