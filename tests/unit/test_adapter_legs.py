"""
Leg loop + combiner (D28): N legs -> N factory calls with identical spec-level
fields, strategy_id = parent+ordinal, the three combiner outcomes, refusal
propagation, and run-to-completion collection.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _adapter_fixtures import adapter_spec, grounded_leg  # noqa: E402
from _librarian_fixtures import stated  # noqa: E402
from agents.librarian.adapter import adapt_spec  # noqa: E402
from agents.librarian.schema import Combiner  # noqa: E402
from agents.quant.config import QuantConfig, RefusalCode, to_rulebook  # noqa: E402


def _two_leg(combiner_kind="equal_average"):
    return adapter_spec(
        legs=(grounded_leg("var_5pct"), grounded_leg("credit_rating")),
        combiner=Combiner(kind=stated(combiner_kind)),
    )


def test_n_legs_produce_n_calls_with_ordinal_ids():
    r = adapt_spec(_two_leg())
    assert len(r.leg_calls) == 2
    assert [lc.strategy_id for lc in r.leg_calls] == [
        "Synthetic Momentum_leg0", "Synthetic Momentum_leg1",
    ]


def test_spec_level_fields_copied_identically_into_every_leg():
    r = adapt_spec(_two_leg())
    a, b = r.leg_calls
    for kwarg in ("weighting", "signal_lag", "min_bonds", "holding_period"):
        assert a.kwargs[kwarg] is b.kwargs[kwarg]  # the same shared Inherited (D28)


def test_combiner_equal_average_is_mean_over_available():
    r = adapt_spec(_two_leg("equal_average"))
    assert r.combiner.kind == "equal_average"
    assert r.combiner.divisor == "available"  # adaptive divisor, ledger item 40


def test_combiner_single_leg_passthrough():
    r = adapt_spec(adapter_spec(combiner=Combiner(kind=stated("single_leg"))))
    assert r.combiner.kind == "single_leg"
    assert not r.refused


def test_combiner_other_refuses():
    r = adapt_spec(adapter_spec(combiner=Combiner(kind=stated("other"))))
    assert r.refused
    assert any(x.code is RefusalCode.UNSUPPORTED_COMBINER for x in r.refusals)


def test_grounded_double_sort_rulebook():
    # A var_5pct sort with a credit_rating control -> the drf-shaped rulebook. A
    # declared double sort names its axis via sort_kind=independent (Guard 1, §3:
    # a control axis with sort_kind=single is a structural contradiction).
    leg = grounded_leg(
        "var_5pct",
        control_axis=grounded_leg("credit_rating").sort_signal,
        sort_kind=stated("independent"),
    )
    r = adapt_spec(adapter_spec(legs=(leg,)))
    lc = r.leg_calls[0]
    assert isinstance(lc.result, QuantConfig)
    rb = to_rulebook(lc.result)
    assert rb["score"] == "var_5pct"
    assert rb["control"] == "rating"
    assert rb["control_groups"] == rb["groups"]  # symmetric 5x5: stated control_n_groups(=5) == groups


def test_asymmetric_double_sort_control_groups_flows():
    # v1.1 tenth transform row: a STATED control_n_groups reaches the factory as
    # control_groups, so an ASYMMETRIC double sort (5x3) is NOT silently symmetrised
    # to 5x5. This is exactly the case Guard 2 cannot catch (symmetric 5x5 hides the
    # drop because the defaulted control_groups coincidentally equals groups).
    leg = grounded_leg(
        "var_5pct",
        control_axis=grounded_leg("credit_rating").sort_signal,
        sort_kind=stated("independent"),
        n_groups=stated(5),
        control_n_groups=stated(3),
    )
    r = adapt_spec(adapter_spec(legs=(leg,)))
    lc = r.leg_calls[0]
    assert isinstance(lc.result, QuantConfig)
    rb = to_rulebook(lc.result)
    assert rb["groups"] == 5
    assert rb["control_groups"] == 3  # STATED 2nd-axis count flows through, not defaulted to groups


def test_run_to_completion_collects_all_adapter_refusals():
    # A spec-level LIMIT conflict (strategy_side) AND a per-leg direction refusal
    # (long_leg=other) -> BOTH collected before returning (D29 run-to-completion).
    leg = grounded_leg("var_5pct", long_leg=stated("other"))
    r = adapt_spec(adapter_spec(legs=(leg,), strategy_side=stated("long_only")))
    codes = {x.code for x in r.refusals}
    assert RefusalCode.ASSUMPTION_MISMATCH in codes  # strategy_side long_only
    assert RefusalCode.REVIEW_REQUIRED in codes       # long_leg other
    assert r.refused


def test_any_leg_refused_makes_strategy_refused():
    # One good leg + one ungrounded leg -> the whole strategy refuses (D28).
    r = adapt_spec(adapter_spec(legs=(grounded_leg("var_5pct"), grounded_leg("maturity"))))
    assert r.refused
    assert any(lc.refused for lc in r.leg_calls)


def test_guard1_belt_refuses_contradictory_leg():
    # A control axis with sort_kind=single is a structural contradiction (Guard 1, §3);
    # the adapter belt refuses it REVIEW_REQUIRED at intake, before the factory (§3 belt).
    leg = grounded_leg(
        "var_5pct",
        control_axis=grounded_leg("credit_rating").sort_signal,
        sort_kind=stated("single"),
    )
    r = adapt_spec(adapter_spec(legs=(leg,)))
    assert r.refused
    assert any(x.code is RefusalCode.REVIEW_REQUIRED and x.field == "sort_kind" for x in r.refusals)
    assert r.leg_calls[0].result is None  # never reached the factory
