"""
Unit tests for QuantConfig: the golden regression (strip reproduces the
already-validated production rulebooks) plus the F1/F2 fixes and malformed-input
raising.
"""

import pytest

from agents.quant.config import (
    Binding,
    Evidence,
    Inherited,
    Locator,
    QuantConfig,
    TrimRule,
    build_quant_config,
    to_rulebook,
)
from agents.quant.config.quant_config import QuantConfigError
from agents.quant.library.characteristic_sort import _apply_defaults

# The production rulebooks the typed layer must reproduce. These *_rulebook
# factories are pure/import-safe (verified: module-level code is constants only).
from scripts.build_str import str_rulebook
from scripts.build_mom6 import mom6_rulebook
from agents.quant.library.bbw_factors import factor_rulebook

_LOC = Locator(1, 0, 1)  # placeholder span for synthetic STATED fixtures (D7 needs a locator to exist)


# --- config builders mirroring the three anchors ---------------------------

def _str_config():
    # Gold-aligned (gold_str_drr_2026.md): deciles, long P10 winners − short P1 losers,
    # sort on the grounded prior_1m_excess_return -> xret (D27 v2). Mirrors str_rulebook.
    return build_quant_config(
        "str",
        Binding("xret", "BOUND", Evidence(column="xret", note="prior_1m_excess_return -> xret (D27 v2)")),
        groups=Inherited(10, "STATED", Evidence(locator=_LOC, quote="we sort bonds into deciles each month")),
        weighting=Inherited("size", "DESIGN", Evidence(note="VW par (BBW §2.4)")),
        long_group=Inherited(9, "STATED", Evidence(locator=_LOC, quote="long the top decile P10")),
        short_group=Inherited(0, "STATED", Evidence(locator=_LOC, quote="short the bottom decile P1")),
        signal_lag=Inherited(0, "STATED", Evidence(locator=_LOC, quote="as-published, signal_lag=0")),
    )


def _drf_config(control_tag="BOUND"):
    if control_tag == "AMBIGUOUS":
        control = Binding(
            "rating", "AMBIGUOUS",
            Evidence(candidates=("rating", "rating_numeric"), chosen="rating",
                     note="two rating columns; chose the numeric spec's canonical name"),
        )
    else:
        control = Binding("rating", "BOUND", Evidence(column="rating", note="credit-rating axis"))
    return build_quant_config(
        "drf",
        Binding("var_5pct", "BOUND", Evidence(column="var_5pct", note="5% VaR")),
        control=control,
        groups=Inherited(5, "STATED", Evidence(locator=_LOC, quote="quintiles")),
        control_groups=Inherited(5, "STATED", Evidence(locator=_LOC, quote="5x5 bivariate sort")),
        weighting=Inherited("size", "DESIGN", Evidence(note="VW par")),
        long_group=Inherited(4, "STATED", Evidence(locator=_LOC, quote="high VaR")),
        short_group=Inherited(0, "STATED", Evidence(locator=_LOC, quote="low VaR")),
        signal_lag=Inherited(0, "STATED", Evidence(locator=_LOC, quote="contemporaneous")),
    )


def _mom6_config():
    return build_quant_config(
        "mom6",
        Binding("mom6", "BOUND", Evidence(column="mom6", note="trailing 6m cumulative return")),
        groups=Inherited(10, "STATED", Evidence(locator=_LOC, quote="decile portfolios")),
        weighting=Inherited("equal", "STATED", Evidence(locator=_LOC, quote="equal-weighted (Jostova)")),
        long_group=Inherited(9, "STATED", Evidence(locator=_LOC, quote="top decile winners")),
        short_group=Inherited(0, "STATED", Evidence(locator=_LOC, quote="bottom decile losers")),
        signal_lag=Inherited(1, "STATED", Evidence(locator=_LOC, quote="skip the most recent month")),
        holding_period=Inherited(6, "STATED", Evidence(locator=_LOC, quote="six-month staggered hold")),
    )


# --- GOLDEN REGRESSION: strip reproduces the production rulebooks -----------

def test_golden_str():
    cfg = _str_config()
    assert isinstance(cfg, QuantConfig)
    assert _apply_defaults(to_rulebook(cfg)) == _apply_defaults(str_rulebook(signal_lag=0))


def test_golden_bbw_drf():
    cfg = _drf_config()
    assert isinstance(cfg, QuantConfig)
    assert _apply_defaults(to_rulebook(cfg)) == _apply_defaults(factor_rulebook("drf"))


def test_golden_mom6():
    cfg = _mom6_config()
    assert isinstance(cfg, QuantConfig)
    prod = mom6_rulebook({"n_groups": 10, "weighting": "equal", "skip_months": 1})
    assert _apply_defaults(to_rulebook(cfg)) == _apply_defaults(prod)
    # holding_period is carried on the config, NOT in the rulebook.
    assert "holding_period" not in to_rulebook(cfg)
    assert cfg.holding_period.value == 6


# --- F1: an AMBIGUOUS-resolved control must NOT be dropped ------------------

def test_ambiguous_control_still_emits_double_sort():
    cfg = _drf_config(control_tag="AMBIGUOUS")
    rb = to_rulebook(cfg)
    assert rb["control"] == "rating"
    assert rb["control_groups"] == 5
    # and it still reproduces the production double-sort rulebook
    assert _apply_defaults(rb) == _apply_defaults(factor_rulebook("drf"))


# --- F2: weighting "size" strips to engine-native "by_size" ----------------

def test_weighting_size_translates_to_by_size():
    assert to_rulebook(_str_config())["weighting"] == "by_size"


def test_weighting_equal_passes_through():
    assert to_rulebook(_mom6_config())["weighting"] == "equal"


# --- malformed-but-supported input raises (agent bug, not a refusal) -------

def test_groups_below_two_raises():
    with pytest.raises(QuantConfigError):
        build_quant_config(
            "bad",
            Binding("score", "BOUND", Evidence(column="score", note="x")),
            groups=Inherited(1, "STATED", Evidence(locator=_LOC, quote="one group")),
        )


def test_long_equals_short_raises():
    with pytest.raises(QuantConfigError):
        build_quant_config(
            "bad",
            Binding("score", "BOUND", Evidence(column="score", note="x")),
            long_group=Inherited(2, "STATED", Evidence(locator=_LOC, quote="two")),
            short_group=Inherited(2, "STATED", Evidence(locator=_LOC, quote="two")),
        )


def test_holding_period_zero_raises():
    with pytest.raises(QuantConfigError):
        build_quant_config(
            "bad",
            Binding("score", "BOUND", Evidence(column="score", note="x")),
            holding_period=Inherited(0, "STATED", Evidence(locator=_LOC, quote="zero")),
        )


def test_negative_signal_lag_raises():
    with pytest.raises(QuantConfigError):
        build_quant_config(
            "bad",
            Binding("score", "BOUND", Evidence(column="score", note="x")),
            signal_lag=Inherited(-1, "STATED", Evidence(locator=_LOC, quote="minus one")),
        )


def test_bool_is_not_accepted_as_int():
    # True is an int subclass in Python; the guard must reject it.
    with pytest.raises(QuantConfigError):
        build_quant_config(
            "bad",
            Binding("score", "BOUND", Evidence(column="score", note="x")),
            signal_lag=Inherited(True, "STATED", Evidence(locator=_LOC, quote="bool")),
        )


# --- T8: hand-built QuantConfig safety nets (bypassing the factory) ---------

def _wrapped_config(**overrides):
    """Build a valid QuantConfig directly via its constructor, so
    __post_init__ (not the factory) is what validates."""
    fields = dict(
        strategy_id="hb",
        score=Binding("score", "BOUND", Evidence(column="score", note="x")),
        groups=Inherited(5, "STATED", Evidence(locator=_LOC, quote="q")),
        weighting=Inherited("size", "DESIGN", Evidence(note="par")),
        signal_lag=Inherited(0, "STATED", Evidence(locator=_LOC, quote="q")),
        min_bonds=Inherited(5, "STATED", Evidence(locator=_LOC, quote="q")),
        long_group=Inherited(4, "STATED", Evidence(locator=_LOC, quote="q")),
        short_group=Inherited(0, "STATED", Evidence(locator=_LOC, quote="q")),
        control_groups=Inherited(5, "STATED", Evidence(locator=_LOC, quote="q")),
        trim_rule=Inherited(TrimRule(method="none"), "UNKNOWN", Evidence(note="none")),
        holding_period=Inherited(1, "STATED", Evidence(locator=_LOC, quote="q")),
        control=None,
    )
    fields.update(overrides)
    return QuantConfig(**fields)


def test_hand_built_valid_config_ok():
    cfg = _wrapped_config()
    assert isinstance(cfg, QuantConfig)


def test_hand_built_missing_score_rejected():
    # the __post_init__ safety net for a MISSING binding that bypassed the factory
    with pytest.raises(QuantConfigError):
        _wrapped_config(score=Binding(None, "MISSING", Evidence(note="no col")))


def test_hand_built_min_bonds_below_one_rejected():
    with pytest.raises(QuantConfigError):
        _wrapped_config(min_bonds=Inherited(0, "STATED", Evidence(locator=_LOC, quote="q")))


def test_hand_built_long_group_out_of_range_rejected():
    with pytest.raises(QuantConfigError):
        _wrapped_config(long_group=Inherited(9, "STATED", Evidence(locator=_LOC, quote="q")))


def test_hand_built_short_group_out_of_range_rejected():
    with pytest.raises(QuantConfigError):
        _wrapped_config(short_group=Inherited(-1, "STATED", Evidence(locator=_LOC, quote="q")))


def test_hand_built_control_groups_below_two_rejected():
    with pytest.raises(QuantConfigError):
        _wrapped_config(
            control=Binding("rating", "BOUND", Evidence(column="rating", note="axis")),
            control_groups=Inherited(1, "STATED", Evidence(locator=_LOC, quote="q")),
        )


def test_hand_built_non_trimrule_value_rejected():
    with pytest.raises(QuantConfigError):
        _wrapped_config(trim_rule=Inherited("nope", "UNKNOWN", Evidence(note="x")))


def test_hand_built_unwrapped_field_rejected():
    # a raw int where an Inherited is expected must be branded, not AttributeError
    with pytest.raises(QuantConfigError):
        _wrapped_config(groups=5)


# --- M1 / L4: the factory trust boundary over raw (non-wrapped) input ------

def test_factory_raw_field_raises_branded_error():
    with pytest.raises(QuantConfigError):
        build_quant_config(
            "s",
            Binding("score", "BOUND", Evidence(column="score", note="x")),
            groups=5,  # raw int, not an Inherited
        )


def test_factory_raw_score_raises_branded_error():
    with pytest.raises(QuantConfigError):
        build_quant_config("s", "score")  # raw str, not a Binding


@pytest.mark.parametrize("bad_id", ["", "   ", None])
def test_factory_empty_strategy_id_rejected(bad_id):
    with pytest.raises(QuantConfigError):
        build_quant_config(
            bad_id,
            Binding("score", "BOUND", Evidence(column="score", note="x")),
        )
