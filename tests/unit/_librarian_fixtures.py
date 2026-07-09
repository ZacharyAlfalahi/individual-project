"""
Shared, quote-bearing builders for Librarian schema tests.

NOT a set of paper-gold fixtures (the brief forbids authoring quote fixtures for
real papers). These are minimal synthetic-but-well-formed objects so the schema
+ validator tests have a valid baseline to mutate. Every STATED value carries a
placeholder quote + locator only because the frozen provenance layer requires it
(D7) -- the strings are deliberately not lifted from any paper.
"""

from __future__ import annotations

from agents.quant.config import Evidence, Inherited, Locator

from agents.librarian.schema import (
    Combiner,
    DescribedSignal,
    Leg,
    LocatedQuote,
    MethodSummary,
    Part1,
    Part2,
    SignalRef,
    SpecHeader,
    StrategySpec,
)


def stated(value, *, quote="synthetic placeholder quote", page=1, cs=0, ce=1):
    return Inherited(value, "STATED", Evidence(quote=quote, locator=Locator(page, cs, ce)))


def unknown(note="not stated; synthetic", reason="not_stated"):
    return Inherited(None, "UNKNOWN", Evidence(note=note, unknown_reason=reason))


def located_quote(text="synthetic quote", page=1, cs=0, ce=1):
    return LocatedQuote(text=text, page=page, char_start=cs, char_end=ce)


def signal_ref(concept="mom6", *, params=None, label="synthetic signal", quotes=None):
    if quotes is None:
        quotes = (located_quote(),)
    return SignalRef(
        concept_id=stated(concept),
        as_described=DescribedSignal(label=label, quotes=tuple(quotes)),
        parameters=params or {},
    )


def build_leg(**overrides):
    """A single well-formed Leg; override any field by name."""
    base = dict(
        sort_signal=signal_ref("mom6", params={"months": stated(6)}),
        control_axis=None,
        sort_kind=stated("single"),
        bucketing_method=stated("equal_count"),
        n_groups=stated(5),
        stripe_aggregation=stated("equal"),
        control_missing_policy=stated("drop"),
        long_leg=stated("highest_signal"),
        signal_transform=stated("none"),
    )
    base.update(overrides)
    return Leg(**base)


def build_part2(legs=None, combiner=None, **overrides):
    if legs is None:
        legs = (build_leg(),)
    if combiner is None:
        combiner = Combiner(kind=stated("single_leg"))
    base = dict(
        eligibility_missing_policy=stated("drop"),
        return_availability_policy=stated("require_next_month_return"),
        signal_lag=stated(0),
        lag_convention=stated("month_end"),
        min_bonds=stated(5),
        min_bonds_granularity=stated("total"),
        tie_break_policy=stated("deterministic_id"),
        weighting_scheme=stated("value"),
        weighting_base=stated("par"),
        weight_timing=stated("formation"),
        strategy_side=stated("long_short"),
        empty_leg_policy=stated("skip"),
        transaction_cost_convention=stated("gross"),
        return_label=stated("realisation"),
        rebalance_frequency=stated("monthly"),
        holding_period=stated(1),
        overlap_convention=stated("overlapping"),
        cohort_weighting=stated("equal"),
        burn_in_policy=stated("keep"),
        missing_return_policy=stated("drop_and_renormalise"),
        realisation_min_survivors=stated(0),
        return_compounding=stated("arithmetic"),
        significance_convention=stated("hac_t_of_mean"),
        hac_lags=stated(0),
        annualisation=stated("arithmetic"),
        rf_convention=stated("none"),
        benchmark_model=stated("none"),
        expost_trim=stated("none"),
    )
    base.update(overrides)
    return Part2(legs=tuple(legs), combiner=combiner, **base)


def build_part1(**overrides):
    base = dict(
        formation_structure=stated("sorted_portfolios"),
        asset_class=stated("corporate_bonds"),
        method_summary=MethodSummary(
            summary=stated("ranks bonds by past return; long-short quintiles; a factor"),
            quotes=(located_quote(),),
        ),
    )
    base.update(overrides)
    return Part1(**base)


def build_header(**overrides):
    base = dict(
        paper_id="SYNTH-0001",
        strategy_label=stated("Synthetic Momentum"),
        registry_version="sig-v1",
        registry_hash="deadbeef",
        silence_table_version="v1",
        canonical_text_hash="cafef00d",
    )
    base.update(overrides)
    return SpecHeader(**base)


def build_spec(header=None, part1=None, part2=None):
    return StrategySpec(
        header=header or build_header(),
        part1=part1 or build_part1(),
        part2=part2 or build_part2(),
    )


class FakeSignalRegistry:
    """A minimal in-memory Signal Concept Registry satisfying SignalRegistryLike
    -- for the registry-aware SignalRef tests. Concrete D22 loader is a separate
    artifact; the validator only needs these two methods."""

    def __init__(self, schemas=None):
        # {concept_id: {param_name: type}}
        self._schemas = schemas or {"mom6": {"months": int}, "str1m": {}, "var5pct": {}}

    def has_concept(self, concept_id: str) -> bool:
        return concept_id in self._schemas

    def parameter_schema(self, concept_id: str):
        return self._schemas[concept_id]
