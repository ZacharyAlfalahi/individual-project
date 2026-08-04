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
    ESTIMATION_FIELDS,
    Combiner,
    DescribedSignal,
    EstimationBlock,
    InstrumentRef,
    InstrumentSet,
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
        control_n_groups=stated(5),
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
        silence_table_version="v1.1",
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


# ---------------------------------------------------------------------------
# Fitted-factor-model (v1.2) builders -- EstimationBlock + InstrumentSet + the
# all-UNKNOWN stub Part2 a KPP-shaped spec carries. Synthetic, not paper-lifted.
# ---------------------------------------------------------------------------


def build_stub_part2():
    """An all-UNKNOWN sort block (single stub leg + single_leg combiner): the
    minimal Part2 a fitted-model spec carries to satisfy the non-empty legs
    guard. Never scored (the KPP path is parallel)."""
    stub_leg = Leg(
        sort_signal=SignalRef(
            concept_id=unknown(), as_described=DescribedSignal(label="schema stub")
        ),
        sort_kind=unknown(),
        bucketing_method=unknown(),
        n_groups=unknown(),
        stripe_aggregation=unknown(),
        control_missing_policy=unknown(),
        long_leg=unknown(),
        signal_transform=unknown(),
        control_n_groups=unknown(),
    )
    return build_part2(
        legs=(stub_leg,),
        combiner=Combiner(kind=unknown()),
        # blank the common block to UNKNOWN so nothing sort-side reads as real.
        **{
            name: unknown()
            for name in (
                "eligibility_missing_policy", "return_availability_policy", "signal_lag",
                "lag_convention", "min_bonds", "min_bonds_granularity", "tie_break_policy",
                "weighting_scheme", "weighting_base", "weight_timing", "strategy_side",
                "empty_leg_policy", "transaction_cost_convention", "return_label",
                "rebalance_frequency", "holding_period", "overlap_convention",
                "cohort_weighting", "burn_in_policy", "missing_return_policy",
                "realisation_min_survivors", "return_compounding", "significance_convention",
                "hac_lags", "annualisation", "rf_convention", "benchmark_model", "expost_trim",
            )
        },
    )


def instrument_ref(concept="past_6m_cumulative_return", *, source_class="bond",
                   transform=None, lag=None, label="synthetic instrument", quotes=None):
    if quotes is None:
        quotes = (located_quote(),)
    return InstrumentRef(
        concept_id=stated(concept),
        source_class=stated(source_class),
        transform=transform if transform is not None else unknown(),
        lag=lag if lag is not None else unknown(),
        as_described=DescribedSignal(label=label, quotes=tuple(quotes)),
    )


def build_estimation_block(**overrides):
    """A well-formed EstimationBlock; STATED headline identity fields, UNKNOWN
    elsewhere by default. Override any field by name."""
    base = {name: unknown() for name in ESTIMATION_FIELDS}
    base.update(
        model_family=stated("instrumented_pca"),
        estimation_algorithm=stated("alternating_least_squares"),
        n_factors_tested=stated({1, 2, 3, 4, 5}),
        n_factors_preferred=stated(5),
        intercept_spec=stated("both"),
        estimation_mode=stated("both"),
        oos_split=stated(36),
        inference_method=stated("wild_bootstrap"),
    )
    base.update(overrides)
    return EstimationBlock(**base)


def build_instrument_set(instruments=None):
    if instruments is None:
        instruments = (
            instrument_ref("past_6m_cumulative_return", source_class="bond"),
            instrument_ref("credit_rating", source_class="bond"),
        )
    return InstrumentSet(instruments=tuple(instruments))


def build_kpp_spec(header=None, part1=None, estimation=None, instruments=None):
    """A KPP-shaped StrategySpec: estimated_factor_model Part1 + stub Part2 +
    estimation + instruments."""
    return StrategySpec(
        header=header or build_header(),
        part1=part1
        or build_part1(formation_structure=stated("estimated_factor_model")),
        part2=build_stub_part2(),
        estimation=estimation or build_estimation_block(),
        instruments=instruments or build_instrument_set(),
    )


class FakeInstrumentRegistry:
    """A minimal in-memory Instrument Concept Registry satisfying
    SignalRegistryLike -- for the estimation validator tests."""

    def __init__(self, ids=None):
        self._ids = set(
            ids
            or {
                "past_6m_cumulative_return", "credit_rating", "bond_var_36m",
                "duration", "spread", "bond_skewness",
            }
        )

    def has_concept(self, concept_id: str) -> bool:
        return concept_id in self._ids

    def parameter_schema(self, concept_id: str):
        return {}
