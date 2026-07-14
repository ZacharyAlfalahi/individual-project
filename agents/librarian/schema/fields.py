"""
Part 1 + Part 2 field names, enum menus, markers, and the ``ALREADY_FINAL_PART2``
set -- the single source of truth for *which* fields exist and *what values*
each menu admits.

Source of record: ``docs/librarian/specs/part2_schema_and_silence_policy_v1_1.md`` (v1.1 adds ``control_n_groups``). 38 Part 2 fields = 10
sort-block + 28 common. Every enum menu carries the ``"other"`` escape (P3: menus
over prose, with a first-class "none of the above"). Markers (D18/D32b) =
``sort_signal`` + ``n_groups`` only.

``ALREADY_FINAL_PART2`` = all 38 fields: nothing was deferred, so the
schema covers the complete Part 2 list.

``PAPER_FACTS_FIELDS`` (v1.1) is a SEPARATE spec-level block (sample window +
claimed metrics) -- extraction output the analysis consumes, never Part 2
execution fields, so it is deliberately NOT in ``ALREADY_FINAL_PART2``.

These constants are consumed by:
  * ``strategy_spec.py`` -- the field slots on ``Part2``/``Leg``/``Combiner``.
  * ``validators/domains.py`` -- the value-set / range for each field.
  * downstream extraction (later clusters) -- the form-filler menus.

No engine coordinates appear here (D3): these are paper-vocabulary menus.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Part 1 field names (D13) -- three fields derived from unowned decisions.
# ---------------------------------------------------------------------------

FORMATION_STRUCTURE = "formation_structure"
ASSET_CLASS = "asset_class"
METHOD_SUMMARY = "method_summary"

PART1_FIELDS: tuple[str, ...] = (
    FORMATION_STRUCTURE,
    ASSET_CLASS,
    METHOD_SUMMARY,
)

# Part 1 enum menus (D13). method_summary is free text (D16), not a menu.
FORMATION_STRUCTURE_MENU: tuple[str, ...] = (
    "sorted_portfolios",
    "estimated_factor_model",
    "trained_predictor",
    "none",
    "other",
)
ASSET_CLASS_MENU: tuple[str, ...] = (
    "corporate_bonds",
    "other",
)

# ---------------------------------------------------------------------------
# Part 2 -- sort block (10 fields, v1.1). Per-leg unless noted (D19: legs + combiner).
# ---------------------------------------------------------------------------

SORT_SIGNAL = "sort_signal"                     # SignalRef (per-leg) -- MARKER
SORT_KIND = "sort_kind"                         # per-leg
BUCKETING_METHOD = "bucketing_method"           # per-leg
N_GROUPS = "n_groups"                           # per-leg int -- MARKER
STRIPE_AGGREGATION = "stripe_aggregation"       # per-leg
CONTROL_MISSING_POLICY = "control_missing_policy"  # per-leg
LONG_LEG = "long_leg"                           # per-leg
SIGNAL_TRANSFORM = "signal_transform"           # per-leg (check-only)
CONTROL_N_GROUPS = "control_n_groups"           # per-leg int (v1.1) -- 2nd-axis group count
COMBINER = "combiner"                           # spec-level

# The per-leg fields that live on a Leg (sort_signal + control_axis are
# SignalRefs, handled specially; the rest are Inherited).
LEG_FIELDS: tuple[str, ...] = (
    SORT_SIGNAL,
    SORT_KIND,
    BUCKETING_METHOD,
    N_GROUPS,
    STRIPE_AGGREGATION,
    CONTROL_MISSING_POLICY,
    LONG_LEG,
    SIGNAL_TRANSFORM,
    CONTROL_N_GROUPS,
)

SORT_BLOCK_FIELDS: tuple[str, ...] = LEG_FIELDS + (COMBINER,)

# Sort-block enum menus (each carrying "other").
SORT_KIND_MENU: tuple[str, ...] = ("single", "independent", "conditional", "other")
BUCKETING_METHOD_MENU: tuple[str, ...] = ("equal_count", "breakpoint", "other")
STRIPE_AGGREGATION_MENU: tuple[str, ...] = ("equal", "count", "value", "other")
CONTROL_MISSING_POLICY_MENU: tuple[str, ...] = ("drop", "pool", "other")
LONG_LEG_MENU: tuple[str, ...] = ("lowest_signal", "highest_signal", "other")
SIGNAL_TRANSFORM_MENU: tuple[str, ...] = ("none", "winsorize", "standardize", "other")
COMBINER_MENU: tuple[str, ...] = ("single_leg", "equal_average", "other")

# ---------------------------------------------------------------------------
# Part 2 -- common block (28 fields, spec-level).
# ---------------------------------------------------------------------------

ELIGIBILITY_MISSING_POLICY = "eligibility_missing_policy"
RETURN_AVAILABILITY_POLICY = "return_availability_policy"
SIGNAL_LAG = "signal_lag"                       # int
LAG_CONVENTION = "lag_convention"
MIN_BONDS = "min_bonds"                         # int
MIN_BONDS_GRANULARITY = "min_bonds_granularity"
TIE_BREAK_POLICY = "tie_break_policy"
WEIGHTING_SCHEME = "weighting_scheme"
WEIGHTING_BASE = "weighting_base"
WEIGHT_TIMING = "weight_timing"
STRATEGY_SIDE = "strategy_side"
EMPTY_LEG_POLICY = "empty_leg_policy"
TRANSACTION_COST_CONVENTION = "transaction_cost_convention"
RETURN_LABEL = "return_label"
REBALANCE_FREQUENCY = "rebalance_frequency"
HOLDING_PERIOD = "holding_period"               # int
OVERLAP_CONVENTION = "overlap_convention"
COHORT_WEIGHTING = "cohort_weighting"
BURN_IN_POLICY = "burn_in_policy"
MISSING_RETURN_POLICY = "missing_return_policy"
REALISATION_MIN_SURVIVORS = "realisation_min_survivors"  # int
RETURN_COMPOUNDING = "return_compounding"
SIGNIFICANCE_CONVENTION = "significance_convention"
HAC_LAGS = "hac_lags"                           # int
ANNUALISATION = "annualisation"
RF_CONVENTION = "rf_convention"
BENCHMARK_MODEL = "benchmark_model"
EXPOST_TRIM = "expost_trim"

COMMON_FIELDS: tuple[str, ...] = (
    ELIGIBILITY_MISSING_POLICY,
    RETURN_AVAILABILITY_POLICY,
    SIGNAL_LAG,
    LAG_CONVENTION,
    MIN_BONDS,
    MIN_BONDS_GRANULARITY,
    TIE_BREAK_POLICY,
    WEIGHTING_SCHEME,
    WEIGHTING_BASE,
    WEIGHT_TIMING,
    STRATEGY_SIDE,
    EMPTY_LEG_POLICY,
    TRANSACTION_COST_CONVENTION,
    RETURN_LABEL,
    REBALANCE_FREQUENCY,
    HOLDING_PERIOD,
    OVERLAP_CONVENTION,
    COHORT_WEIGHTING,
    BURN_IN_POLICY,
    MISSING_RETURN_POLICY,
    REALISATION_MIN_SURVIVORS,
    RETURN_COMPOUNDING,
    SIGNIFICANCE_CONVENTION,
    HAC_LAGS,
    ANNUALISATION,
    RF_CONVENTION,
    BENCHMARK_MODEL,
    EXPOST_TRIM,
)

# Common-block enum menus (each carrying "other"). Integer / composite fields
# (signal_lag, min_bonds, holding_period, realisation_min_survivors, hac_lags,
# significance_convention, expost_trim) have no plain enum menu here -- their
# domains live in domains.yaml.
ELIGIBILITY_MISSING_POLICY_MENU: tuple[str, ...] = (
    "drop", "impute", "carry_forward", "other",
)
RETURN_AVAILABILITY_POLICY_MENU: tuple[str, ...] = (
    "require_next_month_return", "hold_to_recovery", "other",
)
LAG_CONVENTION_MENU: tuple[str, ...] = ("month_end", "business_day", "positional", "other")
MIN_BONDS_GRANULARITY_MENU: tuple[str, ...] = ("total", "per_group", "per_leg", "other")
TIE_BREAK_POLICY_MENU: tuple[str, ...] = ("deterministic_id", "random", "other")
WEIGHTING_SCHEME_MENU: tuple[str, ...] = ("equal", "value", "other")
WEIGHTING_BASE_MENU: tuple[str, ...] = ("par", "market_value", "other")
WEIGHT_TIMING_MENU: tuple[str, ...] = ("formation", "updated", "other")
STRATEGY_SIDE_MENU: tuple[str, ...] = ("long_short", "long_only", "short_only", "other")
EMPTY_LEG_POLICY_MENU: tuple[str, ...] = ("skip", "zero_fill", "one_sided", "other")
TRANSACTION_COST_CONVENTION_MENU: tuple[str, ...] = ("gross", "net_of_costs", "other")
RETURN_LABEL_MENU: tuple[str, ...] = ("formation", "realisation", "other")
REBALANCE_FREQUENCY_MENU: tuple[str, ...] = ("monthly", "quarterly", "weekly", "daily", "other")
OVERLAP_CONVENTION_MENU: tuple[str, ...] = ("overlapping", "non_overlapping", "other")
COHORT_WEIGHTING_MENU: tuple[str, ...] = ("equal", "value", "other")
BURN_IN_POLICY_MENU: tuple[str, ...] = ("keep", "drop_ramp", "scale", "other")
MISSING_RETURN_POLICY_MENU: tuple[str, ...] = (
    "drop_and_renormalise", "hold_to_recovery", "impute_default_return", "other",
)
RETURN_COMPOUNDING_MENU: tuple[str, ...] = ("arithmetic", "geometric", "other")
ANNUALISATION_MENU: tuple[str, ...] = ("arithmetic", "geometric", "other")
RF_CONVENTION_MENU: tuple[str, ...] = ("none", "subtract_rf", "other")
BENCHMARK_MODEL_MENU: tuple[str, ...] = ("none", "capm", "ff5", "bbw4", "other")
# expost_trim's method component (bounds are carried separately).
EXPOST_TRIM_MENU: tuple[str, ...] = ("none", "truncate", "winsorise", "other")

# significance_convention is a composite (t_stat method + kernel + hac_mode);
# the paper-facing t_stat menu component:
SIGNIFICANCE_TSTAT_MENU: tuple[str, ...] = ("hac_t_of_mean", "se_over_sqrt_t", "other")

# ---------------------------------------------------------------------------
# The authoritative set + markers.
# ---------------------------------------------------------------------------

# ALREADY_FINAL_PART2 = ALL 38 Part 2 fields (v1.1: +control_n_groups; nothing
# deferred). paper_facts is a SEPARATE block, not counted here.
ALREADY_FINAL_PART2: frozenset[str] = frozenset(SORT_BLOCK_FIELDS + COMMON_FIELDS)

# Markers (D18/D32b): the two sort-block fields a sort paper cannot fail to state.
MARKERS: frozenset[str] = frozenset({SORT_SIGNAL, N_GROUPS})

# Integer-valued Part 2 fields (no enum menu; range domain in domains.yaml).
INT_FIELDS: frozenset[str] = frozenset({
    N_GROUPS,
    CONTROL_N_GROUPS,
    SIGNAL_LAG,
    MIN_BONDS,
    HOLDING_PERIOD,
    REALISATION_MIN_SURVIVORS,
    HAC_LAGS,
})

# ---------------------------------------------------------------------------
# Part 2 (v1.1) -- paper_facts: a SEPARATE spec-level block, NOT part of the
# Part 2 execution fields. Extraction output (quote-bearing, RQ1-scorable) the
# *analysis* consumes; the adapter never reads it (Guard 2, §5). Kept out of
# ALREADY_FINAL_PART2 so no adapter/domain machinery ever iterates it.
# ---------------------------------------------------------------------------

SAMPLE_START = "sample_start"
SAMPLE_END = "sample_end"
UNIVERSE_FILTER = "universe_filter"
CLAIMED_HEADLINE_METRIC = "claimed_headline_metric"

PAPER_FACTS_FIELDS: frozenset[str] = frozenset({
    SAMPLE_START,
    SAMPLE_END,
    UNIVERSE_FILTER,
    CLAIMED_HEADLINE_METRIC,
})

# Sanity: the counts the schema doc asserts (v1.1).
assert len(SORT_BLOCK_FIELDS) == 10, "sort block must have 10 fields (v1.1: +control_n_groups)"
assert len(COMMON_FIELDS) == 28, "common block must have 28 fields"
assert len(ALREADY_FINAL_PART2) == 38, "Part 2 must have 38 fields total (v1.1)"
# paper_facts is a separate block, disjoint from the Part 2 execution fields.
assert PAPER_FACTS_FIELDS.isdisjoint(ALREADY_FINAL_PART2), "paper_facts must not overlap Part 2"
