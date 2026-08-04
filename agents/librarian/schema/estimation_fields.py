"""
Estimation-block field names, enum menus, and the per-field-type dispatch table
-- the single source of truth for the *fitted latent-factor model* construction
variant (schema v1.2), selected when ``formation_structure == estimated_factor_model``.

This is the Part-2 construction variant KPP (Kelly-Palhares-Pruitt 2023, IPCA on
corporate bonds) needs: the sort schema's ``legs[] + combiner`` (fields.py) does
not apply to a fitted model, so the estimation block replaces it. To keep the
sort schema's version signature (fields.py's 10/28/38 count assertions) BYTE-
IDENTICAL, these constants live in their own module with their own count
assertion -- nothing here touches ``fields.py``.

Two structures:

  * **EstimationBlock** -- the 11 estimation fields (below), each an
    ``Inherited[...]`` on the ``EstimationBlock`` dataclass (strategy_spec.py).
  * **InstrumentSet** -- a *list* of ``InstrumentRef`` (one per characteristic;
    KPP's Table A.I lists ~29). Each ``InstrumentRef`` reuses the SignalRef
    locator discipline (``concept_id: Inherited`` + ``as_described`` quotes) plus
    per-instrument ``source_class`` / ``transform`` / ``lag`` Inheriteds.

Every enum menu carries the ``"other"`` escape (P3), mirroring fields.py.

Field->type dispatch (``ESTIMATION_FIELD_TYPES``) lives here rather than in
``data/prompts/manifest.yaml`` because Scope A runs no extraction round-trip --
the G3 scorer's estimation path reads this table directly. Scope B (the adapter
round-trip) would add ``estimation_enum`` / ``instrument_ref`` field-types +
``fields:`` rows to the manifest with frozen prompt hashes.

Reconciliation (the brief's "free insurance"): each estimation field's engine
counterpart in ``agents/quant/library/configs/kpp_ipca.yaml`` is noted inline.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Estimation-block field names (schema v1.2). 11 fields.
# ---------------------------------------------------------------------------

MODEL_FAMILY = "model_family"                               # enum   (identity)
ESTIMATION_ALGORITHM = "estimation_algorithm"               # enum   -> estimation.init (ALS)
N_FACTORS_TESTED = "n_factors_tested"                       # int_set-> model.K sweep (Table BI)
N_FACTORS_PREFERRED = "n_factors_preferred"                 # int    -> model.K
INTERCEPT_SPEC = "intercept_spec"                           # enum   -> model.alpha (Gamma_alpha)
RETURN_VARIABLE = "return_variable"                         # prose  -> scaling.lane (DtS)
CHARACTERISTIC_PREPROCESSING = "characteristic_preprocessing"  # prose -> data_contract.rank_map
MANAGED_PORTFOLIO_CONSTRUCTION = "managed_portfolio_construction"  # prose (Section 2 method)
ESTIMATION_MODE = "estimation_mode"                         # enum   -> estimation.mode
OOS_SPLIT = "oos_split"                                     # int    -> window.oos_burn_in_months
INFERENCE_METHOD = "inference_method"                       # enum   -> bootstrap{sims,dof}

# Order matters only for to_dict readability (mirrors fields.py's COMMON_FIELDS).
ESTIMATION_FIELDS: tuple[str, ...] = (
    MODEL_FAMILY,
    ESTIMATION_ALGORITHM,
    N_FACTORS_TESTED,
    N_FACTORS_PREFERRED,
    INTERCEPT_SPEC,
    RETURN_VARIABLE,
    CHARACTERISTIC_PREPROCESSING,
    MANAGED_PORTFOLIO_CONSTRUCTION,
    ESTIMATION_MODE,
    OOS_SPLIT,
    INFERENCE_METHOD,
)

# ---------------------------------------------------------------------------
# Enum menus (each carrying "other").
# ---------------------------------------------------------------------------

MODEL_FAMILY_MENU: tuple[str, ...] = ("instrumented_pca", "other")
ESTIMATION_ALGORITHM_MENU: tuple[str, ...] = ("alternating_least_squares", "other")
# restricted = Gamma_alpha constrained to 0 (the no-alpha null); unrestricted =
# nonzero Gamma_alpha (the alternative); both = the paper estimates/tests both.
INTERCEPT_SPEC_MENU: tuple[str, ...] = ("restricted", "unrestricted", "both", "other")
ESTIMATION_MODE_MENU: tuple[str, ...] = ("in_sample", "recursive_oos", "both", "other")
INFERENCE_METHOD_MENU: tuple[str, ...] = ("wild_bootstrap", "none", "other")

# name -> its enum menu (only the enum fields appear here).
ESTIMATION_MENUS: dict[str, tuple[str, ...]] = {
    MODEL_FAMILY: MODEL_FAMILY_MENU,
    ESTIMATION_ALGORITHM: ESTIMATION_ALGORITHM_MENU,
    INTERCEPT_SPEC: INTERCEPT_SPEC_MENU,
    ESTIMATION_MODE: ESTIMATION_MODE_MENU,
    INFERENCE_METHOD: INFERENCE_METHOD_MENU,
}

# ---------------------------------------------------------------------------
# Field-type partition (the G3 scorer's dispatch table).
# ---------------------------------------------------------------------------

ESTIMATION_ENUM_FIELDS: frozenset[str] = frozenset({
    MODEL_FAMILY,
    ESTIMATION_ALGORITHM,
    INTERCEPT_SPEC,
    ESTIMATION_MODE,
    INFERENCE_METHOD,
})
ESTIMATION_INT_FIELDS: frozenset[str] = frozenset({
    N_FACTORS_PREFERRED,
    OOS_SPLIT,
})
ESTIMATION_SET_FIELDS: frozenset[str] = frozenset({
    N_FACTORS_TESTED,
})
# Declared-weaker rubric fields (like method_summary / universe_filter): scored
# NO_POLICY, never enum/int-compared.
ESTIMATION_PROSE_FIELDS: frozenset[str] = frozenset({
    RETURN_VARIABLE,
    CHARACTERISTIC_PREPROCESSING,
    MANAGED_PORTFOLIO_CONSTRUCTION,
})

# name -> {"enum" | "int" | "int_set" | "prose"} -- the scorer reads this directly.
ESTIMATION_FIELD_TYPES: dict[str, str] = {
    **{name: "enum" for name in ESTIMATION_ENUM_FIELDS},
    **{name: "int" for name in ESTIMATION_INT_FIELDS},
    **{name: "int_set" for name in ESTIMATION_SET_FIELDS},
    **{name: "prose" for name in ESTIMATION_PROSE_FIELDS},
}

# ---------------------------------------------------------------------------
# Instrument (per-characteristic) field names + menus.
# ---------------------------------------------------------------------------

CONCEPT_ID = "concept_id"       # Inherited[str] -- instrument-registry id (or "unrecognised")
SOURCE_CLASS = "source_class"   # Inherited[str] -- bond | equity | accounting | macro
TRANSFORM = "transform"         # Inherited[str] -- e.g. rank-standardize
LAG = "lag"                     # Inherited[str] -- reporting / availability lag

# The per-instrument Inherited fields (concept_id handled specially, like a
# SignalRef; the rest are plain Inherited). Order = to_dict readability.
INSTRUMENT_FIELDS: tuple[str, ...] = (
    CONCEPT_ID,
    SOURCE_CLASS,
    TRANSFORM,
    LAG,
)
# The Inherited fields OTHER than concept_id (which rides on InstrumentRef.concept_id).
INSTRUMENT_INHERITED_FIELDS: tuple[str, ...] = (
    SOURCE_CLASS,
    TRANSFORM,
    LAG,
)

SOURCE_CLASS_MENU: tuple[str, ...] = ("bond", "equity", "accounting", "macro", "other")
# Scope-A-inert metadata (data-runnability tag; irrelevant to extraction grading;
# becomes load-bearing only in Scope B). Kept here so the registry/gold vocab is
# single-sourced.
DERIVABLE_ON_MENU: tuple[str, ...] = ("TRACE", "FISD", "CRSP_Compustat", "macro", "other")

# ---------------------------------------------------------------------------
# Sanity: the count the schema doc (v1.2) asserts. This is the estimation block's
# OWN version signature; it is deliberately INDEPENDENT of fields.py's 10/28/38
# so the sort schema signature stays byte-identical.
# ---------------------------------------------------------------------------

assert len(ESTIMATION_FIELDS) == 11, "estimation block must have 11 fields (schema v1.2)"
assert set(ESTIMATION_FIELD_TYPES) == set(ESTIMATION_FIELDS), (
    "every estimation field must have exactly one field-type"
)
assert (
    len(ESTIMATION_ENUM_FIELDS)
    + len(ESTIMATION_INT_FIELDS)
    + len(ESTIMATION_SET_FIELDS)
    + len(ESTIMATION_PROSE_FIELDS)
) == 11, "the field-type partition must cover all 11 fields disjointly"
