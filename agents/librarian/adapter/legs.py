"""
Leg loop + combiner (D28) -- the adapter's structure translation.

Turns a ``StrategySpec``'s Part 2 into N ``build_quant_config`` calls (one per leg)
plus a combiner instruction, threading every field through silence routing
(``silence_routing``), the transform table (``transform_table``), and signal
resolution (``signal_resolution``):

  * spec-level (``common``) fields are translated ONCE and copied identically into
    every leg call (D28: identity transforms carry the librarian's Evidence
    untouched);
  * per-leg (``sort_block``) fields are translated per leg;
  * ``strategy_id = <parent label>_leg<i>`` (parent + ordinal, D28) keys per-field
    per-leg RQ1 scoring;
  * the combiner: ``single_leg`` pass-through / ``equal_average`` mean-over-available
    (adaptive divisor, ledger item 40) / ``other`` -> UNSUPPORTED_COMBINER.

Refusal collection (D29): the adapter collects ALL *adapter-layer* refusals into the
run ``_Batch`` (run-to-completion); the FACTORY stays first-hit per leg (its
``ConfigRefusal`` lands on the leg's ``LegCall.result``). Both feed
``AdaptResult.refused`` (D28: any leg refused -> whole strategy refuses).

Provenance (E1 / tag-reason registry): an IDENTITY transform passes the
librarian's ``Inherited`` through unchanged (STATED stays STATED); a VALUE_CHANGING
transform PRODUCES a new value tagged ``INFERRED(adapter, rule_id)`` -- the adapter
may not emit STATED (sole_producer=librarian) -- carrying the paper's source quote
in evidence for audit.
"""

from __future__ import annotations

from typing import Mapping

from agents.quant.config import (
    Binding,
    ConfigRefusal,
    Evidence,
    Inherited,
    RefusalCode,
    build_quant_config,
)
from agents.quant.config.concept_column import ConceptColumnTable

from ..registries.silence_policy import SilencePolicyTable
from ..registries.standing_substitutions import NO_STANDING_SUBS, StandingSubstitutionTable
from ..schema.strategy_spec import Leg, StrategySpec
from .authorisation import AuthorisationRecords, NO_AUTHORISATIONS
from .result import AppliedStandingSub, CombinerInstruction, LegCall, _Batch
from .signal_resolution import resolve_signal
from .silence_routing import FlagField, OmitField, Proceed, RefuseField, route_field
from .transform_table import Omit, Produced, Review, TransformTable

# The per-leg (sort_block) scalar fields (excludes sort_signal/control_axis, which
# are SignalRefs) and the spec-level (common) fields -- imported from the schema so
# the adapter stays in lockstep with the StrategySpec shape.
from ..schema.strategy_spec import _COMMON_INHERITED_FIELDS, _LEG_INHERITED_FIELDS

# The spec-level engine-hook fields whose translated kwarg is copied into every leg
# call (D28). weighting is a composite handled separately.
_COMMON_ENGINE_IDENTITY: tuple[tuple[str, str], ...] = (
    ("signal_lag", "signal_lag"),
    ("min_bonds", "min_bonds"),
    ("holding_period", "holding_period"),
)


def _refusal(strategy_id: str, code: RefusalCode, field: str, detail: str) -> ConfigRefusal:
    return ConfigRefusal(strategy_id, code, field, detail)


def _handle_refuse_or_flag(routed, strategy_id: str, field: str, batch: _Batch) -> None:
    """Record a RefuseField (hard) or FlagField (soft) from silence routing."""
    if isinstance(routed, RefuseField):
        batch.refuse(_refusal(strategy_id, routed.code, field, routed.detail))
    elif isinstance(routed, FlagField):
        batch.flag(_refusal(strategy_id, RefusalCode.REVIEW_REQUIRED, field, routed.detail))


def _wrap_inferred(value: object, rule_id: str, source: Inherited, note: str) -> Inherited:
    """A value_changing transform output: INFERRED(adapter, rule_id), carrying the
    paper's source quote + locator for audit (the adapter may not emit STATED)."""
    return Inherited(
        value,
        "INFERRED",
        Evidence(
            rule_id=rule_id,
            note=note,
            quote=source.evidence.quote,
            locator=source.evidence.locator,
        ),
    )


def _wrap_design(value: object, note: str) -> Inherited:
    """A standing-substitution transform output (contract §6): DESIGN carrying a note
    (the decision), never a paper quote (the value is a project convention, not a paper
    fact). Per D24 weakest-input derivation, a value derived from a DESIGN input is
    DESIGN(derived_from_design) -- so a standing-substituted weighting is DESIGN, not
    INFERRED. No locator (DESIGN needs a note, not a quote; D7 locators are STATED-only)."""
    return Inherited(value, "DESIGN", Evidence(note=note))


def _resolve_with_auth(signal_ref, concept_table, auth, paper_id, strategy_label, batch) -> Binding:
    """Resolve a SignalRef, honouring an authorisation binding_substitution (D27):
    a matching substitution binds the hand-chosen column and flags the run a
    ``variant``; otherwise the concept->column table decides."""
    concept_id = signal_ref.concept_id.value
    sub = auth.lookup_binding(paper_id, strategy_label, concept_id) if isinstance(concept_id, str) else None
    if sub is not None:
        batch.variant = True
        return resolve_signal(signal_ref, concept_table, override_column=sub.column)
    return resolve_signal(signal_ref, concept_table)


# ---------------------------------------------------------------------------
# Common (spec-level) fields -> the shared engine kwargs
# ---------------------------------------------------------------------------

def _adapt_common(
    spec: StrategySpec,
    tt: TransformTable,
    silence: SilencePolicyTable,
    batch: _Batch,
    standing_subs: StandingSubstitutionTable = NO_STANDING_SUBS,
) -> dict[str, object]:
    """Translate the 28 common fields once. Engine-hook fields (signal_lag,
    min_bonds, holding_period, weighting, expost_trim) produce shared kwargs;
    check-only fields only apply their refuse_on_stated / flag_on_stated overrides.
    Returns the shared kwargs dict (values are ``Inherited`` or ``None``=omit)."""
    part2 = spec.part2
    sid = spec.header.strategy_label.value
    shared: dict[str, object] = {}

    # Identity engine fields: pass the librarian's Inherited through unchanged.
    for field, kwarg in _COMMON_ENGINE_IDENTITY:
        routed = route_field(
            silence.policy_for("common", field), getattr(part2, field), control_present=False
        )
        if isinstance(routed, Proceed):
            shared[kwarg] = routed.inherited
        elif isinstance(routed, OmitField):
            shared[kwarg] = None
        else:
            _handle_refuse_or_flag(routed, sid, field, batch)
            shared[kwarg] = None

    # weighting: composite of weighting_scheme (+ weighting_base).
    shared["weighting"] = _adapt_weighting(part2, tt, silence, sid, batch, standing_subs)

    # expost_trim: v1 -> none/silent omit; else Review (unless a standing delegation authorises it).
    shared["trim"] = _adapt_trim(part2, tt, silence, sid, batch, standing_subs)

    # Remaining common fields are check-only: route them purely for the D23
    # refuse_on_stated / flag_on_stated overrides (no kwarg).
    engine_common = {"signal_lag", "min_bonds", "holding_period", "weighting_scheme",
                     "weighting_base", "expost_trim"}
    for field in _COMMON_INHERITED_FIELDS:
        if field in engine_common:
            continue
        routed = route_field(
            silence.policy_for("common", field), getattr(part2, field), control_present=False
        )
        _handle_refuse_or_flag(routed, sid, field, batch)

    return shared


def _adapt_weighting(part2, tt, silence, sid, batch, standing_subs=NO_STANDING_SUBS) -> Inherited | None:
    scheme_routed = route_field(
        silence.policy_for("common", "weighting_scheme"), part2.weighting_scheme, control_present=False
    )
    if isinstance(scheme_routed, RefuseField):
        _handle_refuse_or_flag(scheme_routed, sid, "weighting_scheme", batch)
        return None
    if isinstance(scheme_routed, OmitField):
        return None  # scheme silent -> factory par default (P5)

    base_routed = route_field(
        silence.policy_for("common", "weighting_base"), part2.weighting_base, control_present=False
    )
    base_value = base_routed.inherited.value if isinstance(base_routed, Proceed) else None

    # Standing substitution (contract §6): a project-wide convention may divert a STATED
    # weighting_base (e.g. market_value -> par) to a DESIGN value BEFORE the vocab lookup,
    # WITHOUT flagging the strategy a variant. Bright line (D27): only over a Proceed
    # (STATED/INFERRED) base -- never a silent/UNKNOWN one (that is an extraction gap, not a
    # divergence to authorise). The output is then DESIGN (D24: DESIGN input -> DESIGN).
    design_sub = None
    paper_base = base_value
    if isinstance(base_routed, Proceed):
        design_sub = standing_subs.substitution_for("weighting_base", base_value)
        if design_sub is not None:
            batch.standing_subs_applied.append(
                AppliedStandingSub(
                    field="weighting_base",
                    paper_value=paper_base,
                    engine_value=design_sub.replacement,
                    substitution_id=design_sub.id,
                )
            )
            base_value = design_sub.replacement

    outcome = tt.apply_weighting(scheme_routed.inherited.value, base_value)
    if isinstance(outcome, Omit):
        return None
    if isinstance(outcome, Review):
        batch.refuse(_refusal(sid, RefusalCode.REVIEW_REQUIRED, "weighting_scheme", outcome.detail))
        return None
    if design_sub is not None:
        # Value derived from a DESIGN-substituted base -> DESIGN(derived_from_design), D24.
        return _wrap_design(
            outcome.value,
            note=(f"standing substitution {design_sub.id!r}: weighting_base {paper_base!r} -> "
                  f"{design_sub.replacement!r} -> weighting={outcome.value!r} ({design_sub.source_decision})"),
        )
    # Produced: value_changing -> INFERRED(adapter). market_value/other ride through
    # to the factory, which refuses OUT_OF_ENUM_WEIGHTING (representability is its job).
    return _wrap_inferred(
        outcome.value, tt.transforms["weighting_scheme"].id, scheme_routed.inherited,
        note=f"adapter weighting: scheme={scheme_routed.inherited.value!r}, base={base_value!r} -> {outcome.value!r}",
    )


def _adapt_trim(part2, tt, silence, sid, batch, standing_subs=NO_STANDING_SUBS) -> Inherited | None:
    routed = route_field(
        silence.policy_for("common", "expost_trim"), part2.expost_trim, control_present=False
    )
    if isinstance(routed, RefuseField):
        _handle_refuse_or_flag(routed, sid, "expost_trim", batch)
        return None
    if isinstance(routed, OmitField):
        return None
    value = routed.inherited.value
    # Standing substitution (contract §6): the ex-post trim is the lab_trim bias toggle
    # (D32a). A STATED trim is DELEGATED to the toggle registry and omitted from the base
    # rulebook (kept trim-free), recorded as an authorised, REGISTERED difference (not a
    # silent drop). Bright line: only over a Proceed (STATED/INFERRED), never UNKNOWN (which
    # took the OmitField branch above). A STATED trim WITHOUT a delegation still Reviews.
    design_sub = standing_subs.substitution_for("expost_trim", value)
    if design_sub is not None:
        batch.standing_subs_applied.append(
            AppliedStandingSub(
                field="expost_trim",
                paper_value=value,
                engine_value=design_sub.replacement,
                substitution_id=design_sub.id,
            )
        )
        value = design_sub.replacement
    outcome = tt.apply_trim(value)
    if isinstance(outcome, Omit):
        return None
    if isinstance(outcome, Review):
        batch.refuse(_refusal(sid, RefusalCode.REVIEW_REQUIRED, "expost_trim", outcome.detail))
        return None
    # (no Produced path in v1 -- apply_trim only omits or reviews)
    return None  # pragma: no cover


# ---------------------------------------------------------------------------
# Per-leg (sort_block) fields -> the per-leg call
# ---------------------------------------------------------------------------

def _adapt_one_leg(
    spec: StrategySpec,
    leg: Leg,
    ordinal: int,
    shared: Mapping[str, object],
    tt: TransformTable,
    concept_table: ConceptColumnTable,
    silence: SilencePolicyTable,
    auth: AuthorisationRecords,
    batch: _Batch,
) -> LegCall:
    """Translate one leg into a ``LegCall`` (its kwargs + the factory outcome)."""
    parent = spec.header.strategy_label.value
    paper_id = spec.header.paper_id
    strategy_id = f"{parent}_leg{ordinal}"
    control_present = leg.control_axis is not None
    leg_refused_before_factory = False

    # --- Guard 1 belt (schema-v1.1 §3): re-assert the validator's sort-structure
    # consistency at intake, so a spec that somehow bypassed validation still cannot
    # configure a contradictory run. A STATED contradiction is disagreement-shaped ->
    # the manual-review lane (REVIEW_REQUIRED), never a silent proceed.
    sort_kind_value = leg.sort_kind.value
    if (sort_kind_value in ("independent", "conditional") and not control_present) or (
        control_present and sort_kind_value == "single"
    ):
        batch.refuse(_refusal(
            strategy_id, RefusalCode.REVIEW_REQUIRED, "sort_kind",
            f"sort-structure contradiction: sort_kind={sort_kind_value!r} vs "
            f"control_axis={'present' if control_present else 'absent'} (Guard 1 belt, §3)",
        ))
        leg_refused_before_factory = True

    # --- sort_signal -> score (Binding). Silence: refuse (the signal is identity).
    sig_routed = route_field(
        silence.policy_for("sort_block", "sort_signal"), leg.sort_signal.concept_id,
        control_present=control_present,
    )
    score: Binding | None = None
    if isinstance(sig_routed, Proceed):
        score = _resolve_with_auth(leg.sort_signal, concept_table, auth, paper_id, parent, batch)
    else:
        _handle_refuse_or_flag(sig_routed, strategy_id, "sort_signal", batch)
        leg_refused_before_factory = True

    # --- control_axis -> control (Binding | None). Structural: no silence row.
    control: Binding | None = None
    if leg.control_axis is not None:
        control = _resolve_with_auth(leg.control_axis, concept_table, auth, paper_id, parent, batch)

    # --- n_groups -> groups (identity). Needed by long_leg (top group = g-1).
    n_groups_routed = route_field(
        silence.policy_for("sort_block", "n_groups"), leg.n_groups, control_present=control_present
    )
    groups: Inherited | None = None
    n_groups_value: object = None
    if isinstance(n_groups_routed, Proceed):
        groups = n_groups_routed.inherited
        n_groups_value = n_groups_routed.inherited.value
    elif isinstance(n_groups_routed, OmitField):
        groups = None  # factory default groups=5
    else:
        _handle_refuse_or_flag(n_groups_routed, strategy_id, "n_groups", batch)
        leg_refused_before_factory = True

    # --- control_n_groups -> control_groups (identity, v1.1). A STATED value MUST
    # reach the factory (D24 totality): silent -> omit (factory default
    # control_groups=groups); STATED -> passed, so an asymmetric double sort (e.g.
    # 5x3) is not silently symmetrised. to_rulebook emits control_groups only when a
    # control axis is present, so a stray value on a single sort is harmless.
    cng_routed = route_field(
        silence.policy_for("sort_block", "control_n_groups"), leg.control_n_groups,
        control_present=control_present,
    )
    control_groups: Inherited | None = None
    if isinstance(cng_routed, Proceed):
        control_groups = cng_routed.inherited
    elif isinstance(cng_routed, OmitField):
        control_groups = None  # factory default control_groups=groups
    else:
        _handle_refuse_or_flag(cng_routed, strategy_id, "control_n_groups", batch)
        leg_refused_before_factory = True

    # --- long_leg -> long_group / short_group (value_changing). Silence: refuse.
    long_group: Inherited | None = None
    short_group: Inherited | None = None
    ll_routed = route_field(
        silence.policy_for("sort_block", "long_leg"), leg.long_leg, control_present=control_present
    )
    if isinstance(ll_routed, Proceed):
        outcome = tt.apply_long_leg(ll_routed.inherited.value, n_groups_value)
        if isinstance(outcome, Produced):
            rid = tt.transforms["long_leg"].id
            long_group = _wrap_inferred(
                outcome.value["long_group"], rid, ll_routed.inherited,
                note=f"adapter direction: long_leg={ll_routed.inherited.value!r}, n_groups={n_groups_value!r} -> long_group",
            )
            short_group = _wrap_inferred(
                outcome.value["short_group"], rid, ll_routed.inherited,
                note=f"adapter direction: long_leg={ll_routed.inherited.value!r}, n_groups={n_groups_value!r} -> short_group",
            )
        else:  # Review -- off-menu direction or silent n_groups
            batch.refuse(_refusal(strategy_id, RefusalCode.REVIEW_REQUIRED, "long_leg", outcome.detail))
            leg_refused_before_factory = True
    else:
        _handle_refuse_or_flag(ll_routed, strategy_id, "long_leg", batch)
        leg_refused_before_factory = True

    # --- per-leg check-only fields: refuse_on_stated / flag_on_stated only.
    for field in _LEG_INHERITED_FIELDS:
        if field in ("n_groups", "long_leg", "control_n_groups"):
            continue  # engine-hook, handled above
        routed = route_field(
            silence.policy_for("sort_block", field), getattr(leg, field), control_present=control_present
        )
        _handle_refuse_or_flag(routed, strategy_id, field, batch)

    # --- assemble the factory kwargs (shared spec-level fields copied in, D28).
    kwargs: dict[str, object] = {
        "score": score,
        "control": control,
        "groups": groups,
        "control_groups": control_groups,  # v1.1: STATED -> passed; silent -> None (factory default = groups)
        "long_group": long_group,
        "short_group": short_group,
        "weighting": shared.get("weighting"),
        "signal_lag": shared.get("signal_lag"),
        "min_bonds": shared.get("min_bonds"),
        "holding_period": shared.get("holding_period"),
        "trim": shared.get("trim"),
    }

    # --- forward the call, UNLESS the adapter already refused this leg before the
    # factory could be reached (e.g. no score binding to pass).
    if leg_refused_before_factory or score is None:
        return LegCall(strategy_id=strategy_id, kwargs=kwargs, result=None)

    result = build_quant_config(
        strategy_id,
        score,
        control=control,
        groups=groups,
        control_groups=control_groups,
        weighting=kwargs["weighting"],  # type: ignore[arg-type]
        signal_lag=kwargs["signal_lag"],  # type: ignore[arg-type]
        min_bonds=kwargs["min_bonds"],  # type: ignore[arg-type]
        long_group=long_group,
        short_group=short_group,
        trim=kwargs["trim"],  # type: ignore[arg-type]
        holding_period=kwargs["holding_period"],  # type: ignore[arg-type]
    )
    return LegCall(strategy_id=strategy_id, kwargs=kwargs, result=result)


# ---------------------------------------------------------------------------
# Combiner (D28)
# ---------------------------------------------------------------------------

def _adapt_combiner(spec: StrategySpec, tt: TransformTable, batch: _Batch) -> CombinerInstruction | None:
    part2 = spec.part2
    sid = spec.header.strategy_label.value
    n_legs = len(part2.legs)
    kind_inh = part2.combiner.kind
    kind = kind_inh.value

    if kind_inh.tag == "UNKNOWN":
        reason = kind_inh.evidence.unknown_reason
        if reason in ("disagreement", "quote_match_failure", "single_response"):
            batch.refuse(_refusal(sid, RefusalCode.REVIEW_REQUIRED, "combiner",
                                  f"combiner extraction needs review (unknown_reason={reason!r})"))
            return None
        # silent: a multi-leg construction with a silent combiner refuses (D32c);
        # a single leg takes the documented default (single_leg).
        if n_legs > 1:
            batch.refuse(_refusal(sid, RefusalCode.REFUSED_ON_SILENCE, "combiner",
                                  "multi-leg construction with a silent combiner -> refuse"))
            return None
        return CombinerInstruction(kind="single_leg")

    if kind in tt.combiner:
        spec_row = tt.combiner[kind]
        return CombinerInstruction(kind=kind, divisor=spec_row.get("divisor"))

    # 'other' (or off-menu) -> adapter-owned refusal (D28).
    batch.refuse(_refusal(sid, RefusalCode.UNSUPPORTED_COMBINER, "combiner",
                          f"combiner={kind!r} is not representable (engine: single_leg | equal_average)"))
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def adapt_legs(
    spec: StrategySpec,
    *,
    transform_table: TransformTable,
    concept_table: ConceptColumnTable,
    silence_table: SilencePolicyTable,
    batch: _Batch,
    auth: AuthorisationRecords = NO_AUTHORISATIONS,
    standing_subs: StandingSubstitutionTable = NO_STANDING_SUBS,
) -> tuple[tuple[LegCall, ...], CombinerInstruction | None]:
    """Translate a spec's Part 2 into per-leg factory calls + a combiner
    instruction, collecting adapter refusals/flags into ``batch``."""
    shared = _adapt_common(spec, transform_table, silence_table, batch, standing_subs)
    leg_calls = tuple(
        _adapt_one_leg(spec, leg, i, shared, transform_table, concept_table, silence_table, auth, batch)
        for i, leg in enumerate(spec.part2.legs)
    )
    combiner = _adapt_combiner(spec, transform_table, batch)
    return leg_calls, combiner
