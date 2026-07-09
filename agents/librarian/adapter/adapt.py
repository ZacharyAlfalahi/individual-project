"""
adapt_spec (D25) -- the deterministic StrategySpec -> engine top-level.

A pure function from a paper-language ``StrategySpec`` to an ``AdaptResult`` (per-leg
``build_quant_config`` calls + a combiner instruction + collected refusals/flags).
It orchestrates the three translations (vocabulary/structure/signal) via
``legs.adapt_legs``, having first:

  1. asserted the registry-version handshake (D27(1)): the spec's stamped
     ``registry_version`` must equal the concept->column table's, else the run
     refuses upfront (a version drift means the columns may no longer mirror the
     concepts) -- a single strategy-level REVIEW_REQUIRED refusal;
  2. applied any human authorisation ``field_override`` records (D27): a matching
     override replaces a Part 2 field with a DESIGN value and flags the strategy a
     ``variant``. The bright line is enforced structurally -- an override may only
     divert a fact the spec carries as STATED/INFERRED (a *correctly extracted*
     fact), never an UNKNOWN (those go to review, not override).

No LLM, no chained transforms, no engine knowledge beyond the factory it calls.
The tables are injected (tests) or loaded from their frozen files (default).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from agents.quant.config import ConfigRefusal, Evidence, Inherited, RefusalCode
from agents.quant.config.concept_column import ConceptColumnTable, load_concept_column_table

from ..errors import LibrarianSchemaError
from ..registries.silence_policy import SilencePolicyTable, load_silence_policy_table
from ..schema.fields import COMMON_FIELDS
from ..schema.strategy_spec import StrategySpec
from .authorisation import AuthorisationRecords, load_authorisation_records
from .legs import adapt_legs
from .result import AdaptResult, _Batch
from .transform_table import TransformTable, load_transform_table

_COMMON_FIELD_SET: frozenset[str] = frozenset(COMMON_FIELDS)


def _apply_field_overrides(
    spec: StrategySpec, auth: AuthorisationRecords, batch: _Batch
) -> StrategySpec:
    """Apply matching ``field_override`` records (D27): replace a Part 2 common
    field with a DESIGN value, flag the strategy a ``variant``. Enforces the bright
    line (never override an UNKNOWN -> that is a suspected extraction error, which
    goes to review)."""
    paper_id = spec.header.paper_id
    label = spec.header.strategy_label.value
    matching = [
        o for o in auth.overrides if (o.paper_id, o.strategy_label) == (paper_id, label)
    ]
    if not matching:
        return spec

    replacements: dict[str, Inherited] = {}
    for ov in matching:
        if ov.field not in _COMMON_FIELD_SET:
            # Per-leg (sort-block) overrides need a leg selector the v1 record
            # format does not carry; refuse to guess.
            raise LibrarianSchemaError(
                f"field_override on {ov.field!r}: v1 supports common (spec-level) field "
                "overrides only (a per-leg override needs a leg selector, not in the v1 format)"
            )
        current = getattr(spec.part2, ov.field)
        if current.tag == "UNKNOWN":
            # The bright line (D27): an override diverts a CORRECTLY extracted fact;
            # it may never patch a suspected extraction error (that goes to review).
            raise LibrarianSchemaError(
                f"field_override on {ov.field!r} targets an UNKNOWN field -- the bright line "
                "(D27) forbids overriding a suspected extraction error; route it to review instead"
            )
        replacements[ov.field] = Inherited(ov.value, "DESIGN", Evidence(note=ov.note))
        batch.variant = True

    part2 = dataclasses.replace(spec.part2, **replacements)
    return dataclasses.replace(spec, part2=part2)


def adapt_spec(
    spec: StrategySpec,
    *,
    concept_table: ConceptColumnTable | None = None,
    silence_table: SilencePolicyTable | None = None,
    transform_table: TransformTable | None = None,
    auth: AuthorisationRecords | None = None,
    data_root: str | Path | None = None,
) -> AdaptResult:
    """Adapt one ``StrategySpec`` to its per-leg factory calls + combiner (D25)."""
    concept_table = concept_table if concept_table is not None else load_concept_column_table()
    silence_table = silence_table if silence_table is not None else load_silence_policy_table()
    transform_table = transform_table if transform_table is not None else load_transform_table()
    auth = auth if auth is not None else load_authorisation_records()

    batch = _Batch()
    label = spec.header.strategy_label.value

    # (1) Registry-version handshake (D27(1)) -- refuse the run upfront on drift.
    if spec.header.registry_version != concept_table.registry_version:
        batch.refuse(
            ConfigRefusal(
                label,
                RefusalCode.REVIEW_REQUIRED,
                "registry_version",
                f"registry version drift: spec stamped {spec.header.registry_version!r} but the "
                f"concept->column table mirrors {concept_table.registry_version!r} -- the columns "
                "may no longer mirror the concepts; refusing upfront",
            )
        )
        return AdaptResult(strategy_label=label, refusals=tuple(batch.refusals))

    # (2) Apply human authorisation field_overrides (-> DESIGN + variant).
    spec = _apply_field_overrides(spec, auth, batch)

    # (3) The three translations, per leg + combiner (D25/D28).
    leg_calls, combiner = adapt_legs(
        spec,
        transform_table=transform_table,
        concept_table=concept_table,
        silence_table=silence_table,
        batch=batch,
        auth=auth,
    )

    return AdaptResult(
        strategy_label=label,
        leg_calls=leg_calls,
        combiner=combiner,
        refusals=tuple(batch.refusals),
        flags=tuple(batch.flags),
        variant=batch.variant,
    )
