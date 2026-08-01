"""G0 — proposal integrity (spec §9). Deterministic; the entry gate. Checks IN ORDER and stops at
the first failure with a structured RefusalCode (no performance is computed on a failed proposal):

  1. schema_valid        — config delta structurally complete           INVALID_SCHEMA
  2. mechanism_authorised— mechanism in the locked library              UNKNOWN_MECHANISM
  3. template_supported  — template registered, allowed by the mechanism,
                           and every config-delta field within its enum  UNSUPPORTED_TEMPLATE /
                                                                          FIELD_OUT_OF_DOMAIN
  4. toggles_preserved   — no bias toggle referenced (no ON->OFF), no
                           forbidden change (base b* = all toggles ON)    TOGGLE_REVERSAL /
                                                                          FORBIDDEN_CHANGE
  5. inputs_available    — every required input present in the data       MISSING_INPUT
  6. not_duplicate       — equivalence key not already seen               DUPLICATE_PROPOSAL

Sets the six G0 booleans. `TOGGLE_IDS` is a magnitude-free import (not on the R2 wall list).
"""

from __future__ import annotations

from agents.auditor.schemas.toggle import TOGGLE_IDS

from ..schemas.equivalence import equivalence_key
from ..schemas.outcomes import RefusalCode
from .gate_result import GateOutcome

_G0_BOOLEANS = (
    "schema_valid", "mechanism_authorised", "template_supported",
    "toggles_preserved", "inputs_available", "not_duplicate",
)
_TOGGLES = frozenset(TOGGLE_IDS)
# Template forbidden-change fields other than a bias toggle (that is TOGGLE_REVERSAL).
_FORBIDDEN_FIELDS = frozenset({"sample_start", "sample_end", "holdout_path", "signal_definition"})


def validate_g0(proposal, case, library, *, seen_keys, available_variables) -> GateOutcome:
    b = {name: True for name in _G0_BOOLEANS}

    def fail(field_name: str, code: RefusalCode) -> GateOutcome:
        b[field_name] = False
        return GateOutcome("G0", passed=False, booleans=dict(b), refusal_code=code)

    cd = proposal.config_delta
    # 1 — schema completeness (Pydantic already decoded; here the delta must be fully specified)
    #     AND the proposal must belong to THIS case (right case_id + parent strategy).
    if (None in (cd.conditioning_variable, cd.conditioning_lag_months, cd.interaction_form)
            or proposal.case_id != case.case_id
            or proposal.parent_strategy_id != case.strategy_id):
        return fail("schema_valid", RefusalCode.INVALID_SCHEMA)

    # 2 — mechanism in the locked library.
    try:
        mech = library.mechanism(proposal.mechanism_ref)
    except KeyError:
        return fail("mechanism_authorised", RefusalCode.UNKNOWN_MECHANISM)

    # 3 — template registered, allowed by the mechanism, fields within domain.
    template = library.templates.get(proposal.template_ref)
    if template is None or proposal.template_ref not in mech["allowed_templates"]:
        return fail("template_supported", RefusalCode.UNSUPPORTED_TEMPLATE)
    pf = template["permitted_fields"]
    if (cd.conditioning_variable not in pf["conditioning_variable"]["allowed"]
            or cd.conditioning_lag_months not in pf["conditioning_lag_months"]["allowed"]
            or cd.interaction_form not in pf["interaction_form"]["allowed"]):
        return fail("template_supported", RefusalCode.FIELD_OUT_OF_DOMAIN)

    # 4 — monotonicity + no forbidden change. The base b* is the corrected lattice point (all
    #     applicable toggles ON); a proposal may never reference a bias toggle or a forbidden field.
    refs = {cd.conditioning_variable, *proposal.required_inputs}
    if refs & _TOGGLES:
        return fail("toggles_preserved", RefusalCode.TOGGLE_REVERSAL)
    forbidden = _FORBIDDEN_FIELDS | (set(template.get("forbidden_changes", [])) - {"any_bias_toggle"})
    if refs & forbidden:
        return fail("toggles_preserved", RefusalCode.FORBIDDEN_CHANGE)

    # 5 — required inputs available in the data.
    for inp in proposal.required_inputs:
        if inp not in available_variables:
            return fail("inputs_available", RefusalCode.MISSING_INPUT)

    # 6 — not a duplicate (same equivalence key as generation's dedup, §4).
    if equivalence_key(proposal) in seen_keys:
        return fail("not_duplicate", RefusalCode.DUPLICATE_PROPOSAL)

    return GateOutcome("G0", passed=True, booleans=dict(b), refusal_code=None)
