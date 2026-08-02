"""G1a — compilation (spec §9; finding F8). Maps a validated ExtensionProposal to an EXECUTABLE
form, or refuses with COMPILATION_FAILED:

  * T4 (execution.mode == double_sort) — the NATIVE `control`/`control_groups` double-sort. The
    audited engine refuses a double-sort held for more than one month, so this compiles only at
    holding_period == 1; the control variable must be a panel column.
  * T1/T2 (month_filter) and T3 (row_filter) — a Scientist-side PANEL TRANSFORM over the corrected
    panel, running the UNMODIFIED engine on the transformed panel (F8). Structurally always
    compilable; the ex-ante / lag obligations are checked at G2.

G1a produces a `CompiledExtension` SPEC (what to run), not the realised QuantConfig/panel — the
engine runs it at G1b. Sets the `compiled` boolean.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..schemas.outcomes import RefusalCode
from .gate_result import GateOutcome


@dataclass(frozen=True)
class PanelTransform:
    kind: str              # "month_filter" (T1/T2) | "row_filter" (T3)
    variable: str          # the conditioning variable partitioned/filtered on
    lag_months: int
    form: str              # interaction_form


@dataclass(frozen=True)
class CompiledExtension:
    proposal_id: str
    parent_strategy_id: str
    template_mode: str                    # month_filter | row_filter | double_sort
    provenance_rule: str                  # from the template's parameter_provenance
    base_quant_config_ref: str            # the corrected parent config the extension builds on
    control: str | None = None            # T4 native double-sort
    control_groups: int | None = None
    panel_transform: PanelTransform | None = None


def _provenance_rule(template: dict) -> str:
    """The single parameter-provenance rule (expanding_window_past_only | ex_ante_committed) — the
    first (only) entry under the template's `parameter_provenance` block (INVARIANT 3)."""
    prov = template.get("parameter_provenance", {})
    for entry in prov.values():
        if isinstance(entry, dict) and "rule" in entry:
            return entry["rule"]
    return "unknown"


def compile_g1a(
    proposal, template, case, *, holding_period: int, available_variables, control_groups: int = 5,
):
    """Return (GateOutcome, CompiledExtension | None). On failure the outcome carries
    COMPILATION_FAILED and the CompiledExtension is None."""
    mode = template.get("execution", {}).get("mode")
    cd = proposal.config_delta
    rule = _provenance_rule(template)

    def fail() -> tuple[GateOutcome, None]:
        return GateOutcome("G1a", passed=False, booleans={"compiled": False},
                           refusal_code=RefusalCode.COMPILATION_FAILED), None

    common = dict(proposal_id=proposal.proposal_id, parent_strategy_id=proposal.parent_strategy_id,
                  template_mode=mode, provenance_rule=rule,
                  base_quant_config_ref=case.corrected_quant_config_ref)

    if mode == "double_sort":
        # Native control double-sort — engine refuses holding>1; control must be a panel column.
        if holding_period != 1 or cd.conditioning_variable not in available_variables:
            return fail()
        ext = CompiledExtension(**common, control=cd.conditioning_variable,
                                control_groups=control_groups)
    elif mode in ("month_filter", "row_filter"):
        ext = CompiledExtension(**common, panel_transform=PanelTransform(
            kind=mode, variable=cd.conditioning_variable,
            lag_months=cd.conditioning_lag_months, form=cd.interaction_form))
    else:
        return fail()

    return GateOutcome("G1a", passed=True, booleans={"compiled": True}), ext
