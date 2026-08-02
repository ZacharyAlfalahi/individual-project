"""Filter-first eligibility (spec §6): a deterministic filter runs BEFORE any ranking, because
pure vector search happily returns a conceptually adjacent mechanism that cannot be executed.

A mechanism is ELIGIBLE for a strategy iff:
  1. the strategy's family is in the mechanism's applicability.strategy_families;
  2. at least one of its allowed_templates is ENGINE-EXECUTABLE for this strategy (F8); and
  3. EVERY required_input has at least one conditioning variable that is BOTH available in the
     data AND reachable through one of the mechanism's EXECUTABLE allowed_templates.

Condition 2 is finding F8: a template counts only if the audited engine can actually RUN it — a
native double-sort (T4) needs holding_period=1; the panel-transform templates (T1/T2 month filter,
T3 row filter) run for any holding. The original census checked template-enum x variable
availability but NOT executability, so it over-counted (F8 / SC-SCI-7). Pure function.
"""

from __future__ import annotations

from dataclasses import dataclass


def template_executable(template: dict, holding_period: int) -> bool:
    """F8 engine-executability. A panel-transform template (T1/T2/T3) runs for any holding; a
    native template (T4 double-sort) runs only when the engine allows it — the audited engine
    refuses a double-sort held for more than one month (UNSUPPORTED_COMBINATION)."""
    ex = template.get("execution", {})
    if ex.get("panel_transform"):
        return True
    if ex.get("native_engine"):
        req = ex.get("requires_holding_period")
        return req is None or holding_period == req
    return False


@dataclass(frozen=True)
class EligibilityResult:
    mechanism_id: str
    eligible: bool
    reasons: tuple[str, ...]              # non-empty iff ineligible
    reachable: dict                       # variable_family -> [(template_id, variable)]


def evaluate(
    mechanism: dict,
    *,
    strategy_family: str,
    holding_period: int,
    templates: dict,
    variable_families: dict,
    available_variables: set[str],
) -> EligibilityResult:
    reasons: list[str] = []

    if strategy_family not in mechanism["applicability"]["strategy_families"]:
        reasons.append(f"strategy_family {strategy_family!r} not supported")

    # F8 — only ENGINE-EXECUTABLE templates count (native double-sort needs holding_period=1).
    allowed = [
        t for t in mechanism["allowed_templates"]
        if t in templates and template_executable(templates[t], holding_period)
    ]
    if not allowed:
        reasons.append(f"no engine-executable allowed_template at holding_period={holding_period}")

    reachable: dict[str, list[tuple[str, str]]] = {}
    for ri in mechanism["applicability"]["required_inputs"]:
        fam = ri["variable_family"]
        fam_vars = set(variable_families.get(fam, []))
        options: list[tuple[str, str]] = []
        for tid in allowed:
            enum = set(templates[tid]["permitted_fields"]["conditioning_variable"]["allowed"])
            for v in sorted(fam_vars & enum & available_variables):
                options.append((tid, v))
        reachable[fam] = options
        if not options:
            reasons.append(f"required_input {fam!r} has no available+reachable variable")

    return EligibilityResult(
        mechanism_id=mechanism["mechanism_id"],
        eligible=not reasons,
        reasons=tuple(reasons),
        reachable=reachable,
    )


def eligible_mechanisms(
    mechanisms,
    *,
    strategy_family: str,
    holding_period: int,
    templates: dict,
    variable_families: dict,
    available_variables: set[str],
) -> list[dict]:
    """The eligible subset for a strategy (filter-first; retrieval ranks WITHIN this set)."""
    return [
        m
        for m in mechanisms
        if evaluate(
            m,
            strategy_family=strategy_family,
            holding_period=holding_period,
            templates=templates,
            variable_families=variable_families,
            available_variables=available_variables,
        ).eligible
    ]
