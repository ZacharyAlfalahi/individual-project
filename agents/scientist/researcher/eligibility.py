"""Filter-first eligibility (spec §6): a deterministic filter runs BEFORE any ranking, because
pure vector search happily returns a conceptually adjacent mechanism that cannot be executed.

A mechanism is ELIGIBLE for a strategy iff:
  1. the strategy's family is in the mechanism's applicability.strategy_families;
  2. at least one of its allowed_templates exists in the registry; and
  3. EVERY required_input has at least one conditioning variable that is BOTH available in the
     data AND reachable through one of the mechanism's allowed_templates (the template's
     conditioning_variable enum contains it).

Condition 3 is exactly the "authored backwards from the census" contract enforced per strategy:
an eligible mechanism has at least one executable (template, variable) option for each input.
Pure function — availability is injected (see library.available_conditioning_variables).
"""

from __future__ import annotations

from dataclasses import dataclass


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
    templates: dict,
    variable_families: dict,
    available_variables: set[str],
) -> EligibilityResult:
    reasons: list[str] = []

    if strategy_family not in mechanism["applicability"]["strategy_families"]:
        reasons.append(f"strategy_family {strategy_family!r} not supported")

    allowed = [t for t in mechanism["allowed_templates"] if t in templates]
    if not allowed:
        reasons.append("no allowed_template exists in the registry")

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
            templates=templates,
            variable_families=variable_families,
            available_variables=available_variables,
        ).eligible
    ]
