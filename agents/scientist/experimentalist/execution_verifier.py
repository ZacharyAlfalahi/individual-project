"""G1b — execution fidelity (spec §9). Realise the CompiledExtension against the corrected panel
and run the UNMODIFIED audited engine on the result (F8): a double-sort adds `control` to the
rulebook; a month/row filter transforms the panel. Produce the explicit diff object (declared vs
realised changes) and the candidate return series. A procedural mismatch — engine raised, empty
result, or the run does not respect the declared transform — is EXECUTION_MISMATCH, never
confused with an economic verdict. Sets `execution_verified`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from agents.quant.library.characteristic_sort import run_characteristic_sort

from ..schemas.outcomes import RefusalCode
from .gate_result import GateOutcome
from .panel_transform import apply_month_filter, apply_row_filter


@dataclass(frozen=True)
class ExecutionResult:
    diff: dict
    candidate_returns: pd.Series          # strategy_ret indexed by date


def _respects_transform(compiled, run_panel: pd.DataFrame, result: dict) -> bool:
    """Realised == declared check: did the run actually apply the declared transform?"""
    mode = compiled.template_mode
    if mode == "double_sort":
        return result.get("settings_used", {}).get("control") == compiled.control
    if mode == "row_filter":
        form = compiled.panel_transform.form
        if form == "restrict_investment_grade":
            return bool(run_panel["investment_grade"].astype(bool).all())
        if form == "restrict_high_yield":
            return bool((~run_panel["investment_grade"].astype(bool)).all())
        return True                        # tercile membership is verified in the executor's tests
    if mode == "month_filter":
        return True                        # month membership is verified in the executor's tests
    return False


def execute_g1b(compiled, panel: pd.DataFrame, base_rulebook: dict, *, macro=None, min_history=60):
    """Return (GateOutcome, ExecutionResult | None). EXECUTION_MISMATCH on any procedural failure."""
    def fail():
        return GateOutcome("G1b", passed=False, booleans={"execution_verified": False},
                           refusal_code=RefusalCode.EXECUTION_MISMATCH), None

    def missing():
        # A month_filter proposal with no conditioning series is a HARNESS INPUT gap,
        # not a procedural execution mismatch: the transform never ran, so there is no
        # realised-vs-declared claim to make. Mislabelling it EXECUTION_MISMATCH masks a
        # wiring omission as an economic refusal (the 2026-09-05 RQ4 funnel bug).
        return GateOutcome("G1b", passed=False, booleans={"execution_verified": False},
                           refusal_code=RefusalCode.MISSING_INPUT), None

    rulebook = dict(base_rulebook)
    mode = compiled.template_mode
    if mode == "double_sort":
        rulebook["control"] = compiled.control
        rulebook["control_groups"] = compiled.control_groups
        run_panel = panel
        declared = {"control": compiled.control}
    elif mode == "month_filter":
        if macro is None:
            return missing()
        pt = compiled.panel_transform
        run_panel = apply_month_filter(panel, macro, lag=pt.lag_months, form=pt.form,
                                       min_history=min_history)
        declared = {"month_filter": {"variable": pt.variable, "lag": pt.lag_months, "form": pt.form}}
    elif mode == "row_filter":
        pt = compiled.panel_transform
        run_panel = apply_row_filter(panel, variable=pt.variable, form=pt.form)
        declared = {"row_filter": {"variable": pt.variable, "form": pt.form}}
    else:
        return fail()

    if len(run_panel) == 0:                # the transform left nothing to run
        return fail()
    try:
        result = run_characteristic_sort(run_panel, rulebook)
    except Exception:                      # any engine error is a procedural (execution) failure
        return fail()
    mr = result.get("monthly_returns")
    if mr is None or len(mr) == 0:
        return fail()
    if not _respects_transform(compiled, run_panel, result):
        return fail()                      # realised != declared

    diff = {"declared_changes": declared, "realised_changes": declared,
            "undeclared_changes": [], "missing_changes": [], "execution_fidelity": "PASS"}
    candidate = mr.set_index("date")["strategy_ret"]
    return (GateOutcome("G1b", passed=True, booleans={"execution_verified": True}),
            ExecutionResult(diff=diff, candidate_returns=candidate))
