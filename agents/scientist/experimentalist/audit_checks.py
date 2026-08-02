"""G2 — audit-clean (spec §9; finding F8 obligations). Given a CompiledExtension, verify the
extension did not introduce a bias:

  1. monotonicity re-assert (post-compile) — the extension references no bias toggle (re-checks
     G0's monotonicity after compilation);
  2. parameter provenance — the threshold/breakpoint rule is EX-ANTE (expanding past-only window
     or ex-ante-committed), NEVER a full-sample median (a look-ahead the extension itself would
     introduce); a ROW filter additionally may not key off a realised-return field;
  3. timing — a MONTH filter's conditioning variable is lagged at least its reporting_delays
     minimum (a contemporaneous / under-lagged regime leaks the future).

Refusals: NEW_TIMING_VIOLATION, PARAMETER_PROVENANCE_VIOLATION. Sets the `audit_clean` boolean.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from agents.auditor.schemas.toggle import TOGGLE_IDS

from ..schemas.outcomes import RefusalCode
from .gate_result import GateOutcome

_TOGGLES = frozenset(TOGGLE_IDS)
_EX_ANTE_RULES = frozenset({"expanding_window_past_only", "ex_ante_committed"})
# Fields derived from realised returns — an ex-ante row filter may NEVER key off one (look-ahead).
_REALISED_RETURN_FIELDS = frozenset({
    "ret", "xret", "next_ret", "return", "realised_return", "mean_return", "sharpe", "alpha",
})

_REPORTING_DELAYS = Path(__file__).resolve().parents[3] / "docs" / "reporting_delays.yaml"


def load_reporting_delays(path=None) -> dict:
    p = Path(path) if path is not None else _REPORTING_DELAYS
    return yaml.safe_load(p.read_text())["reporting_delays"]


def audit_g2(compiled, *, reporting_delays: dict) -> GateOutcome:
    def fail(code: RefusalCode) -> GateOutcome:
        return GateOutcome("G2", passed=False, booleans={"audit_clean": False}, refusal_code=code)

    variable = compiled.control if compiled.control is not None else (
        compiled.panel_transform.variable if compiled.panel_transform else None)

    # 1 — monotonicity re-assert: no bias toggle survived compilation.
    if variable in _TOGGLES:
        return fail(RefusalCode.PARAMETER_PROVENANCE_VIOLATION)

    # 2 — provenance: ex-ante only (never a full-sample median).
    if compiled.provenance_rule not in _EX_ANTE_RULES:
        return fail(RefusalCode.PARAMETER_PROVENANCE_VIOLATION)
    if compiled.template_mode == "row_filter" and variable in _REALISED_RETURN_FIELDS:
        return fail(RefusalCode.PARAMETER_PROVENANCE_VIOLATION)

    # 3 — timing: a month-filter regime variable must be lagged >= its reporting_delays minimum.
    if compiled.template_mode == "month_filter":
        pt = compiled.panel_transform
        required = reporting_delays.get(pt.variable, {}).get("lag_months")
        if required is None or pt.lag_months < required:
            return fail(RefusalCode.NEW_TIMING_VIOLATION)

    return GateOutcome("G2", passed=True, booleans={"audit_clean": True})
