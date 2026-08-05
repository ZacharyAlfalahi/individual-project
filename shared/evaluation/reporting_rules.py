"""
reporting_rules.py — deterministic sentence templates for the licensed inferential
claims (shared/evaluation, spec §8 / D-E17).

Pure functions mapping a typed result to a LICENSED sentence. No LLM. Narrow scope: the
handful of inferential sentences where numeric-token verification is orthogonal to
semantic overclaiming — the same verified alpha supports both "incremental alpha was not
established" and "the factor is fully spanned", and only the first is licensed. Removing
the ability to write the second is what these templates buy; a style note is insufficient
for a generative Reporter.

Every refusal branch is handled, so a typed-refusal result (e.g. the cost turnover stub)
still yields a licensed sentence rather than a gap. `FORBIDDEN_STRINGS` is enforced by a
test over the generated sentences — the unlicensed vocabulary is unwriteable, not merely
discouraged.
"""

from __future__ import annotations

from .contracts import (
    BreakEvenStatus,
    CostResult,
    PredictionAssessment,
    RefusalCode,
    RegimeResult,
    SpanningResult,
)

# Vocabulary that asserts more than the computation licenses. Enforced by test over the
# generated notes (D-E7, D-E10, D-E17). "statistically significant" joined under A7:
# short-sample (holdout) results license point-estimate/direction sentences only, so
# the significance claim must be unwriteable, not merely discouraged.
FORBIDDEN_STRINGS = (
    "spanned",
    "crowded",
    "proven unique",
    "net profitable",
    "implementable",
    "mechanism confirmed",
    "capacity",
    "statistically significant",
)


def spanning_sentence(r: SpanningResult) -> str:
    if not r.estimable:
        if r.refusal_code is RefusalCode.DEVELOPMENT_SCOPE_DIAGNOSTIC:
            return (
                "The spanning regression is a development-window diagnostic and was "
                "not computed on the holdout window: its pre-registered "
                f"{r.min_obs}-month floor cannot be met there by construction of the "
                "walk-forward split (amendment A7)."
            )
        if r.refusal_code is RefusalCode.INSUFFICIENT_OBSERVATIONS:
            return (
                f"The spanning regression was not estimable: {r.n_obs} monthly "
                f"observations fall below the pre-registered {r.min_obs}-month minimum."
            )
        if r.refusal_code is RefusalCode.RANK_DEFICIENT:
            return (
                "The spanning regression was not estimable: the design matrix is "
                f"rank-deficient (rank {r.design_rank} of {r.n_controls + 1}), so the "
                "candidate is a linear combination of the control set — a substantive "
                "finding reported rather than repaired away."
            )
        return "The spanning regression was not estimable."
    # Estimable but the HAC standard error / interval is not computable: the LEAST
    # licensed state. It must NOT fall through to the affirmative sentence (D-E7) — a
    # failure to compute inference is not evidence of an alpha.
    if r.ci_low is None or r.ci_high is None or r.t_hac is None:
        return (
            "Incremental alpha against the fixed six-factor internal control set could "
            "not be established: the HAC standard error was not estimable "
            f"(alpha {r.alpha_monthly:.4f} per month, n = {r.n_obs})."
        )
    if r.ci_low <= 0.0 <= r.ci_high:
        return (
            "Incremental alpha against the fixed six-factor internal control set was not "
            f"established (alpha {r.alpha_monthly:.4f} per month, HAC t = {r.t_hac:.2f}, "
            f"n = {r.n_obs})."
        )
    return (
        "Incremental alpha against the fixed six-factor internal control set was "
        f"{r.alpha_monthly:.4f} per month (HAC t = {r.t_hac:.2f}, n = {r.n_obs})."
    )


def cost_sentence(r: CostResult) -> str:
    if not r.estimable:
        if r.refusal_code is RefusalCode.ARTEFACT_CAPABILITY_MISSING:
            return (
                "Trading cost was not computed: the run artefact does not persist "
                "per-position weights, so turnover is not measurable from it."
            )
        return "Trading cost was not computed."
    parts: list[str] = []
    if r.scenarios:
        s = r.scenarios[0]
        parts.append(
            f"Under the {s.scenario_id} cost scenario ({s.cost_bps_ig:.0f} bp "
            f"{s.unit.value}) the cost-adjusted mean monthly return was "
            f"{s.cost_adjusted_mean_monthly:.4f} (cost drag {s.cost_drag_bps_monthly:.1f} "
            f"bp/month)."
        )
    if r.break_even_alpha_status is BreakEvenStatus.FOUND and r.break_even_alpha_cost_bps is not None:
        parts.append(
            "The control-adjusted edge vanishes at a one-way cost of "
            f"{r.break_even_alpha_cost_bps:.1f} bp (break-even on the turnover-adjusted "
            "alpha intercept)."
        )
    elif r.break_even_mean_status is BreakEvenStatus.FOUND and r.break_even_mean_cost_bps is not None:
        parts.append(
            f"The mean-return edge vanishes at a one-way cost of {r.break_even_mean_cost_bps:.1f} bp."
        )
    else:
        parts.append("A break-even cost was not defined for this candidate on this sample.")
    return " ".join(parts)


def _sign_word(sign: int | None) -> str:
    return {1: "positive", -1: "negative", 0: "zero"}.get(sign if sign is not None else 99, "unspecified")


_SHORT_SAMPLE_QUALIFIER = (
    " Sample below the pre-registered conditional floor (short sample): point estimate "
    "and direction only, no inferential claim (amendment A7)."
)


def regime_sentence(r: RegimeResult) -> str:
    p = r.prediction
    qualifier = _SHORT_SAMPLE_QUALIFIER if r.short_sample else ""
    if not r.applicable:
        return (
            "No falsifiable regime prediction was registered for this candidate; the "
            "regime decomposition is descriptive only." + qualifier
        )
    if not p.evaluable:
        return "The registered regime prediction was not evaluable on this sample." + qualifier
    outcome = "was borne out" if p.hit else "was not borne out"
    tail = " (mechanically implied by the template; excluded from the headline count)" if p.mechanically_implied_by_template else ""
    return (
        f"The registered {_sign_word(p.predicted_sign)} state prediction {outcome}: the "
        f"extension's mean return differed from the corrected parent by "
        f"{p.delta_mean_vs_parent_in_target_state:.4f} in the target state "
        f"(n = {p.n_months_target_state}) versus "
        f"{p.delta_mean_vs_parent_in_other_state:.4f} otherwise "
        f"(n = {p.n_months_other_state}){tail}." + qualifier
    )


def hit_count_sentence(by_parent: dict[str, list[PredictionAssessment]]) -> str:
    """D-E15: exact counts + per-parent breakdown, NO population interval. Mechanically
    implied predictions are excluded from the headline numerator."""
    total_hits = total_eval = 0
    fragments: list[str] = []
    for parent, assessments in by_parent.items():
        scored = [
            a for a in assessments if a.evaluable and not a.mechanically_implied_by_template
        ]
        hits = sum(1 for a in scored if a.hit)
        total_hits += hits
        total_eval += len(scored)
        fragments.append(f"{parent} {hits}/{len(scored)}")
    breakdown = ", ".join(fragments)
    return (
        f"{total_hits} of {total_eval} evaluable predictions had the registered sign "
        f"({breakdown})."
    )
