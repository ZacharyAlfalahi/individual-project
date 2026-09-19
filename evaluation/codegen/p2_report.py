"""P2 coverage-boundary report renderer (WS-C).

Renders a results object (``p2_metrics.P2Metrics`` + the selection + census +
taxonomy) to Markdown. Three binding properties, tested:

  * it carries the contract §6 agreement caveat VERBATIM
    (``docs/extensions/contracts/p1_codegen_ablation.md`` §6);
  * it NEVER emits a Sharpe-as-performance number — P2 has no oracle, so a
    Sharpe would be an unfounded performance claim (the estimand is agreement,
    not performance);
  * it is EMPTY of numbers until a real run: below floor (or with no metrics) it
    renders only the census fate table + eligibility accounting + caveat.

Pure rendering — no model calls, no numeric regeneration from prose (the Reporter
discipline).
"""

from __future__ import annotations

from evaluation.codegen.census import CensusResult
from evaluation.codegen.p2_metrics import P2Metrics
from evaluation.codegen.p2_selector import ArmSelection
from evaluation.codegen.p2_taxonomy import EligibilityAccounting, TaxonomySample

#: The contract §6 agreement caveat — echoed VERBATIM (binding interpretation
#: language). Any drift from the contract text is a test failure by design.
AGREEMENT_CAVEAT = (
    "Inter-model agreement is not correctness. Phase-D precedent: both models\n"
    "agreed control_axis = var_5pct with locating quotes; gold = credit_rating.\n"
    "For generated code the correlated-error risk is higher, not lower — both\n"
    "models share training exposure to the same public implementations.\n"
    "Agreement rates measure task difficulty and ambiguity only, and license no\n"
    "validity claim."
)

#: The census carve-out — P2 produces no performance data, so RQ3's census
#: arithmetic is untouched (echoed from contract §11).
CENSUS_CARVE_OUT = (
    "P2 produces no performance data; RQ3's census arithmetic is unaffected."
)


def _blockquote(text: str) -> str:
    return "\n".join(f"> {line}" if line else ">" for line in text.split("\n"))


def _fmt(x, nd: int = 4) -> str:
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if f != f:  # NaN
        return "NaN"
    return f"{f:.{nd}f}"


def _census_fate_table(fate_table: tuple[dict, ...]) -> list[str]:
    lines = [
        "| paper | disposition | refusal reason | text quality ok | exclusion reason |",
        "|---|---|---|---|---|",
    ]
    for row in fate_table:
        lines.append(
            f"| {row['paper_id']} | {row['disposition']} | "
            f"{row['refusal_reason'] or '—'} | {row['text_quality_ok']} | "
            f"{row['exclusion_reason'] or '—'} |"
        )
    return lines


def _eligibility_section(acc: EligibilityAccounting) -> list[str]:
    lines = ["## Eligibility exclusions (typed, COUNTED — never silent)", ""]
    lines.append(f"Total eligibility exclusions: **{acc.total}**")
    lines.append("")
    if acc.total == 0:
        lines.append("_None — every census paper cleared the text-quality bar._")
        return lines
    lines.append("| reason | count | papers |")
    lines.append("|---|---|---|")
    for reason, count in sorted(acc.by_reason.items()):
        papers = ", ".join(acc.papers_by_reason.get(reason, []))
        lines.append(f"| {reason} | {count} | {papers} |")
    return lines


def _taxonomy_section(sample: TaxonomySample) -> list[str]:
    lines = [
        "## Failure taxonomy (pre-registered stratified sample)",
        "",
        f"Strata axes: `{' × '.join(sample.strata_axes)}`; up to {sample.per_stratum} "
        "member(s) per cell, worst-divergence-first (a frozen order — the sample "
        "precedes inspection). 'Failure' = DISAGREEMENT; P2 has no oracle, so it is "
        "never a correctness failure.",
        "",
    ]
    if not sample.cells:
        lines.append("_Pending a run — no members to stratify yet._")
        return lines
    lines.append("| arm | divergence magnitude | sampled papers |")
    lines.append("|---|---|---|")
    for (arm, mag), members in sorted(sample.cells.items()):
        papers = ", ".join(m.paper_id for m in members) or "—"
        lines.append(f"| {arm} | {mag} | {papers} |")
    return lines


def _runs_section(runs: tuple[dict, ...]) -> list[str]:
    """Every generation run, typed. A run that produced no series is VISIBLE here rather
    than absent from the agreement table."""
    lines = [
        "## Generation runs (typed — one row per member × model)",
        "",
        "| paper | arm | model | code extracted | sandbox status | reason | months | "
        "returned model |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        months = r.get("n_months")
        lines.append(
            f"| {r.get('paper_id')} | {r.get('arm') or '—'} | {r.get('model_id')} | "
            f"{r.get('code_extracted')} | {r.get('sandbox_status') or '—'} | "
            f"{r.get('sandbox_reason') or '—'} | {months if months is not None else '—'} | "
            f"{r.get('returned_model_version') or '—'} |"
        )
    return lines


def _failure_diagnostics_section(runs: tuple[dict, ...]) -> list[str]:
    """The last stderr line of every run that produced no series — the typed "why" behind a
    status. A generation failure, never a correctness claim (P2 has no oracle)."""
    failed = [r for r in runs if r.get("sandbox_status") not in (None, "ok")]
    lines = [
        "## Failure diagnostics (runs that produced no series)",
        "",
        "_The last stderr line of each failed run. These are generation failures; with no "
        "oracle at the coverage boundary they are never correctness failures._",
        "",
        "| paper | model | reason | last stderr line |",
        "|---|---|---|---|",
    ]
    for r in failed:
        tail = [ln.strip() for ln in (r.get("stderr_tail") or "").splitlines() if ln.strip()]
        last = tail[-1].replace("|", r"\|")[:200] if tail else "—"
        lines.append(
            f"| {r.get('paper_id')} | {r.get('model_id')} | {r.get('sandbox_reason') or '—'} | "
            f"`{last}` |"
        )
    return lines


def _compiler_view_lines(config: dict) -> list[str]:
    """State which panel view the compiler side ran under, beside the one the models saw.
    When they differ, any Arm-B divergence mixes implementation difference with panel
    difference — said here rather than left for a reader to discover."""
    compiler, codegen = config.get("compiler_view"), config.get("codegen_panel_view")
    if not compiler or not codegen:
        return []
    lines = ["", f"- Compiler panel view: `{compiler}`", f"- Panel the models saw: `{codegen}`"]
    if config.get("views_match"):
        lines.append("- The two sides saw the SAME panel view, so a divergence is an "
                     "implementation difference.")
    else:
        lines.append("- **The two sides saw DIFFERENT panel views**, so any divergence below "
                     "mixes implementation difference with panel difference and cannot be "
                     "read as implementation fidelity alone.")
    return lines


def _compiler_attempts_section(
    attempts: tuple[dict, ...], compiler_run_config: dict | None = None
) -> list[str]:
    """The Arm-B third implementation. A refused member has NO compiler series, so its
    codegen-vs-compiler comparison does not exist — stated, never quietly omitted."""
    lines = [
        "## Arm-B deterministic-compiler attempts (the third implementation)",
        "",
        "| paper | outcome | refusal codes | months | note |",
        "|---|---|---|---|---|",
    ]
    for a in attempts:
        codes = ", ".join(a.get("refusal_codes") or []) or "—"
        months = a.get("n_months")
        lines.append(
            f"| {a.get('paper_id')} | **{a.get('outcome')}** | {codes} | "
            f"{months if months is not None else '—'} | {a.get('note') or '—'} |"
        )
    if compiler_run_config:
        lines.extend(_compiler_view_lines(compiler_run_config))
    return lines


def render_report(
    metrics: P2Metrics | None,
    selection: ArmSelection | None,
    census: CensusResult | None,
    *,
    taxonomy: TaxonomySample | None = None,
    eligibility: EligibilityAccounting | None = None,
    meta: dict | None = None,
    runs: tuple[dict, ...] | None = None,
    compiler_attempts: tuple[dict, ...] | None = None,
    compiler_run_config: dict | None = None,
) -> str:
    """Render the P2 agreement report from a results object. Below floor (or with
    no metrics), only the census fate table + eligibility accounting + caveat are
    emitted — never an agreement number, never a Sharpe. ``runs`` and
    ``compiler_attempts`` add the executed-run tables (typed run outcomes and the
    Arm-B compiler attempts) when a run produced them."""
    out: list[str] = ["# P2 coverage-boundary agreement report", ""]

    out.append(
        "_Estimand: inter-model AGREEMENT rates + divergence magnitudes + a "
        "failure taxonomy. NEVER performance-as-validity — no oracle exists to "
        "separate 'strategy fails to replicate' from 'code is wrong'._"
    )
    out.append("")
    out.append(f"_{CENSUS_CARVE_OUT}_")
    out.append("")

    if metrics is not None and metrics.floor_override:
        out.append(
            f"> **BELOW-FLOOR DEPARTURE — {metrics.floor_override}.** At this arm size the "
            "registered under-power rule SUPPRESSES the agreement and divergence numbers. "
            "They are computed here under an authorised departure and are "
            "DESCRIPTIVE ONLY — never a registered result. The registered rule at this arm size "
            "is suppression."
        )
        out.append("")

    # The §6 caveat — verbatim, always.
    out.append("## Agreement caveat (contract §6, verbatim)")
    out.append("")
    out.append(_blockquote(AGREEMENT_CAVEAT))
    out.append("")

    if selection is not None:
        out.append("## Arms")
        out.append("")
        out.append(f"- Arm A (whole refusal set, |A| discovered): **{selection.arm_a_size}** — "
                   f"{', '.join(selection.arm_a) or '—'}")
        out.append(f"- Arm B (first min(|A|,max) compilable, zoo order): "
                   f"{', '.join(selection.arm_b) or '—'}")
        out.append(f"- Below floor: **{selection.below_floor}**")
        out.append("")

    if census is not None:
        out.append("## Census fate table")
        out.append("")
        out.extend(_census_fate_table(census.fate_table()))
        out.append("")
        out.extend(
            _eligibility_section(
                eligibility if eligibility is not None else _empty_eligibility(census)
            )
        )
        out.append("")

    if metrics is None or metrics.suppressed:
        reason = (metrics.below_floor_reason if metrics is not None else
                  "no metrics — pending a run")
        out.append("## Agreement / divergence")
        out.append("")
        out.append(f"_SUPPRESSED: {reason}. Only the census fate table + taxonomy are "
                   "emitted (the below-floor rule)._")
        out.append("")
        if runs:
            out.extend(_runs_section(runs))
            out.append("")
        if compiler_attempts:
            out.extend(_compiler_attempts_section(compiler_attempts))
            out.append("")
        if taxonomy is not None:
            out.extend(_taxonomy_section(taxonomy))
        else:
            # §6(a): the taxonomy strata are arm × divergence-magnitude; with divergence
            # suppressed the sampler is empty BY CONSTRUCTION, not pending a run.
            out.append("## Failure taxonomy (pre-registered stratified sample)")
            out.append("")
            out.append("_Empty by construction: the strata are `arm × divergence_magnitude`, "
                       "and divergence is suppressed below floor — there are no strata to "
                       "sample. This is a pre-registered outcome, not a missing run._")
        out.append("")
        _append_provenance(out, meta)
        return "\n".join(out)

    dist = metrics.distribution
    out.append("## Inter-model correlation distribution (PRIMARY)")
    out.append("")
    n_finite = dist.get("n_finite_correlations", 0)
    out.append(f"- n members in the selection: {dist.get('n_members', 0)} "
               f"(measurable correlations: {n_finite}; "
               f"insufficient overlap: {dist.get('n_insufficient_overlap', 0)})")
    if not n_finite:
        out.append("- **No correlation was measurable for any member**, so the distribution "
                   "and strata below carry no measured divergence.")
    out.append(f"- correlation min / median / max: {_fmt(dist.get('correlation_min'))} / "
               f"{_fmt(dist.get('correlation_median'))} / {_fmt(dist.get('correlation_max'))}")
    out.append("")
    out.append("### Per-member agreement (raw — re-thresholdable)")
    out.append("")
    out.append("| paper | arm | n_overlap | correlation | sign agreement | stratum | agrees |")
    out.append("|---|---|---|---|---|---|---|")
    for a in metrics.agreements:
        out.append(
            f"| {a.paper_id} | {a.arm} | {a.n_overlap} | {_fmt(a.correlation)} | "
            f"{_fmt(a.sign_agreement)} | {a.stratum} | {a.agrees} |"
        )
    out.append("")

    out.append("## Binary agreement rate (SECONDARY)")
    out.append("")
    out.append(f"- agreement rate: {_fmt(dist.get('agreement_rate'))} "
               f"({dist.get('n_agree', 0)}/{dist.get('n_scored', 0)} scored pairs)")
    sc = dist.get("strata_counts", {})
    out.append(f"- divergence strata — high: {sc.get('high', 0)}, "
               f"medium: {sc.get('medium', 0)}, low: {sc.get('low', 0)}")
    if not dist.get("n_finite_correlations", 0):
        out.append("  (every member sits in `high` by the non-finite-correlation rule — an "
                   "UNMEASURABLE pair, not an observed high divergence)")
    out.append("")

    if metrics.divergences:
        out.append("## Arm-B codegen-vs-compiler divergence (neither side is truth)")
        out.append("")
        out.append("| paper | model | correlation | tracking error | max abs diff | n_overlap |")
        out.append("|---|---|---|---|---|---|")
        for d in metrics.divergences:
            m = d.metrics
            out.append(
                f"| {d.paper_id} | {d.model_id} | {_fmt(m.get('correlation'))} | "
                f"{_fmt(m.get('tracking_error'))} | {_fmt(m.get('max_abs_diff'))} | "
                f"{m.get('n_overlap', 0)} |"
            )
        out.append("")
    elif compiler_attempts is not None:
        out.append("## Arm-B codegen-vs-compiler divergence (neither side is truth)")
        out.append("")
        if census is not None and not census.compilable_set():
            out.append(
                "_ARM B IS EMPTY: the router compiled no member of this corpus, so there is no "
                "compiler implementation to compare against. The absent arm is the result, not "
                "a gap in the table._"
            )
        else:
            out.append(
                "_NOT COMPUTABLE: no Arm-B member has a deterministic-compiler series (see the "
                "compiler-attempt table below), so this comparison does not exist for this run. "
                "No stand-in implementation is substituted._"
            )
        out.append("")

    if metrics.mde:
        out.append("## MDE by arm size (power guard)")
        out.append("")
        out.append("| arm size (n) | minimum detectable agreement-rate difference | detectable |")
        out.append("|---|---|---|")
        for n, row in sorted(metrics.mde.items()):
            out.append(f"| {n} | {_fmt(row.get('mde'))} | {row.get('detectable')} |")
        out.append("")

    if taxonomy is not None:
        out.extend(_taxonomy_section(taxonomy))
        out.append("")

    if compiler_attempts:
        out.extend(_compiler_attempts_section(compiler_attempts, compiler_run_config))
        out.append("")

    if runs:
        out.extend(_runs_section(runs))
        out.append("")
        if any(r.get("sandbox_status") not in (None, "ok") for r in runs):
            out.extend(_failure_diagnostics_section(runs))
            out.append("")

    _append_provenance(out, meta)
    return "\n".join(out)


def _empty_eligibility(census: CensusResult) -> EligibilityAccounting:
    from evaluation.codegen.p2_taxonomy import eligibility_exclusion_accounting

    return eligibility_exclusion_accounting(census)


def _append_provenance(out: list[str], meta: dict | None) -> None:
    out.append("## Provenance")
    out.append("")
    if not meta:
        out.append("_Pending a run: Phase-F pair + per-call returned_model_version, prompt "
                   "hashes, frozen zoo-list sha256, cache keys._")
        return
    for key in sorted(meta):
        out.append(f"- {key}: {meta[key]}")
