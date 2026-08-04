"""renderer.py — R5 replication-path renderer (docs/reporter/reporter_spec_v0.2.md §7.2, §8, as amended by C9).

Produces a deterministic Markdown note and its claim ledger for one run. Every quantitative
statement is emitted through `emit_claim` (INV-1); every section returns a `RenderedFragment`,
never bare text. The audit section binds directly from the persisted `AuditCore` (C9) — there
is no full `AuditReport` on disk to delegate to `render_report`. Absences are rendered
explicitly from the stage records (INV-3), never omitted.
"""

from __future__ import annotations

from dataclasses import dataclass

from shared.reporting.canonical import NanValue
from shared.reporting.claims import (
    ArtefactType,
    ClaimRecord,
    ClaimSpec,
    EvidenceBlock,
    SourceLocator,
    Unit,
)
from shared.reporting.resolve import PointerResolutionError

from shared.reporting.canonical import canonical_hash

from agents.auditor.schemas.toggle import construction_toggles, data_quality_toggles

from .bundle import ReportBundle, ReportabilityStatus, StageStatus
from .emit import (
    ClaimResolutionError,
    RenderedFragment,
    assert_not_suppressed,
    emit_claim,
    emit_from_mapping,
)
from .legal_states import (
    LegalStateTable,
    PipelineDisposition,
    classify_disposition,
    load_table,
)
from .words import audit_scope_word, effect_direction_word, sign_word

_TOGGLES = ("meas_err", "stale_price", "survivorship", "lib_gap", "lab_trim")

# Below this magnitude a cross-class modulation is an empirical null, not a finding
# (ADR §5.3 / CF-5): the number is still reported (it is a typed leaf), but the prose
# does not call a ~1e-17 structural residual a "modulation".
_CROSS_CLASS_NULL_TOL = 1e-9

# The two headline components (ADR §5.2). Cross-class modulation is handled
# separately — it is a result in its own right (§5.3), belonging to neither headline.
# The fourth element names the toggles that constitute the component's class, so an
# absent (None) component can report WHY it is absent without collapsing the §5.4
# taxonomy (not_applicable vs input_unavailable).
_BIAS_CLASS_COMPONENTS = (
    ("methodological_construction_component", "methodological-construction",
     "how the strategy was built", construction_toggles),
    ("data_quality_component", "data-quality", "how the underlying data was cleaned",
     data_quality_toggles),
)


def _absence_reason(audit: dict, class_toggles: tuple[str, ...]) -> str:
    """Explain WHY a headline component is absent, without collapsing the §5.4
    taxonomy. A component is None only when EVERY toggle of its class is non-runnable
    (a single runnable class toggle makes its singleton coalition present). Distinguish:
      - all such toggles declared not_applicable -> no estimand exists (§5.4);
      - otherwise a class toggle is non-runnable/held -> not opinable (input_unavailable).
    """
    runnable = set(audit.get("runnable_toggles") or [])
    not_applicable = set(audit.get("not_applicable_toggles") or [])
    absent = [t for t in class_toggles if t not in runnable]
    if absent and all(t in not_applicable for t in absent):
        return "not applicable — no estimand of this class exists for this strategy"
    return "not opinable — a toggle of this class is not runnable in this partial audit"


@dataclass(frozen=True)
class RenderedDocument:
    reporter_run_id: str
    disposition: PipelineDisposition
    reportability: ReportabilityStatus
    note_markdown: str
    claims: tuple[ClaimRecord, ...]
    evidence: tuple[EvidenceBlock, ...]
    legal_state_hash: str


def _spec(
    claim_id: str,
    slot_id: str,
    artefact: ArtefactType,
    pointer: str,
    formatter: str,
    unit: Unit,
) -> ClaimSpec:
    return ClaimSpec(
        claim_id=claim_id,
        slot_id=slot_id,
        source_artifact=artefact,
        source_locator=SourceLocator(kind="json_pointer", pointer=pointer),
        formatter_id=formatter,
        unit=unit,
        conditioning_pointer=None,
    )


def _try_emit(
    spec: ClaimSpec, bundle: ReportBundle
) -> tuple[str | None, ClaimRecord | None]:
    """Emit a claim, returning (None, None) if its source value is simply absent (an absent key
    or a null parent). A present-but-null claim value or a non-numeric type is still tolerated as
    absence here (the section renders an explicit absence statement, INV-3)."""
    try:
        return emit_claim(spec, bundle)
    except (ClaimResolutionError, PointerResolutionError):
        return None, None


# --- sections --------------------------------------------------------------------------------

def render_header(bundle: ReportBundle, disposition: PipelineDisposition) -> RenderedFragment:
    lines: list[str] = []
    if bundle.reportability is not ReportabilityStatus.REPORTABLE:
        lines.append(
            f"> **NON-REPORTABLE** ({bundle.reportability.value}). "
            "These figures are not for project reporting."
        )
        lines.append("")
    lines.append(f"# Research note — `{bundle.paper_id}` / `{bundle.strategy_id}`")
    lines.append("")
    lines.append(f"- Run: `{bundle.reporter_run_id}` (phase {bundle.phase})")
    lines.append(f"- Disposition: **{disposition.value}**")
    lines.append(f"- Reporter code version: `{bundle.code_version.short}`")
    lines.append("")
    return RenderedFragment(text="\n".join(lines))


def render_extraction(bundle: ReportBundle) -> RenderedFragment:
    lines = ["## Extraction", ""]
    claims: list[ClaimRecord] = []
    spec_present = bundle.has(ArtefactType.STRATEGY_SPEC)
    if not spec_present:
        lines.append("_No StrategySpec present._")
        return RenderedFragment(text="\n".join(lines) + "\n")

    mean = _spec(
        "extraction.paper_mean",
        "extraction.claimed_mean",
        ArtefactType.STRATEGY_SPEC,
        "/paper_facts/claimed_headline_metric/value/mean",
        "4dp",
        Unit.DECIMAL,
    )
    tstat = _spec(
        "extraction.paper_tstat",
        "extraction.claimed_tstat",
        ArtefactType.STRATEGY_SPEC,
        "/paper_facts/claimed_headline_metric/value/t_stat",
        "2dp",
        Unit.T_STAT,
    )
    mean_tok, mean_rec = _try_emit(mean, bundle)
    tstat_tok, tstat_rec = _try_emit(tstat, bundle)
    if mean_rec is not None and tstat_rec is not None:
        lines.append(
            f"The paper's claimed headline metric is a mean of {mean_tok} "
            f"(t = {tstat_tok})."
        )
        claims.extend([mean_rec, tstat_rec])
    else:
        lines.append(
            "The StrategySpec carries no machine-readable claimed headline metric "
            "(paper_facts absent or unstated)."
        )
    lines.append("")
    return RenderedFragment(text="\n".join(lines), claims=tuple(claims))


def render_compilation(bundle: ReportBundle) -> RenderedFragment:
    lines = ["## Compilation", ""]
    stage = bundle.stages["compilation"]
    if stage.status is StageStatus.REFUSED:
        lines.append(
            f"Compilation was refused (`{stage.refusal_code}`). {stage.evidence.detail}."
        )
    elif stage.status is StageStatus.SUCCEEDED:
        lines.append(
            "The StrategySpec compiled to an audited family and produced a runnable "
            "configuration."
        )
    else:
        lines.append(f"Compilation status: {stage.status.value} ({stage.evidence.detail}).")
    lines.append("")
    return RenderedFragment(text="\n".join(lines))


def render_replication(bundle: ReportBundle) -> RenderedFragment:
    lines = ["## Replication", ""]
    claims: list[ClaimRecord] = []
    if not bundle.has(ArtefactType.STRATEGY_RESULT):
        lines.append("_No strategy result present; the strategy was not executed._")
        return RenderedFragment(text="\n".join(lines) + "\n")

    specs = [
        _spec(
            "replication.sharpe",
            "replication.sharpe",
            ArtefactType.STRATEGY_RESULT,
            "/summary/sharpe",
            "2dp",
            Unit.SHARPE,
        ),
        _spec(
            "replication.tstat",
            "replication.tstat",
            ArtefactType.STRATEGY_RESULT,
            "/summary/t_stat",
            "2dp",
            Unit.T_STAT,
        ),
        _spec(
            "replication.avg",
            "replication.avg_monthly",
            ArtefactType.STRATEGY_RESULT,
            "/summary/average",
            "4dp",
            Unit.DECIMAL,
        ),
    ]
    sharpe_tok, sharpe_rec = _try_emit(specs[0], bundle)
    tstat_tok, tstat_rec = _try_emit(specs[1], bundle)
    avg_tok, avg_rec = _try_emit(specs[2], bundle)
    if None in (sharpe_rec, tstat_rec, avg_rec):
        # INV-3: a partial summary renders an explicit absence, never a `None` in prose.
        lines.append(
            "The strategy result is present but its summary metrics are incomplete; "
            "no primary metric is reported."
        )
        lines.append("")
        return RenderedFragment(text="\n".join(lines))
    claims.extend([sharpe_rec, tstat_rec, avg_rec])
    lines.append(
        f"On the as-published panel the long-short strategy realises a mean monthly "
        f"return of {avg_tok} (t = {tstat_tok}) and a Sharpe ratio of {sharpe_tok}."
    )
    lines.append("")
    lines.append(
        "No formal level comparison between the paper's claimed metric and the pipeline "
        "result was performed by the Reporter; the two are reported side by side above and "
        "in the Extraction section."
    )
    lines.append("")
    return RenderedFragment(text="\n".join(lines), claims=tuple(claims))


def render_audit(bundle: ReportBundle) -> RenderedFragment:
    lines = ["## Audit", ""]
    claims: list[ClaimRecord] = []
    audit = bundle.doc(ArtefactType.AUDIT_REPORT)
    if not isinstance(audit, dict):
        lines.append("_No audit core present._")
        return RenderedFragment(text="\n".join(lines) + "\n")

    scope = audit.get("audit_scope")
    if scope not in ("COMPLETE", "PARTIAL", "REFUSED"):
        # INV-3: an audit core present but carrying no recognised scope (the bundle tolerates
        # this as UNOBSERVED) renders an explicit statement, never crashes on word selection.
        lines.append(
            "The audit core is present but carries no recognised scope; no bias "
            "magnitudes are opinable."
        )
        lines.append("")
        return RenderedFragment(text="\n".join(lines))
    lines.append(f"The audit scope is {audit_scope_word(scope)}.")
    lines.append("")
    if scope == "REFUSED":
        # INV-4: no bias magnitude is opinable under a REFUSED scope — render the refusal, emit
        # no numbers. `assert_not_suppressed` guards any accidental emission below.
        lines.append(
            "The audit was refused, so no first-order bias magnitudes are opinable."
        )
        lines.append("")
        return RenderedFragment(text="\n".join(lines))
    lines.append(
        "First-order bias-attribution effects (corrected minus as-published, on the "
        "primary metric):"
    )
    lines.append("")
    runnable = audit.get("runnable_toggles") or []
    for toggle in _TOGGLES:
        assert_not_suppressed(
            scope == "REFUSED", f"doe magnitude for {toggle} under REFUSED scope"
        )
        if toggle not in runnable:
            continue
        spec = _spec(
            f"audit.doe.{toggle}",
            f"audit.doe_effect.{toggle}",
            ArtefactType.AUDIT_REPORT,
            f"/saturated_bases/doe_effects/{toggle}",
            "4dp",
            Unit.DECIMAL,
        )
        tok, rec = _try_emit(spec, bundle)
        if rec is None:
            continue
        claims.append(rec)
        if isinstance(rec.raw_value, NanValue):
            direction = "undetermined"
        else:
            direction = effect_direction_word(rec.raw_value)
        lines.append(f"- `{toggle}`: {tok} — the correction {direction} the metric.")
    lines.append("")
    return RenderedFragment(text="\n".join(lines), claims=tuple(claims))


def render_bias_class_partition(bundle: ReportBundle) -> RenderedFragment:
    """The bias-class headline (ADR §5.2/§5.3): the endpoint gap split into a
    methodological-construction component, a data-quality component, and a
    cross-class modulation term — never one 'total bias'. A structurally-absent
    component is stated with its reason (not_applicable vs not-opinable, §5.4/§7 —
    never collapsed), never rendered as a zero."""
    lines = ["## Audit — bias-class decomposition", ""]
    claims: list[ClaimRecord] = []
    audit = bundle.doc(ArtefactType.AUDIT_REPORT)
    if not isinstance(audit, dict):
        lines.append("_No audit core present._")
        return RenderedFragment(text="\n".join(lines) + "\n")
    scope = audit.get("audit_scope")
    if scope not in ("COMPLETE", "PARTIAL"):
        # REFUSED or unrecognised => no components opinable (INV-4). No numbers.
        lines.append(
            "No bias-class components are opinable (the audit is not COMPLETE or "
            "PARTIAL)."
        )
        lines.append("")
        return RenderedFragment(text="\n".join(lines))

    if not isinstance(audit.get("bias_class_partition"), dict):
        # A legacy/stale core predating the partition block (ADR §5.2). Stated as an
        # absence (INV-3), never fabricated — emits no numbers.
        lines.append("The bias-class partition is not present in this audit core.")
        lines.append("")
        return RenderedFragment(text="\n".join(lines))

    lines.append(
        "The endpoint gap is partitioned by bias class into three components. They "
        "are incommensurable — a data-quality correction and a construction choice — "
        "and are never summed into one total:"
    )
    lines.append("")

    for pointer_key, label, gloss, class_toggles in _BIAS_CLASS_COMPONENTS:
        spec = _spec(
            f"audit.bias_class.{pointer_key}",
            f"audit.bias_class.{pointer_key}",
            ArtefactType.AUDIT_REPORT,
            f"/bias_class_partition/{pointer_key}",
            "4dp",
            Unit.DECIMAL,
        )
        tok, rec = _try_emit(spec, bundle)
        if rec is None:
            # Structural absence (null component): every toggle of this class is
            # non-runnable. Stated explicitly (INV-3), never as a 0 (ADR §5.4), and
            # WITHOUT collapsing not_applicable into input_unavailable (§7).
            lines.append(
                f"- {label} component: {_absence_reason(audit, class_toggles())}."
            )
            continue
        claims.append(rec)
        lines.append(f"- {label} component ({gloss}): {tok}.")

    # Cross-class modulation — a result in its own right (§5.3).
    cc_spec = _spec(
        "audit.bias_class.cross_class_modulation",
        "audit.bias_class.cross_class_modulation",
        ArtefactType.AUDIT_REPORT,
        "/bias_class_partition/cross_class_modulation",
        "4dp",
        Unit.DECIMAL,
    )
    cc_tok, cc_rec = _try_emit(cc_spec, bundle)
    if cc_rec is None:
        lines.append(
            "- cross-class modulation: not applicable — it requires both a "
            "data-quality and a construction estimand."
        )
    else:
        claims.append(cc_rec)
        if isinstance(cc_rec.raw_value, NanValue):
            qualifier = "undetermined"
        elif abs(cc_rec.raw_value) < _CROSS_CLASS_NULL_TOL:
            qualifier = "within numerical tolerance of zero (no material modulation)"
        else:
            qualifier = (
                "data-quality cleaning modulates a construction bias by this amount"
            )
        lines.append(f"- cross-class modulation: {cc_tok} — {qualifier}.")
    lines.append("")
    return RenderedFragment(text="\n".join(lines), claims=tuple(claims))


def render_audit_refusals(bundle: ReportBundle) -> RenderedFragment:
    lines = ["## Audit — refusals", ""]
    audit = bundle.doc(ArtefactType.AUDIT_REPORT)
    if not isinstance(audit, dict):
        lines.append("_No audit core present._")
        return RenderedFragment(text="\n".join(lines) + "\n")
    runnable = set(audit.get("runnable_toggles") or [])
    not_opinable = [t for t in _TOGGLES if t not in runnable]
    if not_opinable:
        lines.append("The following toggles were not opinable: " + ", ".join(not_opinable) + ".")
    else:
        lines.append("Every registered toggle was opinable.")
    lines.append("")
    return RenderedFragment(text="\n".join(lines))


def render_stage_summary(bundle: ReportBundle) -> RenderedFragment:
    lines = ["## Stage summary", ""]
    for stage, record in bundle.stages.items():
        lines.append(f"- **{stage}**: {record.status.value} — {record.evidence.detail}")
    lines.append("")
    return RenderedFragment(text="\n".join(lines))


def render_limitations(bundle: ReportBundle) -> RenderedFragment:
    lines = ["## Limitations", ""]
    lines.append(
        "Cross-agent run identity is asserted manually, not mechanically established: "
        f"{bundle.join_rationale}"
    )
    if bundle.reportability is not ReportabilityStatus.REPORTABLE:
        lines.append("")
        lines.append(
            f"This run is {bundle.reportability.value}; its figures are excluded from "
            "reportable results."
        )
    lines.append("")
    return RenderedFragment(text="\n".join(lines))


def render_provenance_footer(
    bundle: ReportBundle, table: LegalStateTable
) -> RenderedFragment:
    lines = ["## Provenance", ""]
    lines.append("Source artefacts (sha256):")
    for artefact in bundle.artefacts:
        lines.append(f"- `{artefact.key}` — `{artefact.sha256}`")
    if bundle.upstream_stamps.quant is not None:
        lines.append(f"- quant git: `{bundle.upstream_stamps.quant.short}`")
    if bundle.upstream_stamps.audit is not None:
        lines.append(f"- auditor git: `{bundle.upstream_stamps.audit.short}`")
    if bundle.upstream_stamps.auditor_prereg_tag:
        lines.append(f"- auditor prereg tag: `{bundle.upstream_stamps.auditor_prereg_tag}`")
    lines.append(f"- reporter code version: `{bundle.code_version.short}`")
    lines.append(f"- legal-state table: `{table.content_hash}`")
    lines.append("")
    return RenderedFragment(text="\n".join(lines))


# --- extension path (fixture-only, D1) -------------------------------------------------------

def _load_mechanism_library():
    """Load the real mechanism library (`prereg/mechanism_library/`) for verbatim mechanism
    claims and the version hash. Imported lazily so the replication path pays no cost."""
    from agents.scientist.researcher.library import load_library

    return load_library()


def _proposal_id(proposal_report, index: int) -> str:
    return proposal_report.record.get("proposal_id", f"proposal_{index}")


def render_proposals(bundle: ReportBundle, library) -> RenderedFragment:
    lines = ["## Proposals", ""]
    evidence = []
    for i, pr in enumerate(bundle.proposals):
        pid = _proposal_id(pr, i)
        outcome = pr.record.get("final_outcome")
        refusal = pr.record.get("refusal_code")
        lines.append(f"### {pid}")
        suffix = f" (refusal `{refusal}`)" if refusal else ""
        lines.append(f"- Outcome: **{outcome}**{suffix}")
        if pr.gate_outcomes:
            last = pr.gate_outcomes[-1]
            lines.append(
                f"- Gate reached: `{last.get('gate')}` "
                f"({'passed' if last.get('passed') else 'failed'})"
            )
        if pr.proposal and library is not None:
            mref = pr.proposal.get("mechanism_ref")
            if mref:
                mech = library.mechanism(mref)  # KeyError propagates (INV-13)
                claim_text = str(mech.get("claim", "")).strip()
                lines.append(f"- Mechanism `{mref}` — {mech.get('title', '')}")
                lines.append(f"  > {claim_text}")
                evidence.append(
                    EvidenceBlock(
                        evidence_id=f"{pid}.mechanism",
                        source_artifact=ArtefactType.MECHANISM_LIBRARY,
                        source_artifact_sha256=library.version_hash,
                        source_locator=SourceLocator(
                            kind="json_pointer", pointer=f"/{mref}/claim"
                        ),
                        verbatim_text=claim_text,
                        verification=None,
                    )
                )
        lines.append("")
    return RenderedFragment(text="\n".join(lines), evidence=tuple(evidence))


def render_diagnostics(bundle: ReportBundle) -> RenderedFragment:
    lines = ["## Diagnostics", ""]
    claims: list[ClaimRecord] = []
    fields = (("alpha", Unit.DECIMAL, "4dp", "alpha"), ("alpha_t", Unit.T_STAT, "2dp", "t"))
    for i, pr in enumerate(bundle.proposals):
        if not pr.diagnostics:
            continue
        pid = _proposal_id(pr, i)
        mapping = dict(pr.diagnostics)
        sha = canonical_hash(mapping)
        for key, unit, fid, label in fields:
            if key not in mapping:
                continue
            tok, rec = emit_from_mapping(
                claim_id=f"{pid}.crowding.{key}",
                slot_id=f"diagnostics.crowding.{key}",
                source_artifact=ArtefactType.CROWDING_DIAGNOSTIC,
                pointer=f"/{key}",
                formatter_id=fid,
                unit=unit,
                mapping=mapping,
                sha256=sha,
            )
            claims.append(rec)
            lines.append(f"- `{pid}` crowding {label}: {tok}")
    # Capacity / regime have no diagnostic module — render their UNOBSERVED stage record.
    for stage in ("diagnostics_capacity", "diagnostics_regime"):
        record = bundle.stages[stage]
        lines.append(f"- {stage}: {record.status.value} ({record.evidence.detail})")
    lines.append("")
    return RenderedFragment(text="\n".join(lines), claims=tuple(claims))


def render_holdout(bundle: ReportBundle) -> RenderedFragment:
    # The word "success" must not appear (§7.2): a sign, a CI and a paired difference only.
    lines = ["## Holdout", ""]
    claims: list[ClaimRecord] = []
    for i, pr in enumerate(bundle.proposals):
        if pr.holdout is None:
            continue
        pid = _proposal_id(pr, i)
        hv = pr.holdout.to_dict()
        sha = canonical_hash(hv)
        low_tok, low_rec = emit_from_mapping(
            claim_id=f"{pid}.holdout.ci_low",
            slot_id="holdout.sharpe_ci_low",
            source_artifact=ArtefactType.HOLDOUT_VIEW,
            pointer="/sharpe_ci_low",
            formatter_id="2dp",
            unit=Unit.SHARPE,
            mapping=hv,
            sha256=sha,
        )
        high_tok, high_rec = emit_from_mapping(
            claim_id=f"{pid}.holdout.ci_high",
            slot_id="holdout.sharpe_ci_high",
            source_artifact=ArtefactType.HOLDOUT_VIEW,
            pointer="/sharpe_ci_high",
            formatter_id="2dp",
            unit=Unit.SHARPE,
            mapping=hv,
            sha256=sha,
        )
        claims.extend([low_rec, high_rec])
        lines.append(
            f"- `{pid}`: out-of-sample Sharpe sign is {sign_word(pr.holdout.sharpe_sign)}."
        )
        lines.append(
            f"- `{pid}`: the out-of-sample Sharpe interval spans [{low_tok}, {high_tok}]."
        )
        if pr.holdout.paired_difference is not None:
            pd_tok, pd_rec = emit_from_mapping(
                claim_id=f"{pid}.holdout.paired",
                slot_id="holdout.paired_difference",
                source_artifact=ArtefactType.HOLDOUT_VIEW,
                pointer="/paired_difference",
                formatter_id="4dp",
                unit=Unit.DECIMAL,
                mapping=hv,
                sha256=sha,
            )
            claims.append(pd_rec)
            lines.append(
                f"- `{pid}`: paired difference vs the corrected parent is {pd_tok}."
            )
    lines.append("")
    return RenderedFragment(text="\n".join(lines), claims=tuple(claims))


# --- orchestration ---------------------------------------------------------------------------

def render_note(bundle: ReportBundle) -> RenderedDocument:
    """Render the full replication-path note and its ledger for `bundle`."""
    table = load_table()
    disposition = classify_disposition(bundle.stages, bundle.audit_scope(), table=table)

    fragments = [render_header(bundle, disposition), render_extraction(bundle)]
    fragments.append(render_compilation(bundle))
    if bundle.has(ArtefactType.STRATEGY_RESULT):
        fragments.append(render_replication(bundle))
    if bundle.has(ArtefactType.AUDIT_REPORT):
        fragments.append(render_audit(bundle))
        fragments.append(render_bias_class_partition(bundle))
        fragments.append(render_audit_refusals(bundle))
    if bundle.proposals:
        library = _load_mechanism_library()
        fragments.append(render_proposals(bundle, library))
        fragments.append(render_diagnostics(bundle))
        fragments.append(render_holdout(bundle))
    fragments.append(render_stage_summary(bundle))
    fragments.append(render_limitations(bundle))
    fragments.append(render_provenance_footer(bundle, table))

    note = "\n".join(f.text.rstrip("\n") + "\n" for f in fragments).rstrip("\n") + "\n"
    claims = tuple(c for f in fragments for c in f.claims)
    evidence = tuple(e for f in fragments for e in f.evidence)
    return RenderedDocument(
        reporter_run_id=bundle.reporter_run_id,
        disposition=disposition,
        reportability=bundle.reportability,
        note_markdown=note,
        claims=claims,
        evidence=evidence,
        legal_state_hash=table.content_hash,
    )
