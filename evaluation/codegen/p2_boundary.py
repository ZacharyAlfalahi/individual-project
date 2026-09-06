"""P2 coverage-boundary below-floor result assembly (WS-C).

When ``|Arm A| < below_floor_min_arm_a`` the selection is under-powered: the
registered rule SUPPRESSES agreement/divergence and NO generation is performed. This
module assembles the entire reportable artefact from the census alone — the suppressed
metrics, the typed eligibility-exclusion accounting, and the rendered report — plus the
derived ``reportable`` flag and a zero-spend accounting.

Pure: no model calls, no disk, no randomness — it reuses the already-built pieces
(``p2_metrics.compute_p2_metrics`` suppressed branch, ``p2_taxonomy`` accounting,
``p2_report.render_report``). The point of the whole path is a $0, zero-call close-out:
the *evidence* (phase=report extraction over a frozen census) can be reportable even
though the *agreement numbers* are suppressed by rule.
"""

from __future__ import annotations

from evaluation.codegen.census import CensusResult
from evaluation.codegen.p2_metrics import compute_p2_metrics
from evaluation.codegen.p2_report import render_report
from evaluation.codegen.p2_selector import ArmSelection
from evaluation.codegen.p2_taxonomy import eligibility_exclusion_accounting


def derive_reportable(
    provenance: dict[str, dict], corpus_status: str, zoo_sha_ok: bool
) -> bool:
    """The *evidence* is reportable — even though the agreement *numbers* are suppressed —
    iff every routed member's spec provenance is a ``phase=report`` emission, the corpus
    selection is ``frozen``, and the frozen zoo-list hash matches. An empty provenance map
    (no routed member carries a real spec) is NOT reportable."""
    if not provenance:
        return False
    all_report = all((p or {}).get("manifest_phase") == "report" for p in provenance.values())
    return bool(all_report and corpus_status == "frozen" and zoo_sha_ok)


def assemble_below_floor(
    census: CensusResult,
    selection: ArmSelection,
    thresholds: dict,
    *,
    provenance: dict[str, dict],
    corpus_status: str,
    zoo_sha_ok: bool,
    meta: dict | None = None,
) -> tuple[dict, str]:
    """Assemble the below-floor P2 boundary result + rendered report.

    Returns ``(result_dict, report_md)``. ``result_dict`` carries the suppression reason
    verbatim, the observed dispositions, the typed eligibility accounting, the full census
    fate table, the derived ``reportable`` flag, the real-spec provenance, and an explicit
    ``model_calls: 0`` / ``spend_usd: 0.0`` — the zero-spend accounting that keeps this path
    from ever becoming a paid run by accident.

    Raises ``ValueError`` if called on an ABOVE-floor selection (that path REQUIRES
    generation — use the driver, not this module)."""
    if not selection.below_floor:
        raise ValueError(
            "assemble_below_floor called on an ABOVE-floor selection — generation is "
            "required there; use run_p2_driver, not the boundary assembly"
        )
    metrics = compute_p2_metrics(census, selection, {}, None, thresholds)
    assert metrics.suppressed, "a below-floor selection must yield suppressed metrics"

    accounting = eligibility_exclusion_accounting(census)
    reportable = derive_reportable(provenance, corpus_status, zoo_sha_ok)

    # §6(a): taxonomy is empty by construction below floor — pass taxonomy=None so the
    # renderer emits the one-line "empty by construction" reason (not a "pending run").
    report_md = render_report(
        metrics, selection, census, taxonomy=None, eligibility=accounting,
        meta=(dict(meta) if meta else None),
    )

    result = {
        "experiment": "p2_coverage_boundary",
        "suppressed": True,
        "below_floor": True,
        "below_floor_reason": metrics.below_floor_reason,
        "arm_a_size": selection.arm_a_size,
        "arm_a": list(selection.arm_a),
        "arm_b": list(selection.arm_b),
        "observed_dispositions": {
            "compilable": len(census.compilable_set()),
            "refused": len(census.refusal_set()),
            "eligibility_excluded": len(census.eligibility_exclusions()),
        },
        "eligibility_accounting": accounting.to_dict(),
        "census_fate_table": [dict(r) for r in census.fate_table()],
        "metrics": metrics.to_dict(),
        "spec_provenance": provenance,
        "reportable": reportable,
        "model_calls": 0,
        "spend_usd": 0.0,
    }
    return result, report_md
