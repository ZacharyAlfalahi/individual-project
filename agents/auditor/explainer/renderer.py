"""
renderer.py — a deterministic AuditReport renderer.

The design's LLM explainer renders the completed statistical
verdict in prose, downstream of the deterministic verdict and with zero gating
power (§11). This module is the deterministic renderer for that slot: it emits prose whose
every numeric token is pulled directly from the typed AuditReport, so
`numeric_verifier.verify_numbers` passes by construction.

No language model is involved — this keeps the whole stack deterministic.
An LLM explainer, when configured, produces richer prose over the same report and
is gated by the same verifier.
"""

from __future__ import annotations

from ..schemas.audit_report import AuditReport


def render_report(report: AuditReport) -> str:
    """Render a per-strategy AuditReport to prose citing only typed report numbers."""
    return render_report_dict(report.to_dict())


def render_report_dict(d: dict) -> str:
    """The renderer over the report's own serialised form — the same dict the numeric
    verifier checks against, and the same dict a run writes to `<anchor>_report.json`.
    A stored report therefore re-renders exactly from this dict, without re-running the
    lattice."""
    econ = d["economic"]
    fdr = d["fdr"]
    comp = d["compression"]

    lines: list[str] = []
    lines.append(f"Audit scope: {d['audit_scope']}.")
    if d["conditioning_statement"]:
        lines.append(d["conditioning_statement"])

    lines.append(
        f"The registered endpoint gap on the primary metric is "
        f"{econ['endpoint_gap']:.4f} (economic band: {econ['gap_band']})."
    )

    for label, r in d["inference"].items():
        # The t and p come from `method`; the interval does NOT — checks/inference.py
        # always reads it from BootstrapResult.doe_ci. Naming one method for both would
        # attribute a block-bootstrap percentile interval to HAC, so they are named apart.
        interval_method = "the block bootstrap" if r["method"] == "HAC" else r["method"]
        # A bootstrap-routed or inert coordinate carries no t; render it as absent rather than
        # raising inside the fallback renderer (which exists to keep a report renderable).
        t_txt = f"{r['t_stat']:.4f}" if r.get("t_stat") is not None else "n/a"
        p_txt = f"{r['p_value']:.4f}" if r.get("p_value") is not None else "n/a"
        lines.append(
            f"Effect {label}: point {r['point']:.4f}, t {t_txt}, "
            f"p-value {p_txt} via {r['method']}; "
            f"interval [{r['ci_low']:.4f}, {r['ci_high']:.4f}] via {interval_method}."
        )

    lines.append(
        f"BH-FDR rejected {fdr['n_rejected']} of {fdr['n_family']} within-strategy "
        f"tests at q {fdr['q']}."
    )
    # Adequacy is decided on the interval's UPPER BOUND (checks/compression.py), so the
    # point alone can sit under D_max while the verdict is inadequate; the interval
    # reconciles the point with the verdict, so it is printed alongside them.
    lines.append(
        f"Compression statistic D is {comp['d_point']:.4f}, interval "
        f"[{comp['d_ci_low']:.4f}, {comp['d_ci_high']:.4f}], against D_max "
        f"{comp['d_max']} applied to the interval's upper bound; compression "
        f"{'adequate' if comp['compression_adequate'] else 'inadequate'}."
    )
    # A NaN metric renders as the token "nan", which the numeric verifier's number
    # regex deliberately does not match — so it is unverifiable-by-omission, never a
    # fabricated number. All finite values below trace to typed report fields.
    lines.append(
        f"Deflated Sharpe of the corrected endpoint is "
        f"{econ['deflated_sharpe_corrected']:.4f}."
    )
    return "\n".join(lines)
