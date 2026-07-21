"""
renderer.py — a deterministic AuditReport renderer.

The design's step 19 is an LLM explainer that renders the completed statistical
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
    d = report.to_dict()
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
        lines.append(
            f"Effect {label}: point {r['point']:.4f}, p-value {r['p_value']:.4f}, "
            f"interval [{r['ci_low']:.4f}, {r['ci_high']:.4f}] via {r['method']}."
        )

    lines.append(
        f"BH-FDR rejected {fdr['n_rejected']} of {fdr['n_family']} confirmatory "
        f"tests at q {fdr['q']}."
    )
    lines.append(
        f"Compression statistic D is {comp['d_point']:.4f} against D_max "
        f"{comp['d_max']}; compression "
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
