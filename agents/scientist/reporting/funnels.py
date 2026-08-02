"""The two funnels (spec §12) — denominators ALWAYS visible.

  * economic_funnel — Generated -> Valid -> Compiled -> Executed -> Audit-clean -> BH survivor ->
    CPCV-qualified -> Holdout evaluated. Each stage is a strict subset of the prior, so the counts
    are monotone non-increasing; the drop at each stage is the finding.
  * agent_quality_funnel — the generation-quality view (large n, no market data): requested vs
    valid-unique, with the invalid and duplicate rates (§8.2 "generation failure is data").

Reads typed EvaluationRecords / GenerationResults only — never prose.
"""

from __future__ import annotations

_G0_FIELDS = ("schema_valid", "mechanism_authorised", "template_supported",
              "toggles_preserved", "inputs_available", "not_duplicate")


def economic_funnel(records) -> dict:
    """Stage counts over EvaluationRecords, monotone non-increasing."""
    b = [r.booleans for r in records]
    return {
        "generated": len(records),
        "valid": sum(1 for x in b if all(getattr(x, f) for f in _G0_FIELDS)),
        "compiled": sum(1 for x in b if x.compiled),
        "executed": sum(1 for x in b if x.execution_verified),
        "audit_clean": sum(1 for x in b if x.audit_clean),
        "bh_survivor": sum(1 for x in b if x.bh_survived),
        "cpcv_qualified": sum(1 for x in b if x.cpcv_qualified),
        "holdout_evaluated": sum(1 for x in b if x.holdout_evaluated),
    }


def agent_quality_funnel(generation_results) -> dict:
    """Invalid + duplicate rates over the generation results (denominators visible)."""
    req = sum(g.n_requested for g in generation_results)
    return {
        "requested": req,
        "valid_unique": sum(g.n_valid_unique for g in generation_results),
        "invalid": sum(g.n_invalid for g in generation_results),
        "duplicate": sum(g.n_duplicate for g in generation_results),
        "invalid_rate": (sum(g.n_invalid for g in generation_results) / req) if req else 0.0,
        "duplicate_rate": (sum(g.n_duplicate for g in generation_results) / req) if req else 0.0,
    }
