"""INVARIANT 1 — THE WALL (spec §3.1). The generative context is assembled from ONLY the
allow-listed fields; no realised performance statistic or performance-derived ranking may enter.
Severing magnitudes at generation is what makes N_trials = m a STRUCTURAL fact rather than an
honour-system claim.

The ScientistCase is already magnitude-free (the seam stripped every magnitude), so the wall is
largely maintained by construction; this module makes it explicit — it emits ONLY the allow-list
keys, and `_assert_no_magnitudes` recursively rejects any magnitude-bearing key, so a future edit
that tries to thread a Sharpe/alpha/effect/p-value into the context fails loudly here.
"""

from __future__ import annotations

# spec §3.1 ALLOWED_RESEARCHER_FIELDS — the only keys a generative call may receive.
ALLOWED_RESEARCHER_FIELDS = (
    "strategy_spec_economic",
    "corrected_quant_config",
    "failed_check_ids",
    "failed_check_verdicts",
    "toggle_definitions",
    "mechanism_documents",
    "available_templates",
    "required_inputs_available",
)

# Magnitude-free descriptions of what correction each toggle represents (static; no numbers).
TOGGLE_DEFINITIONS = {
    "meas_err": "measurement-error corrections (decimal-shift + bounce-back) on trade prices",
    "stale_price": "stale-price mask (exclude month-end prices with no recent trade)",
    "survivorship": "retain distress (default) exits rather than dropping them",
    "lib_gap": "lag the signal by one month to avoid using contemporaneous information",
    "lab_trim": "retain all realised returns (no ex-post return trimming)",
}

# The same banned magnitude keys the ScientistCase wall test uses (belt-and-braces, key-level).
_BANNED_KEYS = frozenset({
    "sharpe", "alpha", "alpha_bbw4", "t_stat", "p_value", "p_raw", "p_bh",
    "effect", "effect_signed", "effect_magnitude", "doe", "bh_adjusted_p", "mean_return",
})


def _mechanism_document(m: dict) -> dict:
    """Strip a mechanism entry to the non-magnitude, generation-relevant fields (its VALUES carry
    no magnitudes — a mechanism is an economic claim, not a result)."""
    return {
        "mechanism_id": m["mechanism_id"],
        "title": m["title"],
        "claim": m["claim"],
        "sources": [
            {"source_id": s["source_id"], "kind": s["kind"], "cite": s["verification"]["cite"]}
            for s in m["sources"]
        ],
        "allowed_templates": list(m["allowed_templates"]),
        "required_inputs": list(m["applicability"]["required_inputs"]),
        "forbidden_uses": list(m["forbidden_uses"]),
        "caveats": list(m.get("caveats", [])),   # source-fidelity limitations (e.g. BPW crisis sample)
    }


def _template_document(t: dict) -> dict:
    return {
        "template_id": t["template_id"],
        "permitted_fields": t["permitted_fields"],
        "required_invariants": t["required_invariants"],
        "forbidden_changes": t["forbidden_changes"],
    }


def build_context(case, eligibility_results, library, *, strategy_spec_economic=None) -> dict:
    """Assemble the generative context for a ScientistCase from allow-listed fields only."""
    eligible = [r for r in eligibility_results if r.eligible]
    ctx = {
        "strategy_spec_economic": dict(strategy_spec_economic or {}),
        "corrected_quant_config": case.corrected_quant_config_ref,   # a REF string, not numbers
        "failed_check_ids": list(case.failed_check_ids),
        "failed_check_verdicts": dict(case.failed_check_verdicts),   # PASS/FAIL/REFUSED only
        "toggle_definitions": {t: TOGGLE_DEFINITIONS.get(t, t) for t in case.applicable_toggles},
        "mechanism_documents": [_mechanism_document(library.mechanism(r.mechanism_id)) for r in eligible],
        "available_templates": [_template_document(t) for t in library.templates.values()],
        "required_inputs_available": {r.mechanism_id: r.reachable for r in eligible},
    }
    _assert_no_magnitudes(ctx)
    return ctx


def _all_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _all_keys(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _all_keys(v)


def _assert_no_magnitudes(ctx: dict) -> None:
    extra = set(ctx) - set(ALLOWED_RESEARCHER_FIELDS)
    if extra:
        raise AssertionError(f"context has non-allow-listed fields: {sorted(extra)}")
    for k in _all_keys(ctx):
        if k.lower() in _BANNED_KEYS:
            raise AssertionError(f"magnitude-bearing key leaked into researcher context: {k!r}")
