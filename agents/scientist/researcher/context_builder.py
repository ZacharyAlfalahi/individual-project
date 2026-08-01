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

import re

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

# Magnitude-bearing key patterns (SUBSTRING regex, so `sharpe_ratio` / `net_sharpe` / `annual_alpha`
# / `tstat` / `information_ratio` are caught, not just the exact tokens). Verified against every
# legitimate context key: none contains any of these substrings, so there are no false positives.
_BANNED_KEY_RE = re.compile(
    r"sharpe|alpha|tstat|t_stat|t-stat|pvalue|p_value|p-value|p_raw|p_bh|"
    r"bh_adjusted|bhadjusted|effect|\bdoe\b|mean_return|return|information_ratio|"
    r"info_ratio|drawdown|volatility|premium"
)
# Magnitude patterns inside a free-form STRING value (used only on strategy_spec_economic, so
# citations elsewhere — years, page numbers — are never scanned): a number with %/bp, or a
# performance token immediately followed by a number.
_MAGNITUDE_STR_RE = re.compile(
    r"\d+(\.\d+)?\s*(%|bp|bps)\b|"                                     # 17 bps, 8.3%
    r"(sharpe|alpha|t-?stat|p-?value|information[ _-]?ratio)\b.{0,12}?-?\d"  # 'sharpe of 1.7'
)


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


def _iter_kv(obj, key=None):
    """Yield (key, value) for every node — `key` is the dict key holding `value` (None for a
    list element). Lets the wall inspect VALUES, not only keys (a magnitude leaks via either)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield (str(k), v)
            yield from _iter_kv(v, str(k))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield (key, v)
            yield from _iter_kv(v, key)


def _assert_no_magnitudes(ctx: dict) -> None:
    """Enforce INVARIANT 1 on the assembled context: no non-allow-listed field, no
    magnitude-bearing key (substring), and no float value anywhere (a magnitude-free context uses
    only config INTS — lag options — so any float is a suspected performance number). The one
    free-form caller field, strategy_spec_economic, is locked down hardest (no numerics at all,
    plus a magnitude-string scan)."""
    extra = set(ctx) - set(ALLOWED_RESEARCHER_FIELDS)
    if extra:
        raise AssertionError(f"context has non-allow-listed fields: {sorted(extra)}")
    for key, val in _iter_kv(ctx):
        if key is not None and _BANNED_KEY_RE.search(key.lower()):
            raise AssertionError(f"magnitude-bearing key leaked into researcher context: {key!r}")
        if isinstance(val, bool):
            continue                          # a bool flag is not a magnitude
        if isinstance(val, float):
            raise AssertionError(
                f"float value (suspected magnitude) in researcher context: {val!r} under {key!r}"
            )
    _assert_spec_economic_clean(ctx.get("strategy_spec_economic", {}))


def _assert_spec_economic_clean(spec) -> None:
    """strategy_spec_economic is the only free-form caller-supplied allow-listed field, so it is
    validated hardest: NO numeric values (a strategy's economic design is descriptors/enums, never
    a realised Sharpe/return), and no magnitude-bearing string."""
    for _key, val in _iter_kv(spec):
        if isinstance(val, bool):
            continue
        if isinstance(val, (int, float)):
            raise AssertionError(
                f"numeric value in strategy_spec_economic (magnitudes forbidden): {val!r}"
            )
        if isinstance(val, str) and _MAGNITUDE_STR_RE.search(val.lower()):
            raise AssertionError(f"magnitude-bearing string in strategy_spec_economic: {val!r}")
