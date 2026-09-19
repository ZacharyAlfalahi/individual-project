"""
Layered RQ2 coverage (evaluation contract §5.2).

The candidate set 𝒞 nests ℰ ⊆ ℬ ⊆ 𝒮 ⊆ 𝒞:
  * 𝒮 (semantically understood) = 𝒞 minus SEMANTIC refusals
  * ℬ (bound to columns)        = 𝒮 minus BINDING refusals
  * ℰ (executed)                = ℬ minus EXECUTION refusals
with C_semantic = |𝒮|/|𝒞|, C_binding = |ℬ|/|𝒮|, C_execution = |ℰ|/|ℬ|,
C_end_to_end = |ℰ|/|𝒞| (the product). Each refusal is attributed to a layer by
its TYPED `RefusalCode` (§5.2); a strategy fails at the EARLIEST layer among its
codes (semantic < binding < execution).

This module is the corpus-agnostic, reusable artefact: the anchor-set run
(𝒞 = 3 supported golds ⇒ every layer = 1.0) exercises it end-to-end; the corpus
headline reuses the identical classifier once the frozen Librarian enumeration +
enumeration golds instantiate the real 𝒞 (deferred — phase_f).

TAXONOMY NOTE (D27): the contract §5.2 names "unrecognised-concept → semantic",
but there is no `UNRECOGNISED_CONCEPT` code — per D27 every concept→column miss
(unrecognised concept, parameter mismatch, no row) rides `MISSING_BINDING`
(→ binding). So the semantic layer is populated only by `REVIEW_REQUIRED` /
`REFUSED_ON_SILENCE`; splitting an unrecognised concept out of
`MISSING_BINDING` would require inspecting the refusal's structured evidence, a
deferred refinement recorded here rather than silently mis-mapped.
"""

from __future__ import annotations

from .refusal import RefusalCode

# §5.2 typed-code → layer. All eight members are mapped (completeness asserted below).
_SEMANTIC = frozenset({RefusalCode.REVIEW_REQUIRED, RefusalCode.REFUSED_ON_SILENCE})
_BINDING = frozenset({RefusalCode.MISSING_BINDING})
_EXECUTION = frozenset({
    RefusalCode.OUT_OF_ENUM_WEIGHTING,
    RefusalCode.UNSUPPORTED_TRIM_VARIANT,
    RefusalCode.UNSUPPORTED_COMBINATION,
    RefusalCode.UNSUPPORTED_COMBINER,
    RefusalCode.ASSUMPTION_MISMATCH,
})

LAYERS = ("semantic", "binding", "execution")

REFUSAL_LAYER: dict[RefusalCode, str] = {
    **{c: "semantic" for c in _SEMANTIC},
    **{c: "binding" for c in _BINDING},
    **{c: "execution" for c in _EXECUTION},
}

# Completeness: every RefusalCode has exactly one layer (fail-loud at import if a
# future additive member is unmapped — the coverage layer must never silently drop one).
_UNMAPPED = set(RefusalCode) - set(REFUSAL_LAYER)
if _UNMAPPED:
    raise RuntimeError(f"coverage.REFUSAL_LAYER is missing a layer for {_UNMAPPED}")


def layer_of(code) -> str:
    """The §5.2 layer for a `RefusalCode` (or its string value). Fail-loud on unknown."""
    if not isinstance(code, RefusalCode):
        code = RefusalCode(code)
    return REFUSAL_LAYER[code]


def _fail_layer(codes: list) -> str | None:
    """The earliest (semantic < binding < execution) layer at which a strategy
    fails, or None if it has no refusals (⇒ executed)."""
    if not codes:
        return None
    return min((layer_of(c) for c in codes), key=LAYERS.index)


def layered_coverage(strategies: list[dict]) -> dict:
    """Layered coverage over a candidate set. Each `strategies` item is a dict with
    a `refusal_codes` list (empty ⇒ executed). Returns the four C_* fractions, the
    nested cardinalities |𝒞|/|𝒮|/|ℬ|/|ℰ|, and the refusal counts by layer. A fraction
    over an empty denominator is `None` (never a fabricated 1.0)."""
    n = len(strategies)
    fails = [_fail_layer(s.get("refusal_codes", [])) for s in strategies]

    by_layer = {layer: sum(1 for f in fails if f == layer) for layer in LAYERS}
    n_semantic = n - by_layer["semantic"]                      # |𝒮|
    n_binding = n_semantic - by_layer["binding"]               # |ℬ|
    n_executed = n_binding - by_layer["execution"]             # |ℰ|

    def frac(num: int, den: int) -> float | None:
        return (num / den) if den else None

    return {
        "n_candidates": n,
        "n_semantic": n_semantic,
        "n_binding": n_binding,
        "n_executed": n_executed,
        "C_semantic": frac(n_semantic, n),
        "C_binding": frac(n_binding, n_semantic),
        "C_execution": frac(n_executed, n_binding),
        "C_end_to_end": frac(n_executed, n),
        "refusals_by_layer": by_layer,
    }
