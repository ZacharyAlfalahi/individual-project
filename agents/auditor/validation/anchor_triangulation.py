"""
anchor_triangulation.py — external triangulation against the locked anchors (build
step 12a, §10.4).

Assessed on point estimates and bootstrap intervals (both prior-free, both on
common support) against the primary-prior result. The gate is PER-ANCHOR and
LOCKED — there is NO aggregate fraction rule (the "≥2/3" rule was withdrawn once
`str` was reclassified as pilot; over n=2 untouched anchors a fraction gate is
brittle, D-A38). Each locked anchor (`mom6`, `drf`) is assessed against its OWN
pre-registered sign and order-of-magnitude criteria plus the 25% Sharpe-triangulation
band. The Auditor is externally triangulated only if NEITHER locked anchor exhibits
a material unexplained contradiction.

`str` is pilot evidence (already observed) and is reported DESCRIPTIVELY in the
three-anchor table, never in the locked gate (D-A33). Silent iteration until the
anchors agree is p-hacking and is forbidden (§13.3): a miss is reported and
investigated, never repaired.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnchorExpectation:
    """A locked anchor's pre-registered, frozen expectation (§13.2). Criteria are
    committed before the result is examined (O-A10)."""

    anchor: str
    dominant_bias: str
    expected_sign: int                       # +1 / −1 for the dominant effect
    expected_magnitude_range: tuple[float, float]  # order-of-magnitude band on |effect|
    expected_sharpe: float
    is_locked: bool = True                   # False => pilot (str), descriptive only


@dataclass(frozen=True)
class AnchorVerdict:
    anchor: str
    sign_ok: bool
    magnitude_ok: bool
    sharpe_ok: bool
    contradiction: bool
    is_locked: bool
    detail: str

    def to_dict(self) -> dict:
        return {
            "anchor": self.anchor,
            "sign_ok": self.sign_ok,
            "magnitude_ok": self.magnitude_ok,
            "sharpe_ok": self.sharpe_ok,
            "contradiction": self.contradiction,
            "is_locked": self.is_locked,
            "detail": self.detail,
        }


def triangulate_anchor(
    expectation: AnchorExpectation,
    observed_effect: float,
    observed_sharpe: float,
    *,
    sharpe_band: float = 0.25,
) -> AnchorVerdict:
    """Assess one anchor against its frozen criteria. A `contradiction` is a
    LOCKED anchor whose sign disagrees or whose magnitude/Sharpe fall materially
    outside the pre-registered bands. Pilot anchors never register a contradiction
    (they are descriptive)."""
    sign_ok = (observed_effect > 0) == (expectation.expected_sign > 0) or observed_effect == 0
    lo, hi = expectation.expected_magnitude_range
    magnitude_ok = lo <= abs(observed_effect) <= hi
    if expectation.expected_sharpe == 0:
        sharpe_ok = abs(observed_sharpe) <= sharpe_band
    else:
        rel = abs(observed_sharpe - expectation.expected_sharpe) / abs(expectation.expected_sharpe)
        sharpe_ok = rel <= sharpe_band

    contradiction = expectation.is_locked and not (sign_ok and magnitude_ok and sharpe_ok)
    detail = (
        f"sign {'ok' if sign_ok else 'MISMATCH'}; "
        f"|effect|={abs(observed_effect):.4g} vs [{lo:g},{hi:g}] "
        f"{'ok' if magnitude_ok else 'OUT'}; "
        f"sharpe {'ok' if sharpe_ok else 'OUT'}"
    )
    return AnchorVerdict(
        anchor=expectation.anchor,
        sign_ok=sign_ok,
        magnitude_ok=magnitude_ok,
        sharpe_ok=sharpe_ok,
        contradiction=contradiction,
        is_locked=expectation.is_locked,
        detail=detail,
    )


@dataclass(frozen=True)
class TriangulationReport:
    verdicts: tuple[AnchorVerdict, ...]
    externally_triangulated: bool           # True iff no locked anchor contradicts

    def to_dict(self) -> dict:
        return {
            "externally_triangulated": self.externally_triangulated,
            "verdicts": [v.to_dict() for v in self.verdicts],
        }


def run_triangulation(verdicts: list[AnchorVerdict]) -> TriangulationReport:
    """The Auditor is externally triangulated iff NEITHER locked anchor exhibits a
    material unexplained contradiction (§10.4). No aggregate fraction rule."""
    triangulated = not any(v.contradiction for v in verdicts if v.is_locked)
    return TriangulationReport(tuple(verdicts), triangulated)
