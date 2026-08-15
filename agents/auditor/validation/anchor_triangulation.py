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
    expected_sharpe: float | None = None     # D-4: None drops the Sharpe check entirely
    #                                          (a committed 0 would instead assert Sharpe ~ 0)
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
    if expectation.expected_sharpe is None:
        sharpe_ok = True  # D-4: no Sharpe expectation was pre-registered -> the check is dropped
    elif expectation.expected_sharpe == 0:
        sharpe_ok = abs(observed_sharpe) <= sharpe_band
    else:
        rel = abs(observed_sharpe - expectation.expected_sharpe) / abs(expectation.expected_sharpe)
        sharpe_ok = rel <= sharpe_band

    contradiction = expectation.is_locked and not (sign_ok and magnitude_ok and sharpe_ok)
    sharpe_detail = "n/a" if expectation.expected_sharpe is None else ("ok" if sharpe_ok else "OUT")
    detail = (
        f"sign {'ok' if sign_ok else 'MISMATCH'}; "
        f"|effect|={abs(observed_effect):.4g} vs [{lo:g},{hi:g}] "
        f"{'ok' if magnitude_ok else 'OUT'}; "
        f"sharpe {sharpe_detail}"
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


# --------------------------------------------------------------------------
# Loader — the ratified per-anchor gate criteria (O-A10, §10.4) from thresholds.yaml
# --------------------------------------------------------------------------

_ANCHOR_GATE_SECTION = "§10.4 (O-A10 per-anchor locked gate criteria)"


def load_anchor_gate(path=None) -> dict[str, AnchorExpectation]:
    """Load the ratified per-anchor gate criteria (O-A10, §10.4) fail-loud from
    `auditor.anchor_gate.anchors`, returning `{anchor_id: AnchorExpectation}`.

    The loader enforces, AT LOAD, the guards a malformed pre-registration would
    otherwise slip past (so a committed block can never silently produce an
    always-fail or always-lock gate):

      * D-1 / D-2 — `expected_magnitude_range` is an UNSIGNED, ascending band on
        |effect| in DECIMAL monthly units (`0 <= lo <= hi`). Direction is carried by
        `expected_sign` (+1/-1) and is NEVER folded into the band. A signed band
        (e.g. `[-0.90, -0.30]`) or a `%/mo` value (e.g. `[0.30, 0.90]` for 0.30 %/mo)
        is rejected here rather than silently registering `magnitude_ok=False`.
      * D-3 — `is_locked` is REQUIRED. The dataclass default is `True` (fail-open toward
        locking); omitting the key in the pre-registration would silently lock a pilot
        anchor (`str`) into the gate and let it register a forbidden contradiction
        (D-A32/D-A33). The loader raises if it is absent.
      * D-4 — `expected_sharpe` is OPTIONAL; omit it to drop the Sharpe check entirely
        (a committed `0` would instead assert Sharpe ~ 0, a stricter test).

    Raises `AuditorThresholdError` on any missing/malformed field — never defaults.
    The fail-loud helpers are imported locally so importing this module (widely used)
    stays dependency-light; only calling the loader touches `thresholds.yaml`."""
    from agents.auditor.thresholds import (
        AuditorThresholdError,
        _auditor_block,
        _require,
        _require_bool,
        _require_int,
        _require_number,
        _require_str,
    )

    block = _auditor_block(path)
    gate = block.get("anchor_gate")
    if not isinstance(gate, dict) or not gate:
        raise AuditorThresholdError("auditor.anchor_gate", _ANCHOR_GATE_SECTION)
    anchors = gate.get("anchors")
    if not isinstance(anchors, dict) or not anchors:
        raise AuditorThresholdError("auditor.anchor_gate.anchors", _ANCHOR_GATE_SECTION)

    out: dict[str, AnchorExpectation] = {}
    for aid in anchors:
        base = ("anchor_gate", "anchors", aid)
        stem = f"auditor.anchor_gate.anchors.{aid}"

        dominant = _require_str(
            _require(block, base + ("dominant_bias",), _ANCHOR_GATE_SECTION),
            f"{stem}.dominant_bias", _ANCHOR_GATE_SECTION,
        )
        sign = _require_int(
            _require(block, base + ("expected_sign",), _ANCHOR_GATE_SECTION),
            f"{stem}.expected_sign", _ANCHOR_GATE_SECTION,
        )
        if sign not in (-1, 1):
            raise AuditorThresholdError(
                f"{stem}.expected_sign (must be +1 or -1; got {sign})", _ANCHOR_GATE_SECTION,
            )

        rng = _require(block, base + ("expected_magnitude_range",), _ANCHOR_GATE_SECTION)
        if not (isinstance(rng, (list, tuple)) and len(rng) == 2):
            raise AuditorThresholdError(
                f"{stem}.expected_magnitude_range (must be a [lo, hi] pair)", _ANCHOR_GATE_SECTION,
            )
        lo = _require_number(rng[0], f"{stem}.expected_magnitude_range[0]", _ANCHOR_GATE_SECTION)
        hi = _require_number(rng[1], f"{stem}.expected_magnitude_range[1]", _ANCHOR_GATE_SECTION)
        if not (0.0 <= lo <= hi):
            raise AuditorThresholdError(
                f"{stem}.expected_magnitude_range (must be unsigned + ascending, 0 <= lo <= hi; "
                f"got [{lo}, {hi}] — the band is on |effect|; carry direction in expected_sign "
                f"and use DECIMAL monthly units, e.g. 0.0030 for 0.30 %/mo; see D-1/D-2)",
                _ANCHOR_GATE_SECTION,
            )

        # D-3: is_locked is REQUIRED — no fail-open default at the pre-registration boundary.
        is_locked = _require_bool(
            _require(block, base + ("is_locked",), _ANCHOR_GATE_SECTION),
            f"{stem}.is_locked", _ANCHOR_GATE_SECTION,
        )

        # A LOCKED gate must be falsifiable. With lo = 0, |effect| >= 0 always clears the
        # lower bound AND sign_ok holds at effect == 0, so a zero-lo locked gate can barely
        # register a contradiction — it fits the anchor whatever the result. Require lo > 0
        # for locked anchors; a pilot (is_locked=false) is descriptive, so lo = 0 is fine.
        if is_locked and not (lo > 0.0):
            raise AuditorThresholdError(
                f"{stem}.expected_magnitude_range (a LOCKED anchor needs lo > 0; lo = 0 makes the "
                f"gate nearly unfalsifiable — |effect| >= 0 always clears a zero lower bound and "
                f"sign_ok holds at effect = 0. Widen lo above 0, or set is_locked: false for a pilot)",
                _ANCHOR_GATE_SECTION,
            )

        # D-4: expected_sharpe is OPTIONAL — absent (or explicit null) drops the Sharpe check.
        spec = anchors[aid]
        raw_sharpe = spec.get("expected_sharpe") if isinstance(spec, dict) else None
        expected_sharpe = (
            None if raw_sharpe is None
            else _require_number(raw_sharpe, f"{stem}.expected_sharpe", _ANCHOR_GATE_SECTION)
        )

        out[aid] = AnchorExpectation(
            anchor=aid,
            dominant_bias=dominant,
            expected_sign=sign,
            expected_magnitude_range=(lo, hi),
            expected_sharpe=expected_sharpe,
            is_locked=is_locked,
        )
    return out
