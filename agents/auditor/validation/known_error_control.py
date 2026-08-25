"""
known_error_control.py — §4.7 arm-2 known-error positive control on real data (BBW).

BBW is the one real corpus paper carrying a DOCUMENTED defect known before the run: a
look-ahead / lead error (DMR 2023 p.4-5; DRR's DRF-specific critique), which the `drf`
hypothesis-registry row (config/hypothesis_registry.yaml) already records as "a look-ahead
lead error (a non-lattice lead_lag defect)". The integrated pipeline must REPRODUCE that
defect and its correction must FIX it. This module is the wired, falsifiable instantiation
of that expectation — resolving the §4.7 "success criterion not yet frozen" note:

  * WHICH raw-vs-corrected differential — the lead/lag collapse/restore measured by
    ``scripts/run_leadlag_gate.py``: inject the as-published look-ahead over BBW's documented
    window and correlate against the correctly-aligned factor (the "as-published" arm), then
    round-trip re-align (the "corrected" arm). A two-build differential, not a level.
  * WHAT gap counts as a pass — the PRE-REGISTERED band ``validation.gate_thresholds.lead_lag``:
    the as-published correlation collapses BELOW ``collapse_max`` (the defect is reproduced) AND
    the corrected round-trip restores it ABOVE ``restore_min`` (the correction fixes it).

Nothing is restated here: the band is READ from ``docs/thresholds.yaml`` and the anchor is the
registered ``drf`` factor. ``drf`` is the PRIMARY positive-control anchor (DRR's DRF-specific
critique); ``crf`` and ``lrf`` corroborate but do not gate (per the per-anchor, no-aggregate-
fraction philosophy of ``anchor_triangulation``). This chose the lead_lag route over a
``meas_err`` return-differential band: the drf ``meas_err`` DOE effect is ~null (+0.006 %/mo,
0/8 FDR) and its magnitude band was retracted for cause (drf is sign-only), so a ``meas_err``
positive control would not fire — the lead_lag defect is the one BBW error that is documented,
banded, and reproduced.

This is a DETERMINISTIC evaluator (zero LLM, no gating I/O beyond the thresholds read): it
GRADES the gate's emitted report, it does not run the gate. Because BBW helped shape
development, a PASS licenses only that the connected pipeline routes BBW to its pre-specified
defect and correction — never that the system discovers an unknown problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_THRESHOLDS_PATH = Path(__file__).resolve().parents[3] / "docs" / "thresholds.yaml"

PRIMARY_ANCHOR = "drf"                       # DRR's DRF-specific documented look-ahead critique
CORROBORATING_ANCHORS = ("crf", "lrf")       # reported beside drf; do NOT gate the control


@dataclass(frozen=True)
class LeadLagCriterion:
    """The pre-registered lead/lag pass band, read from
    ``validation.gate_thresholds.lead_lag`` — never restated in code."""

    collapse_max: float                      # as-published corr must drop BELOW this
    restore_min: float                       # round-trip corrected corr must rise ABOVE this


@dataclass(frozen=True)
class AnchorControlOutcome:
    anchor: str
    corr_as_published: float                 # corr(correct, defective) — injected look-ahead arm
    corr_corrected: float                    # corr(correct, round-trip restored) — correction arm
    collapsed: bool                          # as-published corr < collapse_max (defect reproduced)
    restored: bool                           # corrected corr >= restore_min (correction fixed it)
    passed: bool                             # collapsed AND restored

    def to_dict(self) -> dict:
        return {
            "anchor": self.anchor,
            "corr_as_published": self.corr_as_published,
            "corr_corrected": self.corr_corrected,
            "collapsed": self.collapsed,
            "restored": self.restored,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class KnownErrorControlVerdict:
    """The §4.7 arm-2 verdict. ``passed`` is the PRIMARY (drf) outcome — the positive
    control's decision; corroborators are reported, not gating."""

    criterion: LeadLagCriterion
    primary: AnchorControlOutcome
    corroborators: tuple[AnchorControlOutcome, ...]
    passed: bool
    detail: str

    def to_dict(self) -> dict:
        return {
            "control": "known_error_positive_control",
            "paper": "BBW",
            "defect": "lead_lag_look_ahead",
            "criterion": {"collapse_max": self.criterion.collapse_max,
                          "restore_min": self.criterion.restore_min},
            "primary": self.primary.to_dict(),
            "corroborators": [c.to_dict() for c in self.corroborators],
            "passed": self.passed,
            "detail": self.detail,
        }


def load_lead_lag_criterion(thresholds_path: str | Path | None = None) -> LeadLagCriterion:
    """Read the pre-registered lead/lag band, HARD-RAISING if absent (never default a
    criterion: same fail-loud discipline as the other auditor-validation loaders)."""
    import yaml

    path = Path(thresholds_path) if thresholds_path is not None else _THRESHOLDS_PATH
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        g = data["validation"]["gate_thresholds"]["lead_lag"]
        return LeadLagCriterion(float(g["collapse_max"]), float(g["restore_min"]))
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            "validation.gate_thresholds.lead_lag missing/malformed in thresholds.yaml — the "
            "known-error positive-control criterion is unregistered (fail loud, do not default)"
        ) from exc


def _grade(anchor: str, arm: dict, criterion: LeadLagCriterion) -> AnchorControlOutcome:
    """Grade one arm of ``leadlag_gate.json`` against the loaded band. The pass is
    RE-DERIVED from the raw correlations here (an independent check), not taken from the
    gate's own ``collapsed_below_max`` / ``restored_above_min`` booleans."""
    d = float(arm["corr_correct_vs_defective"])
    r = float(arm["corr_correct_vs_restored"])
    collapsed = d < criterion.collapse_max
    restored = r >= criterion.restore_min
    return AnchorControlOutcome(anchor, d, r, collapsed, restored, collapsed and restored)


def evaluate_known_error_control(
    gate_report: dict, *, thresholds_path: str | Path | None = None
) -> KnownErrorControlVerdict:
    """Grade a ``run_leadlag_gate.py`` report as the §4.7 known-error positive control.

    The control PASSES iff the PRIMARY anchor (drf) reproduces the documented look-ahead
    defect (as-published correlation collapses below ``collapse_max``) AND its correction
    restores the factor (round-trip correlation above ``restore_min``). Raises if the report
    lacks the drf arm — a missing anchor is a failure to run, never a silent pass."""
    criterion = load_lead_lag_criterion(thresholds_path)
    arms = gate_report.get("arms") if isinstance(gate_report, dict) else None
    if not isinstance(arms, dict) or PRIMARY_ANCHOR not in arms:
        raise ValueError(
            f"lead/lag gate report is missing the {PRIMARY_ANCHOR!r} arm — cannot grade the "
            "known-error positive control (expected scripts/run_leadlag_gate.py output)"
        )
    primary = _grade(PRIMARY_ANCHOR, arms[PRIMARY_ANCHOR], criterion)
    corroborators = tuple(
        _grade(a, arms[a], criterion) for a in CORROBORATING_ANCHORS if a in arms
    )
    detail = (
        f"{PRIMARY_ANCHOR}: as-published corr={primary.corr_as_published:.3f} "
        f"(< {criterion.collapse_max}? {primary.collapsed}), corrected corr="
        f"{primary.corr_corrected:.3f} (>= {criterion.restore_min}? {primary.restored}) -> "
        f"{'defect REPRODUCED and correction FIXED it' if primary.passed else 'NOT reproduced/fixed'}"
    )
    return KnownErrorControlVerdict(criterion, primary, corroborators, primary.passed, detail)
