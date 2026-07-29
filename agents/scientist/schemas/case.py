"""
case.py — `ScientistCase`, the Scientist's INPUT contract (spec §5.1).

Assembled ONLY by the handoff seam (`shared/handoff/scientist_case.py`, R2) — the single
place an audit magnitude becomes a verdict. The assembled case then travels to the Researcher.

INVARIANT 1 (the wall), enforced by omission: there is NO differential-magnitude field
anywhere in this object — no Sharpe gap, effect size, p-value or t-statistic. It carries
references, toggle IDs, and PASS/FAIL/REFUSED verdicts ONLY. That omission is load-bearing:
it is exactly what lets `agents/scientist/` stay import-isolated from the magnitude-bearing
Auditor schemas (R2 / `test_wall_import_invariant`). DO NOT add a magnitude field here — a
Sharpe gap in this object would defeat the whole architecture (§13.10, prohibition 10).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

# Toggle IDENTITIES only. `agents.auditor.schemas.toggle` carries no magnitudes and is NOT on
# the wall's forbidden list — it is the permitted, required reuse (deliverable A).
from agents.auditor.schemas.toggle import TOGGLE_IDS, ToggleId

from .outcomes import VERDICTS, Verdict


@dataclass(frozen=True)
class DevelopmentWindow:
    """The locked development window (§5.1). e.g. start='2002-07', end='2021-12'."""

    start: str
    end: str

    def to_dict(self) -> dict:
        return {"start": self.start, "end": self.end}


@dataclass(frozen=True)
class HoldoutStatus:
    """Whether the holdout is currently accessible (§5.1). False during development."""

    accessible: bool

    def __post_init__(self) -> None:
        if not isinstance(self.accessible, bool):
            raise TypeError(f"HoldoutStatus.accessible must be a bool; got {self.accessible!r}")

    def to_dict(self) -> dict:
        return {"accessible": self.accessible}


@dataclass(frozen=True)
class ScientistCase:
    """One correction-sensitive strategy at its corrected lattice point, ready for the
    Researcher (§5.1). References, never copied narratives; verdicts, never magnitudes.

      failed_check_ids       — the toggles whose entry-rule verdict is FAIL (correction-
                               sensitive); a non-empty tuple is the entry condition (§1).
      failed_check_verdicts  — {toggle: FAIL} for exactly those ids (matches §5.1's
                               `{lib_gap: FAIL}`). `Verdict` admits PASS/FAIL/REFUSED because
                               it also types the seam's full per-toggle map; this field records
                               the FAILs (the entry trigger) only.
      applicable_toggles     — every toggle applicable to the strategy (runnable ∪ refused),
                               canonical TOGGLE_IDS order (§5.1 lists all five for str).
    """

    case_id: str
    strategy_id: str
    corrected_quant_config_ref: str
    corrected_run_ref: str
    audit_report_ref: str
    failed_check_ids: tuple[ToggleId, ...]
    failed_check_verdicts: Mapping[ToggleId, Verdict]
    applicable_toggles: tuple[ToggleId, ...]
    development_window: DevelopmentWindow
    holdout_status: HoldoutStatus

    def __post_init__(self) -> None:
        for t in self.applicable_toggles:
            if t not in TOGGLE_IDS:
                raise ValueError(
                    f"applicable_toggles: {t!r} is not a registered toggle {TOGGLE_IDS}"
                )
        for t in self.failed_check_ids:
            if t not in self.applicable_toggles:
                raise ValueError(
                    f"failed_check_ids: {t!r} is not among applicable_toggles "
                    f"{self.applicable_toggles}"
                )
        if set(self.failed_check_verdicts) != set(self.failed_check_ids):
            raise ValueError(
                "failed_check_verdicts keys must equal failed_check_ids; got "
                f"{tuple(self.failed_check_verdicts)} vs {self.failed_check_ids}"
            )
        for t, v in self.failed_check_verdicts.items():
            if v not in VERDICTS:
                raise ValueError(f"verdict for {t!r} must be one of {VERDICTS}; got {v!r}")
        # Defensive copy so the frozen case cannot be mutated through an aliased dict.
        object.__setattr__(self, "failed_check_verdicts", dict(self.failed_check_verdicts))

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "strategy_id": self.strategy_id,
            "corrected_quant_config_ref": self.corrected_quant_config_ref,
            "corrected_run_ref": self.corrected_run_ref,
            "audit_report_ref": self.audit_report_ref,
            "failed_check_ids": list(self.failed_check_ids),
            "failed_check_verdicts": dict(self.failed_check_verdicts),
            "applicable_toggles": list(self.applicable_toggles),
            "development_window": self.development_window.to_dict(),
            "holdout_status": self.holdout_status.to_dict(),
        }
