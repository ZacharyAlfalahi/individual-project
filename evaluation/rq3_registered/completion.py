"""Audit completion rate, conservative and permissive (registered: auditor design §11).

The registration asks for BOTH readings of an incomplete audit:

  * **conservative** — a refusal counts as a failure: ``completed / (completed + refused)``;
  * **permissive** — a refusal is excluded from numerator AND denominator:
    ``completed / completed``, i.e. 1.0 whenever anything completed.

The rate is reported **per arm**, never pooled. The lattice audits and the fitted-model
(IPCA) differential have different denominators and different refusal behaviour — the
lattice arm refuses only an incomplete audit, the IPCA arm carries typed refusals — so a single
system-level number would read as the former while being neither.

A ``not_applicable`` cell (``str``'s ``meas_err``: no estimand exists) is neither completed
nor refused. It leaves both rates untouched and is reported separately, because counting it
either way would answer a different question.
"""

from __future__ import annotations

from dataclasses import dataclass


class CompletionInputError(ValueError):
    """An arm's counts are inconsistent — raised rather than silently normalised."""


@dataclass(frozen=True)
class ArmCompletion:
    """One arm's completion counts. ``unit`` names what is being counted (audits, cell
    pairs) so two arms' rates are never read as the same denominator."""

    arm: str
    unit: str
    completed: int
    refused: int
    not_applicable: int = 0
    detail: str = ""

    def __post_init__(self) -> None:
        for name in ("completed", "refused", "not_applicable"):
            if getattr(self, name) < 0:
                raise CompletionInputError(f"{self.arm}: {name} is negative")
        if self.completed + self.refused == 0:
            raise CompletionInputError(
                f"{self.arm}: nothing completed and nothing refused — there is no rate to take"
            )

    @property
    def conservative(self) -> float:
        """Refusal = failure."""
        return self.completed / (self.completed + self.refused)

    @property
    def permissive(self) -> float:
        """Refusal excluded from both numerator and denominator."""
        return 1.0 if self.completed else 0.0

    def to_dict(self) -> dict:
        return {
            "arm": self.arm,
            "unit": self.unit,
            "completed": self.completed,
            "refused": self.refused,
            "not_applicable": self.not_applicable,
            "conservative": self.conservative,
            "conservative_fraction": f"{self.completed}/{self.completed + self.refused}",
            "permissive": self.permissive,
            "permissive_fraction": f"{self.completed}/{self.completed}",
            "detail": self.detail,
        }


def completion_block(arms: list[ArmCompletion]) -> dict:
    """The per-arm completion block. Deliberately emits NO pooled rate."""
    if not arms:
        raise CompletionInputError("no arms supplied")
    return {
        "definition": {
            "conservative": "completed / (completed + refused) — a refusal counts as a failure",
            "permissive": "completed / completed — refusals excluded from both sides",
            "not_applicable": "neither completed nor refused; excluded from both rates",
        },
        "pooling": ("NOT POOLED — the arms count different units and refuse differently; a "
                    "single system-level rate would read as the lattice arm's while being "
                    "neither arm's"),
        "arms": [a.to_dict() for a in arms],
    }


def lattice_arm(reports: dict[str, dict]) -> ArmCompletion:
    """The correction-lattice arm from ``{strategy: report}``: one audit per strategy,
    complete iff ``audit_scope == 'COMPLETE'``."""
    completed = sum(1 for r in reports.values() if r.get("audit_scope") == "COMPLETE")
    refused = len(reports) - completed
    not_applicable = sum(len(r.get("not_applicable_toggles") or []) for r in reports.values())
    return ArmCompletion(
        arm="correction_lattice",
        unit="audits (one per audited strategy)",
        completed=completed,
        refused=refused,
        not_applicable=not_applicable,
        detail=("audit_scope per strategy; not_applicable counts toggle cells with no "
                "estimand (str's meas_err), which are not audits"),
    )


def ipca_arm(n_runnable: int, n_refused: int, *, refusal_code: str = "") -> ArmCompletion:
    """The fitted-model differential arm, whose cell pairs DO refuse."""
    return ArmCompletion(
        arm="ipca_differential",
        unit="cell pairs",
        completed=n_runnable,
        refused=n_refused,
        detail=(f"typed refusals: {refusal_code}" if refusal_code else ""),
    )
