"""
audit_core.py — the OUTPUT schema for the analytical spine.

`AuditCore` is the typed verdict the instrument core produces for ONE strategy:
the derived audit scope and its conditioning, the common-support diagnostics, the
three lattice readings (corner marginals, saturated bases, Shapley), and the
per-toggle invariance results. It is never an input.

`is_no_op` lives HERE (on InvarianceResult), never on the input `ToggleFacts` — so
it is structurally impossible to declare a no-op in a spec (§3.3, §3.5). This is
the scoped precursor to the full §12 AuditReport (which adds inference,
Bayesian, prevalence, economic and recovery fields).

Every numeric field is reachable from `to_dict()`, so the numeric verifier (§11)
can assert that every number in the explainer's prose traces to a typed field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .decomposition import CornerMarginals, SaturatedBasis, ShapleyResult
from .toggle import ToggleId, ToggleState

AuditScope = Literal["COMPLETE", "PARTIAL", "REFUSED"]


@dataclass(frozen=True)
class SupportInfo:
    """Common-support diagnostics (§4.2). The common intersection drives every
    cross-cell quantity; the native ranges are descriptive. `gate_passed` is the
    §4.2 minimum-support gate; `downgraded` records that cross-cell nonlinear
    inference was refused / reduced to descriptive because support was too short
    (a lattice can be complete yet statistically unusable)."""

    t_common: int
    reference_native_months: int
    common_fraction: float
    min_common_months: int
    min_common_fraction_of_reference: float
    native_min_months: int
    native_max_months: int
    gate_passed: bool
    downgraded: bool

    def to_dict(self) -> dict:
        return {
            "t_common": self.t_common,
            "reference_native_months": self.reference_native_months,
            "common_fraction": float(self.common_fraction),
            "min_common_months": self.min_common_months,
            "min_common_fraction_of_reference": float(self.min_common_fraction_of_reference),
            "native_min_months": self.native_min_months,
            "native_max_months": self.native_max_months,
            "gate_passed": self.gate_passed,
            "downgraded": self.downgraded,
        }


@dataclass(frozen=True)
class InvarianceResult:
    """The §3.5 invariance verdict for one `expect_no_op` toggle. The claim being
    proved is *different config, identical behaviour* — BOTH halves are required.
    `is_no_op` is True iff the config hashes differ AND every observable output
    hash matches AND the membership proxy was actually verifiable.

    `membership_verified` guards the multi-leg case: an `equal_average` combiner
    that dropped its `n_bonds` column would make the membership half trivially true
    (empty==empty), so a no-op is NEVER certified when the proxy is unavailable —
    the conservative direction (a membership change must not slip through)."""

    toggle_id: ToggleId
    config_hashes_differ: bool
    returns_identical: bool
    n_bonds_identical: bool
    metrics_identical: bool
    is_no_op: bool
    membership_verified: bool = True
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "toggle_id": self.toggle_id,
            "config_hashes_differ": self.config_hashes_differ,
            "returns_identical": self.returns_identical,
            "n_bonds_identical": self.n_bonds_identical,
            "metrics_identical": self.metrics_identical,
            "membership_verified": self.membership_verified,
            "is_no_op": self.is_no_op,
            "note": self.note,
        }


def _bias_class_partition_dict(harsanyi, runnable_toggles) -> dict:
    """Serialised bias_class partition for the runnable lattice (ADR §5.2). Local
    import so the schema layer never eagerly loads the checks layer at module init."""
    from ..checks.algebra import bias_class_partition

    return bias_class_partition(harsanyi, runnable_toggles).to_dict()


@dataclass(frozen=True, eq=False)
class AuditCore:
    """The instrument-core verdict for one strategy (steps 2-11)."""

    strategy_label: str
    audit_scope: AuditScope
    runnable_toggles: tuple[ToggleId, ...]
    conditioning_signature: tuple[tuple[ToggleId, ToggleState], ...]
    conditioning_statement: str | None
    primary_metric: str
    support: SupportInfo
    corner_marginals: CornerMarginals
    saturated: SaturatedBasis
    shapley: ShapleyResult
    invariance: tuple[InvarianceResult, ...]
    pre_registration_tag: str | None = None
    # Toggles excluded because NO estimand exists for this strategy (ADR §5.4). This
    # is a first-class part of the verdict ("the paper could not have committed this
    # bias"), distinct from a construct failure (input_unavailable) or a measured
    # no-op. Empty for every all-runnable audit; declared per strategy, never derived.
    not_applicable_toggles: tuple[ToggleId, ...] = ()

    def to_dict(self) -> dict:
        return {
            "strategy_label": self.strategy_label,
            "audit_scope": self.audit_scope,
            "runnable_toggles": list(self.runnable_toggles),
            "not_applicable_toggles": list(self.not_applicable_toggles),
            "conditioning_signature": [
                {"toggle_id": t, "fixed_state": s}
                for (t, s) in self.conditioning_signature
            ],
            "conditioning_statement": self.conditioning_statement,
            "primary_metric": self.primary_metric,
            "support": self.support.to_dict(),
            "corner_marginals": self.corner_marginals.to_dict(),
            "saturated_bases": self.saturated.to_dict(),
            # The bias_class headline (ADR §5.2): the endpoint gap split into
            # methodological-construction / data-quality / cross-class components.
            # Derived from the saturated Harsanyi dividends over the runnable
            # lattice — a pure recombination, no re-fit. A local import breaks the
            # schemas<->checks module ordering cleanly.
            "bias_class_partition": _bias_class_partition_dict(
                self.saturated.harsanyi, self.runnable_toggles
            ),
            "shapley": self.shapley.to_dict(),
            "invariance": [r.to_dict() for r in self.invariance],
            "pre_registration_tag": self.pre_registration_tag,
        }


def conditioning_statement_for(
    signature: tuple[tuple[ToggleId, ToggleState], ...],
) -> str | None:
    """The human-readable conditioning statement that must travel with every
    number a PARTIAL audit produces (§3.4, §8.2.3b). None for a COMPLETE audit
    (empty signature)."""
    if not signature:
        return None
    held = ", ".join(f"{t} held at {s}" for (t, s) in signature)
    return (
        "Conditional audit: the reported gap decomposes the correction of the "
        f"runnable toggles with {held}. It is NOT the full as-published -> "
        "corrected gap."
    )
