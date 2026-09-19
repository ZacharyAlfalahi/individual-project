"""
Adapter output types (D25/D28).

``adapt_spec`` returns an ``AdaptResult``: the per-leg factory calls, the combiner
instruction, and the collected refusals/flags. The adapter assembles the wrapped
factory kwargs and *forwards the call* to ``build_quant_config`` per leg -- it
never constructs a ``QuantConfig`` itself (D25: only the factory emits
representability refusals and fills defaults). Each ``LegCall`` therefore
carries both the assembled ``kwargs`` (the G1 attribution surface -- exactly what
the adapter produced) and the factory's ``QuantConfig | ConfigRefusal`` return
(the G2 round-trip surface -- ``to_rulebook`` diffs against the golden rulebooks).

Refusals are the single ``ConfigRefusal`` type Quant-side (D29 -> one RQ2
aggregation table), partitioned into two tuples:

  * ``refusals`` -- HARD refusals; any one makes the strategy ``refused`` (D28:
    any leg refused -> whole strategy refuses; partial averaging is a silent
    construction change). Sources: silence-on-a-load-bearing-field
    (REFUSED_ON_SILENCE), review-lane UNKNOWNs (REVIEW_REQUIRED), combiner=other
    (UNSUPPORTED_COMBINER), and every factory refusal (MISSING_BINDING, etc.).
  * ``flags`` -- SOFT review flags; the strategy still runs but is flagged
    (the silence table's ``flag_on_stated``, e.g. transaction_cost=net_of_costs).

``variant`` is set when any authorisation record (D27) injected a DESIGN value --
the strategy is then excluded from replication-fidelity aggregates (D23).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from agents.quant.config import ConfigRefusal, QuantConfig


@dataclass(frozen=True)
class CombinerInstruction:
    """How the legs combine into the strategy return (D28). ``single_leg`` is a
    pass-through of the one leg; ``equal_average`` is a by-date arithmetic mean
    over the *available* legs (adaptive divisor -- ledger item 40)."""

    kind: str                # "single_leg" | "equal_average"
    divisor: str | None = None  # "available" for equal_average; None for single_leg

    def to_dict(self) -> dict:
        out: dict = {"kind": self.kind}
        if self.divisor is not None:
            out["divisor"] = self.divisor
        return out


@dataclass(frozen=True)
class LegCall:
    """One leg's adapter output: the ``strategy_id`` (parent label + ordinal, D28),
    the assembled factory ``kwargs`` (provenance-wrapped fields; ``None`` = omitted
    so the factory fills its default), and the factory ``result``."""

    strategy_id: str
    kwargs: Mapping[str, object]
    result: QuantConfig | ConfigRefusal | None = None

    @property
    def refused(self) -> bool:
        return isinstance(self.result, ConfigRefusal)


@dataclass(frozen=True)
class AppliedStandingSub:
    """One standing substitution the adapter applied to this strategy (contract §6):
    a STATED paper value diverted to a DESIGN engine value by a project-wide convention
    (e.g. weighting_base market_value -> par). Recorded so the G2 register can emit its
    row; carries NO variant effect (standing subs stay IN fidelity aggregates, D23)."""

    field: str
    paper_value: object
    engine_value: object
    substitution_id: str


@dataclass(frozen=True)
class AdaptResult:
    """The adapter's whole-strategy output (D25/D28)."""

    strategy_label: str
    leg_calls: tuple[LegCall, ...] = ()
    combiner: CombinerInstruction | None = None
    refusals: tuple[ConfigRefusal, ...] = ()   # HARD -> strategy refused
    flags: tuple[ConfigRefusal, ...] = ()       # SOFT review flags -> proceed
    variant: bool = False
    # Standing substitutions applied (contract §6): DESIGN, non-variant. Read by the
    # G2 register; deliberately NOT emitted by to_dict (keeps the wall artifact stable).
    standing_subs_applied: tuple[AppliedStandingSub, ...] = ()

    @property
    def refused(self) -> bool:
        """True iff any hard adapter refusal exists OR any leg's factory call
        refused (D28: any leg refused -> whole strategy refuses)."""
        return bool(self.refusals) or any(lc.refused for lc in self.leg_calls)

    def to_dict(self) -> dict:
        return {
            "strategy_label": self.strategy_label,
            "refused": self.refused,
            "variant": self.variant,
            "combiner": self.combiner.to_dict() if self.combiner is not None else None,
            "n_legs": len(self.leg_calls),
            "refusals": [r.to_dict() for r in self.refusals],
            "flags": [f.to_dict() for f in self.flags],
        }


# A small internal accumulator the leg loop threads through (not part of the
# public surface). Kept here so result.py owns every adapter outcome shape.
@dataclass
class _Batch:
    """Run-to-completion refusal/flag accumulator (D29): the adapter collects ALL
    adapter-layer refusals before returning; the factory stays first-hit per leg."""

    refusals: list = field(default_factory=list)
    flags: list = field(default_factory=list)
    variant: bool = False
    standing_subs_applied: list = field(default_factory=list)

    def refuse(self, refusal: ConfigRefusal) -> None:
        self.refusals.append(refusal)

    def flag(self, refusal: ConfigRefusal) -> None:
        self.flags.append(refusal)
