"""
StrategySpec -- the Librarian's output object: a typed, frozen,
provenance-carrying description of ONE runnable strategy (D20), in *paper
language* (D3). Everything downstream (the Quant-side adapter, RQ1 scoring)
consumes it.

Shape (D13/D17/D19; paper_facts added v1.1):

    StrategySpec
      +- header:      SpecHeader        -- run provenance (ids, hashes, model, timestamp)
      +- part1:       Part1             -- formation_structure, asset_class, method_summary
      +- part2:       Part2             -- common spec-level fields + legs[] + combiner
      +- paper_facts: PaperFacts | None -- (v1.1) sample window + claimed metrics;
                                           extraction output the analysis consumes,
                                           NEVER read by the adapter (Guard 2, §5)

    Part2
      +- <28 common fields>     -- each Inherited[...]
      +- legs:     (Leg, ...)   -- one long-short construction per leg (D19)
      +- combiner: Combiner     -- how the legs combine

    Leg
      +- sort_signal:      SignalRef        -- MARKER (D22)
      +- control_axis:     SignalRef | None -- the 2nd sort of a double sort
      +- <8 per-leg sort fields>            -- each Inherited[...] (n_groups = MARKER;
                                               v1.1 adds control_n_groups)

Every *fact-bearing* field is an ``Inherited[T]`` (D6 -- no parallel Fact[T]),
reusing the frozen provenance layer. Each ``__post_init__`` type-guards that its
fact-bearing fields are actually ``Inherited`` (mirrors ``quant_config.py``'s
wrapper guards): a hand-built spec that passes a raw value is branded a
``LibrarianSchemaError``, not left to surface as an off-taxonomy ``AttributeError``
three layers downstream.

Guards here are *shape* guards only. Value/domain checks (menu membership, int
ranges), the D8 "no DESIGN" negative, and registry-aware SignalRef membership
are the ``validators/`` package's job -- kept separate so the schema is a pure
data contract and the policy is one auditable pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents.quant.config import Inherited

from ..errors import LibrarianSchemaError
from .signal_ref import LocatedQuote, SignalRef


def _inherited_to_dict(inh: Inherited) -> dict:
    """Serialise an Inherited to a JSON/YAML-safe dict (value + tag + evidence)."""
    return {"value": inh.value, "tag": inh.tag, "evidence": inh.evidence.to_dict()}


def _require_inherited(value: object, name: str) -> None:
    if not isinstance(value, Inherited):
        raise LibrarianSchemaError(
            f"{name} must be an Inherited (fact-bearing fields are provenance-wrapped, D6); "
            f"got {type(value).__name__}"
        )


# ---------------------------------------------------------------------------
# Header -- run provenance. Plain str/None except strategy_label (Inherited).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpecHeader:
    """Run + provenance stamps for one spec. Every field is a plain ``str``/
    ``None`` scalar EXCEPT ``strategy_label`` -- the paper's name for the
    strategy, itself a quote-bearing fact (D20: RQ1 scoring is keyed by it), so
    it is an ``Inherited[str]``."""

    paper_id: str
    strategy_label: Inherited  # Inherited[str] -- the paper's name for the strategy
    registry_version: str
    registry_hash: str
    silence_table_version: str
    canonical_text_hash: str
    prompt_template_hashes: str | None = None
    model_ids: str | None = None
    run_id: str | None = None
    timestamp: str | None = None
    trace_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.paper_id, str) or self.paper_id.strip() == "":
            raise LibrarianSchemaError("SpecHeader.paper_id must be a non-empty string")
        _require_inherited(self.strategy_label, "SpecHeader.strategy_label")
        for name, val in (
            ("registry_version", self.registry_version),
            ("registry_hash", self.registry_hash),
            ("silence_table_version", self.silence_table_version),
            ("canonical_text_hash", self.canonical_text_hash),
        ):
            if not isinstance(val, str) or val.strip() == "":
                raise LibrarianSchemaError(f"SpecHeader.{name} must be a non-empty string")
        for name, val in (
            ("prompt_template_hashes", self.prompt_template_hashes),
            ("model_ids", self.model_ids),
            ("run_id", self.run_id),
            ("timestamp", self.timestamp),
            ("trace_sha256", self.trace_sha256),
        ):
            if val is not None and not isinstance(val, str):
                raise LibrarianSchemaError(f"SpecHeader.{name} must be a str or None")

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "strategy_label": _inherited_to_dict(self.strategy_label),
            "registry_version": self.registry_version,
            "registry_hash": self.registry_hash,
            "silence_table_version": self.silence_table_version,
            "canonical_text_hash": self.canonical_text_hash,
            "prompt_template_hashes": self.prompt_template_hashes,
            "model_ids": self.model_ids,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "trace_sha256": self.trace_sha256,
        }


# ---------------------------------------------------------------------------
# Part 1 -- three fields (D13). method_summary is free text (D16).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MethodSummary:
    """The one Part-1 field that cannot be a menu (D16): an LLM-written,
    per-strategy, three-slot summary, with 1-3 supporting located quotes.

    Provenance-wrapped like the other facts: ``summary`` is an
    ``Inherited[str]`` so its tag/evidence travel with it; the located quotes
    are additionally carried for the audit trail."""

    summary: Inherited  # Inherited[str]
    quotes: tuple[LocatedQuote, ...] = ()

    def __post_init__(self) -> None:
        _require_inherited(self.summary, "MethodSummary.summary")
        if isinstance(self.quotes, list):
            object.__setattr__(self, "quotes", tuple(self.quotes))
        if not isinstance(self.quotes, tuple):
            raise LibrarianSchemaError("MethodSummary.quotes must be a tuple of LocatedQuote")
        for q in self.quotes:
            if not isinstance(q, LocatedQuote):
                raise LibrarianSchemaError(
                    f"MethodSummary.quotes entries must be LocatedQuote; got {type(q).__name__}"
                )
        if len(self.quotes) > 3:
            raise LibrarianSchemaError("MethodSummary supports at most 3 supporting quotes (D16)")

    def to_dict(self) -> dict:
        return {
            "summary": _inherited_to_dict(self.summary),
            "quotes": [q.to_dict() for q in self.quotes],
        }


@dataclass(frozen=True)
class Part1:
    """Part 1: formation_structure, asset_class (both Inherited[str] menu picks),
    method_summary (a MethodSummary, D16)."""

    formation_structure: Inherited  # Inherited[str]
    asset_class: Inherited          # Inherited[str]
    method_summary: MethodSummary

    def __post_init__(self) -> None:
        _require_inherited(self.formation_structure, "Part1.formation_structure")
        _require_inherited(self.asset_class, "Part1.asset_class")
        if not isinstance(self.method_summary, MethodSummary):
            raise LibrarianSchemaError(
                "Part1.method_summary must be a MethodSummary; "
                f"got {type(self.method_summary).__name__}"
            )

    def to_dict(self) -> dict:
        return {
            "formation_structure": _inherited_to_dict(self.formation_structure),
            "asset_class": _inherited_to_dict(self.asset_class),
            "method_summary": self.method_summary.to_dict(),
        }


# ---------------------------------------------------------------------------
# Part 2 -- sort block: Leg + Combiner (D19).
# ---------------------------------------------------------------------------

# The per-leg Inherited fields (sort_signal / control_axis are SignalRefs,
# handled separately). Order matters only for to_dict readability.
_LEG_INHERITED_FIELDS: tuple[str, ...] = (
    "sort_kind",
    "bucketing_method",
    "n_groups",
    "stripe_aggregation",
    "control_missing_policy",
    "long_leg",
    "signal_transform",
    "control_n_groups",
)


@dataclass(frozen=True)
class Leg:
    """One long-short construction from one grid (D19). Carries the per-leg sort
    fields; ``sort_signal`` is a SignalRef (MARKER, D22) and ``control_axis`` is
    a SignalRef or None (the 2nd axis of a double sort). The remaining fields are
    ``Inherited[...]`` (``n_groups`` is the other MARKER; ``control_n_groups``,
    v1.1, is the 2nd-axis group count -- sibling to ``control_axis``).

    ``control_n_groups`` is declared before the defaulted ``control_axis`` so the
    dataclass keeps all non-default fields ahead of the one field with a default."""

    sort_signal: SignalRef
    sort_kind: Inherited          # Inherited[str]
    bucketing_method: Inherited   # Inherited[str]
    n_groups: Inherited           # Inherited[int] -- MARKER
    stripe_aggregation: Inherited  # Inherited[str]
    control_missing_policy: Inherited  # Inherited[str]
    long_leg: Inherited           # Inherited[str]
    signal_transform: Inherited   # Inherited[str]
    control_n_groups: Inherited   # Inherited[int] (v1.1) -- 2nd-axis group count
    control_axis: SignalRef | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sort_signal, SignalRef):
            raise LibrarianSchemaError(
                f"Leg.sort_signal must be a SignalRef; got {type(self.sort_signal).__name__}"
            )
        if self.control_axis is not None and not isinstance(self.control_axis, SignalRef):
            raise LibrarianSchemaError(
                "Leg.control_axis must be a SignalRef or None; "
                f"got {type(self.control_axis).__name__}"
            )
        for name in _LEG_INHERITED_FIELDS:
            _require_inherited(getattr(self, name), f"Leg.{name}")

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"sort_signal": self.sort_signal.to_dict()}
        out["control_axis"] = (
            self.control_axis.to_dict() if self.control_axis is not None else None
        )
        for name in _LEG_INHERITED_FIELDS:
            out[name] = _inherited_to_dict(getattr(self, name))
        return out


@dataclass(frozen=True)
class Combiner:
    """How the legs combine into the strategy return (D19). Wraps an
    ``Inherited["single_leg"|"equal_average"|"other"]``. ``other`` is a refusal
    at the adapter; here it is a legal (menu-valid) value."""

    kind: Inherited  # Inherited[str]

    def __post_init__(self) -> None:
        _require_inherited(self.kind, "Combiner.kind")

    def to_dict(self) -> dict:
        return {"kind": _inherited_to_dict(self.kind)}


# ---------------------------------------------------------------------------
# Part 2 -- common block + legs + combiner.
# ---------------------------------------------------------------------------

# The 28 common fields, all Inherited[...]. Listed here so the type-guard and
# to_dict stay in lockstep with fields.py (a missing field would be caught by
# validate_librarian_spec's completeness check, not silently dropped).
_COMMON_INHERITED_FIELDS: tuple[str, ...] = (
    "eligibility_missing_policy",
    "return_availability_policy",
    "signal_lag",
    "lag_convention",
    "min_bonds",
    "min_bonds_granularity",
    "tie_break_policy",
    "weighting_scheme",
    "weighting_base",
    "weight_timing",
    "strategy_side",
    "empty_leg_policy",
    "transaction_cost_convention",
    "return_label",
    "rebalance_frequency",
    "holding_period",
    "overlap_convention",
    "cohort_weighting",
    "burn_in_policy",
    "missing_return_policy",
    "realisation_min_survivors",
    "return_compounding",
    "significance_convention",
    "hac_lags",
    "annualisation",
    "rf_convention",
    "benchmark_model",
    "expost_trim",
)


@dataclass(frozen=True)
class Part2:
    """Part 2: the 28 spec-level common fields (each ``Inherited[...]``), plus
    the sort block (``legs`` + ``combiner``, D19). ``legs`` is a non-empty tuple
    of ``Leg``; anchor golds are one-element lists (D19 ripple)."""

    # common fields (spec-level)
    eligibility_missing_policy: Inherited
    return_availability_policy: Inherited
    signal_lag: Inherited
    lag_convention: Inherited
    min_bonds: Inherited
    min_bonds_granularity: Inherited
    tie_break_policy: Inherited
    weighting_scheme: Inherited
    weighting_base: Inherited
    weight_timing: Inherited
    strategy_side: Inherited
    empty_leg_policy: Inherited
    transaction_cost_convention: Inherited
    return_label: Inherited
    rebalance_frequency: Inherited
    holding_period: Inherited
    overlap_convention: Inherited
    cohort_weighting: Inherited
    burn_in_policy: Inherited
    missing_return_policy: Inherited
    realisation_min_survivors: Inherited
    return_compounding: Inherited
    significance_convention: Inherited
    hac_lags: Inherited
    annualisation: Inherited
    rf_convention: Inherited
    benchmark_model: Inherited
    expost_trim: Inherited
    # sort block
    combiner: Combiner
    legs: tuple[Leg, ...] = ()

    def __post_init__(self) -> None:
        for name in _COMMON_INHERITED_FIELDS:
            _require_inherited(getattr(self, name), f"Part2.{name}")
        if isinstance(self.legs, list):
            object.__setattr__(self, "legs", tuple(self.legs))
        if not isinstance(self.legs, tuple) or len(self.legs) == 0:
            raise LibrarianSchemaError("Part2.legs must be a non-empty tuple of Leg (D19)")
        for i, leg in enumerate(self.legs):
            if not isinstance(leg, Leg):
                raise LibrarianSchemaError(
                    f"Part2.legs[{i}] must be a Leg; got {type(leg).__name__}"
                )
        if not isinstance(self.combiner, Combiner):
            raise LibrarianSchemaError(
                f"Part2.combiner must be a Combiner; got {type(self.combiner).__name__}"
            )

    def to_dict(self) -> dict:
        out: dict[str, Any] = {}
        for name in _COMMON_INHERITED_FIELDS:
            out[name] = _inherited_to_dict(getattr(self, name))
        out["legs"] = [leg.to_dict() for leg in self.legs]
        out["combiner"] = self.combiner.to_dict()
        return out


# ---------------------------------------------------------------------------
# Part 2 (v1.1) -- paper_facts: a SEPARATE spec-level block (NOT a Part 2
# execution field). Sample window + claimed metrics, quote-bearing and
# RQ1-scorable; the *analysis* (fidelity harness, Reporter's numeric verifier)
# consumes them, the adapter NEVER does (Guard 2, §5 -- structurally unreachable
# because it lives on ``spec.paper_facts``, a sibling the leg/common walk never
# touches). Anti-bloat rule (schema doc, verbatim): no fifth field without a
# named consumer and a log amendment.
# ---------------------------------------------------------------------------

# The paper_facts Inherited fields, in to_dict order. claimed_headline_metric's
# {mean, t_stat, unit} composite rides inside its single Inherited.value (a dict),
# exactly as significance_convention carries a composite value.
_PAPER_FACTS_INHERITED_FIELDS: tuple[str, ...] = (
    "sample_start",
    "sample_end",
    "universe_filter",
    "claimed_headline_metric",
)


@dataclass(frozen=True)
class PaperFacts:
    """(v1.1) The paper's own reported facts: sample window + universe + claimed
    headline metric. Each field is an ``Inherited[...]`` with the same quote+locator
    discipline as the rest of the spec (STATED or UNKNOWN(not_stated)). These are
    extraction OUTPUT the analysis reads; they are never adapter inputs (Guard 2)."""

    sample_start: Inherited            # Inherited[date-as-stated]
    sample_end: Inherited              # Inherited[date-as-stated]
    universe_filter: Inherited         # Inherited[str] (free text, weaker-checked)
    claimed_headline_metric: Inherited  # Inherited[{mean, t_stat, unit}]

    def __post_init__(self) -> None:
        for name in _PAPER_FACTS_INHERITED_FIELDS:
            _require_inherited(getattr(self, name), f"PaperFacts.{name}")

    def to_dict(self) -> dict:
        return {name: _inherited_to_dict(getattr(self, name)) for name in _PAPER_FACTS_INHERITED_FIELDS}


# ---------------------------------------------------------------------------
# The whole spec.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StrategySpec:
    """One runnable strategy in paper language (D20). = header + Part1 + Part2,
    plus (v1.1) an optional ``paper_facts`` block. ``paper_facts`` defaults to
    ``None`` so v1 call sites keep working; when present it is analysis-only data
    the adapter never reads (Guard 2, §5)."""

    header: SpecHeader
    part1: Part1
    part2: Part2
    paper_facts: PaperFacts | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.header, SpecHeader):
            raise LibrarianSchemaError(
                f"StrategySpec.header must be a SpecHeader; got {type(self.header).__name__}"
            )
        if not isinstance(self.part1, Part1):
            raise LibrarianSchemaError(
                f"StrategySpec.part1 must be a Part1; got {type(self.part1).__name__}"
            )
        if not isinstance(self.part2, Part2):
            raise LibrarianSchemaError(
                f"StrategySpec.part2 must be a Part2; got {type(self.part2).__name__}"
            )
        if self.paper_facts is not None and not isinstance(self.paper_facts, PaperFacts):
            raise LibrarianSchemaError(
                "StrategySpec.paper_facts must be a PaperFacts or None; "
                f"got {type(self.paper_facts).__name__}"
            )

    def to_dict(self) -> dict:
        return {
            "header": self.header.to_dict(),
            "part1": self.part1.to_dict(),
            "part2": self.part2.to_dict(),
            "paper_facts": self.paper_facts.to_dict() if self.paper_facts is not None else None,
        }
