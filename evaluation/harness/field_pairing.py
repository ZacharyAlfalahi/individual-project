"""
G3 gold<->run field pairing (evaluation contract §3, D34/D37).

Joins a hand-authored gold ``StrategySpec`` to one run's trace records. Two
problems to solve: the gold is walked as dotted paths while the trace is keyed by
FLAT field names, and the gold's walker does not reach every scoreable field.

**Gold-side walk.** ``spec_validators._iter_inherited`` yields (dotted_path,
Inherited) for the header, Part 1, all 28 Part-2 common fields, the combiner, the
leg fields, and the signal_ref concept_ids + parameters. It does NOT walk
``paper_facts``. That omission is CORRECT for the validator -- Guard 2 keeps
paper_facts off the adapter path -- so it is extended here rather than widened at
source, where widening would silently change validator semantics (it would begin
applying the D8 negatives inside paper_facts).

**The absent-control_axis case.** On a single-sort anchor the gold's
``leg.control_axis`` is ``None``, so the walker yields no path for it at all --
while the run may well have extracted one. Dropping the key would silently
discard exactly the interesting case: a model asserting a control axis the paper
does not have. ``iter_gold_fields`` therefore injects an explicit ``None``
comparand, making "gold says there is no control axis" a first-class fact that
the compare policy can score as a fabrication.

**Mapping is table-driven with a terminal raise.** An unmapped path is a hard
error, never a silent skip -- the same discipline ``gold_loader`` applies to a
schema field missing from a gold. A schema change surfaces here loudly instead of
quietly shrinking the scored universe (contract §8 non-elasticity).
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agents.librarian.schema.strategy_spec import (  # noqa: E402
    _PAPER_FACTS_INHERITED_FIELDS,
)
from agents.librarian.validators.spec_validators import _iter_inherited  # noqa: E402


class PairingError(RuntimeError):
    """A gold field could not be mapped to a trace key. A build error, surfaced
    loudly -- a silently dropped field would shrink the scored universe."""


# D37 ruling 4: enumeration input, re-located but never dual-model extracted.
# Scoring it would measure the lister, not the reader. Counted as an explicit
# exclusion rather than dropped, so the exclusion is visible in the totals.
EXCLUDED_PATHS: frozenset[str] = frozenset({"header.strategy_label"})

# D37 ruling 7: prose, scored by rubric (declared weaker, excluded from headline
# accuracy). No rubric is authored, so these are never compared -- and never
# given a synthesised text-similarity score, which is the number the contract's
# non-elasticity clause exists to prevent.
RUBRIC_FIELDS: frozenset[str] = frozenset({"method_summary", "universe_filter"})

_LEG_SIGNAL_RE = re.compile(r"^part2\.legs\[(\d+)\]\.(sort_signal|control_axis)\.concept_id$")
_LEG_PARAM_RE = re.compile(r"^part2\.legs\[(\d+)\]\.(?:sort_signal|control_axis)\.parameters\.(.+)$")
_LEG_FIELD_RE = re.compile(r"^part2\.legs\[(\d+)\]\.(.+)$")
_PART_RE = re.compile(r"^part[12]\.(.+)$")
_PAPER_FACTS_RE = re.compile(r"^paper_facts\.(.+)$")


@dataclass(frozen=True, order=True)
class FieldKey:
    """The join key: the trace-side flat field name plus the leg it belongs to.

    ``leg_index`` is ``None`` for spec-level fields. It is carried even though
    every anchor is single-leg today, because a multi-leg spec would otherwise
    collapse two legs' fields onto one key and score them as one field."""

    name: str
    leg_index: int | None = None


def trace_key(dotted_path: str) -> FieldKey | None:
    """Map a gold dotted path to its trace key, or ``None`` for a path that
    carries no trace record BY DESIGN. Raises ``PairingError`` on anything
    unmapped."""
    if dotted_path in EXCLUDED_PATHS:
        return None
    if dotted_path == "part1.method_summary.summary":
        return FieldKey("method_summary")
    if dotted_path == "part2.combiner.kind":
        return FieldKey("combiner")

    m = _LEG_SIGNAL_RE.match(dotted_path)
    if m:
        return FieldKey(m.group(2), int(m.group(1)))
    m = _LEG_PARAM_RE.match(dotted_path)
    if m:
        # fill_signal_ref pushes parameter traces under the BARE parameter name
        # (signal_filler.py), so a registry parameter shares the flat trace
        # namespace with schema fields. Every v1 concept has an empty parameter
        # schema, so this cannot fire today; load_run's duplicate guard catches it
        # if a future concept introduces a colliding name.
        return FieldKey(m.group(2), int(m.group(1)))
    m = _LEG_FIELD_RE.match(dotted_path)
    if m:
        return FieldKey(m.group(2), int(m.group(1)))
    m = _PAPER_FACTS_RE.match(dotted_path)
    if m:
        return FieldKey(m.group(1))
    m = _PART_RE.match(dotted_path)
    if m:
        return FieldKey(m.group(1))

    raise PairingError(
        f"gold path {dotted_path!r} has no trace-key mapping -- a schema field with no "
        "home is a build error, never a silent skip"
    )


def iter_gold_fields(spec) -> Iterator[tuple[str, object]]:
    """Every scoreable gold field as (dotted_path, Inherited | None).

    ``_iter_inherited`` plus the two gaps it leaves: the ``paper_facts`` block,
    and an explicit ``None`` for an absent ``control_axis`` (see module docstring).
    The ``None`` is yielded as the VALUE, not as an ``Inherited`` -- the compare
    policy treats "gold has no control axis" as a fact to be matched, not as a
    missing input."""
    yield from _iter_inherited(spec)

    if getattr(spec, "paper_facts", None) is not None:
        for name in _PAPER_FACTS_INHERITED_FIELDS:
            yield f"paper_facts.{name}", getattr(spec.paper_facts, name)

    for idx, leg in enumerate(spec.part2.legs):
        if getattr(leg, "control_axis", None) is None:
            yield f"part2.legs[{idx}].control_axis.concept_id", None


@dataclass(frozen=True)
class PairedField:
    """One gold field joined to its run record (or to nothing)."""

    key: FieldKey
    dotted_path: str
    gold: object          # Inherited, or None for an absent control_axis
    run: object           # RunField, or None when the run has no record
    excluded_rubric: bool

    @property
    def gold_tag(self) -> str:
        if self.gold is None:
            return "UNKNOWN"          # "the paper has no control axis" == silence
        return self.gold.tag

    @property
    def gold_value(self):
        return None if self.gold is None else self.gold.value


# --- run-side leg encoding (multi-leg join, D34) ------------------------------
#
# A multi-leg run keys its per-leg records ``legs[{j}].{field}`` (the gold path
# ``part2.legs[i].<field>`` minus ``part2.``); a single-leg run keeps the BARE
# ``<field>`` names. load_run rejects duplicate flat names, so a 3-leg run MUST use
# the leg-scoped form. The gold->run leg permutation is resolved by ``match_legs``.

_RUN_LEG_RE = re.compile(r"^legs\[(\d+)\]\.")


def _run_concept(artefacts, j: int, field: str) -> object:
    """The run's normalised concept value for leg j's sort_signal/control_axis,
    taken from whichever model answered (matching is identity-only, not scoring)."""
    rf = artefacts.fields.get(f"legs[{j}].{field}")
    if rf is None:
        return None
    if getattr(rf, "a_answered", False):
        return rf.normalised_a
    if getattr(rf, "b_answered", False):
        return rf.normalised_b
    return None


def _run_leg_proxy(artefacts, j: int):
    """A lightweight stand-in exposing ``.sort_signal.concept_id.value`` /
    ``.control_axis.concept_id.value`` -- the only attributes ``match_legs`` reads."""
    from types import SimpleNamespace

    def sig(field: str):
        return SimpleNamespace(concept_id=SimpleNamespace(
            value=_run_concept(artefacts, j, field)))

    return SimpleNamespace(sort_signal=sig("sort_signal"), control_axis=sig("control_axis"))


def _build_run_legs(artefacts) -> tuple[list[int], list]:
    """Reconstruct the run's legs from its leg-scoped keys. Returns the ACTUAL run
    leg indices (sorted) alongside the proxies. The indices need not be a contiguous
    0..n-1 range, so the caller MUST translate a match position back through this
    list to the real ``legs[{idx}]`` record -- ``match_legs`` returns positions into
    the proxy list, not run indices."""
    indices = sorted({
        int(m.group(1)) for k in artefacts.fields
        if (m := _RUN_LEG_RE.match(k)) is not None
    })
    return indices, [_run_leg_proxy(artefacts, j) for j in indices]


def _run_field(artefacts, key: FieldKey, gold_to_run: dict[int, int], multi: bool):
    """The run record for one gold field. Spec-level and single-leg fields use the
    BARE name (byte-identical to the historical lookup); a multi-leg leg field uses
    the matched run leg's ``legs[{run_j}].{name}`` record (or None if unmatched)."""
    if key.leg_index is None or not multi:
        return artefacts.fields.get(key.name)
    run_j = gold_to_run.get(key.leg_index)
    if run_j is None:
        return None
    return artefacts.fields.get(f"legs[{run_j}].{key.name}")


def pair_fields(spec, artefacts) -> tuple[list[PairedField], list[str]]:
    """Join a gold spec to a run's fields.

    Returns (paired, excluded_paths). The universe is fixed by the GOLD, never by
    the run (contract §8): a field the run never emitted still appears, with
    ``run=None``, so it lands in the coverage denominator rather than vanishing.

    Single-leg specs use the bare-name run lookup (unchanged). A multi-leg spec
    (CRF) resolves the gold->run leg permutation with ``match_legs`` (order-invariant,
    keyed on sort_signal then control_axis) and looks each leg field up under the
    matched run leg's ``legs[{run_j}].{field}`` record -- so two gold legs never
    collapse onto one trace record. ``FieldKey`` carries ``leg_index``, so each
    leg's fields stay distinct in ``seen``."""
    multi = len(spec.part2.legs) > 1
    gold_to_run: dict[int, int] = {}
    if multi:
        run_indices, run_legs = _build_run_legs(artefacts)
        # match_legs returns (gold_i, POSITION) into run_legs; translate the position
        # back to the ACTUAL run leg index so _run_field reads the right legs[{idx}]
        # record even when the run's leg indices are non-contiguous (fail-safe, not
        # an assumed invariant).
        gold_to_run = {gi: run_indices[pos]
                       for gi, pos in match_legs(spec.part2.legs, run_legs)}

    paired: list[PairedField] = []
    excluded: list[str] = []
    seen: set[FieldKey] = set()

    for path, gold in iter_gold_fields(spec):
        key = trace_key(path)
        if key is None:
            excluded.append(path)
            continue
        if key in seen:
            raise PairingError(
                f"duplicate trace key {key} from gold path {path!r} -- two gold fields "
                "map onto one trace record, so one would be scored twice"
            )
        seen.add(key)
        paired.append(
            PairedField(
                key=key,
                dotted_path=path,
                gold=gold,
                run=_run_field(artefacts, key, gold_to_run, multi),
                excluded_rubric=key.name in RUBRIC_FIELDS,
            )
        )
    return paired, excluded


def match_legs(gold_legs, run_legs) -> list[tuple[int, int]]:
    """Order-invariant leg matching (D34).

    Identity for n<=1, which is every anchor today. The permutation search lands
    with it because retrofitting order-invariance into a scorer whose row identity
    assumes positional legs is far more expensive than carrying it from the start.
    Ties break to the lexicographically-first permutation -- deterministic."""
    if len(gold_legs) <= 1 or len(run_legs) <= 1:
        return [(i, i) for i in range(min(len(gold_legs), len(run_legs)))]

    from itertools import permutations

    def _concept(leg, attr) -> object:
        sig = getattr(leg, attr, None)
        cid = getattr(sig, "concept_id", None)
        return getattr(cid, "value", None)

    def score(g, r) -> int:
        gs, rs = _concept(g, "sort_signal"), _concept(r, "sort_signal")
        gc, rc = _concept(g, "control_axis"), _concept(r, "control_axis")
        # The sort signal is the leg's primary identity (the crux field), so an
        # agreeing signal outweighs the scalar fields. But when legs SHARE a sort
        # signal -- CRF's three legs all sort on credit_rating -- the sort-signal
        # term is constant across every permutation, so the DISTINGUISHING axis is
        # the control_axis. Add it as a lower-weighted tiebreak; without it, all
        # permutations tie and the arbitrary identity pairing can pair gold's VaR
        # leg to a run's ILLIQ leg.
        s = 100 if gs is not None and gs == rs else 0
        s += 10 if gc is not None and gc == rc else 0
        return s

    best, best_score = None, -1
    for perm in permutations(range(len(run_legs))):
        total = sum(score(gold_legs[i], run_legs[perm[i]])
                    for i in range(min(len(gold_legs), len(run_legs))))
        if total > best_score:
            best, best_score = perm, total
    return [(i, best[i]) for i in range(min(len(gold_legs), len(run_legs)))]
