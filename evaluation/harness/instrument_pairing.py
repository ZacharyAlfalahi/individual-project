"""
Instrument set-matching (schema v1.2) -- the fitted-model analogue of
``field_pairing.match_legs``.

KPP's instrument set is an unordered list of ~29 characteristics; a run may emit
them in any order and may over- or under-claim. ``match_instruments`` is an
order-invariant set match keyed on ``concept_id.value`` (D22 exact + binary; a
concept either matches an id or it does not). Unmatched golds -> misses (the
coverage denominator); unmatched runs -> false positives (the over-claim
numerator). This feeds the same 2x2 the sort scorer uses, on the instrument axis.

Kept as a SEPARATE module from ``field_pairing`` (which raises on any >1-leg spec)
so the sort pairing path is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InstrumentMatch:
    """One matched (gold, run) instrument pair, keyed on a shared concept_id."""

    concept_id: str
    gold: object   # InstrumentRef
    run: object    # InstrumentRef


@dataclass(frozen=True)
class InstrumentPairing:
    matches: tuple[InstrumentMatch, ...]
    missed_gold_ids: tuple[str, ...]         # gold concept_ids the run did not emit
    false_positive_run_ids: tuple[str, ...]  # run concept_ids not in the gold


def _by_concept(instruments) -> dict[str, list]:
    out: dict[str, list] = {}
    for ins in instruments:
        cid = ins.concept_id.value
        out.setdefault(cid, []).append(ins)
    return out


def match_instruments(gold_instruments, run_instruments) -> InstrumentPairing:
    """Order-invariant set match keyed on ``concept_id.value``.

    Duplicate ids on either side (not expected -- ids are unique in a well-formed
    set) are paired positionally up to the smaller count; the surplus falls to
    misses / false positives so nothing is silently dropped."""
    gold_by = _by_concept(gold_instruments)
    run_by = _by_concept(run_instruments)

    matches: list[InstrumentMatch] = []
    missed: list[str] = []
    false_pos: list[str] = []

    for cid in sorted(set(gold_by) | set(run_by)):
        gs = gold_by.get(cid, [])
        rs = run_by.get(cid, [])
        n = min(len(gs), len(rs))
        for i in range(n):
            matches.append(InstrumentMatch(concept_id=cid, gold=gs[i], run=rs[i]))
        missed.extend(cid for _ in gs[n:])
        false_pos.extend(cid for _ in rs[n:])

    return InstrumentPairing(
        matches=tuple(matches),
        missed_gold_ids=tuple(missed),
        false_positive_run_ids=tuple(false_pos),
    )
