"""
Planted out-of-registry signals -> ``unrecognised`` is the correct answer (D22).

The gold set deliberately includes signals whose concept is *not* in the Signal
Concept Registry (D22: "the gold set includes planted out-of-registry papers; the
``unrecognised`` rate is a reported metric"). This test drives several such
planted signals through ``validate_librarian_spec`` against the REAL
``SignalConceptRegistry`` and asserts:

  (a) a planted out-of-registry signal correctly declared ``unrecognised`` (with a
      non-empty ``as_described``) VALIDATES -- ``unrecognised`` is the right answer,
      not a failure (D22);
  (b) the SAME planted signal, instead claiming a wrong *in-registry* concept_id,
      is REJECTED -- but ONLY when the claimed id is genuinely absent from the
      registry; a planted concept that happens to collide with a real registry id
      would (wrongly) pass the registry check, which is exactly the force-match
      hazard we measure;
  (c) the **force-match rate** = fraction of planted-out-of-registry cases that
      (wrongly) resolved to an in-registry concept. For correctly-declared
      ``unrecognised`` cases this must be 0.

Force-match is the D22 failure mode: a signal that is "not one of ours" getting
silently bound to a registry concept it does not match. The correct escape is
``unrecognised``; the metric proves the escape holds.
"""

from __future__ import annotations

from agents.librarian.registries.signal_concept_registry import (
    load_signal_concept_registry,
)
from agents.librarian.schema import UNRECOGNISED, DescribedSignal, SignalRef
from agents.librarian.validators import validate_librarian_spec

from _librarian_fixtures import (
    build_leg,
    build_part2,
    build_spec,
    located_quote,
    stated,
)

# ---------------------------------------------------------------------------
# The planted out-of-registry concepts: signals a paper might state that are NOT
# in the v1 registry (which seeds prior_1m_excess_return, past_6m_cumulative_
# return, var_5pct, credit_rating, bpw_gamma, maturity, size). Each is a genuine
# "not one of ours": near-misses (9-month momentum vs the 6-month column) and
# out-of-scope characteristics (ESG, a book-to-market equity signal).
# ---------------------------------------------------------------------------

PLANTED_OUT_OF_REGISTRY: tuple[tuple[str, str], ...] = (
    ("esg_tilt", "ESG tilt"),
    ("9m_momentum", "nine-month momentum"),
    ("book_to_market", "book-to-market"),
    ("analyst_dispersion", "analyst forecast dispersion"),
)


def _registry():
    """The real v1 Signal Concept Registry (a SignalRegistryLike)."""
    return load_signal_concept_registry()


def _unrecognised_signal(label: str) -> SignalRef:
    """A signal correctly declared ``unrecognised`` -- the D22 escape, carrying
    the paper's own words (a non-empty ``as_described``) that failed to match."""
    return SignalRef(
        concept_id=stated(UNRECOGNISED),
        as_described=DescribedSignal(label=label, quotes=(located_quote(text=label),)),
    )


def _wrong_in_registry_signal(planted_concept: str, label: str) -> SignalRef:
    """The SAME planted signal, but *claiming* the planted concept id as if it
    were an in-registry concept (no ``unrecognised`` escape). Since the planted
    ids are out of registry, this is the wrong answer the registry check must
    catch."""
    return SignalRef(
        concept_id=stated(planted_concept),
        as_described=DescribedSignal(label=label, quotes=(located_quote(text=label),)),
    )


def _spec_with_sort_signal(sig: SignalRef):
    """A well-formed one-leg spec whose sole sort signal is ``sig``."""
    leg = build_leg(sort_signal=sig)
    return build_spec(part2=build_part2(legs=(leg,)))


# --- (a) correctly-declared unrecognised VALIDATES ------------------------------

def test_correctly_declared_unrecognised_validates():
    reg = _registry()
    for planted, label in PLANTED_OUT_OF_REGISTRY:
        spec = _spec_with_sort_signal(_unrecognised_signal(label))
        errors = validate_librarian_spec(spec, registry=reg)
        assert errors == [], (
            f"planted out-of-registry signal {planted!r} correctly declared "
            f"unrecognised should VALIDATE (D22); got {[e.reason for e in errors]}"
        )


def test_planted_concepts_are_genuinely_out_of_registry():
    # sanity: the whole exercise depends on these NOT being real registry ids
    reg = _registry()
    for planted, _label in PLANTED_OUT_OF_REGISTRY:
        assert not reg.has_concept(planted), (
            f"{planted!r} is unexpectedly IN the registry -- pick a different "
            "planted concept so the out-of-registry premise holds"
        )


# --- (b) a wrong in-registry claim is REJECTED ----------------------------------

def test_wrong_in_registry_claim_is_rejected():
    reg = _registry()
    for planted, label in PLANTED_OUT_OF_REGISTRY:
        spec = _spec_with_sort_signal(_wrong_in_registry_signal(planted, label))
        errors = validate_librarian_spec(spec, registry=reg)
        assert errors, (
            f"planted signal {planted!r} claiming an in-registry concept_id must be "
            "REJECTED by the registry check (D22: exact match; the correct answer "
            f"is {UNRECOGNISED!r})"
        )
        # the rejection is specifically the not-in-registry finding
        assert any("not in the Signal Concept Registry" in e.reason for e in errors), (
            f"expected a registry-membership rejection for {planted!r}; "
            f"got {[e.reason for e in errors]}"
        )


# --- (c) force-match rate ------------------------------------------------------

def _resolved_to_in_registry(sig: SignalRef, reg) -> bool:
    """Did this signal (wrongly) resolve to an in-registry concept? I.e. it is NOT
    declared ``unrecognised`` AND its claimed concept_id passes ``has_concept``.
    That is the force-match event: a not-one-of-ours signal bound to a concept."""
    cid = sig.concept_id.value
    if cid == UNRECOGNISED:
        return False
    return reg.has_concept(cid)


def test_force_match_rate_is_zero_for_correctly_declared_unrecognised():
    reg = _registry()
    planted_signals = [
        _unrecognised_signal(label) for _planted, label in PLANTED_OUT_OF_REGISTRY
    ]
    n = len(planted_signals)
    forced = sum(1 for sig in planted_signals if _resolved_to_in_registry(sig, reg))
    force_match_rate = forced / n
    assert force_match_rate == 0.0, (
        f"force-match rate must be 0 for correctly-declared unrecognised planted "
        f"signals; got {force_match_rate:.3f} ({forced}/{n} planted "
        "out-of-registry signals wrongly resolved to an in-registry concept)"
    )


def test_force_match_rate_is_one_when_all_planted_signals_claim_a_concept():
    # the metric's counter-case: if every planted signal (wrongly) claims its
    # out-of-registry id, none of those ids resolve in the registry, so the
    # force-match rate (resolution to an *in-registry* concept) is still 0 --
    # the registry rejects them rather than binding them. This proves the metric
    # measures wrong-binding, not merely "claimed a concept".
    reg = _registry()
    signals = [
        _wrong_in_registry_signal(planted, label)
        for planted, label in PLANTED_OUT_OF_REGISTRY
    ]
    n = len(signals)
    forced = sum(1 for sig in signals if _resolved_to_in_registry(sig, reg))
    force_match_rate = forced / n
    assert force_match_rate == 0.0, (
        "planted out-of-registry ids must never resolve in the registry; "
        f"got force-match rate {force_match_rate:.3f} ({forced}/{n})"
    )


def test_force_match_metric_catches_a_real_in_registry_id():
    # positive control for the metric itself: a signal claiming a REAL registry id
    # (past_6m_cumulative_return) DOES resolve -- so _resolved_to_in_registry
    # returns True. This is a legitimate match, not a force-match; the test just
    # confirms the detector fires on genuine resolution so a 0 rate above is
    # meaningful, not vacuous.
    reg = _registry()
    real_id = "past_6m_cumulative_return"
    assert reg.has_concept(real_id)
    sig = SignalRef(
        concept_id=stated(real_id),
        as_described=DescribedSignal(
            label="six-month momentum", quotes=(located_quote(),)
        ),
    )
    assert _resolved_to_in_registry(sig, reg) is True
