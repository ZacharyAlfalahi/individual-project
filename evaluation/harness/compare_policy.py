"""
G3 value-comparison policy (D34 correctness rules, D37 conventions).

D34 fixes what "correct" means: exact match for enums and ints, SignalRef =
concept_id + canonical parameters, method_summary by rubric (declared weaker,
excluded from headline accuracy).

**Both sides go through the pipeline's own normaliser.** ``form_filler.normalise``
IS the D9 definition of value equality in this system; reimplementing it here
would create a second, drifting definition of "the same value" -- and the scorer
would then be measuring its own comparator rather than the extractor. Reused, not
re-derived (the same argument ``canonical_yaml`` makes for reusing the engine's
own default-filler on the G2 side).

**Unnormalisable gold is a real category** (D37 ruling 2). ``normalise`` RAISES on
a value its field type cannot hold -- gold's ``hac_lags = "floor(T^0.25)"`` is a
faithful transcription of the paper into an INT field, and no int can represent
it. That is a schema limitation, not a reader error, so it is neither correct nor
wrong: it is excluded from selective accuracy's numerator AND denominator, named
and counted. The raise is caught rather than pre-empted by a regex, because a
"does this look like a formula?" test would be a second copy of ``normalise``'s
contract, free to drift from it.

**``None`` is a first-class comparand, not a missing input.** The four
gold/run None-vs-SignalRef cells are enumerated explicitly. A
``if gold is None or run is None: return NOT_COMPARABLE`` guard would silently
swallow the single most interesting result in the corpus -- a model asserting a
control axis on a strategy that has none (observed on mom6, where the run named
the sort signal itself as the control axis).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agents.librarian.errors import LibrarianSchemaError  # noqa: E402
from agents.librarian.pipeline.form_filler import normalise  # noqa: E402
from agents.librarian.schema import fields as F  # noqa: E402

SIGNAL_FIELDS: frozenset[str] = frozenset({"sort_signal", "control_axis"})


class Comparability(str, Enum):
    COMPARABLE = "comparable"
    GOLD_UNNORMALISABLE = "gold_unnormalisable"
    RUN_UNNORMALISABLE = "run_unnormalisable"
    NO_POLICY = "no_policy"


@dataclass(frozen=True)
class CompareOutcome:
    """``equal`` is ``None`` iff ``comparability != COMPARABLE`` -- a tri-state,
    so "not scorable" can never be silently read as "not equal"."""

    equal: bool | None
    comparability: Comparability
    gold_normalised: object = None
    run_normalised: object = None
    note: str = ""


_SENTINEL = object()


def _value_kind_for(name: str) -> str | None:
    """The explicit value_kind ``normalise`` needs where the field name alone is
    not enough. ``normalise`` now dispatches paper_facts on name too, so this is
    belt-and-braces for the composite."""
    if name == F.CLAIMED_HEADLINE_METRIC:
        return "paper_metric"
    return None


def _safe_normalise(name: str, value: object) -> tuple[object, str | None]:
    """``normalise``, catching the decoding-contract raise. Returns
    ``(_SENTINEL, message)`` when the value cannot be normalised."""
    try:
        return normalise(name, value, _value_kind_for(name)), None
    except LibrarianSchemaError as exc:
        return _SENTINEL, str(exc)


def _concept_of(signal) -> object:
    """The concept_id value for a signal axis, or None when there is no signal.

    Accepts either shape, because the two sides arrive differently:
    ``_iter_inherited`` walks INTO a SignalRef and yields the ``concept_id``
    field directly, so the gold side is usually a bare concept string; a synthetic
    caller may pass the whole ``SignalRef``. Treating a bare string as "no signal"
    would score every correct sort_signal as a fabrication."""
    if signal is None:
        return None
    if hasattr(signal, "concept_id"):
        concept = signal.concept_id
        return getattr(concept, "value", concept)
    return getattr(signal, "value", signal)


def _canonical_params(signal) -> tuple:
    """A SignalRef's parameters as a canonical sorted tuple (D34: concept +
    canonical parameters). Empty for every v1 concept, so this is exercised only
    by a synthetic fixture -- a real-data test of it would be vacuous. A bare
    concept value carries no parameters."""
    if signal is None or not hasattr(signal, "parameters"):
        return ()
    params = getattr(signal, "parameters", None) or {}
    out = []
    for pname in sorted(params):
        pval = params[pname]
        pval = getattr(pval, "value", pval)
        norm, err = _safe_normalise(pname, pval)
        out.append((pname, pval if err else norm))
    return tuple(out)


def compare_signal(gold_signal, run_concept) -> CompareOutcome:
    """The four None-vs-signal cells, enumerated.

    The run side is a concept_id (the trace carries the merged concept string),
    the gold side is a SignalRef or ``None``."""
    gold_concept = _concept_of(gold_signal)

    if gold_concept is None and run_concept is None:
        return CompareOutcome(True, Comparability.COMPARABLE, None, None,
                              "both report no signal on this axis")
    if gold_concept is None and run_concept is not None:
        # The paper has no control axis; the model asserted one. A fabrication,
        # and the reason this function does not short-circuit on None.
        return CompareOutcome(False, Comparability.COMPARABLE, None, run_concept,
                              "gold has no signal on this axis; the run asserted one")
    if gold_concept is not None and run_concept is None:
        return CompareOutcome(False, Comparability.COMPARABLE, gold_concept, None,
                              "gold names a signal; the run reported none")

    g_norm, g_err = _safe_normalise("concept_id", gold_concept)
    r_norm, r_err = _safe_normalise("concept_id", run_concept)
    if g_err:
        return CompareOutcome(None, Comparability.GOLD_UNNORMALISABLE, note=g_err)
    if r_err:
        return CompareOutcome(None, Comparability.RUN_UNNORMALISABLE, note=r_err)

    # D34: SignalRef = concept_id + canonical parameters. Every v1 registry
    # concept has an EMPTY parameter schema, so concept equality is the whole
    # comparison today. The run side's parameters are not reconstructable from
    # this call anyway -- fill_signal_ref emits them as SEPARATE trace records
    # under bare parameter names -- so rather than compare against a silently
    # empty tuple (which would read as "parameters matched" when nothing was
    # checked), a parameterised gold concept is declared not-scorable until the
    # run-side parameter join is built.
    gold_params = _canonical_params(gold_signal)
    if gold_params:
        return CompareOutcome(
            None, Comparability.NO_POLICY, g_norm, r_norm,
            note=(f"gold concept carries parameters {gold_params!r}; the run's parameters "
                  "live in separate trace records and are not joined yet (D34 requires "
                  "concept + canonical parameters, so concept-only would overstate a match)"),
        )
    return CompareOutcome(g_norm == r_norm, Comparability.COMPARABLE, g_norm, r_norm)


def compare_field(key, gold_value: object, run_value: object) -> CompareOutcome:
    """Compare one field's gold value to the run's shipped (normalised) value."""
    name = key.name if hasattr(key, "name") else str(key)

    if name in ("method_summary", "universe_filter"):
        return CompareOutcome(
            None, Comparability.NO_POLICY,
            note="prose: scored by rubric (D34, declared weaker); no rubric authored",
        )

    if name in SIGNAL_FIELDS:
        return compare_signal(gold_value, run_value)

    g_norm, g_err = _safe_normalise(name, gold_value)
    if g_err:
        return CompareOutcome(
            None, Comparability.GOLD_UNNORMALISABLE,
            note=f"gold value {gold_value!r} is not representable in this field's type: {g_err}",
        )
    r_norm, r_err = _safe_normalise(name, run_value)
    if r_err:
        return CompareOutcome(
            None, Comparability.RUN_UNNORMALISABLE, gold_normalised=g_norm,
            note=f"run value {run_value!r} did not normalise: {r_err}",
        )
    return CompareOutcome(g_norm == r_norm, Comparability.COMPARABLE, g_norm, r_norm)
