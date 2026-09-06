"""
Estimation-field value comparison (schema v1.2) -- the fitted-model analogue of
``compare_policy.compare_field``, dispatching on ``ESTIMATION_FIELD_TYPES``.

Kept SEPARATE from ``compare_policy`` (whose ``compare_field`` dispatches through
``form_filler.normalise``, which has no estimation-field entries) so the sort
comparator is untouched. The ``CompareOutcome`` / ``Comparability`` result types
ARE reused from ``compare_policy`` -- a defect here cannot change the sort path,
but the outcome vocabulary stays single-sourced.

Dispatch:
  * ``enum``    -> exact match on the trimmed token.
  * ``int``     -> exact int match.
  * ``int_set`` -> set equality (order-invariant K grid).
  * ``prose``   -> ``NO_POLICY`` (declared-weaker rubric, like method_summary).
"""

from __future__ import annotations

from agents.librarian.schema.estimation_fields import ESTIMATION_FIELD_TYPES

from evaluation.harness.compare_policy import Comparability, CompareOutcome


def _as_int(v: object) -> tuple[int | None, str | None]:
    try:
        return int(v), None
    except (TypeError, ValueError) as exc:
        return None, f"not an int: {v!r} ({exc})"


def _as_int_set(v: object) -> tuple[frozenset | None, str | None]:
    try:
        return frozenset(int(x) for x in v), None
    except (TypeError, ValueError) as exc:
        return None, f"not a set of ints: {v!r} ({exc})"


def compare_estimation_value(name: str, gold_value: object, run_value: object,
                             *, rubric_judgement: bool | None = None) -> CompareOutcome:
    """Compare one estimation field's gold value to a run value, dispatching on the
    field's type. Both values are the shipped (``Inherited.value``) tokens; None
    handling (gold-silent / run-abstained) is the scoring layer's job, not here.

    ``rubric_judgement`` (kpp_prose_rubric.md §2.2):
    a rubric adjudication of a SHIPPED prose answer -- when supplied, the prose
    branch returns ``COMPARABLE`` + that bool (the two judged rows of the §2.2
    table). ``None`` (the default, and the only legal value for non-prose kinds)
    keeps the ``NO_POLICY`` channel: abstained / never-asked / not-yet-adjudicated
    rows are coverage information, NEVER read as "not equal" (the CompareOutcome
    docstring's guarantee; D37 conv. 3)."""
    ftype = ESTIMATION_FIELD_TYPES.get(name)
    if ftype is None:
        return CompareOutcome(None, Comparability.NO_POLICY, note=f"{name!r} is not an estimation field")

    if ftype == "prose":
        if rubric_judgement is not None:
            return CompareOutcome(
                bool(rubric_judgement), Comparability.COMPARABLE,
                note="rubric adjudication (kpp_prose_rubric.md; "
                     "declared-weaker class, never headline)",
            )
        return CompareOutcome(
            None, Comparability.NO_POLICY,
            note="prose estimation field: declared-weaker rubric (not adjudicated)",
        )
    if rubric_judgement is not None:
        raise ValueError(
            f"rubric_judgement supplied for non-prose field {name!r} -- the rubric "
            "adjudicates prose only (kpp_prose_rubric.md §0)")

    if ftype == "enum":
        g = str(gold_value).strip()
        r = str(run_value).strip()
        return CompareOutcome(g == r, Comparability.COMPARABLE, g, r)

    if ftype == "int":
        g, g_err = _as_int(gold_value)
        if g_err:
            return CompareOutcome(None, Comparability.GOLD_UNNORMALISABLE, note=g_err)
        r, r_err = _as_int(run_value)
        if r_err:
            return CompareOutcome(None, Comparability.RUN_UNNORMALISABLE, gold_normalised=g, note=r_err)
        return CompareOutcome(g == r, Comparability.COMPARABLE, g, r)

    if ftype == "int_set":
        g, g_err = _as_int_set(gold_value)
        if g_err:
            return CompareOutcome(None, Comparability.GOLD_UNNORMALISABLE, note=g_err)
        r, r_err = _as_int_set(run_value)
        if r_err:
            return CompareOutcome(None, Comparability.RUN_UNNORMALISABLE, gold_normalised=g, note=r_err)
        return CompareOutcome(g == r, Comparability.COMPARABLE, g, r)

    return CompareOutcome(None, Comparability.NO_POLICY, note=f"unknown estimation field type {ftype!r}")


def compare_source_class(gold_value: object, run_value: object) -> CompareOutcome:
    """Per-instrument source_class exact match (bond|equity|accounting|macro)."""
    g = str(gold_value).strip()
    r = str(run_value).strip()
    return CompareOutcome(g == r, Comparability.COMPARABLE, g, r)
