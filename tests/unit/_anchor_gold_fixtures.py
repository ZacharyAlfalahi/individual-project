"""
Anchor gold specs as StrategySpec fixtures, for the ledger-check no-false-positives
test (`test_ledger_check.py::test_anchor_specs_no_false_positives`).

These are **ledger-check inputs, not full-fidelity gold reconstructions.** A real
`gold_*.md -> StrategySpec` loader is G2's job; `check_assumptions` only reads the
13 finalised ledger rows, so a faithful transcription of just those fields is
sufficient here and far safer than parsing the irregular, human-authored gold
Markdown.

Two load-bearing construction rules (see the plan / Z review):

  1. **All-UNKNOWN baseline; override only gold-STATED fields.** Every common and
     per-leg field starts as ``unknown()`` (driven off the schema's own field-name
     tuples, so a future schema field is auto-covered as UNKNOWN). We then override
     ONLY the fields the gold actually STATES. We never inherit the synthetic
     STATED engine-default values from ``build_part2`` / ``build_leg`` -- if the
     ledger table later grows a row, an anchor may fire only because a *gold* states
     an incompatible value, never because a synthetic default was STATED-at-engine.

  2. **Pointer quotes, never real paper text.** Every ``stated(...)`` uses an
     evidence pointer (``"see evaluation/gold_specs/<file>#<field>"``), not the
     paper's verbatim quote. `_librarian_fixtures.py` is explicitly not a home for
     real paper-gold quote fixtures; the checker never inspects the quote on the
     zero-mismatch path, so a pointer is enough and keeps copyrighted text out of
     the test tree.

The STATED values below are transcribed from
``evaluation/gold_specs/gold_{str,drf,mom6}_*.md`` (the 13 ledger-checked fields
only). Every unlisted field is UNKNOWN in the gold and stays UNKNOWN here.
"""

from __future__ import annotations

from agents.librarian.schema import Combiner
from agents.librarian.schema.strategy_spec import (
    _COMMON_INHERITED_FIELDS,
    _LEG_INHERITED_FIELDS,
)

from _librarian_fixtures import (
    build_leg,
    build_part2,
    build_spec,
    signal_ref,
    stated,
    unknown,
)


def _ptr(gold_file: str, field: str) -> str:
    """An evidence pointer into the gold doc -- NOT the paper's verbatim quote."""
    return f"see evaluation/gold_specs/{gold_file}#{field}"


def _all_unknown_common() -> dict:
    return {f: unknown() for f in _COMMON_INHERITED_FIELDS}


def _all_unknown_leg() -> dict:
    return {f: unknown() for f in _LEG_INHERITED_FIELDS}


def _spec(
    gold_file: str,
    *,
    sort_signal_concept: str,
    n_groups_val: int,
    leg_stated: dict,
    common_stated: dict,
    control_axis_concept: str | None = None,
    control_n_groups_val: int | None = None,
):
    """Build one anchor StrategySpec from an all-UNKNOWN baseline, overriding only
    the gold-STATED fields (+ the structurally-required sort_signal / n_groups /
    combiner, and the control axis for an independent double sort)."""

    def q(field: str) -> str:
        return _ptr(gold_file, field)

    leg_kwargs = _all_unknown_leg()
    leg_kwargs["sort_signal"] = signal_ref(sort_signal_concept, label=q("sort_signal"))
    leg_kwargs["n_groups"] = stated(n_groups_val, quote=q("n_groups"))
    if control_axis_concept is not None:
        leg_kwargs["control_axis"] = signal_ref(control_axis_concept, label=q("control_axis"))
        leg_kwargs["control_n_groups"] = stated(control_n_groups_val, quote=q("control_n_groups"))
    else:
        leg_kwargs["control_axis"] = None  # control_n_groups stays UNKNOWN from the baseline
    for field, value in leg_stated.items():
        leg_kwargs[field] = stated(value, quote=q(field))
    leg = build_leg(**leg_kwargs)

    common = _all_unknown_common()
    for field, value in common_stated.items():
        common[field] = stated(value, quote=q(field))

    part2 = build_part2(
        legs=(leg,),
        combiner=Combiner(kind=stated("single_leg", quote=q("combiner"))),
        **common,
    )
    return build_spec(part2=part2)


# --- the three anchors (STATED fields per the gold matrix; rest UNKNOWN) -----

def _str_spec():
    # gold_str_drr_2026.md: single sort, deciles, long-short; return realisation;
    # return availability = require next-month return. Everything else UNKNOWN.
    return _spec(
        "gold_str_drr_2026.md",
        sort_signal_concept="prior_1m_excess_return",
        n_groups_val=10,
        leg_stated={"sort_kind": "single"},
        common_stated={
            "return_availability_policy": "require_next_month_return",
            "return_label": "realisation",
        },
    )


def _drf_spec():
    # gold_drf_bbw_2019.md: INDEPENDENT 5x5 double sort (control = credit_rating);
    # the factor is stated independent (Table 3's conditional analysis is NOT the
    # factor). Every ledger-checked common field is UNKNOWN.
    return _spec(
        "gold_drf_bbw_2019.md",
        sort_signal_concept="var_5pct",
        n_groups_val=5,
        control_axis_concept="credit_rating",
        control_n_groups_val=5,
        leg_stated={"sort_kind": "independent"},
        common_stated={},
    )


def _mom6_spec():
    # gold_mom6_jnps_2013.md: single decile sort; overlapping 6-month cohorts,
    # equal cohort weighting; return realisation. Rest UNKNOWN.
    return _spec(
        "gold_mom6_jnps_2013.md",
        sort_signal_concept="past_6m_cumulative_return",
        n_groups_val=10,
        leg_stated={"sort_kind": "single"},
        common_stated={
            "return_label": "realisation",
            "overlap_convention": "overlapping",
            "cohort_weighting": "equal",
        },
    )


_ANCHORS = {"str": _str_spec, "drf": _drf_spec, "mom6": _mom6_spec}


def anchor_gold_spec(anchor_id: str):
    """The anchor's StrategySpec (ledger-check fields transcribed from its gold)."""
    try:
        return _ANCHORS[anchor_id]()
    except KeyError:
        raise ValueError(
            f"unknown anchor_id {anchor_id!r}; expected one of {sorted(_ANCHORS)}"
        ) from None
