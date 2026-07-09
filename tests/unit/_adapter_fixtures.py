"""
Adapter-test fixtures: thin wrappers over ``_librarian_fixtures`` with the two
overrides every adapter test needs -- ``registry_version="v1"`` (so the
concept->column handshake passes) and a GROUNDED sort-signal concept (one of the
four v1 concept->column rows: var_5pct, credit_rating, past_6m_cumulative_return,
bpw_gamma). Synthetic, never lifted from a paper.
"""

from __future__ import annotations

from _librarian_fixtures import (
    build_header,
    build_leg,
    build_part2,
    build_spec,
    signal_ref,
    stated,
)

# The four concept ids the v1 concept->column table grounds -> their columns.
GROUNDED = {
    "var_5pct": "var_5pct",
    "credit_rating": "rating",
    "past_6m_cumulative_return": "mom6",
    "bpw_gamma": "gamma",
}


def grounded_leg(concept: str = "past_6m_cumulative_return", **over):
    """A single well-formed leg whose sort_signal is a grounded concept."""
    base = dict(sort_signal=signal_ref(concept), n_groups=stated(5))
    base.update(over)
    return build_leg(**base)


def adapter_spec(legs=None, header=None, **part2_over):
    """A well-formed spec that passes the registry handshake, with grounded legs."""
    if legs is None:
        legs = (grounded_leg(),)
    return build_spec(
        header=header or build_header(registry_version="v1"),
        part2=build_part2(legs=tuple(legs), **part2_over),
    )
