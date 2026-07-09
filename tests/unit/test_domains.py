"""Unit tests for the D24 closed-domain declarations + validator."""

import pytest

from agents.librarian.errors import LibrarianSchemaError
from agents.librarian.schema.fields import (
    ALREADY_FINAL_PART2,
    INT_FIELDS,
    SORT_SIGNAL,
)
from agents.librarian.validators import (
    iter_domain_cases,
    load_domains,
    validate_value,
)


@pytest.fixture(scope="module")
def domains():
    return load_domains()


# --- every Part 2 field has a domain (bar the SignalRef) --------------------

def test_every_part2_scalar_field_has_a_domain(domains):
    part2 = domains["part2"]
    # sort_signal is a SignalRef, not a scalar -- its "domain" is the signal
    # registry, so it has no scalar-domain row by design.
    expected = ALREADY_FINAL_PART2 - {SORT_SIGNAL}
    missing = expected - set(part2.keys())
    assert missing == set(), f"Part 2 fields missing a domain: {sorted(missing)}"


def test_enum_domains_include_other(domains):
    for name, dom in domains["part2"].items():
        if dom.is_enum:
            assert "other" in dom.values, f"{name} enum must include 'other' (P3)"


def test_int_fields_are_numeric(domains):
    part2 = domains["part2"]
    for name in INT_FIELDS:
        assert part2[name].is_numeric, f"{name} should be a numeric domain"


# --- validate_value accepts in-domain, rejects out-of-domain ---------------

def test_validate_value_enum(domains):
    assert validate_value("bucketing_method", "equal_count", domains)
    assert validate_value("bucketing_method", "other", domains)
    assert not validate_value("bucketing_method", "nyse_breakpoint", domains)
    assert not validate_value("bucketing_method", 5, domains)  # wrong type


def test_validate_value_int_range(domains):
    assert validate_value("n_groups", 5, domains)
    assert validate_value("n_groups", 2, domains)  # min
    assert not validate_value("n_groups", 1, domains)  # below min
    assert not validate_value("n_groups", True, domains)  # bool rejected
    assert not validate_value("n_groups", 2.5, domains)  # float in an int domain


def test_validate_value_unknown_field_raises(domains):
    with pytest.raises(LibrarianSchemaError):
        validate_value("no_such_field", 1, domains)


# --- engine-side domains (behind the wall) ---------------------------------

def test_engine_weighting_ceiling(domains):
    assert validate_value("weighting", "by_size", domains, family="engine")
    assert validate_value("weighting", "equal", domains, family="engine")
    assert not validate_value("weighting", "rank", domains, family="engine")


def test_parametric_engine_range_checks_lower_bound_only(domains):
    # long_group is [0, groups) -- parametric upper bound not checkable here.
    long_group = domains["engine"]["long_group"]
    assert long_group.is_parametric
    assert long_group.contains(0)
    assert not long_group.contains(-1)
    # a large value passes the standalone check (upper bound deferred to the gate)
    assert long_group.contains(999)


# --- iter_domain_cases (the G1 seam) ---------------------------------------

def test_iter_domain_cases_round_trips(domains):
    cases = dict(iter_domain_cases(domains))
    # every scalar Part 2 field appears
    assert set(cases.keys()) == set(domains["part2"].keys())
    # every representative value is actually in its domain
    for field, reps in cases.items():
        for v in reps:
            assert validate_value(field, v, domains), f"{field}={v!r} not in-domain"


def test_iter_domain_cases_enum_yields_full_set(domains):
    cases = dict(iter_domain_cases(domains))
    assert set(cases["long_leg"]) == {"lowest_signal", "highest_signal", "other"}
