"""
Unit tests for the silence-policy table loader (D26 / D32c, v1).

Covers: the v1 file loads; ``verify_hash`` against the recorded canonical sha256
passes; the byte-``content_hash`` is deterministic; and a spread of policy
lookups return the right shape -- ``long_leg`` -> refuse, ``sort_kind`` ->
conditional_refuse (with resolvable branches), ``weighting_scheme`` ->
tag_and_proceed(value), plus the refuse_on_stated / flag_on_stated overrides.
"""

import pytest

from agents.librarian.errors import LibrarianSchemaError
from agents.librarian.registries import (
    POLICIES,
    FieldPolicy,
    SilencePolicyTable,
    load_silence_policy_table,
)

# The canonical sha256 of config/silence_policy_v1.yaml (version v1.1), recorded in
# docs/librarian/specs/part2_schema_and_silence_policy_v1_1.md.
_RECORDED_SHA256 = "3bffb06d9ed58c50623a21272e0728483b90ce065c764c00372ede2f812a9978"


@pytest.fixture(scope="module")
def table():
    return load_silence_policy_table()


# --- loads + version --------------------------------------------------------

def test_table_loads_with_version(table):
    assert table.version == "v1.1"
    assert isinstance(table, SilencePolicyTable)


# --- the recorded hash ------------------------------------------------------

def test_verify_hash_matches_recorded(table):
    assert table.verify_hash(_RECORDED_SHA256)
    assert table.content_hash == _RECORDED_SHA256


def test_verify_hash_rejects_wrong_hash(table):
    assert not table.verify_hash("0" * 64)


def test_content_hash_deterministic():
    a = load_silence_policy_table()
    b = load_silence_policy_table()
    assert a.content_hash == b.content_hash


# --- policy lookups return the right shape ----------------------------------

def test_long_leg_refuses(table):
    fp = table.policy_for("sort_block", "long_leg")
    assert fp.policy == "refuse"
    assert fp.reason  # a load-bearing-direction reason is recorded


def test_sort_signal_refuses(table):
    assert table.policy_for("sort_block", "sort_signal").policy == "refuse"


def test_sort_kind_conditional_refuse(table):
    fp = table.policy_for("sort_block", "sort_kind")
    assert fp.policy == "conditional_refuse"
    assert fp.is_conditional
    # no control axis -> tag_and_proceed(single)
    no_ctrl = fp.resolve("when_no_control")
    assert no_ctrl.policy == "tag_and_proceed"
    assert no_ctrl.default == "single"
    # control axis present + silent -> refuse (never silently 'independent')
    assert fp.resolve("when_control_present").policy == "refuse"


def test_sort_kind_unknown_context_raises(table):
    fp = table.policy_for("sort_block", "sort_kind")
    with pytest.raises(LibrarianSchemaError):
        fp.resolve("when_unicorn")


def test_control_n_groups_tag_and_proceed(table):
    # v1.1: the 2nd-axis group count omits on silence (factory defaults control_groups=groups).
    fp = table.policy_for("sort_block", "control_n_groups")
    assert fp.policy == "tag_and_proceed"
    assert fp.default == "n_groups"


def test_paper_facts_not_routable(table):
    # v1.1: paper_facts is an analysis-only block, NOT a routable silence field (the
    # loader's routable blocks are sort_block/common only) -- the adapter can never
    # reach it (Guard 2, structural). control_axis likewise has no routable row.
    from agents.librarian.schema.fields import PAPER_FACTS_FIELDS
    for field in PAPER_FACTS_FIELDS:
        with pytest.raises(LibrarianSchemaError):
            table.policy_for("common", field)
        with pytest.raises(LibrarianSchemaError):
            table.policy_for("sort_block", field)
    with pytest.raises(LibrarianSchemaError):
        table.policy_for("sort_block", "control_axis")


def test_weighting_scheme_tag_and_proceed(table):
    fp = table.policy_for("common", "weighting_scheme")
    assert fp.policy == "tag_and_proceed"
    assert fp.default == "value"
    # a non-conditional field resolves to itself.
    assert fp.resolve("ignored") is fp


def test_strategy_side_refuse_on_stated(table):
    # D23 refuse-on-conflict: default on silence, refuse on a STATED
    # non-representable value.
    fp = table.policy_for("common", "strategy_side")
    assert fp.policy == "tag_and_proceed"
    assert set(fp.refuse_on_stated) == {"long_only", "short_only"}


def test_transaction_cost_flag_on_stated(table):
    fp = table.policy_for("common", "transaction_cost_convention")
    assert fp.policy == "tag_and_proceed"
    assert set(fp.flag_on_stated) == {"net_of_costs"}


def test_every_policy_is_in_the_closed_vocabulary(table):
    for block, fields in table.policies.items():
        for name, fp in fields.items():
            assert fp.policy in POLICIES, (block, name, fp.policy)


# --- error paths ------------------------------------------------------------

def test_unknown_block_raises(table):
    with pytest.raises(LibrarianSchemaError):
        table.policy_for("no_such_block", "x")


def test_unknown_field_raises(table):
    with pytest.raises(LibrarianSchemaError):
        table.policy_for("common", "no_such_field")


def test_missing_file_raises(tmp_path):
    with pytest.raises(LibrarianSchemaError):
        load_silence_policy_table(tmp_path / "does_not_exist.yaml")


def test_bad_policy_value_is_build_error():
    with pytest.raises(LibrarianSchemaError):
        FieldPolicy(block="common", field="x", policy="teleport")


def test_conditional_refuse_without_branches_is_build_error():
    with pytest.raises(LibrarianSchemaError):
        FieldPolicy(block="sort_block", field="x", policy="conditional_refuse")
