"""
Estimation silence-policy table (schema v1.2) -- the fitted-model silences live in
a SEPARATE file (``config/silence_policy_estimation_v1.yaml``, block
``("estimation",)``), loaded via the additive ``blocks`` parameter. The headline
isolation assertion: the sort table (``config/silence_policy_v1.yaml``) and its
frozen byte-hash are provably untouched.
"""

from pathlib import Path

from agents.librarian.registries.silence_policy import load_silence_policy_table
from agents.librarian.schema.estimation_fields import ESTIMATION_FIELDS

_REPO = Path(__file__).resolve().parents[2]
_EST_PATH = _REPO / "config" / "silence_policy_estimation_v1.yaml"

# Recorded in docs/librarian/specs/schema_v1_2_estimation_block.md.
_EST_HASH = "d96aec2f9e9486c422003ff46095e521d50ea455ffd09430f9355197254aa80b"
# The sort table's frozen byte-hash (docs/librarian/specs/part2_schema_and_silence_policy_v1_1.md).
_SORT_HASH = "3bffb06d9ed58c50623a21272e0728483b90ce065c764c00372ede2f812a9978"

# Load-bearing identity fields refuse on silence (mirrors sort_signal / long_leg).
_REFUSE_FIELDS = {
    "model_family", "estimation_algorithm", "n_factors_tested",
    "intercept_spec", "estimation_mode",
}


def _table():
    return load_silence_policy_table(_EST_PATH, blocks=("estimation",))


def test_all_eleven_estimation_fields_have_a_policy():
    tbl = _table()
    for name in ESTIMATION_FIELDS:
        pol = tbl.policy_for("estimation", name)  # raises if absent
        assert pol.policy in ("refuse", "tag_and_proceed")


def test_identity_fields_refuse_on_silence():
    tbl = _table()
    for name in ESTIMATION_FIELDS:
        expected = "refuse" if name in _REFUSE_FIELDS else "tag_and_proceed"
        assert tbl.policy_for("estimation", name).policy == expected, name


def test_estimation_byte_hash_recorded():
    assert _table().verify_hash(_EST_HASH)


def test_sort_silence_table_hash_unchanged():
    # The default load (sort table) must be byte-identical to its frozen hash --
    # adding the estimation file did not touch config/silence_policy_v1.yaml.
    assert load_silence_policy_table().verify_hash(_SORT_HASH)


def test_estimation_version_is_v12():
    assert _table().version == "v1.2"
