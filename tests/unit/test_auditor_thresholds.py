"""Stage 1 — the fail-loud Auditor thresholds loader.

The integrity requirement (design §13.2, user decision): a missing pre-registration
constant must RAISE, never default. A silent default would launder a post-hoc
constant, the exact sin the instrument exposes. These tests pin that behaviour and
the happy path against an explicit temp thresholds file.
"""

from __future__ import annotations

import textwrap

import pytest

from agents.auditor.thresholds import (
    AuditorThresholdError,
    load_bootstrap_config,
    load_primary_metric,
    load_shapley_pct_denominator_min,
    load_support_gate,
)


def _write(tmp_path, body: str):
    p = tmp_path / "thresholds.yaml"
    p.write_text(textwrap.dedent(body))
    return p


FULL = """
    auditor:
      primary_metric: sharpe
      support_gate:
        min_common_months: 60
        min_common_fraction_of_reference: 0.5
      bootstrap:
        n_replicates: 1000
        min_effective_blocks: 10
        block_length_months: 6
      shapley:
        percentage_denominator_min: 0.05
    """


def test_happy_path_reads_all_constants(tmp_path):
    path = _write(tmp_path, FULL)
    gate = load_support_gate(path)
    assert gate.min_common_months == 60
    assert gate.min_common_fraction_of_reference == 0.5

    boot = load_bootstrap_config(path)
    assert (boot.n_replicates, boot.min_effective_blocks, boot.block_length_months) == (
        1000,
        10,
        6,
    )
    assert load_primary_metric(path) == "sharpe"
    assert load_shapley_pct_denominator_min(path) == 0.05


def test_missing_auditor_block_raises(tmp_path):
    path = _write(tmp_path, "trace_cleaning:\n  price_floor: 1.0\n")
    with pytest.raises(AuditorThresholdError, match="auditor"):
        load_support_gate(path)


def test_missing_nested_key_raises_naming_the_key(tmp_path):
    path = _write(
        tmp_path,
        """
        auditor:
          support_gate:
            min_common_months: 60
        """,
    )
    with pytest.raises(AuditorThresholdError, match="min_common_fraction_of_reference"):
        load_support_gate(path)


def test_present_but_wrong_type_raises(tmp_path):
    path = _write(
        tmp_path,
        """
        auditor:
          support_gate:
            min_common_months: "sixty"
            min_common_fraction_of_reference: 0.5
        """,
    )
    with pytest.raises(AuditorThresholdError, match="not an int"):
        load_support_gate(path)


def test_bool_is_rejected_as_int(tmp_path):
    # bool is an int subclass; a YAML `true` must not pass as a count.
    path = _write(
        tmp_path,
        """
        auditor:
          bootstrap:
            n_replicates: true
            min_effective_blocks: 10
            block_length_months: 6
        """,
    )
    with pytest.raises(AuditorThresholdError, match="not an int"):
        load_bootstrap_config(path)


def test_the_real_thresholds_file_has_no_auditor_block_yet():
    # Documents the current state: the auditor block is NOT pre-registered, so the
    # loader fails loud against the repo file. When the researcher pre-registers
    # and git-tags it, this test flips to xfail-and-update (a deliberate tripwire).
    with pytest.raises(AuditorThresholdError):
        load_support_gate()
