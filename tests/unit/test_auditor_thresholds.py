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
    load_vartheta,
    load_vartheta_grid,
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


def test_vartheta_grid_happy_path_contains_headline(tmp_path):
    path = _write(
        tmp_path,
        """
        auditor:
          practical_significance:
            vartheta: 0.001
            vartheta_sensitivity_grid: [0.0005, 0.001, 0.0015, 0.002]
        """,
    )
    grid = load_vartheta_grid(path)
    assert grid == (0.0005, 0.001, 0.0015, 0.002)
    assert load_vartheta(path) in grid


def test_vartheta_grid_without_headline_raises(tmp_path):
    path = _write(
        tmp_path,
        """
        auditor:
          practical_significance:
            vartheta: 0.001
            vartheta_sensitivity_grid: [0.0005, 0.0015, 0.002]
        """,
    )
    with pytest.raises(AuditorThresholdError, match="headline vartheta"):
        load_vartheta_grid(path)


def test_vartheta_grid_missing_raises(tmp_path):
    path = _write(
        tmp_path,
        """
        auditor:
          practical_significance:
            vartheta: 0.001
        """,
    )
    with pytest.raises(AuditorThresholdError, match="vartheta_sensitivity_grid"):
        load_vartheta_grid(path)


def test_the_real_thresholds_file_is_pre_registered():
    # The auditor: block is now pre-registered (git tag auditor-prereg).
    # The real file must load a valid support gate, and the whole AuditorConfig must
    # assemble fail-loud-free. (This flipped from the earlier "no block yet" tripwire.)
    gate = load_support_gate()
    assert gate.min_common_months > 0 and 0 < gate.min_common_fraction_of_reference <= 1.0

    from agents.auditor.checks.report import AuditorConfig
    cfg = AuditorConfig.from_thresholds()
    assert cfg.primary_metric == "average"
    assert cfg.vartheta > 0 and cfg.d_max > 0 and 0 < cfg.fdr_q < 1
    # the neighbouring-threshold sweep grid is pre-registered and brackets the headline.
    assert cfg.vartheta in cfg.vartheta_grid and len(cfg.vartheta_grid) >= 2


def test_inert_relative_tol_is_registered_and_fail_loud(tmp_path):
    """The inert tolerance decides whether a DOE coordinate is tested at all, so it is registered.

    Two halves: the shipped thresholds carry it, and a file without it raises rather than falling
    back to a module default (which would score a run against an unregistered constant).
    """
    from agents.auditor.thresholds import load_inert_relative_tol

    assert load_inert_relative_tol() > 0

    bare = tmp_path / "thresholds.yaml"
    bare.write_text("auditor:\n  primary_metric: average\n", encoding="utf-8")
    with pytest.raises(AuditorThresholdError, match="inert_relative_tol"):
        load_inert_relative_tol(bare)


def test_auditor_config_carries_the_registered_inert_tolerance():
    """`from_thresholds` threads the registered value onto the config the audited path uses."""
    from agents.auditor.checks.report import AuditorConfig
    from agents.auditor.thresholds import load_inert_relative_tol

    assert AuditorConfig.from_thresholds().inert_relative_tol == load_inert_relative_tol()


@pytest.mark.parametrize("value,label", [
    (".inf", "infinite"),
    ("-1.0e-12", "negative"),
    ("0", "zero"),
    ("true", "boolean"),
    ("'1e-12'", "string"),
])
def test_inert_relative_tol_rejects_malformed_values(tmp_path, value, label):
    """Fail-loud on shapes a bare `float(value) > 0` check would wave through.

    An INFINITE tolerance is the dangerous one: it marks every coordinate inert (p = 1, t = 0), so
    a typo would silently suppress the tests this constant exists to route, with no error anywhere.
    """
    from agents.auditor.thresholds import load_inert_relative_tol

    path = tmp_path / f"thresholds_{label}.yaml"
    path.write_text(f"auditor:\n  inert_relative_tol: {value}\n", encoding="utf-8")
    with pytest.raises(AuditorThresholdError, match="inert_relative_tol"):
        load_inert_relative_tol(path)
