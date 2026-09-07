"""Relocator thresholds loader tests (evaluation/harness/relocate_thresholds.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from evaluation.harness.relocate_thresholds import (                        # noqa: E402
    RelocatorThresholdError,
    load_relocator_calibration_config,
    load_relocator_config,
)

_FULL = """
librarian:
  relocator_diagnostic:
    metric: sequencematcher_ratio_l1
    accept_bar: {bar}
    min_quote_chars: 30
    min_anchor_chars: 10
    bar_grid_hundredths: {{start: 50, stop: 98, step: 2}}
    recall_floor_light: 0.90
    recall_floor_moderate: 0.75
"""


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "thresholds.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_loader_round_trips_full_block(tmp_path):
    p = _write(tmp_path, _FULL.format(bar="0.84"))
    cal = load_relocator_calibration_config(p)
    assert cal.metric == "sequencematcher_ratio_l1"
    assert cal.min_quote_chars == 30 and cal.min_anchor_chars == 10
    assert cal.bar_grid[0] == 0.50 and cal.bar_grid[-1] == 0.98 and len(cal.bar_grid) == 25
    assert cal.recall_floor_light == 0.90 and cal.recall_floor_moderate == 0.75
    cfg = load_relocator_config(p)
    assert cfg.accept_bar == 0.84


def test_loader_raises_on_missing_block_and_malformed_keys(tmp_path):
    with pytest.raises(RelocatorThresholdError, match="not registered"):
        load_relocator_config(_write(tmp_path, "librarian:\n  g3: {min_cell_n: 20}\n"))
    with pytest.raises(RelocatorThresholdError, match="has not been frozen"):
        load_relocator_config(_write(tmp_path, _FULL.format(bar="null")))
    with pytest.raises(RelocatorThresholdError, match="accept_bar"):
        load_relocator_config(_write(tmp_path, _FULL.format(bar="1.5")))
    with pytest.raises(RelocatorThresholdError, match="min_quote_chars"):
        load_relocator_config(_write(
            tmp_path, _FULL.format(bar="0.8").replace("min_quote_chars: 30",
                                                      "min_quote_chars: 0")))
    with pytest.raises(RelocatorThresholdError, match="metric"):
        load_relocator_config(_write(
            tmp_path, _FULL.format(bar="0.8").replace("sequencematcher_ratio_l1", "levenshtein")))


def test_calibration_loader_ignores_null_accept_bar(tmp_path):
    p = _write(tmp_path, _FULL.format(bar="null"))
    cal = load_relocator_calibration_config(p)
    assert cal.bar_grid == tuple(b / 100 for b in range(50, 99, 2))


def test_qr2_protocol_reads_its_own_bar(tmp_path):
    # The qr1 bar stays null; the qr2 sub-key governs the qr2 protocol only.
    text = _FULL.format(bar="null") + "    qr2:\n      accept_bar: 0.90\n"
    p = _write(tmp_path, text)
    with pytest.raises(RelocatorThresholdError, match="qr1 accept_bar is null"):
        load_relocator_config(p)
    assert load_relocator_config(p, protocol="qr2").accept_bar == 0.90

    # A null qr2 bar locks the qr2 pass; an absent qr2 block names the cause.
    p2 = _write(tmp_path, _FULL.format(bar="null") + "    qr2:\n      accept_bar: null\n")
    with pytest.raises(RelocatorThresholdError, match="qr2 accept_bar is null"):
        load_relocator_config(p2, protocol="qr2")
    with pytest.raises(RelocatorThresholdError, match="qr2 block absent"):
        load_relocator_config(_write(tmp_path, _FULL.format(bar="null")), protocol="qr2")
    with pytest.raises(RelocatorThresholdError, match="unknown relocator protocol"):
        load_relocator_config(p, protocol="qr3")


def test_committed_yaml_block_parses():
    # Guards the committed block's structure through BOTH freeze stages (the
    # calibration loader accepts a null bar, so this passes at stage A and B).
    cal = load_relocator_calibration_config()
    assert cal.metric == "sequencematcher_ratio_l1"
    assert cal.min_quote_chars == 30
