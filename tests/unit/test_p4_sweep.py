"""
Tests for scripts/run_p4_grid_sweep.py — the P4 optimisation-overfitting sweep
(WS-D). SYNTHETIC ONLY: no test loads a panel or touches data/ (the only file
reads are the committed gate YAML, thresholds, and tmp_path fixtures); the
gold cross-check is disabled (gold_check=False) everywhere except the dry-run
smoke test, which parses the committed gold Markdown (not data).
"""

import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_p4_grid_sweep as p4  # noqa: E402

GATE_FILE = REPO_ROOT / "docs" / "p4_execution_gate.yaml"

# A synthetic thresholds dict mirroring docs/thresholds.yaml's p4_grids + signals
# centres — resolve_grids(gold_check=False) must be drivable from this alone.
SYNTH_THRESHOLDS = {
    "p4_grids": {
        "execution_gate_file": "docs/p4_execution_gate.yaml",
        "max_cells_per_module": 200,
        "mom6": {
            "formation_months": {"offsets": [-3, 0, 3, 6]},
            "total_signal_gap_months": {"levels": [1, 2]},
            "min_obs_rule": "equals_formation",
        },
        "str": {"reversal_window_months": {"levels": [1, 2, 3]}},
        "drf": {
            "window": {"offsets": [-12, -6, 0, 6, 12]},
            "min_obs": {"offsets": [-6, 0, 6]},
            "rank_rule": "ceil_5pct_of_window",
        },
    },
    "signals": {
        "mom6": {"formation_months": 6, "min_obs": 6},
        "var_5pct": {"window": 36, "min_obs": 24, "rank": 2, "multiplier": -1.0},
    },
}


# ---------------------------------------------------------------------------
# 1. Gate: the committed gate file refuses by default.
# ---------------------------------------------------------------------------

def test_gate_refuses_by_default():
    gate = p4.load_gate(GATE_FILE)
    refusal = p4.check_gate(gate, REPO_ROOT)
    assert refusal is not None
    assert "P4_GATE_REFUSED" in refusal
    # The committed gate has panel_rebuild_complete: false — the first typed check.
    assert "panel_rebuild_incomplete" in refusal


# ---------------------------------------------------------------------------
# 2. Gate: hashes are pinned — null / wrong sha refuse; correct sha allows.
# ---------------------------------------------------------------------------

def test_gate_requires_hash_match(tmp_path):
    names = (
        "monthly_panel_total_return_report.json",
        "monthly_panel_maximal_report.json",
    )
    dev = tmp_path / "data" / "development"
    dev.mkdir(parents=True)
    contents = {}
    for i, name in enumerate(names):
        payload = json.dumps({"synthetic": True, "i": i}).encode()
        (dev / name).write_bytes(payload)
        contents[name] = hashlib.sha256(payload).hexdigest()

    # Flags true but a null sha -> refusal NAMING the null entry.
    gate_null = {
        "panel_rebuild_complete": True,
        "scope_approved": True,
        "panel_report_sha256": {names[0]: None, names[1]: contents[names[1]]},
    }
    refusal = p4.check_gate(gate_null, tmp_path)
    assert refusal is not None
    assert "null_report_hash" in refusal
    assert names[0] in refusal

    # Flags true but a wrong sha -> refusal naming the mismatch.
    gate_wrong = {
        "panel_rebuild_complete": True,
        "scope_approved": True,
        "panel_report_sha256": {names[0]: "0" * 64, names[1]: contents[names[1]]},
    }
    refusal = p4.check_gate(gate_wrong, tmp_path)
    assert refusal is not None
    assert "report_hash_mismatch" in refusal
    assert names[0] in refusal

    # Flags true + every recorded sha matches the file bytes -> allowed (None).
    gate_ok = {
        "panel_rebuild_complete": True,
        "scope_approved": True,
        "panel_report_sha256": dict(contents),
    }
    assert p4.check_gate(gate_ok, tmp_path) is None

    # Flags true but the hash map is empty -> refusal (a vacuous gate is a hole).
    gate_empty = {
        "panel_rebuild_complete": True,
        "scope_approved": True,
        "panel_report_sha256": {},
    }
    refusal = p4.check_gate(gate_empty, tmp_path)
    assert refusal is not None
    assert "no_report_hashes" in refusal


# ---------------------------------------------------------------------------
# 3. Holdout guard: COMPONENT-EQUALITY semantics (mirrors run_anchor_descriptive
#    exactly: `"holdout" in p.parts`). A path with "holdout" as a component
#    raises; a filename that merely CONTAINS the substring ("holdout_notes.txt"
#    is one component, not equal to "holdout") does NOT raise — pinned here.
# ---------------------------------------------------------------------------

def test_holdout_guard_raises():
    with pytest.raises(RuntimeError, match="holdout"):
        p4._guard_no_holdout([Path("data/holdout/monthly_panel.parquet")])
    with pytest.raises(RuntimeError, match="holdout"):
        p4._guard_no_holdout(
            [Path("data/development/x.parquet"), Path("/abs/data/holdout/y.parquet")]
        )
    # Substring-in-a-component is NOT a holdout component: no raise.
    p4._guard_no_holdout(
        [Path("docs/holdout_notes.txt"), Path("data/development/panel.parquet")]
    )
    p4._guard_no_holdout([])


# ---------------------------------------------------------------------------
# 4. Grid resolution from the synthetic thresholds dict (gold_check=False).
#    NOTE (pinned deviation from the build brief): the drf cross product is
#    5 windows x 3 min_obs = 15 COMBOS, of which (window=24, min_obs=30)
#    violates min_obs <= window and is FILTERED (recorded) -> 14 cells, and the
#    grand total is 8 + 3 + 14 = 25 (the brief's "15 / 26" is the pre-filter
#    count). compute_var_5pct raises on min_obs > window, so the filter is the
#    only faithful resolution.
# ---------------------------------------------------------------------------

def test_grid_resolution():
    grids, meta = p4.resolve_grids(SYNTH_THRESHOLDS, gold_check=False)

    # mom6: {3,6,9,12} x gaps {1,2}, min_obs == formation -> 8 cells.
    mom6 = grids["mom6"]
    assert len(mom6) == 8
    assert {c["formation_months"] for c in mom6} == {3, 6, 9, 12}
    assert {c["total_signal_gap_months"] for c in mom6} == {1, 2}
    assert all(c["min_obs"] == c["formation_months"] for c in mom6)

    # str: 3 cells.
    assert grids["str"] == [
        {"reversal_window_months": 1},
        {"reversal_window_months": 2},
        {"reversal_window_months": 3},
    ]

    # drf: 14 cells (15 combos minus the filtered (24, 30)); rank rule pinned.
    drf = grids["drf"]
    assert len(drf) == 14
    assert all(c["min_obs"] <= c["window"] for c in drf)
    rank_by_window = {c["window"]: c["rank"] for c in drf}
    assert rank_by_window[36] == 2
    assert rank_by_window[48] == 3
    assert rank_by_window[24] == 2 and rank_by_window[30] == 2 and rank_by_window[42] == 3
    assert meta["filtered"]["drf"] == [{"window": 24, "min_obs": 30}]

    # Totals + published centres are members of their own grids.
    assert sum(len(c) for c in grids.values()) == 25
    assert meta["trial_counts"] == {"mom6": 8, "str": 3, "drf": 14}
    for module, pub in meta["published"].items():
        assert pub in grids[module]
    assert meta["published"]["mom6"] == {
        "formation_months": 6, "total_signal_gap_months": 1, "min_obs": 6,
    }
    assert meta["published"]["drf"] == {"window": 36, "min_obs": 24, "rank": 2}
    assert meta["gold_check"] == {"status": "skipped"}

    # max_cells_per_module is enforced.
    tiny = copy.deepcopy(SYNTH_THRESHOLDS)
    tiny["p4_grids"]["max_cells_per_module"] = 5
    with pytest.raises(p4.P4GridError, match="max_cells_per_module"):
        p4.resolve_grids(tiny, gold_check=False)

    # D0 floor: a gap level of 0 (lib_gap reintroduction) is refused.
    bad_gap = copy.deepcopy(SYNTH_THRESHOLDS)
    bad_gap["p4_grids"]["mom6"]["total_signal_gap_months"]["levels"] = [0, 1]
    with pytest.raises(p4.P4GridError, match="D0"):
        p4.resolve_grids(bad_gap, gold_check=False)


# ---------------------------------------------------------------------------
# 5. Cache key: stable on identical inputs, sensitive to every identity part.
# ---------------------------------------------------------------------------

def test_cell_cache_key_stable_and_param_sensitive():
    params = {"formation_months": 6, "total_signal_gap_months": 1, "min_obs": 6}
    key = p4.cell_cache_key("mom6", params, "panelsha", "cornerhash")
    assert key == p4.cell_cache_key("mom6", dict(params), "panelsha", "cornerhash")
    assert len(key) == 64 and all(ch in "0123456789abcdef" for ch in key)

    # Key order inside params must not matter (canonical sorted-keys JSON).
    reordered = {"min_obs": 6, "formation_months": 6, "total_signal_gap_months": 1}
    assert key == p4.cell_cache_key("mom6", reordered, "panelsha", "cornerhash")

    changed = dict(params, total_signal_gap_months=2)
    assert key != p4.cell_cache_key("mom6", changed, "panelsha", "cornerhash")
    assert key != p4.cell_cache_key("drf", params, "panelsha", "cornerhash")
    assert key != p4.cell_cache_key("mom6", params, "otherpanel", "cornerhash")
    assert key != p4.cell_cache_key("mom6", params, "panelsha", "othercorner")
    assert key != p4.cell_cache_key(
        "mom6", params, "panelsha", "cornerhash", pipeline_version="p4-v2"
    )


# ---------------------------------------------------------------------------
# 6. deflate_module on synthetic cells (seeded rng; one dominant cell).
# ---------------------------------------------------------------------------

_DATES = [d.date().isoformat() for d in pd.date_range("2002-01-31", periods=120, freq="ME")]


def _synth_cell(params: dict, mean: float, sd: float, seed: int) -> dict:
    """A synthetic cell record with the summary convention deflate_module reads:
    sharpe = realised mean / sd(ddof=1) * sqrt(12) (the summarize_returns
    functional), average = realised monthly mean, n_months = T."""
    rng = np.random.default_rng(seed)
    x = rng.normal(mean, sd, len(_DATES))
    mu = float(x.mean())
    sigma = float(x.std(ddof=1))
    return {
        "params": dict(params),
        "summary": {
            "sharpe": mu / sigma * math.sqrt(12.0),
            "average": mu,
            "n_months": len(_DATES),
        },
        "monthly": [(d, float(v)) for d, v in zip(_DATES, x)],
    }


def test_deflate_module_synthetic():
    # mom6-like grid: {3,6,9,12} x {1,2} = 8 cells; (9, 2) dominant positive.
    dominant = {"formation_months": 9, "total_signal_gap_months": 2, "min_obs": 9}
    published = {"formation_months": 6, "total_signal_gap_months": 1, "min_obs": 6}
    cells = []
    seed = 0
    for f in (3, 6, 9, 12):
        for g in (1, 2):
            params = {"formation_months": f, "total_signal_gap_months": g, "min_obs": f}
            if params == dominant:
                cells.append(_synth_cell(params, 0.020, 0.020, seed))
            else:
                cells.append(_synth_cell(params, 0.000, 0.030, seed))
            seed += 1

    out = p4.deflate_module("mom6", cells, published)

    assert out["n_trials"] == len(cells) == 8
    assert out["pbo"]["n_candidates"] == 8
    assert out["direction"] == 1
    assert out["grid_best"]["params"] == dominant
    assert out["published"]["params"] == published

    by_params = {p4._cell_id(c["params"]): c["summary"] for c in cells}
    best_sr = by_params[p4._cell_id(dominant)]["sharpe"]
    pub_sr = by_params[p4._cell_id(published)]["sharpe"]
    best_mean = by_params[p4._cell_id(dominant)]["average"]
    pub_mean = by_params[p4._cell_id(published)]["average"]
    # uplift = grid-best - published, computed on the recorded summaries.
    assert out["uplift"]["sharpe"] == pytest.approx(best_sr - pub_sr)
    assert out["uplift"]["mean"] == pytest.approx(best_mean - pub_mean)
    assert out["uplift"]["sharpe_in_claimed_direction"] == pytest.approx(best_sr - pub_sr)

    assert isinstance(out["dsr_grid_best"], float) and math.isfinite(out["dsr_grid_best"])
    assert isinstance(out["dsr_grid_best_oriented"], float)
    assert math.isfinite(out["dsr_grid_best_oriented"])
    assert isinstance(out["published"]["psr_n_trials_1"], float)
    assert 0.0 <= out["pbo"]["pbo"] <= 1.0
    assert out["pbo"]["months_in_matrix"] == 120
    assert out["pbo"]["months_dropped_inner_join"] == 0
    # purge from grid maxima: max formation 12 + max gap 2 + holding 6 = 20;
    # embargo = max(1, holding 6) = 6.
    assert out["pbo"]["purge"] == 20
    assert out["pbo"]["embargo"] == 6

    # A published-params dict not present in the grid fails loud.
    with pytest.raises(ValueError, match="published cell"):
        p4.deflate_module(
            "mom6", cells,
            {"formation_months": 7, "total_signal_gap_months": 1, "min_obs": 7},
        )

    # str-like case, direction -1: the grid-best is the most NEGATIVE Sharpe.
    str_cells = [
        _synth_cell({"reversal_window_months": 1}, 0.000, 0.030, 100),
        _synth_cell({"reversal_window_months": 2}, -0.020, 0.020, 101),  # dominant negative
        _synth_cell({"reversal_window_months": 3}, 0.000, 0.030, 102),
    ]
    out_str = p4.deflate_module("str", str_cells, {"reversal_window_months": 1})
    assert out_str["direction"] == -1
    assert out_str["n_trials"] == 3
    assert out_str["grid_best"]["params"] == {"reversal_window_months": 2}
    assert out_str["grid_best"]["sharpe"] == pytest.approx(
        min(c["summary"]["sharpe"] for c in str_cells)
    )
    # Oriented uplift is non-negative by construction of the argmax.
    assert out_str["uplift"]["sharpe_in_claimed_direction"] >= 0.0
    assert isinstance(out_str["dsr_grid_best_oriented"], float)
    assert math.isfinite(out_str["dsr_grid_best_oriented"])
    assert 0.0 <= out_str["pbo"]["pbo"] <= 1.0
    # purge: max k 3 + 1 = 4; embargo max(1, holding 1) = 1.
    assert out_str["pbo"]["purge"] == 4
    assert out_str["pbo"]["embargo"] == 1


# ---------------------------------------------------------------------------
# 7. --dry-run exits 0 and never opens anything under data/ (pd.read_parquet is
#    the only panel entry point; it is monkeypatched to raise if called).
# ---------------------------------------------------------------------------

def test_dry_run_touches_no_data(monkeypatch, capsys):
    def _boom(*args, **kwargs):  # pragma: no cover - failure path
        raise AssertionError(f"pd.read_parquet called during --dry-run: {args}")

    monkeypatch.setattr(pd, "read_parquet", _boom)

    rc = p4.main(["--dry-run"])
    assert rc == 0

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["mode"] == "dry-run"
    assert payload["status"] == "EXPLORATORY"
    assert payload["d1_disclosure"] == p4.D1_DISCLOSURE
    # Real thresholds resolve to the pinned counts (drf = 14 post-filter).
    assert payload["trial_counts"] == {"mom6": 8, "str": 3, "drf": 14, "total": 25}
    # The gate is displayed but NOT enforced in dry-run (enforcing would hash
    # files under data/development/).
    assert payload["gate"]["enforced_in_dry_run"] is False
    # Cache-key preview exists per module and uses the placeholder panel sha
    # (64-hex keys — no panel was read to produce them).
    for module in ("mom6", "str", "drf"):
        key = payload["cache_key_preview_first_cell"][module]
        assert len(key) == 64
