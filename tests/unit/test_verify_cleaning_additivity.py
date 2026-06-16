"""
Unit tests for scripts/verify_cleaning_additivity.py — the cross-stage
row-count reconciliation over the corrected TRACE cleaning chain.

The checking logic is a pure function over the three stage-report dicts, so
these tests exercise it directly with synthetic reports (no pipeline run).
"""
import copy

from verify_cleaning_additivity import check_additivity


def _reports():
    """A mutually consistent set of (cleaning, decimal_shift, bounce_back)
    stage reports for the development / corrected chain."""
    cleaning = {
        "thresholds_sha256": "abc",
        "rows": {
            "raw_total": 1000,
            "dropped_trc_st": 100,
            "dropped_cancelled_original": 50,
            "dropped_asof_cd": 80,
            "dropped_wis_fl": 10,
            "dropped_interdealer_duplicate": 60,
            "dropped_invalid_date": 0,
            "development_rows": 500,        # → decimal_shift input
            "holdout_rows": 200,
            "final_clean_total": 700,       # == dev + holdout
        },
    }
    decimal_shift = {
        "thresholds_sha256": "abc",
        "rows_dev": {
            "input_rows": 500,              # == preprocess development_rows
            "dropped_pre_ceiling": 5,
            "dropped_floor": 15,
            "kept_no_shift": 400,
            "kept_shift_div10": 50,
            "kept_shift_div100": 30,
            "output_rows": 480,             # == 400+50+30 == 500-5-15 → bounce input
        },
    }
    bounce_back = {
        "thresholds_sha256": "abc",
        "rows_dev": {
            "input_rows": 480,              # == decimal_shift output_rows
            "kept_rows": 470,
            "dropped_bounce_back": 10,
        },
    }
    return cleaning, decimal_shift, bounce_back


def test_consistent_chain_passes():
    res = check_additivity(*_reports())
    assert res["ok"] is True
    assert res["failures"] == []
    assert res["warnings"] == []


def test_broken_handoff_fails():
    # decimal-shift ingested one fewer row than preprocess emitted (silent loss
    # at the parquet handoff). Keep decimal-shift's OWN internal sum consistent
    # so that ONLY the handoff check trips.
    c, d, b = _reports()
    d = copy.deepcopy(d)
    d["rows_dev"]["input_rows"] = 499
    d["rows_dev"]["dropped_floor"] = 14    # 499 == 480 + 5 + 14 (internal holds)
    res = check_additivity(c, d, b)
    assert res["ok"] is False
    names = [f["name"] for f in res["failures"]]
    assert "handoff: preprocess.development_rows == decimal_shift.input_rows" in names


def test_internal_sum_break_fails():
    c, d, b = _reports()
    b = copy.deepcopy(b)
    b["rows_dev"]["dropped_bounce_back"] = 11   # 470 + 11 != 480
    res = check_additivity(c, d, b)
    assert res["ok"] is False
    assert any(f["name"] == "bounce_back: input == kept + dropped"
               for f in res["failures"])


def test_thresholds_sha_mismatch_warns_but_arithmetic_still_passes():
    c, d, b = _reports()
    b = copy.deepcopy(b)
    b["thresholds_sha256"] = "stale-different-run"
    res = check_additivity(c, d, b)
    assert res["ok"] is True                     # arithmetic still holds
    assert any("thresholds_sha256 differs" in w for w in res["warnings"])


def test_missing_field_reports_clear_failure():
    # A stale/old-format report missing a required field must yield a clear
    # "missing field — re-run the stage" failure, not a bare KeyError.
    c, d, b = _reports()
    d = copy.deepcopy(d)
    del d["rows_dev"]["output_rows"]
    res = check_additivity(c, d, b)
    assert res["ok"] is False
    assert any("decimal_shift.rows_dev.output_rows" in f["name"]
               for f in res["failures"])


def test_null_field_reports_clear_failure():
    # A JSON null (None) value must be treated the same as a missing field.
    c, d, b = _reports()
    b = copy.deepcopy(b)
    b["rows_dev"]["kept_rows"] = None
    res = check_additivity(c, d, b)
    assert res["ok"] is False
    assert any("bounce_back.rows_dev.kept_rows" in f["name"]
               for f in res["failures"])


def test_counts_caveat_is_surfaced_as_warning():
    # A stage that declares its own counts_caveat (e.g. a stale report from an
    # older counting method) must be surfaced so a resulting failure is
    # self-explanatory rather than looking like a live logic bug.
    c, d, b = _reports()
    d = copy.deepcopy(d)
    d["counts_caveat"] = "older float-equality counting undercounts div10/div100"
    res = check_additivity(c, d, b)
    assert any("counts_caveat" in w for w in res["warnings"])
