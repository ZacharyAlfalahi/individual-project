"""Unit tests for the per-paper baseline-profile registry (spec v4 B1/D1)."""

import pytest

from agents.quant.library.baseline_profiles import (
    PROFILE_IDS,
    baseline_signature,
    load_profile,
    profile_provenance,
)


def test_known_profiles_load():
    assert set(PROFILE_IDS) == {"bbw_2019", "jostova_2013"}
    bbw = load_profile("bbw_2019")
    assert bbw["consumed_by"] == ["drf", "crf"]      # drf/crf share it (spec D1)
    ops = [s["op"] for s in bbw["steps"]]
    assert "price_range" in ops and "min_volume" in ops
    jos = load_profile("jostova_2013")
    assert jos["consumed_by"] == ["mom6"]
    jops = [s["op"] for s in jos["steps"]]
    assert "month_end_price" in jops
    # Jostova states NO price range (deliberate absence, spec D3).
    assert "price_range" not in jops
    assert "price_range" in jos["absent"]


def test_unknown_profile_raises():
    with pytest.raises(KeyError):
        load_profile("nonexistent_2099")


def test_all_steps_stated():
    # Both profiles are all-STATED, so the B3 assumed-default guard is dormant.
    for pid in PROFILE_IDS:
        assert profile_provenance(pid) == {"STATED"}


def test_baseline_signature_deterministic_and_distinct():
    s1 = baseline_signature("bbw_2019")
    s2 = baseline_signature("bbw_2019")
    assert s1 == s2 and len(s1) == 64
    assert baseline_signature("jostova_2013") != s1


def test_invalid_provenance_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("profiles:\n  x:\n    steps:\n      - {op: foo, provenance: MADE_UP}\n")
    with pytest.raises(ValueError, match="provenance"):
        load_profile("x", path=p)
