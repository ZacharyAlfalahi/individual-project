"""The §6.2 randomisation-FPR driver: the artefact carries what the claim it backs depends on.

The library's own `FprResult.to_dict()` omits the seed and the p-values, and computes no uniformity
statistic, so a `results` artefact built straight from it could not be reproduced or re-derived and
would not support the registered claim's second clause. These tests pin exactly that contract, plus
determinism, at a scale small enough for the unit suite (the registered cell is R=50/Q=49).
"""
from __future__ import annotations

import json
import warnings

import pytest

from scripts.run_ipca_randomisation_fpr import main, uniformity

TINY = ("--r-datasets", "3", "--q-permutations", "5")


def _run(tmp_path, *extra):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert main(["--seed", "7", *TINY, "--out", str(tmp_path), *extra]) == 0
    return json.loads((tmp_path / "randomisation_fpr.json").read_text())


def test_artefact_carries_seed_p_values_and_uniformity(tmp_path):
    out = _run(tmp_path)
    inner = out["randomisation_fpr"]
    # the three things to_dict() leaves out
    assert out["seed"] == 7
    assert len(out["p_values"]) == inner["r_datasets"] == 3
    assert out["uniformity"]["gating"] is False
    assert "ks_d" in out["uniformity"] and "deciles" in out["uniformity"]
    # provenance a reader needs to place the run
    assert out["at_registered_scale"] is False
    assert out["reduced_fallback_registered"]["used"] is True
    assert out["substrate"].startswith("synthetic")
    assert inner["q_permutations"] == 5


def test_run_is_deterministic_given_the_seed(tmp_path):
    a = _run(tmp_path / "a")
    b = _run(tmp_path / "b")
    assert a["p_values"] == b["p_values"]
    assert a["randomisation_fpr"] == b["randomisation_fpr"]


def test_band_note_matches_whether_the_lower_limit_bites(tmp_path):
    out = _run(tmp_path)
    lo, _hi = out["randomisation_fpr"]["acceptance_band_counts"]
    one_sided = out["acceptance_band_is_one_sided_in_practice"]
    assert one_sided is (lo == 0)
    # the note must not assert the P(X=0) reasoning when the lower limit does bite
    assert ("only an excess of rejections can fail" in out["band_note"]) is one_sided
    assert ("two-sided" in out["band_note"]) is not one_sided


@pytest.mark.parametrize("p, expect_uniform", [
    ((0.1, 0.3, 0.5, 0.7, 0.9), True),
    ((0.01, 0.01, 0.02, 0.01, 0.02), False),
])
def test_uniformity_statistic_separates_uniform_from_clustered(p, expect_uniform):
    u = uniformity(p, q=49)
    assert (u["ks_p"] > 0.05) is expect_uniform
    assert sum(u["deciles"]) == len(p)
    assert "1/(Q+1)" in u["caveat"]      # the discreteness caveat travels with the number
