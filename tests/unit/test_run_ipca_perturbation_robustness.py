"""The §6.3 perturbation-robustness driver: the artefact keeps the honest naming attached.

§6.3 is not a false-positive rate — mean-zero return noise does not give a mean-zero bracket
through a nonlinear fitted model.
The whole value of the artefact is that a reader cannot pick the centre up without also picking up
that a non-null centre is EXPECTED, and cannot read the zero-coverage fraction without learning that
over-coverage is expected too. These tests pin that, plus the substrate disclosure and determinism,
at a scale the unit suite can afford (registered is 200 draws and R=50).
"""
from __future__ import annotations

import json
import warnings

from scripts.run_ipca_perturbation_robustness import main

TINY = ("--n-draws", "4", "--r-datasets", "2")


def _run(tmp_path, seed="7"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert main(["--seed", seed, *TINY, "--out", str(tmp_path)]) == 0
    return json.loads((tmp_path / "perturbation_robustness.json").read_text())


def test_non_null_centre_is_never_labelled_a_false_positive_rate(tmp_path):
    out = _run(tmp_path)
    pert = out["perturbation_robustness"]
    assert pert["is_fpr"] is False
    assert "never a false positive" in pert["interpretation"]
    assert "EXPECTED" in out["reading_notes"]["centre"]
    # the centre is reported whatever it is; the point is the label, not the value
    assert isinstance(pert["i_mean"], float)


def test_zero_coverage_travels_with_its_over_coverage_note(tmp_path):
    out = _run(tmp_path)
    zc = out["bootstrap_zero_coverage"]
    assert zc["n_usable"] <= zc["r_datasets"] == 2
    assert 0.0 <= zc["coverage_fraction"] <= 1.0
    assert "over-coverage" in out["reading_notes"]["zero_coverage"]
    assert out["bootstrap"]["n_replicates"] > 0     # the shipped §5.3 interval, not a stand-in


def test_the_substrate_choice_is_disclosed_not_silent(tmp_path):
    out = _run(tmp_path)
    choice = out["substrate_is_a_driver_choice"]
    assert choice["registered"] == "n_draws and noise_sd only"
    assert "matched-twin" in choice["chosen"] and choice["why"]
    assert out["at_registered_scale"] is False
    assert out["draws_retained"] is False and "deterministic" in out["reproduction"]


def test_run_is_deterministic_given_the_seed_and_moves_with_it(tmp_path):
    a = _run(tmp_path / "a")
    b = _run(tmp_path / "b")
    assert a["perturbation_robustness"] == b["perturbation_robustness"]
    assert a["bootstrap_zero_coverage"] == b["bootstrap_zero_coverage"]
    c = _run(tmp_path / "c", seed="8")
    assert c["perturbation_robustness"]["i_obs"] != a["perturbation_robustness"]["i_obs"]
