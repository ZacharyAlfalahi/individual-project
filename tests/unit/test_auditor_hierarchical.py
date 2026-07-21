"""Inc2-F — the hierarchical prevalence model (§8.2)."""

from __future__ import annotations

from agents.auditor.checks.hierarchical import (
    HierarchyPriors,
    StrategyEffect,
    fit_coordinate_hierarchy,
    fit_hierarchy,
)

PRIORS = HierarchyPriors(mu_scale=1.0, tau_scale=0.1)


def _effects(values, var=1e-4, no_ops=0):
    out = [StrategyEffect(f"s{i}", v, var, False) for i, v in enumerate(values)]
    for k in range(no_ops):
        out.append(StrategyEffect(f"noop{k}", 0.0, 1e-12, True))
    return out


# --------------------------------------------------------------------------
# Magnitude recovery
# --------------------------------------------------------------------------

def test_mu_recovers_common_mean():
    # All strategies near 0.05 => μ posterior near 0.05.
    eff = _effects([0.048, 0.052, 0.05, 0.049, 0.051])
    res = fit_coordinate_hierarchy(eff, vartheta=0.01, priors=PRIORS, seed=1)
    assert 0.03 < res.mu_mean < 0.07
    assert res.mu_ci_low < res.mu_mean < res.mu_ci_high


def test_wide_tau_when_effects_heterogeneous():
    homogeneous = fit_coordinate_hierarchy(
        _effects([0.05, 0.05, 0.05, 0.05, 0.05]), vartheta=0.01, priors=PRIORS, seed=2
    )
    heterogeneous = fit_coordinate_hierarchy(
        _effects([-0.1, 0.2, 0.0, 0.15, -0.05]), vartheta=0.01, priors=PRIORS, seed=2
    )
    assert heterogeneous.tau_mean > homogeneous.tau_mean


# --------------------------------------------------------------------------
# Prevalence denominator (D-A31): no-ops count in the denominator, not the numerator
# --------------------------------------------------------------------------

def test_prevalence_large_when_all_susceptible_are_material():
    # 5 strategies all with a big effect vs ϑ => prevalence near 1.
    res = fit_coordinate_hierarchy(
        _effects([0.3, 0.3, 0.3, 0.3, 0.3], var=1e-5), vartheta=0.05, priors=PRIORS, seed=3
    )
    assert res.prevalence > 0.9
    assert res.n_runnable == 5 and res.n_susceptible == 5


def test_no_ops_inflate_denominator_not_numerator():
    # 1 material strategy + 4 proven no-ops => prevalence ≈ 1/5 = 0.2 (D-A31 limit case).
    eff = _effects([0.3], var=1e-6, no_ops=4)
    res = fit_coordinate_hierarchy(eff, vartheta=0.05, priors=PRIORS, seed=4)
    assert res.n_runnable == 5 and res.n_susceptible == 1
    assert abs(res.prevalence - 0.2) < 0.08  # ~0.2, not ~1.0


def test_all_no_ops_gives_zero_prevalence():
    eff = _effects([], no_ops=3)
    res = fit_coordinate_hierarchy(eff, vartheta=0.05, priors=PRIORS, seed=5)
    assert res.prevalence == 0.0
    assert res.n_susceptible == 0


def test_prevalence_small_when_effects_below_threshold():
    # Effects all well below ϑ => low prevalence.
    res = fit_coordinate_hierarchy(
        _effects([0.001, 0.002, 0.0, 0.001, -0.001], var=1e-8),
        vartheta=0.05, priors=PRIORS, seed=6,
    )
    assert res.prevalence < 0.3


# --------------------------------------------------------------------------
# Corpus wrapper + serialisation
# --------------------------------------------------------------------------

def test_fit_hierarchy_over_multiple_coordinates():
    corpus = {
        "meas_err": _effects([0.1, 0.12, 0.11, 0.09, 0.1], var=1e-5),
        "lib_gap": _effects([0.0, 0.001, 0.0, 0.0, 0.0], var=1e-8),
    }
    out = fit_hierarchy(corpus, vartheta=0.05, priors=PRIORS, seed=0)
    assert set(out) == {"meas_err", "lib_gap"}
    assert out["meas_err"].prevalence > out["lib_gap"].prevalence
    d = out["meas_err"].to_dict()
    assert "conditional on susceptibility" in d["label_note"]
