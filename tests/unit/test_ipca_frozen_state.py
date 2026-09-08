"""
FrozenIPCAState — the frozen-state contract (spec §4.2, D-A49).

Pins: hash determinism + sensitivity; the manifest validates against the four-category
inventory (and is NON-tautological — it checks the actual arrays); the restricted α=0 spec is
enforced; K/L must agree with λ; the frozen arrays are read-only and the source fit is untouched;
the consumption boundary (projection_params exposes only gamma_beta/gamma_alpha, never factors).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from agents.auditor.ipca_differential import inventory as inv
from agents.auditor.ipca_differential.frozen_state import (
    FrozenIPCAState,
    FrozenStateError,
)
from agents.auditor.thresholds import load_ipca_lambda
from agents.quant.library.ipca import IPCAFit

LAM = load_ipca_lambda()   # real registered λ: K=5, L=8, 7 characteristics


def _fit(K: int = 5, L: int = 8, T: int = 30, seed: int = 0, alpha: bool = False) -> IPCAFit:
    rng = np.random.default_rng(seed)
    gb, _ = np.linalg.qr(rng.standard_normal((L, K)))    # (L, K) orthonormal columns
    factors = rng.standard_normal((K, T))
    ga = None if not alpha else np.zeros((L, 1))
    return IPCAFit(
        gamma_beta=gb, gamma_alpha=ga, factors=factors, months=np.arange(1, T + 1, dtype=np.int64),
        n_iter=3, converged=True, tol_final=1e-5, init_kind="cold",
        weighting="per_month_normalized",
    )


def test_freeze_and_manifest_validates():
    state = FrozenIPCAState.freeze(_fit(), LAM)
    state.validate_manifest(LAM)   # must not raise
    m = state.manifest()
    assert m["factor_count"] == LAM.factor_count == 5
    assert m["instrument_count"] == LAM.instrument_count == 8
    assert tuple(m["characteristic_order"]) == LAM.characteristic_order
    assert m["fitted_fields"] == list(inv.CAT1_FITTED_FIELDS)
    assert m["excluded_fields"] == list(inv.CAT3_EXCLUDED_FIELDS)
    assert state.verify_hash(LAM)


def test_hash_is_deterministic_and_sensitive():
    a = FrozenIPCAState.freeze(_fit(seed=1), LAM)
    b = FrozenIPCAState.freeze(_fit(seed=1), LAM)      # same fit → same hash
    assert a.content_hash == b.content_hash
    c = FrozenIPCAState.freeze(_fit(seed=2), LAM)      # different fit → different hash
    assert a.content_hash != c.content_hash


def test_frozen_arrays_are_readonly_and_source_untouched():
    fit = _fit()
    assert fit.gamma_beta.flags.writeable            # source is writeable before freeze
    state = FrozenIPCAState.freeze(fit, LAM)
    assert not state.gamma_beta.flags.writeable      # frozen copy is read-only
    assert not state.factors.flags.writeable
    assert not state.months.flags.writeable
    assert fit.gamma_beta.flags.writeable            # freeze copied — source still writeable
    with pytest.raises(ValueError):
        state.gamma_beta[0, 0] = 1.0                 # cannot mutate the frozen array


def test_projection_params_excludes_factors():
    """The consumption boundary: evaluation may read only gamma_beta / gamma_alpha."""
    state = FrozenIPCAState.freeze(_fit(), LAM)
    pp = state.projection_params()
    assert set(pp._fields) == {"gamma_beta", "gamma_alpha"}
    assert "factors" not in pp._fields
    np.testing.assert_array_equal(pp.gamma_beta, state.gamma_beta)
    assert pp.gamma_alpha is None


def test_restricted_alpha_zero_is_enforced():
    with pytest.raises(FrozenStateError, match="α=0"):
        FrozenIPCAState.freeze(_fit(alpha=True), LAM)


def test_factor_count_must_match_lambda():
    with pytest.raises(FrozenStateError, match="K=4"):
        FrozenIPCAState.freeze(_fit(K=4), LAM)


def test_instrument_count_must_match_lambda():
    with pytest.raises(FrozenStateError, match="L="):
        FrozenIPCAState.freeze(_fit(L=7), LAM)


def test_validate_manifest_detects_tampered_hash():
    """A directly-constructed state with a wrong content_hash fails validation (non-tautological)."""
    good = FrozenIPCAState.freeze(_fit(), LAM)
    tampered = dataclasses.replace(good, content_hash="0" * 64)
    assert not tampered.verify_hash(LAM)
    with pytest.raises(FrozenStateError, match="hash"):
        tampered.validate_manifest(LAM)


def test_validate_manifest_detects_excluded_field_leak():
    """If gamma_alpha somehow becomes non-None (α leaked in), validation rejects it."""
    good = FrozenIPCAState.freeze(_fit(), LAM)
    leaked = dataclasses.replace(good, gamma_alpha=np.zeros((8, 1)))
    with pytest.raises(FrozenStateError, match="gamma_alpha"):
        leaked.validate_manifest(LAM)


def test_inventory_doc_mirrors_module():
    """The human-readable inventory doc must name every category-1/3/4 field in the module."""
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    doc_path = repo / "docs" / "auditor" / "ipca_frozen_state_inventory.md"
    if not doc_path.exists():
        pytest.skip("inventory doc not shipped with the repository")
    doc = doc_path.read_text()
    for name in inv.CAT1_FITTED_FIELDS + inv.CAT3_EXCLUDED_FIELDS + inv.CAT4_LAMBDA:
        assert name in doc, f"inventory field {name!r} missing from the audit doc"
