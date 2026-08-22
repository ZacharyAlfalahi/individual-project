"""Unit tests for the factor-level hypothesis registry + runtime gate (Part F)."""

import textwrap

import pytest

from agents.auditor.validation.hypothesis_registry import (
    FactorHypothesisAbsent,
    HypothesisRegistryError,
    load_hypothesis_registry,
    require_factor_registered,
)


def test_committed_registry_loads_and_seeds_from_22july():
    reg = load_hypothesis_registry()
    assert set(reg) == {"str", "mom6", "drf", "traded_liquidity"}
    # str: pilot, lib_gap, sign_only, NOT in the confirmatory family (D-A33)
    s = reg["str"]
    assert s.dominant_bias == "lib_gap" and s.expected_sign == -1
    assert s.magnitude_mode == "sign_only" and s.is_locked is False
    assert s.is_confirmatory is False
    # mom6: locked band from DRR published figure
    m = reg["mom6"]
    assert m.dominant_bias == "lab_trim" and m.expected_sign == -1
    assert m.magnitude_mode == "band" and m.expected_magnitude_range == (0.0, 0.003)
    assert m.is_locked and m.is_confirmatory
    # drf: locked but SIGN-ONLY (citation resolution ed32122)
    d = reg["drf"]
    assert d.dominant_bias == "meas_err" and d.expected_sign == -1
    assert d.magnitude_mode == "sign_only" and d.expected_magnitude_range is None
    assert d.is_locked and d.is_confirmatory
    # negative control — falsifiable `separated` specificity gate (D20 §4.5 remediation)
    tl = reg["traded_liquidity"]
    assert tl.expected_sign == 0
    assert tl.magnitude_mode == "separated" and tl.expected_magnitude_range is None
    # every row carries the registering commit
    assert all(h.registered_commit for h in reg.values())


def test_runtime_gate_returns_for_registered_factor():
    assert require_factor_registered("mom6").dominant_bias == "lab_trim"


def test_runtime_gate_refuses_unregistered_factor():
    with pytest.raises(FactorHypothesisAbsent) as exc:
        require_factor_registered("nonexistent_factor")
    assert "no row" in str(exc.value)
    assert exc.value.factor_id == "nonexistent_factor"


def test_runtime_gate_refuses_pilot_factor_fail_closed():
    # str is a PILOT (is_locked=false) — it can never yield a confirmatory outcome,
    # so the gate must REFUSE rather than return its row (the fail-open the docstrings
    # forbid). Pilot reporting reads load_hypothesis_registry() directly instead.
    with pytest.raises(FactorHypothesisAbsent) as exc:
        require_factor_registered("str")
    assert "not confirmatory" in str(exc.value)
    assert exc.value.factor_id == "str"


def _write(tmp_path, body):
    p = tmp_path / "reg.yaml"
    p.write_text(textwrap.dedent(body))
    return p


def test_locked_band_lo0_with_definite_sign_is_falsifiable(tmp_path):
    # mom6's shape: locked band [0, hi] with a DEFINITE sign (−1) — falsifiable via
    # sign + upper bound, so lo=0 is allowed (the ex-ante endpoint is ≈0).
    p = _write(tmp_path, """
        version: v1
        factors:
          x: {dominant_bias: lab_trim, expected_sign: -1,
              expected_magnitude_range: [0.0, 0.01], is_locked: true,
              status: locked, source: s, registered_commit: abc}
    """)
    reg = load_hypothesis_registry(p)
    assert reg["x"].magnitude_mode == "band" and reg["x"].expected_magnitude_range == (0.0, 0.01)


def test_locked_sign0_band_at_lo0_rejected(tmp_path):
    # An UNDEFINED sign (0) + lo=0 IS unfalsifiable (any small effect passes).
    p = _write(tmp_path, """
        version: v1
        factors:
          x: {dominant_bias: none, expected_sign: 0,
              expected_magnitude_range: [0.0, 0.01], is_locked: true,
              status: locked, source: s, registered_commit: abc}
    """)
    with pytest.raises(HypothesisRegistryError, match="unfalsifiable"):
        load_hypothesis_registry(p)


def test_signed_or_descending_band_rejected(tmp_path):
    p = _write(tmp_path, """
        version: v1
        factors:
          x: {dominant_bias: meas_err, expected_sign: -1,
              expected_magnitude_range: [-0.009, -0.003], is_locked: true,
              status: locked, source: s, registered_commit: abc}
    """)
    with pytest.raises(HypothesisRegistryError, match="unsigned ascending"):
        load_hypothesis_registry(p)


def test_missing_is_locked_rejected(tmp_path):
    p = _write(tmp_path, """
        version: v1
        factors:
          x: {dominant_bias: meas_err, expected_sign: -1, magnitude: sign_only,
              status: locked, source: s, registered_commit: abc}
    """)
    with pytest.raises(HypothesisRegistryError, match="is_locked"):
        load_hypothesis_registry(p)


def test_unknown_status_rejected(tmp_path):
    # A mislabelled status (e.g. capitalised "Pilot") would slip past is_confirmatory's
    # exact-string check and corrupt the gate — the loader must refuse it fail-loud.
    p = _write(tmp_path, """
        version: v1
        factors:
          x: {dominant_bias: lab_trim, expected_sign: -1, magnitude: sign_only,
              is_locked: true, status: Pilot, source: s, registered_commit: abc}
    """)
    with pytest.raises(HypothesisRegistryError, match="status must be one of"):
        load_hypothesis_registry(p)


def test_sign_only_locked_is_allowed(tmp_path):
    # drf's shape: locked + sign_only (no band) — must load (the band-lo>0 guard
    # only applies to banded factors).
    p = _write(tmp_path, """
        version: v1
        factors:
          drf: {dominant_bias: meas_err, expected_sign: -1, magnitude: sign_only,
                is_locked: true, status: locked, source: s, registered_commit: abc}
    """)
    reg = load_hypothesis_registry(p)
    assert reg["drf"].magnitude_mode == "sign_only" and reg["drf"].is_confirmatory


def test_separated_mode_loads(tmp_path):
    # The negative control's shape: locked, sign 0, magnitude `separated` (no band).
    # It must load — the band falsifiability guard applies only to banded factors;
    # `separated` carries its falsifiable threshold in the evaluator (±vartheta), not a band.
    p = _write(tmp_path, """
        version: v1
        factors:
          c: {dominant_bias: none, expected_sign: 0, magnitude: separated,
              is_locked: true, status: negative_control, source: s, registered_commit: abc}
    """)
    reg = load_hypothesis_registry(p)
    assert reg["c"].magnitude_mode == "separated" and reg["c"].expected_magnitude_range is None
