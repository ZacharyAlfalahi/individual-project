"""
Unit tests for the bias-toggle registry RunConfig dataclass.

Covers:
  - Polarity convention (OFF = biased; ON = corrected)
  - YAML round-trip identity
  - hash() stability across runs
  - frozen=True enforced (mutation raises)
  - Invalid combinations rejected
  - Endpoint convenience constructors (uncorrected, corrected)
"""

import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent.parent / "agents" / "quant" / "library"),
)
from run_config import (  # noqa: E402
    ConstructionConfig,
    EvaluationConfig,
    PanelViewConfig,
    RunConfig,
    corrected,
    uncorrected,
)


# ---------------------------------------------------------------------------
# Polarity convention
# ---------------------------------------------------------------------------

class TestPolarityConvention:
    def test_uncorrected_endpoint_is_all_off(self):
        """uncorrected() = as-published baseline: raw family, no stale mask,
        no terminal rows, signal_lag=0, expost_trim=as_published."""
        c = uncorrected()
        assert c.panel_view.price_family == "raw"
        assert c.panel_view.stale_mask is False
        assert c.panel_view.include_terminal_rows is False
        assert c.construction.signal_lag == 0
        assert c.construction.expost_trim == "as_published"
        assert c.evaluation.mt_flag == "informational"

    def test_corrected_endpoint_is_all_on_except_survivorship(self):
        """corrected() pre-FISD: corr family, stale mask on, signal_lag=1,
        no expost_trim. include_terminal_rows stays False because exit_reason
        is NaN pre-FISD."""
        c = corrected()
        assert c.panel_view.price_family == "corr"
        assert c.panel_view.stale_mask is True
        assert c.panel_view.include_terminal_rows is False
        assert c.construction.signal_lag == 1
        assert c.construction.expost_trim == "none"

    def test_uncorrected_and_corrected_differ_at_every_toggle(self):
        u, c = uncorrected(), corrected()
        assert u.panel_view.price_family != c.panel_view.price_family
        assert u.panel_view.stale_mask != c.panel_view.stale_mask
        assert u.construction.signal_lag != c.construction.signal_lag
        assert u.construction.expost_trim != c.construction.expost_trim


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_invalid_price_family_raises(self):
        with pytest.raises(ValueError, match="price_family"):
            PanelViewConfig(price_family="raw_corr", stale_mask=False,
                            include_terminal_rows=False)

    def test_stale_mask_must_be_bool(self):
        with pytest.raises(TypeError, match="stale_mask"):
            PanelViewConfig(price_family="raw", stale_mask=1,  # type: ignore
                            include_terminal_rows=False)

    def test_negative_signal_lag_raises(self):
        with pytest.raises(ValueError, match="signal_lag"):
            ConstructionConfig(signal_lag=-1, expost_trim="none")

    def test_bool_signal_lag_rejected(self):
        # True isinstance int in Python; ensure we don't accept bool by accident
        with pytest.raises(TypeError, match="signal_lag"):
            ConstructionConfig(signal_lag=True, expost_trim="none")  # type: ignore

    def test_invalid_expost_trim_raises(self):
        with pytest.raises(ValueError, match="expost_trim"):
            ConstructionConfig(signal_lag=0, expost_trim="winsorise")  # type: ignore

    def test_mt_flag_must_be_informational(self):
        with pytest.raises(ValueError, match="mt_flag"):
            EvaluationConfig(mt_flag="gates")  # type: ignore


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------

class TestImmutability:
    def test_runconfig_is_frozen(self):
        c = uncorrected()
        with pytest.raises(FrozenInstanceError):
            c.panel_view = PanelViewConfig(  # type: ignore
                price_family="corr", stale_mask=True, include_terminal_rows=False
            )

    def test_sub_configs_are_frozen(self):
        pv = PanelViewConfig(price_family="raw", stale_mask=False,
                             include_terminal_rows=False)
        with pytest.raises(FrozenInstanceError):
            pv.stale_mask = True  # type: ignore


# ---------------------------------------------------------------------------
# YAML round-trip
# ---------------------------------------------------------------------------

class TestYAMLRoundTrip:
    def test_uncorrected_round_trips(self):
        c = uncorrected()
        y = c.to_yaml()
        c2 = RunConfig.from_yaml(y)
        assert c == c2

    def test_corrected_round_trips(self):
        c = corrected()
        y = c.to_yaml()
        c2 = RunConfig.from_yaml(y)
        assert c == c2

    def test_round_trip_preserves_hash(self):
        c = corrected()
        c2 = RunConfig.from_yaml(c.to_yaml())
        assert c.hash() == c2.hash()

    def test_from_dict_accepts_wrapped_and_unwrapped(self):
        c = uncorrected()
        wrapped = c.to_dict()
        unwrapped = wrapped["run_config"]
        assert RunConfig.from_dict(wrapped) == c
        assert RunConfig.from_dict(unwrapped) == c

    def test_from_dict_missing_required_key_raises(self):
        with pytest.raises(KeyError, match="panel_view"):
            RunConfig.from_dict({"construction": {"signal_lag": 0, "expost_trim": "none"}})


# ---------------------------------------------------------------------------
# Hash stability
# ---------------------------------------------------------------------------

class TestHashStability:
    def test_same_config_same_hash(self):
        c1 = corrected()
        c2 = corrected()
        assert c1.hash() == c2.hash()

    def test_different_configs_different_hash(self):
        assert uncorrected().hash() != corrected().hash()

    def test_field_change_changes_hash(self):
        c1 = uncorrected()
        c2 = RunConfig(
            panel_view=PanelViewConfig(
                price_family="raw", stale_mask=True,
                include_terminal_rows=False,
            ),
            construction=c1.construction,
            evaluation=c1.evaluation,
        )
        assert c1.hash() != c2.hash()

    def test_hash_is_sha256_hex(self):
        h = corrected().hash()
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)
