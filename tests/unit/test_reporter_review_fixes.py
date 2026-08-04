"""Regression tests for the Reporter note/ledger/registry rendering."""

from __future__ import annotations

import unicodedata

import pytest

from agents.reporter.bundle import _stamp_from_log
from agents.reporter.format import fmt
from agents.reporter.thresholds import ReporterThresholdError, load_reporter_params
from shared.reporting.canonical import (
    CanonicalisationError,
    NanValue,
    canonical_json,
)
from shared.reporting.resolve import PointerResolutionError, resolve_json_pointer

_NFD = unicodedata.normalize("NFD", "é")
_NFC = unicodedata.normalize("NFC", "é")


# --- Major 1: RFC-6901 index must be ASCII, fail closed otherwise ---------------------------

@pytest.mark.parametrize("token", ["٢", "２", "²", "₁", "-", "00", "01", "1a"])
def test_non_ascii_or_malformed_index_fails_closed(token):
    with pytest.raises(PointerResolutionError):
        resolve_json_pointer({"xs": [10, 20, 30]}, f"/xs/{token}")


def test_valid_ascii_index_still_resolves():
    assert resolve_json_pointer({"xs": [10, 20, 30]}, "/xs/2") == 30
    assert resolve_json_pointer({"xs": [10, 20, 30]}, "/xs/0") == 10


# --- Major 2: git_short must be validated, never fabricate full ------------------------------

def test_empty_git_short_derives_from_full():
    s = _stamp_from_log({"git_commit": "a" * 40, "git_short": ""})
    assert s.full == "a" * 40
    assert s.short == "aaaaaaa"
    assert s.source_form == "full"


def test_garbage_git_short_rejected_and_derived():
    s = _stamp_from_log({"git_commit": "a" * 40, "git_short": "zzz"})
    assert s.short == "aaaaaaa"
    assert s.source_form == "full"


def test_neither_git_form():
    s = _stamp_from_log({"unrelated": 1})
    assert s.full is None and s.short is None and s.source_form == "none"


# --- Minor 3/4: NaN reason NFC, NFC key collision rejected ----------------------------------

def test_nan_reason_is_nfc_normalised():
    assert _NFD != _NFC
    assert canonical_json(NanValue("re" + _NFD)) == canonical_json(NanValue("re" + _NFC))


def test_nfc_key_collision_raises():
    d = {f"k{_NFD}": 1, f"k{_NFC}": 2}
    assert len(d) == 2  # distinct source strings
    with pytest.raises(CanonicalisationError):
        canonical_json(d)


# --- Minor 7: negative precision is a typed error -------------------------------------------

def test_negative_precision_raises():
    with pytest.raises(ValueError):
        fmt(1.23, precision=-1)


# --- Minor 8: missing thresholds file is the module's typed error ---------------------------

def test_missing_thresholds_file_raises_typed(tmp_path):
    with pytest.raises(ReporterThresholdError):
        load_reporter_params(tmp_path / "does_not_exist.yaml")
