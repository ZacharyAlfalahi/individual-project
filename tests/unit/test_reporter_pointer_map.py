"""Pointer key-map proof (C1-C3): the audit-core pointers resolve against the REAL serialised
AuditCore on disk, and the spec's stale/renamed pointers fail closed (INV-13).

This tests the Reporter's understanding of the actual `AuditCore.to_dict()` / quant `<anchor>.json`
shapes against genuine committed artefacts, not a hand-built dict — so a wrong key-map belief is
caught here, not in production.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared.reporting.resolve import PointerResolutionError, resolve_json_pointer

_REPO = Path(__file__).resolve().parents[2]
_CORE = _REPO / "results/auditor/drf/drf_core.json"
_QRES = _REPO / "results/quant/drf/drf.json"


@pytest.mark.skipif(not _CORE.exists(), reason="committed audit-core artefact absent")
def test_audit_core_pointer_map_resolves():
    d = json.loads(_CORE.read_text())
    # C1: core is flattened — audit_scope / saturated_bases / shapley are top-level.
    assert resolve_json_pointer(d, "/audit_scope") == "COMPLETE"
    # C2: corner_marginals.bracket -> interaction_bracket.
    assert isinstance(
        resolve_json_pointer(d, "/corner_marginals/interaction_bracket/meas_err"),
        (int, float),
    )
    # C2: shapley outer key NOT renamed; inner keys renamed.
    assert isinstance(
        resolve_json_pointer(d, "/shapley/shapley_values/meas_err"), (int, float)
    )
    resolve_json_pointer(d, "/shapley/shapley_share_of_registered_endpoint_gap/meas_err")
    resolve_json_pointer(d, "/shapley/registered_endpoint_gap")
    resolve_json_pointer(d, "/shapley/shares_reported")
    # C2/C3: saturated_bases.doe_effects, subset labels joined by × (U+00D7), empty set ∅.
    resolve_json_pointer(d, "/saturated_bases/doe_effects/meas_err")
    resolve_json_pointer(d, "/saturated_bases/doe_effects/meas_err×stale_price")
    resolve_json_pointer(d, "/saturated_bases/doe_effects/∅")
    # ADR §5.2: the bias_class partition is a top-level block; the three components
    # are typed leaves the Reporter binds by json_pointer (all present on this
    # all-runnable COMPLETE audit).
    assert isinstance(
        resolve_json_pointer(d, "/bias_class_partition/methodological_construction_component"),
        (int, float),
    )
    assert isinstance(
        resolve_json_pointer(d, "/bias_class_partition/data_quality_component"),
        (int, float),
    )
    assert isinstance(
        resolve_json_pointer(d, "/bias_class_partition/cross_class_modulation"),
        (int, float),
    )
    assert resolve_json_pointer(d, "/bias_class_partition/data_quality_present") is True


@pytest.mark.skipif(not _CORE.exists(), reason="committed audit-core artefact absent")
def test_stale_or_renamed_pointers_fail_closed():
    d = json.loads(_CORE.read_text())
    for bad in (
        "/core/audit_scope",  # there is no `core` key (flattened)
        "/saturated/doe",  # pre-rename object-graph name
        "/saturated_bases/doe",  # inner un-renamed
        "/shapley_values/meas_err",  # shapley_values is not top-level
        "/corner_marginals/bracket/meas_err",  # bracket un-renamed
        "/economic/deflated_sharpe_corrected",  # full-report layer not persisted (C9)
        "/fdr/decisions",  # full-report layer not persisted (C9)
        "/bias_class_partition/total_bias",  # banned term; not a partition key
        "/bias_class_partition/method_component",  # wrong key (it's the full name)
    ):
        with pytest.raises(PointerResolutionError):
            resolve_json_pointer(d, bad)


@pytest.mark.skipif(not _QRES.exists(), reason="committed quant result artefact absent")
def test_quant_result_pointers_resolve():
    d = json.loads(_QRES.read_text())
    # The persisted quant <anchor>.json is a plain summary-bearing dict — direct json_pointer.
    assert isinstance(resolve_json_pointer(d, "/summary/sharpe"), (int, float))
    assert isinstance(resolve_json_pointer(d, "/summary/t_stat"), (int, float))
    assert isinstance(resolve_json_pointer(d, "/strategy_label"), str)
