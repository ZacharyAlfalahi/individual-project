"""
Canonical-YAML rulebook serialisation + byte-equality (G2, decision D30).

D30 specifies the round-trip gate as "canonical-YAML serialisation of ``to_rulebook``
output dicts, byte-equal per leg". The two sides of the comparison genuinely carry
different key sets by design -- ``to_rulebook`` emits ``min_bonds`` / ``trim_rule``
(provenance-relevant), the golden rulebook builders emit ``nw_lags`` (engine-call) --
so the single reconciler is ``characteristic_sort._apply_defaults``: it fills BOTH
sides to the engine's canonical superset (the same fill the engine applies at run
time, already load-bearing across the golden regression tests). After defaults, both
carry identical keys; sorted-key YAML then makes the bytes deterministic.
"""

from __future__ import annotations

import difflib

import yaml

from agents.quant.library.characteristic_sort import _apply_defaults


def canonical_rulebook_yaml(rulebook: dict) -> str:
    """Deterministic canonical-YAML bytes for a rulebook: fill engine defaults
    (the key-set reconciler), then sorted-key block YAML. Rulebooks are
    int/str/None/nested-dict only (no floats), so the serialisation is exact."""
    filled = _apply_defaults(dict(rulebook))  # copy: _apply_defaults may mutate
    return yaml.safe_dump(filled, sort_keys=True, default_flow_style=False, allow_unicode=True)


def rulebooks_equal(produced: dict, expected: dict) -> bool:
    """True iff the two rulebooks are equal after default-filling (semantic check)."""
    return _apply_defaults(dict(produced)) == _apply_defaults(dict(expected))


def assert_rulebook_byte_equal(produced: dict, expected: dict) -> None:
    """The G2 gate: assert ``produced`` == ``expected`` as rulebooks. Dict-equality
    first (a clean, readable failure), then canonical-YAML byte-equality (the D30
    literal). Raises ``AssertionError`` with a unified diff on mismatch."""
    pd, ed = _apply_defaults(dict(produced)), _apply_defaults(dict(expected))
    if pd != ed:
        raise AssertionError(
            "rulebook dicts differ after default-filling:\n"
            f"  produced={pd}\n  expected={ed}"
        )
    p_yaml, e_yaml = canonical_rulebook_yaml(produced), canonical_rulebook_yaml(expected)
    if p_yaml != e_yaml:
        diff = "".join(
            difflib.unified_diff(
                e_yaml.splitlines(keepends=True),
                p_yaml.splitlines(keepends=True),
                fromfile="expected",
                tofile="produced",
            )
        )
        raise AssertionError(f"canonical-YAML bytes differ:\n{diff}")
