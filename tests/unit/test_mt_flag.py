"""Check 5 (§7.5) — the multiple-testing flag (D5 resolved 2026-09-03).

Pins: the three registered conditions (inclusive |t| band; > max free params;
exact casefolded zoo match), NOT_EVALUATED semantics for absent inputs, the
fail-loud names loader, the thresholds loader, and the CLI sidecar shape.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from agents.auditor.checks.mt_flag import (  # noqa: E402
    ZooNamesError,
    compute_mt_flag,
    load_zoo_names,
)
from agents.auditor.thresholds import load_mt_flag_params  # noqa: E402

_NAMES = _REPO_ROOT / "data" / "dickerson_zoo" / "names.csv"
_needs_names = pytest.mark.skipif(
    not _NAMES.exists(),
    reason="machine-local names.csv absent (regenerate: scripts/build_dickerson_names.py)",
)

_BAND = (2.0, 2.5)


def _flag(**kw):
    defaults = dict(t_stat=None, free_parameters=None, zoo_names=None,
                    t_band=_BAND, max_free_parameters=2)
    defaults.update(kw)
    return compute_mt_flag(kw.pop("label", "x"), **defaults)


# --- conditions --------------------------------------------------------------

def test_band_inclusive_bounds_and_abs():
    assert _flag(t_stat=2.0).t_band.met is True          # boundary ties flag
    assert _flag(t_stat=2.5).t_band.met is True
    assert _flag(t_stat=2.25).t_band.met is True
    assert _flag(t_stat=-2.2).t_band.met is True         # |t|
    assert _flag(t_stat=1.99).t_band.met is False
    assert _flag(t_stat=2.51).t_band.met is False
    assert _flag(t_stat=6.16).t_band.met is False        # str's actual headline t


def test_free_parameters_strictly_greater_than_max():
    assert _flag(free_parameters=2).free_parameters.met is False   # at the max: no flag
    assert _flag(free_parameters=3).free_parameters.met is True
    assert _flag(free_parameters=0).free_parameters.met is False
    with pytest.raises(ValueError):
        _flag(free_parameters=-1)


def test_zoo_match_casefolds_both_sides():
    zoo = frozenset({"drf", "b_amd*"})
    r = compute_mt_flag("  DRF ", t_stat=None, free_parameters=None,
                        zoo_names=zoo, t_band=_BAND, max_free_parameters=2)
    assert r.zoo_name_match.met is True
    r2 = compute_mt_flag("mom6", t_stat=None, free_parameters=None,
                         zoo_names=zoo, t_band=_BAND, max_free_parameters=2)
    assert r2.zoo_name_match.met is False


def test_not_evaluated_semantics_and_flag():
    r = _flag()                                          # nothing supplied
    assert (r.t_band.evaluated, r.free_parameters.evaluated,
            r.zoo_name_match.evaluated) == (False, False, False)
    assert r.t_band.met is None
    assert r.flag is False                               # no evaluated condition met
    # One evaluated-and-met condition flags regardless of the unevaluated rest.
    assert _flag(t_stat=2.2).flag is True
    # Evaluated-but-unmet conditions do not flag.
    assert _flag(t_stat=6.0, free_parameters=1).flag is False


def test_invalid_band_rejected():
    with pytest.raises(ValueError):
        _flag(t_band=(2.5, 2.0))


def test_result_dict_is_informational():
    d = _flag(t_stat=2.2).to_dict()
    assert d["informational"] is True
    assert d["check"] == "mt_flag"
    assert d["conditions"]["t_band"]["met"] is True


# --- names loader ------------------------------------------------------------

def test_load_zoo_names_skips_comments_and_header(tmp_path):
    p = tmp_path / "names.csv"
    p.write_text("# comment\nname\nAlpha\n beta \n\n", encoding="utf-8")
    assert load_zoo_names(p) == frozenset({"alpha", "beta"})


def test_load_zoo_names_fail_loud(tmp_path):
    with pytest.raises(ZooNamesError):
        load_zoo_names(tmp_path / "absent.csv")
    empty = tmp_path / "empty.csv"
    empty.write_text("# only comments\nname\n", encoding="utf-8")
    with pytest.raises(ZooNamesError):
        load_zoo_names(empty)


@_needs_names
def test_real_names_file_has_108_labels():
    assert len(load_zoo_names(_NAMES)) == 108            # the registered zoo count


# --- thresholds loader -------------------------------------------------------

def test_load_mt_flag_params_from_registered_file():
    p = load_mt_flag_params()
    assert p.t_band == (2.0, 2.5)                        # §7.5 registered band
    assert p.max_free_parameters == 2
    assert p.zoo_names_path == "data/dickerson_zoo/names.csv"
    assert p.strip_trailing_asterisk is True             # 2026-09-03 normalisation
    # mom6: fact-anchored (gold skip month); drf: verified
    # (var_95 same 36m horizon/5% tail; daily-vs-monthly divergence recorded).
    assert p.aliases == {"mom6": "mom6_1", "drf": "var_95"}


# --- normalisation (registered 2026-09-03) -----------------------------

def test_strip_trailing_asterisk_is_orthographic_only(tmp_path):
    p = tmp_path / "names.csv"
    p.write_text("name\nstr*\nmom9_1*\nplain\n", encoding="utf-8")
    assert load_zoo_names(p, strip_trailing_asterisk=True) == \
        frozenset({"str", "mom9_1", "plain"})
    assert load_zoo_names(p) == frozenset({"str*", "mom9_1*", "plain"})   # default off


def test_alias_matching_and_detail():
    zoo = frozenset({"mom6_1"})
    r = compute_mt_flag("mom6", t_stat=None, free_parameters=None, zoo_names=zoo,
                        t_band=_BAND, max_free_parameters=2,
                        aliases={"mom6": "mom6_1"})
    assert r.zoo_name_match.met is True
    assert "alias" in r.zoo_name_match.detail
    # No alias entry -> no match; the alias table never fuzzy-matches.
    r2 = compute_mt_flag("drf", t_stat=None, free_parameters=None, zoo_names=zoo,
                         t_band=_BAND, max_free_parameters=2,
                         aliases={"mom6": "mom6_1"})
    assert r2.zoo_name_match.met is False


@_needs_names
def test_real_anchors_match_under_registered_normalisation():
    """All three audited anchors match under the registered rules: str via the
    *-strip (direct), mom6 via the mom6_1 alias, drf via the verified
    var_95 alias (2026-09-04; same-horizon/different-estimator, divergence
    recorded in citations_verified.md §2)."""
    p = load_mt_flag_params()
    zoo = load_zoo_names(_NAMES, strip_trailing_asterisk=p.strip_trailing_asterisk)

    def match(label):
        return compute_mt_flag(label, t_stat=None, free_parameters=None,
                               zoo_names=zoo, t_band=p.t_band,
                               max_free_parameters=p.max_free_parameters,
                               aliases=p.aliases).zoo_name_match.met

    assert match("str") is True
    assert match("mom6") is True
    assert match("drf") is True


# --- CLI sidecar -------------------------------------------------------------

@_needs_names
def test_cli_writes_sidecar(tmp_path):
    from scripts.run_mt_flag import main

    out = tmp_path / "mt_flag_drf.json"
    rc = main(["--strategy", "drf", "--t-stat", "2.3", "--out", str(out)])
    assert rc == 0
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["informational"] is True
    assert d["conditions"]["t_band"]["met"] is True
    assert d["conditions"]["free_parameters"]["evaluated"] is False
    assert d["inputs"]["t_band"] == [2.0, 2.5]
