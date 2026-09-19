"""Unit tests for the return-basis / output-path flags on the data-layer scripts:
build_{bbw_factors,mktb,str,mom6} --basis; run_leadlag_gate and
run_dickerson_gross_error_check --factors-dir/--out; run_mom6_lab_gate --basis/--out;
run_accrual_validation --total/--out; run_anchor_descriptive --total-return-panel/--out.

For each script: (a) the flags parse; (b) with a flag, inputs resolve to the right
basis_inputs.PANELS entry and outputs land under results/consistent_basis/<basis>/...;
(c) with no flag, the resolved paths equal the module defaults. LIGHT: no real panel is ever read — the heavy
run functions are monkeypatched and the end-to-end main() checks use tiny synthetic parquets.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import scripts.basis_inputs as B
import build_bbw_factors
import build_mktb
import build_mom6
import build_str
import run_accrual_validation
import run_anchor_descriptive
import run_dickerson_gross_error_check
import run_leadlag_gate
import run_mom6_lab_gate

REPO = B.REPO_ROOT
DEV = REPO / "data" / "development"
DEFAULT_FACTORS = DEV / "factors"
DEFAULT_HEADLINES = DEV / "headlines"
DEFAULT_TR_PANEL = DEV / "monthly_panel_total_return.parquet"
CLEAN_PANEL = DEV / "monthly_panel_maximal.parquet"
DATES = pd.date_range("2005-01-31", periods=24, freq="ME")

BUILDERS = [
    pytest.param(build_bbw_factors, "bbw_factors", id="bbw_factors"),
    pytest.param(build_mktb, "mktb", id="mktb"),
    pytest.param(build_str, "str", id="str"),
    pytest.param(build_mom6, "mom6", id="mom6"),
]


def _tiny_panel(path: Path) -> Path:
    """A 2-bond x 24-month maximal-format stub (enough columns for every script's projection)."""
    rows = [(c, d) for c in ("AAA111", "BBB222") for d in DATES]
    df = pd.DataFrame(rows, columns=["cusip", "date"])
    df["size"] = 1.0e6
    df["universe_eligible"] = True
    df["xret_raw"] = df["xret_corr"] = df["ret_raw"] = df["ret_corr"] = 0.001
    df["day_count_fallback"] = False
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return path


def _tiny_signal(path: Path, col: str) -> Path:
    rows = [(c, d) for c in ("AAA111", "BBB222") for d in DATES]
    df = pd.DataFrame(rows, columns=["cusip", "date"])
    df[col] = 0.5
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return path


# ---------------------------------------------------------------------------
# 1. factor builders — --basis
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod, stem", BUILDERS)
def test_builder_parses_basis_flag(mod, stem):
    assert mod.parse_args([]).basis is None
    for basis in B.BASES:
        assert mod.parse_args(["--basis", basis]).basis == basis
    with pytest.raises(SystemExit):
        mod.parse_args(["--basis", "total"])


@pytest.mark.parametrize("mod, stem", BUILDERS)
def test_builder_basis_resolves_panel_and_consistent_basis_outputs(mod, stem):
    for basis in B.BASES:
        panel, out_file, report_out = mod.resolve_paths(basis)
        assert panel == B.PANELS[basis][0] and panel.is_absolute()
        root = REPO / "results" / "consistent_basis" / basis / "factors"
        assert out_file == root / f"{stem}.parquet"
        assert report_out == root / f"{stem}_report.json"


@pytest.mark.parametrize("mod, stem", BUILDERS)
def test_builder_no_flag_resolves_default_paths(mod, stem):
    panel, out_file, report_out = mod.resolve_paths()
    assert (panel, out_file, report_out) == (mod.PANEL_FILE, mod.OUT_FILE, mod.REPORT_OUT)
    assert out_file == DEFAULT_FACTORS / f"{stem}.parquet"
    assert report_out == DEFAULT_FACTORS / f"{stem}_report.json"
    if "BBW_ANCHOR_PANEL" not in os.environ:
        assert panel == (DEFAULT_TR_PANEL if DEFAULT_TR_PANEL.exists() else CLEAN_PANEL)


@pytest.mark.parametrize("mod, stem", BUILDERS)
def test_builder_env_var_honoured_without_flag(mod, stem, tmp_path):
    target = tmp_path / "anchor_override.parquet"
    try:
        with mock.patch.dict(os.environ, {"BBW_ANCHOR_PANEL": str(target)}):
            importlib.reload(mod)
            assert mod.resolve_paths()[0] == target
            assert mod.resolve_paths("clean")[0] == B.PANELS["clean"][0]  # the flag wins
    finally:
        importlib.reload(mod)  # env restored -> module back to its import-time state


def _install_builder_fakes(mod, stem, monkeypatch, tmp_path):
    """Replace the heavy run unit with a synthetic one and point side inputs at tiny stubs."""
    if mod is build_mktb:
        monkeypatch.setattr(mod, "compute_dual_family", lambda panel: pd.DataFrame({
            "date": DATES, "mktb_raw": 0.002, "mktb_corr": 0.003, "n_bonds_raw": 2, "n_bonds_corr": 2}))
    elif mod is build_str:
        monkeypatch.setattr(mod, "run_family", lambda maximal, fam: {
            "monthly": pd.DataFrame({"date": DATES, f"str_{fam}": 0.01, f"n_bonds_{fam}": 2}),
            "summary": {"t_stat": 2.0, "sharpe": 0.5}})
    elif mod is build_mom6:
        monkeypatch.setattr(mod, "SIGNAL_FILE", _tiny_signal(tmp_path / "sig" / "mom6.parquet", "mom6_corr"))
        monkeypatch.setattr(mod, "run_family", lambda maximal, signal, fam, cfg: {
            "monthly": pd.DataFrame({"date": DATES, f"mom6_{fam}": 0.001, f"n_bonds_{fam}": 2}),
            "summary": {"t_stat": 0.4, "sharpe": 0.1}})
    else:
        monkeypatch.setattr(mod, "VAR_FILE", _tiny_signal(tmp_path / "sig" / "var_5pct.parquet", "var_5pct_corr"))
        monkeypatch.setattr(mod, "GAMMA_FILE", _tiny_signal(tmp_path / "sig" / "gamma.parquet", "gamma_corr"))
        names = mod.FACTORS + ["crf"]

        def fake_run_family(maximal, signals, fam):
            monthly = {n: pd.DataFrame({"date": DATES, f"{n}_{fam}": 0.004}) for n in names}
            return monthly, {n: {"t_stat": 1.5} for n in names}
        monkeypatch.setattr(mod, "run_family", fake_run_family)
    # Sentinel default outputs, so a basis run can be proven NOT to write them.
    monkeypatch.setattr(mod, "OUT_FILE", tmp_path / "default" / f"{stem}.parquet")
    monkeypatch.setattr(mod, "REPORT_OUT", tmp_path / "default" / f"{stem}_report.json")


@pytest.mark.parametrize("mod, stem", BUILDERS)
@pytest.mark.parametrize("basis", B.BASES)
def test_builder_main_with_basis_writes_under_basis_root_with_provenance(mod, stem, basis, monkeypatch, tmp_path):
    _install_builder_fakes(mod, stem, monkeypatch, tmp_path)
    panel = _tiny_panel(tmp_path / "panels" / f"{basis}.parquet")  # outside the repo: relative_to must not crash
    monkeypatch.setitem(B.PANELS, basis, (panel, tmp_path / "panels" / "profiles.parquet"))
    monkeypatch.setattr(B, "RESULTS_ROOT", tmp_path / "consistent_basis")

    mod.main(["--basis", basis])

    out_dir = tmp_path / "consistent_basis" / basis / "factors"
    assert (out_dir / f"{stem}.parquet").is_file()
    report = json.loads((out_dir / f"{stem}_report.json").read_text())
    assert report["basis"] == basis
    assert report["input_panel"] == str(panel)
    assert report["input_panel_sha256"] == B.sha256(panel)
    assert not (tmp_path / "default").exists()


@pytest.mark.parametrize("mod, stem", BUILDERS)
def test_builder_main_without_flag_uses_module_defaults_and_no_basis_keys(mod, stem, monkeypatch, tmp_path):
    _install_builder_fakes(mod, stem, monkeypatch, tmp_path)
    monkeypatch.setattr(mod, "PANEL_FILE", _tiny_panel(tmp_path / "panels" / "anchor.parquet"))
    monkeypatch.setattr(B, "RESULTS_ROOT", tmp_path / "consistent_basis")

    mod.main([])

    assert (tmp_path / "default" / f"{stem}.parquet").is_file()
    report = json.loads((tmp_path / "default" / f"{stem}_report.json").read_text())
    assert "basis" not in report and "input_panel_sha256" not in report
    if mod in (build_mktb, build_str):  # these two record input_panel without a basis too
        assert report["input_panel"] == str(tmp_path / "panels" / "anchor.parquet")
    assert not (tmp_path / "consistent_basis").exists()


# ---------------------------------------------------------------------------
# 2. run_leadlag_gate / run_dickerson_gross_error_check — --factors-dir / --out
# ---------------------------------------------------------------------------

def test_leadlag_no_flag_resolves_default_paths():
    args = run_leadlag_gate.parse_args([])
    assert args.factors_dir == DEFAULT_FACTORS
    assert args.out == DEFAULT_HEADLINES / "leadlag_gate.json"
    assert run_leadlag_gate.resolve_paths(args) == (DEFAULT_FACTORS / "bbw_factors.parquet",
                                                    DEFAULT_HEADLINES / "leadlag_gate.json")
    assert run_leadlag_gate.input_provenance(run_leadlag_gate.FACTORS_FILE) == {}


def test_leadlag_flags_resolve_basis_paths():
    out = B.basis_dir("clean", "headlines", "leadlag_gate.json")
    args = run_leadlag_gate.parse_args(["--factors-dir", str(B.factors_dir("clean")), "--out", str(out)])
    factors_file, resolved_out = run_leadlag_gate.resolve_paths(args)
    assert factors_file == REPO / "results" / "consistent_basis" / "clean" / "factors" / "bbw_factors.parquet"
    assert resolved_out == out


def test_leadlag_non_default_factors_with_default_out_refused():
    args = run_leadlag_gate.parse_args(["--factors-dir", str(B.factors_dir("clean"))])
    with pytest.raises(SystemExit, match="default gate report"):
        run_leadlag_gate.resolve_paths(args)


def test_leadlag_main_reads_factors_dir_and_writes_out(monkeypatch, tmp_path):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2002-01-31", "2017-12-31", freq="ME")
    factors_dir = tmp_path / "factors"
    factors_dir.mkdir()
    pd.DataFrame({"date": dates, **{c: rng.normal(0, 0.01, len(dates))
                                     for c in ("drf_corr", "crf_corr", "lrf_corr")}}
                 ).to_parquet(factors_dir / "bbw_factors.parquet")
    monkeypatch.setattr(run_leadlag_gate, "OUT", tmp_path / "default" / "leadlag_gate.json")
    out = tmp_path / "basis" / "leadlag_gate.json"

    with pytest.raises(SystemExit):
        run_leadlag_gate.main(["--factors-dir", str(factors_dir), "--out", str(out)])

    report = json.loads(out.read_text())
    assert report["factors_file"] == str(factors_dir / "bbw_factors.parquet")
    assert report["factors_file_sha256"] == B.sha256(factors_dir / "bbw_factors.parquet")
    assert set(report["arms"]) == {"drf", "crf", "lrf"}
    assert not (tmp_path / "default").exists()


def test_dickerson_no_flag_resolves_default_paths():
    args = run_dickerson_gross_error_check.parse_args([])
    assert args.factors_dir == DEFAULT_FACTORS
    assert args.out == DEFAULT_HEADLINES / "dickerson_gross_error.json"


def test_dickerson_flags_route_inputs_and_output(monkeypatch, tmp_path):
    gec = run_dickerson_gross_error_check
    monkeypatch.setattr(gec, "EXTERNAL_FILE", tmp_path / "external.parquet")  # absent -> inputs_missing
    monkeypatch.setattr(gec, "FACTORS_DIR", tmp_path / "default_factors")
    monkeypatch.setattr(gec, "OUT", tmp_path / "default" / "dickerson_gross_error.json")
    factors_dir = tmp_path / "basis_factors"
    out = tmp_path / "basis" / "dickerson_gross_error.json"

    with pytest.raises(SystemExit) as exc:
        gec.main(["--factors-dir", str(factors_dir), "--out", str(out)])
    assert exc.value.code == 0

    report = json.loads(out.read_text())
    assert report["factors_dir"] == str(factors_dir)
    assert str(factors_dir / "str.parquet") in report["missing_files"]
    assert not any("default_factors" in m for m in report["missing_files"])
    assert not (tmp_path / "default").exists()


def test_dickerson_non_default_factors_with_default_out_refused(monkeypatch, tmp_path):
    gec = run_dickerson_gross_error_check
    monkeypatch.setattr(gec, "FACTORS_DIR", tmp_path / "default_factors")
    monkeypatch.setattr(gec, "OUT", tmp_path / "default" / "dickerson_gross_error.json")
    with pytest.raises(SystemExit, match="default report"):
        gec.main(["--factors-dir", str(tmp_path / "basis_factors")])
    assert not (tmp_path / "default").exists()


def test_dickerson_records_sha256_of_non_default_factor_files(monkeypatch, tmp_path):
    gec = run_dickerson_gross_error_check
    monkeypatch.setattr(gec, "EXTERNAL_FILE", tmp_path / "external.parquet")  # absent -> inputs_missing
    monkeypatch.setattr(gec, "FACTORS_DIR", tmp_path / "default_factors")
    factors_dir = tmp_path / "basis_factors"
    present = _tiny_signal(factors_dir / "str.parquet", "str_corr")
    out = tmp_path / "basis" / "dickerson_gross_error.json"
    with pytest.raises(SystemExit):
        gec.main(["--factors-dir", str(factors_dir), "--out", str(out)])
    assert json.loads(out.read_text())["factors_sha256"] == {"str.parquet": B.sha256(present)}


def test_dickerson_no_flag_writes_default_out_without_factors_dir_key(monkeypatch, tmp_path):
    gec = run_dickerson_gross_error_check
    monkeypatch.setattr(gec, "EXTERNAL_FILE", tmp_path / "external.parquet")
    monkeypatch.setattr(gec, "FACTORS_DIR", tmp_path / "default_factors")
    monkeypatch.setattr(gec, "OUT", tmp_path / "default" / "dickerson_gross_error.json")
    with pytest.raises(SystemExit):
        gec.main()
    report = json.loads((tmp_path / "default" / "dickerson_gross_error.json").read_text())
    assert "factors_dir" not in report


# ---------------------------------------------------------------------------
# 3. run_mom6_lab_gate — --basis / --out
# ---------------------------------------------------------------------------

def test_mom6_lab_flags_parse_and_resolve():
    args = run_mom6_lab_gate.parse_args([])
    assert args.basis is None and args.out is None
    assert run_mom6_lab_gate.resolve_out() == DEFAULT_HEADLINES / "mom6_lab_gate.json"
    assert run_mom6_lab_gate.resolve_out("clean") == B.basis_dir("clean", "headlines", "mom6_lab_gate.json")
    assert run_mom6_lab_gate.resolve_out("clean", Path("x.json")) == Path("x.json")
    with pytest.raises(SystemExit, match="default headline"):     # a basis run never writes the default headline
        run_mom6_lab_gate.resolve_out("total_return", DEFAULT_HEADLINES / "mom6_lab_gate.json")
    assert run_mom6_lab_gate.resolve_panel() == run_mom6_lab_gate.PANEL_FILE
    for basis in B.BASES:
        assert run_mom6_lab_gate.parse_args(["--basis", basis]).basis == basis
        assert run_mom6_lab_gate.resolve_panel(basis) == B.PANELS[basis][0]
    with pytest.raises(SystemExit):
        run_mom6_lab_gate.parse_args(["--basis", "tr"])


@pytest.mark.parametrize("with_out", [True, False], ids=["explicit_out", "basis_default_out"])
def test_mom6_lab_main_with_basis_reads_basis_panel_and_writes_out(monkeypatch, tmp_path, with_out):
    lab = run_mom6_lab_gate
    monkeypatch.setattr(B, "RESULTS_ROOT", tmp_path / "cb")
    rng = np.random.default_rng(1)
    panel = _tiny_panel(tmp_path / "panels" / "clean.parquet")
    monkeypatch.setitem(B.PANELS, "clean", (panel, tmp_path / "panels" / "profiles.parquet"))
    monkeypatch.setattr(lab, "SIGNAL_FILE", _tiny_signal(tmp_path / "sig" / "mom6.parquet", "mom6_corr"))
    monkeypatch.setattr(lab, "OUT", tmp_path / "default" / "mom6_lab_gate.json")
    seen = {}

    def fake_view(maximal, cfg, signals=None):
        seen["rows"] = len(maximal)
        return maximal.assign(ret=maximal["ret_corr"], mom6=0.5)
    monkeypatch.setattr(lab, "view", fake_view)
    monkeypatch.setattr(lab, "winsorize_returns", lambda ret, **kw: ret)
    levels = iter((0.0, 0.003, 0.001))  # none, ex_post, ex_ante

    def fake_premium(panel_, cfg):
        level = next(levels)
        ser = pd.Series(level + rng.normal(0, 1e-4, len(DATES)), index=DATES)
        return {"mean_pct": level * 100, "t_stat": 1.0, "n_months": len(DATES), "series": ser}
    monkeypatch.setattr(lab, "_premium", fake_premium)
    if with_out:
        out = tmp_path / "basis" / "mom6_lab_gate.json"
        argv = ["--basis", "clean", "--out", str(out)]
    else:                                   # --basis alone lands under the consistent-basis tree, never OUT
        out = tmp_path / "cb" / "clean" / "headlines" / "mom6_lab_gate.json"
        argv = ["--basis", "clean"]

    with pytest.raises(SystemExit):
        lab.main(argv)

    assert seen["rows"] == 48  # the tiny basis panel was the one read
    report = json.loads(out.read_text())
    assert report["basis"] == "clean"
    assert report["input_panel"] == str(panel)
    assert report["input_panel_sha256"] == B.sha256(panel)
    assert not (tmp_path / "default").exists()


# ---------------------------------------------------------------------------
# 4. run_accrual_validation — --total / --out
# ---------------------------------------------------------------------------

def test_accrual_flags_parse_and_default_paths():
    args = run_accrual_validation.parse_args([])
    assert args.total == DEFAULT_TR_PANEL
    assert args.out == DEFAULT_HEADLINES / "accrual_validation.json"
    tr = B.PANELS["total_return"][0]
    out = B.shared_dir("accrual_validation.json")
    args = run_accrual_validation.parse_args(["--total", str(tr), "--out", str(out)])
    assert args.total == tr and args.out == out


def test_accrual_main_reads_total_flag_and_records_provenance(monkeypatch, tmp_path):
    acc = run_accrual_validation
    clean = _tiny_panel(tmp_path / "panels" / "clean.parquet")
    total = _tiny_panel(tmp_path / "panels" / "total.parquet")
    for name, col in (("var_5pct", "var_5pct_corr"), ("gamma_illiq", "gamma_corr"), ("mom6", "mom6_corr")):
        _tiny_signal(tmp_path / "dev" / "signals" / f"{name}.parquet", col)
    monkeypatch.setattr(acc, "CLEAN", clean)
    monkeypatch.setattr(acc, "DEV", tmp_path / "dev")
    monkeypatch.setattr(acc, "OUT", tmp_path / "default" / "accrual_validation.json")
    monkeypatch.setattr(acc, "_bbw_panel", lambda df, sig: df[["cusip", "date"]].assign(rating=1.0))
    monkeypatch.setattr(acc, "bbw_means", lambda p: ({"drf": 0.1, "lrf": 0.1, "crf": 0.1}, {}))
    monkeypatch.setattr(acc, "compute_market_factor", lambda df, col: pd.DataFrame({"mktb": [0.001]}))
    monkeypatch.setattr(acc, "_assign_groups", lambda d, score, ident, n: pd.Series(4, index=d.index))
    monkeypatch.setattr(acc, "mom6_gap", lambda df, sig: 0.1)
    monkeypatch.setattr(acc, "leadlag_drf_corr", lambda comp: 0.3)
    out = tmp_path / "basis" / "accrual_validation.json"

    with pytest.raises(SystemExit):
        acc.main(["--total", str(total), "--out", str(out)])

    inputs = json.loads(out.read_text())["inputs"]
    assert inputs["total_panel"] == {"path": str(total), "sha256": B.sha256(total)}
    assert inputs["clean_panel"] == {"path": str(clean), "sha256": B.sha256(clean)}
    assert not (tmp_path / "default").exists()


def test_accrual_non_default_total_with_default_out_refused(monkeypatch, tmp_path):
    acc = run_accrual_validation
    monkeypatch.setattr(acc, "OUT", tmp_path / "default" / "accrual_validation.json")
    monkeypatch.setattr(acc, "CLEAN", tmp_path / "never_read.parquet")
    with pytest.raises(SystemExit) as exc:
        acc.main(["--total", str(tmp_path / "other_total.parquet")])
    assert exc.value.code == 2
    assert not (tmp_path / "default").exists()


def test_accrual_provenance_is_repo_relative_inside_repo():
    assert run_accrual_validation._rel(DEFAULT_TR_PANEL) == "data/development/monthly_panel_total_return.parquet"


# ---------------------------------------------------------------------------
# 5. run_anchor_descriptive — --total-return-panel / --out
# ---------------------------------------------------------------------------

def test_anchor_no_flag_resolves_default_paths():
    ad = run_anchor_descriptive
    args = ad.parse_args([])
    assert args.total_return_panel == DEFAULT_TR_PANEL and args.out is None
    assert ad.resolve_bases(args.total_return_panel) == ad.BASES == {
        "clean": CLEAN_PANEL, "total_return": DEFAULT_TR_PANEL}
    assert ad.default_out() == REPO / "results" / "quant" / "descriptive" / "run" / "anchor_descriptive.json"


def test_anchor_flags_resolve_basis_panel_and_out():
    ad = run_anchor_descriptive
    tr = B.PANELS["total_return"][0]
    out = B.shared_dir("anchor_descriptive.json")
    args = ad.parse_args(["--total-return-panel", str(tr), "--out", str(out)])
    bases = ad.resolve_bases(args.total_return_panel)
    assert bases == {"clean": B.PANELS["clean"][0], "total_return": tr}
    assert args.out == out


def test_anchor_holdout_override_refused():
    with pytest.raises(RuntimeError, match="holdout"):
        run_anchor_descriptive._guard_no_holdout(
            run_anchor_descriptive.resolve_bases(Path("data/holdout/panel.parquet")))


def test_anchor_main_records_both_panels_with_sha256(monkeypatch, tmp_path):
    ad = run_anchor_descriptive
    clean = _tiny_panel(tmp_path / "panels" / "clean.parquet")
    tr = _tiny_panel(tmp_path / "panels" / "tr_default_flat.parquet")
    monkeypatch.setitem(ad.BASES, "clean", clean)
    monkeypatch.setattr(ad, "MOM6_SIGNAL", _tiny_signal(tmp_path / "sig" / "mom6.parquet", "mom6_corr"))
    monkeypatch.setattr(ad, "VAR_SIGNAL", _tiny_signal(tmp_path / "sig" / "var.parquet", "var_5pct_corr"))
    monkeypatch.setattr(ad, "GAMMA_SIGNAL", _tiny_signal(tmp_path / "sig" / "gamma.parquet", "gamma_corr"))
    lvl = {fam: {"mean_pct_per_month": 0.1, "t_stat": 1.0, "n_months": 24} for fam in ad.FAMILIES}
    monkeypatch.setattr(ad, "run_bbw", lambda m, s: {"drf": lvl, "lrf": lvl, "crf": lvl})
    monkeypatch.setattr(ad, "run_mktb", lambda m: lvl)
    monkeypatch.setattr(ad, "run_str", lambda m: lvl)
    monkeypatch.setattr(ad, "run_mom6", lambda m, s: lvl)
    out = tmp_path / "basis" / "anchor_descriptive.json"

    ad.main(["--total-return-panel", str(tr), "--out", str(out)])

    report = json.loads(out.read_text())
    assert report["bases"] == {"clean": str(clean), "total_return": str(tr)}
    assert report["bases_sha256"] == {"clean": B.sha256(clean), "total_return": B.sha256(tr)}
