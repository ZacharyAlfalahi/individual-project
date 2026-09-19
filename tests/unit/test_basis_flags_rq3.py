"""--basis / --out / --factors-dir / --audit-reports flags on the RQ3 experiment drivers.

Pins, for each driver: flag parsing; that a given basis routes the inputs through
``basis_inputs.load_basis_inputs(basis)``; that no flag keeps the default loader and output paths; the
negative-control injection path on a tiny synthetic panel; and that the output JSON records the basis.
No real panel is loaded and no experiment runs: every loader, lattice and factor build is a sentinel.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import agents.auditor.ipca_differential.runner as RUNNER  # noqa: E402
import agents.auditor.validation.negative_control_run as NCR  # noqa: E402
import run_auditor as RA  # noqa: E402
import run_integration_arms as IA  # noqa: E402
import run_ipca_differential as IPCA  # noqa: E402
import run_lab_trim_invariance_gate as G  # noqa: E402
import run_rq3_controls as RC  # noqa: E402
import scripts.basis_inputs as SBI  # noqa: E402  (the module every driver imports)

_PROV = SBI.BasisProvenance("total_return", {"data/development/p.parquet": "abc"})


class _Stop(Exception):
    """Raised by a sentinel once the loader call has been observed — nothing downstream runs."""


@pytest.fixture
def calls(monkeypatch):
    """Sentinel loaders: record which loader was called (and with which basis); no panel is read."""
    log: list = []
    maximal, signals, reg = pd.DataFrame({"m": [1]}), pd.DataFrame({"s": [1]}), {"r": 1}

    def fake_basis(basis):
        log.append(("basis", basis))
        return maximal, signals, reg

    def fake_dev():
        log.append(("dev", None))
        return maximal, signals, reg

    def fake_prov(basis):
        log.append(("prov", basis))
        return SBI.BasisProvenance(basis, dict(_PROV.panels))

    monkeypatch.setattr(SBI, "load_basis_inputs", fake_basis)
    monkeypatch.setattr(SBI, "basis_provenance", fake_prov)
    monkeypatch.setattr(RUNNER, "load_dev_inputs", fake_dev)
    for mod in (RA, G, IPCA):
        monkeypatch.setattr(mod, "load_dev_inputs", fake_dev)
    return log


def test_clean_provenance_panels_are_the_ones_the_dev_loader_reads():
    assert SBI.PANELS["clean"] == (RUNNER.MAXIMAL_PANEL, RUNNER.DEV / "monthly_panel_profiles.parquet")


def test_all_drivers_share_one_basis_inputs_module():
    assert RC.basis_inputs is SBI and G.basis_inputs is SBI and IA.basis_inputs is SBI
    assert IPCA.basis_inputs is SBI and RA.basis_inputs is SBI


# --- negative_control_run: the injection path -------------------------------------------------

def _lrf_frame(rng, n=240, bias=0.0):
    dates = pd.date_range("2002-01-31", periods=n, freq="ME")
    return pd.DataFrame({"date": dates, "strategy_ret": rng.normal(0.002 + bias, 0.01, n),
                         "n_bonds": np.full(n, 50)})


def test_injected_maximal_and_signals_bypass_the_dev_loader(monkeypatch, calls):
    rng = np.random.default_rng(7)
    frames = {"raw": _lrf_frame(rng), "corr": _lrf_frame(rng)}
    maximal = pd.DataFrame({"cusip": ["A"], "date": pd.to_datetime(["2020-01-31"]), "ret_raw": [0.01]})
    signals = pd.DataFrame({"cusip": ["A"], "date": pd.to_datetime(["2020-01-31"]),
                            "gamma_illiq_raw": [1.0], "gamma_illiq_corr": [2.0], "mom6_raw": [0.0]})
    seen = []

    def fake_lrf(m, s, fam):
        seen.append((m, list(s.columns), fam))
        return frames[fam]

    monkeypatch.setattr(NCR, "lrf_family_returns", fake_lrf)
    r = NCR.run_negative_control_from_dev(seed=0, maximal=maximal, signals=signals,
                                          return_basis="total_return")
    assert ("dev", None) not in calls                       # the clean dev loader never ran
    assert [f for _, _, f in seen] == ["raw", "corr"]
    assert all(m is maximal for m, _, _ in seen)
    assert seen[0][1] == ["cusip", "date", "gamma_raw", "gamma_corr"]   # only gamma, renamed
    assert r["return_basis"] == "total_return"
    assert r["estimand"]["panel"] == "maximal_total_return_dev"
    assert r["basis"] == "real_dev_data" and isinstance(r["passed"], bool)
    assert r["bootstrap"]["n_months_common"] == 240


def test_no_injection_uses_the_dev_loader_and_keeps_the_default_result_shape(monkeypatch, calls):
    rng = np.random.default_rng(8)
    frames = {"raw": _lrf_frame(rng), "corr": _lrf_frame(rng)}
    sig = pd.DataFrame({"cusip": ["A"], "date": pd.to_datetime(["2020-01-31"]),
                        "gamma_illiq_raw": [1.0], "gamma_illiq_corr": [2.0]})
    monkeypatch.setattr(RUNNER, "load_dev_inputs", lambda: (calls.append(("dev", None)) or (None, sig, {})))
    monkeypatch.setattr(NCR, "lrf_family_returns", lambda m, s, fam: frames[fam])
    r = NCR.run_negative_control_from_dev(seed=0)
    assert calls == [("dev", None)]
    assert "return_basis" not in r and r["estimand"]["panel"] == "maximal_clean_dev"


def test_half_injection_refused():
    with pytest.raises(NCR.NegativeControlRunError, match="together"):
        NCR.run_negative_control_from_dev(maximal=pd.DataFrame())


# --- run_rq3_controls ----------------------------------------------------------------------------

def _gate_report():
    arm = {"corr_correct_vs_defective": 0.25, "corr_correct_vs_restored": 1.0}
    return {"run_timestamp": "t", "arms": {k: dict(arm) for k in ("drf", "crf", "lrf")}}


def _neg_result():
    return {"control": "negative_control_specificity", "factor": "traded_liquidity",
            "basis": "real_dev_data", "status": "fired", "ci_low": -1e-4, "ci_high": 4e-4,
            "vartheta": 1e-3, "passed": True, "absolute_gap": 4e-4, "point_gap_mean": 1e-4,
            "bootstrap": {"n_months_common": 233, "n_replicates": 1000}}


def _factors(tmp_path: Path) -> Path:
    d = tmp_path / "factors"
    d.mkdir()
    pd.DataFrame({"lrf_raw": [0.01, 0.02], "lrf_corr": [0.02, 0.02]}).to_parquet(d / "bbw_factors.parquet")
    return d


def test_rq3_negative_control_basis_injects_and_records(monkeypatch, calls, tmp_path):
    seen = {}
    monkeypatch.setattr(RC, "run_negative_control_from_dev",
                        lambda **kw: (seen.update(kw) or _neg_result()))
    r = RC.negative_control(basis="clean", factors_dir=_factors(tmp_path))
    assert ("basis", "clean") in calls and ("dev", None) not in calls
    assert seen["return_basis"] == "clean" and seen["maximal"] is not None and seen["signals"] is not None
    assert r["basis_provenance"]["basis"] == "clean"
    s = r["sensitivity_total_return"]
    assert s["basis"] == "clean" and s["n_months"] == 2 and s["point_gap_mean"] == pytest.approx(0.005)
    assert s["factors_file"].endswith("bbw_factors.parquet") and len(s["factors_sha256"]) == 64


def test_rq3_negative_control_default_uses_the_default_inputs(monkeypatch, calls):
    seen = {}
    monkeypatch.setattr(RC, "run_negative_control_from_dev",
                        lambda **kw: (seen.update(kw) or _neg_result()))
    monkeypatch.setattr(RC, "total_return_sensitivity", lambda *a, **k: ("default", a, k))
    r = RC.negative_control()
    assert set(seen) == {"seed", "thresholds_path"}                 # no injection kwargs
    assert not [c for c in calls if c[0] in ("basis", "prov")]
    assert "basis_provenance" not in r
    assert r["sensitivity_total_return"] == ("default", (), {})    # default factors file, default call


def test_rq3_build_controls_forwards_only_given_kwargs():
    got = []
    RC.build_controls(_gate_report(), negative_control_fn=lambda **kw: got.append(kw) or _neg_result())
    RC.build_controls(_gate_report(), basis="total_return", factors_dir=Path("x"),
                      negative_control_fn=lambda **kw: got.append(kw) or _neg_result())
    assert got[0] == {"thresholds_path": None}
    assert got[1] == {"thresholds_path": None, "basis": "total_return", "factors_dir": Path("x")}


def test_rq3_default_out_paths():
    assert RC.default_out() == REPO_ROOT / "results" / "auditor" / "rq3_controls.json"
    assert RC.default_out("total_return") == SBI.RESULTS_ROOT / "total_return" / "auditor" / \
        "rq3_controls.json"


def test_rq3_main_basis_end_to_end(monkeypatch, calls, tmp_path):
    captured = {}

    def fake_neg(**kw):
        captured.update(kw)
        return _neg_result()

    monkeypatch.setattr(RC, "negative_control", fake_neg)
    lead = tmp_path / "leadlag.json"
    lead.write_text(json.dumps(_gate_report()))
    out = tmp_path / "o.json"
    assert RC.main(["--leadlag", str(lead), "--out", str(out), "--basis", "total_return"]) == 0
    assert captured["basis"] == "total_return"
    assert captured["factors_dir"] == SBI.factors_dir("total_return")   # basis factors dir by default
    r = json.loads(out.read_text())
    assert r["provenance"]["basis"] == "total_return"

    captured.clear()
    fdir = tmp_path / "f"
    assert RC.main(["--leadlag", str(lead), "--out", str(out), "--factors-dir", str(fdir)]) == 0
    assert "basis" not in captured and captured["factors_dir"] == fdir
    assert "basis" not in json.loads(out.read_text())["provenance"]


def test_rq3_default_leadlag_paths():
    assert RC.default_leadlag() == RC._LEADLAG
    assert RC.default_leadlag("clean") == SBI.RESULTS_ROOT / "clean" / "headlines" / "leadlag_gate.json"


def test_rq3_main_basis_reads_its_own_leadlag_gate_and_refuses_the_default_gate(monkeypatch, calls, tmp_path):
    monkeypatch.setattr(RC, "negative_control", lambda **kw: _neg_result())
    monkeypatch.setattr(SBI, "RESULTS_ROOT", tmp_path / "cb")
    gate = SBI.basis_dir("total_return", "headlines", "leadlag_gate.json")
    gate.parent.mkdir(parents=True)
    gate.write_text(json.dumps(_gate_report()))
    out = tmp_path / "o.json"
    assert RC.main(["--basis", "total_return", "--out", str(out)]) == 0        # no --leadlag given
    prov = json.loads(out.read_text())["provenance"]
    assert prov["leadlag_gate"] == str(gate) and prov["leadlag_gate_sha256"] == SBI.sha256(gate)

    default_gate = tmp_path / "default_leadlag.json"
    default_gate.write_text(json.dumps(_gate_report()))
    monkeypatch.setattr(RC, "_LEADLAG", default_gate)
    with pytest.raises(RC.RQ3ControlsError, match="default lead/lag gate"):
        RC.main(["--basis", "total_return", "--leadlag", str(default_gate), "--out", str(out)])
    assert RC.main(["--leadlag", str(default_gate), "--out", str(out)]) == 0   # no basis: default input


def test_rq3_main_without_flags_forwards_nothing(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(RC, "negative_control", lambda **kw: captured.update(kw) or _neg_result())
    lead = tmp_path / "leadlag.json"
    lead.write_text(json.dumps(_gate_report()))
    assert RC.main(["--leadlag", str(lead), "--out", str(tmp_path / "o.json")]) == 0
    assert captured == {"thresholds_path": None}


def test_rq3_rejects_unknown_basis(tmp_path):
    with pytest.raises(SystemExit):
        RC.main(["--basis", "tr", "--out", str(tmp_path / "o.json")])


# --- run_lab_trim_invariance_gate ---------------------------------------------------------------

class _FakeCore:
    def __init__(self, cd):
        self._cd = cd

    def to_dict(self):
        return self._cd


def _core_dict():
    bases = {b: {"∅": 0.01, "lab_trim": 0.0} for b in ("harsanyi_dividends", "walsh_coefficients", "doe_effects")}
    return {"saturated_bases": bases, "shapley": {"shapley_values": {"lab_trim": 0.0}},
            "invariance": [{"toggle_id": "lab_trim", "is_no_op": True, "note": "n"}]}


@pytest.fixture
def lab_trim_stubs(monkeypatch):
    monkeypatch.setattr(G, "load_anchor_strategy", lambda a: f"strategy:{a}")
    monkeypatch.setattr(G, "load_anchor_expost_trim_off", lambda a: None)
    monkeypatch.setattr(G, "load_anchor_meas_err_off_family", lambda a: "raw")
    monkeypatch.setattr(G, "audit_anchor", lambda *a, **k: _FakeCore(_core_dict()))


def _audit_reports(tmp_path: Path, per_anchor: bool) -> Path:
    for a in ("str", "drf"):
        d = tmp_path / (f"full_run_{a}_tr" if per_anchor else "reports")
        d.mkdir(exist_ok=True)
        (d / f"{a}_report.json").write_text(json.dumps(_core_dict()))
    return tmp_path / ("full_run_{anchor}_tr" if per_anchor else "reports")


def test_lab_trim_basis_uses_basis_loader_and_records_it(lab_trim_stubs, calls, tmp_path):
    reports = _audit_reports(tmp_path, per_anchor=True)
    out = tmp_path / "gate.json"
    art = G.main(["--basis", "total_return", "--audit-reports", str(reports), "--out", str(out)])
    assert ("basis", "total_return") in calls and ("dev", None) not in calls
    written = json.loads(out.read_text())
    assert written["basis"] == "total_return" == art["basis"]
    assert written["basis_provenance"]["basis"] == "total_return"
    assert written["audit_reports"] == str(reports)
    assert written["anchors"]["drf"]["audit_report"]["path"].endswith("full_run_drf_tr/drf_report.json")


def test_lab_trim_default_loader_and_record(lab_trim_stubs, calls, tmp_path):
    out = tmp_path / "gate.json"
    reports = _audit_reports(tmp_path, per_anchor=False)
    art = G.main(["--audit-reports", str(reports), "--out", str(out)])
    assert calls == [("dev", None)]
    assert art["basis"] == "clean" and "basis_provenance" not in art


def test_lab_trim_default_paths():
    assert G.default_out() == REPO_ROOT / "results" / "auditor" / "lab_trim_invariance_gate.json"
    assert G.RECORDED_AUDIT == REPO_ROOT / "results" / "auditor" / "{anchor}_corrected"
    assert G.default_out("clean") == SBI.RESULTS_ROOT / "clean" / "auditor" / "lab_trim_invariance_gate.json"
    assert G.report_path(G.RECORDED_AUDIT, "str") == \
        REPO_ROOT / "results" / "auditor" / "str_corrected" / "str_report.json"
    assert G.report_path(Path("r/full_run_{anchor}_x"), "drf") == Path("r/full_run_drf_x/drf_report.json")


def test_lab_trim_total_return_requires_its_own_audit_reports(calls):
    with pytest.raises(SystemExit):
        G.main(["--basis", "total_return"])
    assert calls == []                                        # refused before any load


# --- run_integration_arms -----------------------------------------------------------------------

def test_arm_b_loader_dispatch(monkeypatch, calls):
    def stop(maximal, signals):
        calls.append(("prep", None))
        raise _Stop

    monkeypatch.setattr(IA, "_prep_panel", stop)
    with pytest.raises(_Stop):
        IA.run_integrated_leadlag_positive(basis="clean")
    with pytest.raises(_Stop):
        IA.run_integrated_leadlag_positive()
    assert calls == [("basis", "clean"), ("prep", None), ("dev", None), ("prep", None)]


def test_integration_arms_main_basis_and_out(monkeypatch, tmp_path):
    got = []
    fake = {"clean_variant_null": {"passed": True},
            "integrated_known_error_positive_control": {"passed": True, "detail": "d", "basis": "total_return"}}
    monkeypatch.setattr(IA, "run_integration_arms", lambda **kw: got.append(kw) or fake)
    out = tmp_path / "arms.json"
    assert IA.main(["--replicates", "5", "--basis", "total_return", "--out", str(out)]) == 0
    assert got == [{"replicates": 5, "basis": "total_return"}]
    assert json.loads(out.read_text())["integrated_known_error_positive_control"]["basis"] == "total_return"
    assert IA.main(["--replicates", "5", "--out", str(out)]) == 0
    assert got[1] == {"replicates": 5}


def test_integration_arms_default_out():
    assert IA.default_out() == REPO_ROOT / "results" / "scientist" / "integration_arms.json"
    assert IA.default_out("total_return") == SBI.RESULTS_ROOT / "total_return" / "scientist" / "integration_arms.json"


# --- run_ipca_differential ----------------------------------------------------------------------

class _NoPairs:
    smoke_pair = ("meas_err", "drf")

    def runnable_pairs(self):
        return []

    def refused_pairs(self):
        return []


def test_ipca_run_all_basis_loader_and_provenance(monkeypatch, calls, tmp_path):
    monkeypatch.setattr(IPCA, "load_ipca_execution_pairs", lambda: _NoPairs())
    char_calls = []
    real_char = IPCA._characteristics_maximal

    def fake_char(basis):
        char_calls.append(basis)
        return "CLEAN_MAXIMAL" if basis == "total_return" else None
    monkeypatch.setattr(IPCA, "_characteristics_maximal", fake_char)
    assert IPCA.run_all(tmp_path / "tr", basis="total_return") == 0
    assert ("basis", "total_return") in calls and ("dev", None) not in calls
    assert char_calls == ["total_return"]
    log = json.loads((tmp_path / "tr" / "results_9pairs.json").read_text())["run_log"]
    assert log["basis_provenance"]["basis"] == "total_return"
    assert "xret" in log["basis_instrument_note"] and "clean-price panel" in log["basis_instrument_note"]
    assert "instrument_panels" in log                                           # clean panels recorded
    assert log["bootstrap_seed"] == 20260612 and log["stability_seed"] == 20260612   # registered seeds, independent of basis
    # the clean-instrument panel is only ever loaded on the total-return basis
    assert real_char(None) is None and real_char("clean") is None

    calls.clear()
    char_calls.clear()
    assert IPCA.run_all(tmp_path / "dflt") == 0
    assert calls == [("dev", None)]
    assert char_calls == [None]
    log = json.loads((tmp_path / "dflt" / "results_9pairs.json").read_text())["run_log"]
    assert "basis_provenance" not in log and "basis_instrument_note" not in log


def test_ipca_default_out_dir():
    assert IPCA.default_out_dir() == REPO_ROOT / "results" / "ipca_differential" / "run"
    assert IPCA.default_out_dir("clean") == SBI.RESULTS_ROOT / "clean" / "ipca_differential"


def test_ipca_main_flags(monkeypatch, tmp_path):
    got = []
    monkeypatch.setattr(IPCA, "run_all", lambda *a, **k: got.append(("all", a, k)) or 0)
    monkeypatch.setattr(IPCA, "run_smoke", lambda *a, **k: got.append(("smoke", a, k)) or 0)
    for argv in (["--all", "--basis", "total_return", "--out", str(tmp_path)], ["--all"],
                 ["--smoke", "--basis", "clean"], ["--smoke"]):
        with pytest.raises(SystemExit):
            IPCA.main(argv)
    assert got == [("all", (tmp_path,), {"basis": "total_return"}), ("all", (None,), {}),
                   ("smoke", (), {"basis": "clean"}), ("smoke", (), {})]
    with pytest.raises(SystemExit):
        IPCA.main(["--smoke", "--out", str(tmp_path)])            # --out is --all only
    assert len(got) == 4


def test_ipca_default_total_return_panel_carries_every_view_column():
    """The IPCA feed consumes view() output, which needs only these panel columns; the default-flat
    TR panel must carry each (schema-only read — no rows)."""
    import pyarrow.parquet as pq

    from agents.quant.library.views import _FAMILY_INDEXED_PANEL_COLUMNS

    panel = SBI.PANELS["total_return"][0]
    if not panel.is_file():
        pytest.skip("requires the licensed default-flat total-return panel (data/development/) — "
                    "not shipped; see README")
    cols = set(pq.read_schema(panel).names)
    need = {f"{b}_{fam}" for b in _FAMILY_INDEXED_PANEL_COLUMNS for fam in ("raw", "corr")}
    need |= {"cusip", "date", "size", "rf_monthly", "rating", "time_to_maturity", "exit_reason", "universe_eligible"}
    assert need <= cols, sorted(need - cols)


# --- run_auditor --------------------------------------------------------------------------------

def test_run_auditor_basis_loader_log_and_out(monkeypatch, calls, tmp_path):
    rec = {"anchor": "str", "audit_scope": "s", "core_sync_1": {"passed": True}, "stale_invariance": None,
           "core": {"k": 1}}
    monkeypatch.setattr(RA, "run_anchor", lambda *a, **k: dict(rec))
    assert RA.run_all(["str"], out_dir=tmp_path / "tr", basis="total_return") == 0
    assert ("basis", "total_return") in calls and ("dev", None) not in calls
    log = json.loads((tmp_path / "tr" / "run_log.json").read_text())
    assert log["basis_provenance"]["basis"] == "total_return"

    calls.clear()
    assert RA.run_all(["str"], out_dir=tmp_path / "dflt") == 0
    assert calls == [("dev", None)]
    assert "basis_provenance" not in json.loads((tmp_path / "dflt" / "run_log.json").read_text())

    assert RA.default_out_dir() == REPO_ROOT / "results" / "auditor" / "full_run"
    assert RA.default_out_dir("clean") == SBI.RESULTS_ROOT / "clean" / "auditor" / "full_run"


def test_run_auditor_main_flags(monkeypatch):
    got = []
    monkeypatch.setattr(RA, "run_all", lambda anchors, **k: got.append((tuple(anchors), k)) or 0)
    assert RA.main(["--anchor", "drf", "--basis", "clean"]) == 0
    assert RA.main(["--anchor", "drf"]) == 0
    assert got[0] == (("drf",), {"out_dir": None, "tol": RA.DEFAULT_TOL, "basis": "clean"})
    assert got[1] == (("drf",), {"out_dir": None, "tol": RA.DEFAULT_TOL})


def test_ipca_run_all_passes_clean_instruments_to_each_runnable_pair(monkeypatch, calls, tmp_path):
    """Complements test_ipca_run_all_basis_loader_and_provenance (which has no runnable pair): on the
    total-return basis every pair receives the clean characteristics panel; without a basis it receives none."""
    class _OnePair(_NoPairs):
        def runnable_pairs(self):
            return [("survivorship", "mom6")]

    seen = {}

    def fake_pair(bias, anchor, maximal, signals, reg, characteristics_maximal=None):
        seen.update(bias=bias, anchor=anchor, cm=characteristics_maximal)
        raise _Stop()

    monkeypatch.setattr(IPCA, "load_ipca_execution_pairs", lambda: _OnePair())
    monkeypatch.setattr(IPCA, "_characteristics_maximal",
                        lambda basis: "CLEAN_MAXIMAL" if basis == "total_return" else None)
    monkeypatch.setattr(RUNNER, "run_pair_full", fake_pair)
    with pytest.raises(_Stop):
        IPCA.run_all(tmp_path / "tr", basis="total_return")
    assert seen == {"bias": "survivorship", "anchor": "mom6", "cm": "CLEAN_MAXIMAL"}
    with pytest.raises(_Stop):
        IPCA.run_all(tmp_path / "dflt")
    assert seen["cm"] is None
