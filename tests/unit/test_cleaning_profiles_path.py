"""
End-to-end tests for the as-published cleaning profiles through the SHARED
preprocess_trace engine (FL-D21a: one code path, a profile is a config).

A single synthetic raw CSV exercises every profile-differentiating mechanism:

  A  pre-2012 correction pair    — bbw keeps the C replacement (keep_replacement_
                                   pre2012), jostova/raw delete both (BKMX rung-2)
  B  post-2012 C record          — a CANCELLATION in every mode (G3 regime shift)
  C  pre-2012 asof-'R' reversal  — profiles net out the ORIGINAL (P3, FL-D21d);
                                   raw drops only the reversal record (as-built)
  D  asof-'A' late execution     — retained by profiles (P4, FL-D21h), dropped by raw
  E  when-issued                 — dropped by bbw (STATED), kept by jostova (R2)
  F  BBW screens                 — $4 price and $9,999 volume dropped by bbw only
  G  Jostova screens             — commission 'Y' and negative price dropped
  H  interdealer B/S pair        — dedup ON drops the buy; dedup OFF keeps both (E1)
  Z  clean row                   — survives everywhere

The raw profile's behaviour is pinned separately by test_preprocess_trace.py;
here it serves as the as-built comparator.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import preprocess_trace as pt
from preprocess_trace import RAW_PROFILE, build_profile, run_pandas

_CFG = {"asof_cd_keep": "", "holdout_start_year": 2022, "holdout_end_year": 2025}

_COLUMNS = [
    "bond_sym_id", "cusip_id", "company_symbol", "trd_exctn_dt", "trd_exctn_tm",
    "rptd_pr", "entrd_vol_qt", "sub_prdct", "rpt_side_cd", "trdg_mkt_cd",
    "trd_mod_3", "bloomberg_identifier", "scrty_type_cd",
    "msg_seq_nb", "orig_msg_seq_nb", "trc_st", "asof_cd", "wis_fl",
    "cmsn_trd", "pr_trd_dt", "lckd_in_ind", "spcl_trd_fl", "sale_cndtn_cd",
    "days_to_sttl_ct",
]


def _row(bond, date, msg, *, pr, vol, trc="T", orig="", asof="", wis="N",
         side="S", cmsn="N", pr_trd="", lckd="", spcl="", cndtn="@", sttl="1"):
    return {
        "bond_sym_id": bond, "cusip_id": f"CUSIP{bond}", "company_symbol": bond,
        "trd_exctn_dt": date, "trd_exctn_tm": "10:00:00",
        "rptd_pr": pr, "entrd_vol_qt": vol, "sub_prdct": "CORP",
        "rpt_side_cd": side, "trdg_mkt_cd": "S1", "trd_mod_3": "",
        "bloomberg_identifier": f"BB{bond}", "scrty_type_cd": "C",
        "msg_seq_nb": msg, "orig_msg_seq_nb": orig, "trc_st": trc,
        "asof_cd": asof, "wis_fl": wis, "cmsn_trd": cmsn, "pr_trd_dt": pr_trd,
        "lckd_in_ind": lckd, "spcl_trd_fl": spcl, "sale_cndtn_cd": cndtn,
        "days_to_sttl_ct": sttl,
    }


def _make_raw_csv(path: Path) -> None:
    rows = [
        # A — pre-2012 correction pair (keep-replacement discriminator)
        _row("BA", "2010-03-01", "A1", pr=100.0, vol=11000.0),
        _row("BA", "2010-03-01", "A2", pr=101.0, vol=11000.0, trc="C", orig="A1"),
        # B — post-2012 C record = cancellation in every mode
        _row("BB", "2015-03-02", "B1", pr=100.0, vol=12000.0),
        _row("BB", "2015-03-02", "B2", pr=100.0, vol=12000.0, trc="C", orig="B1"),
        # C — pre-2012 asof-'R' reversal; original value-matched via pr_trd_dt
        _row("BC", "2010-04-01", "C1", pr=98.0, vol=13000.0),
        _row("BC", "2010-05-01", "C2", pr=98.0, vol=13000.0, asof="R",
             pr_trd="2010-04-01"),
        # D — asof-'A' late execution
        _row("BD", "2015-04-01", "D1", pr=99.0, vol=14000.0, asof="A"),
        # E — when-issued
        _row("BE", "2015-05-01", "E1", pr=97.0, vol=15000.0, wis="Y"),
        # F — BBW screens (price floor, volume floor)
        _row("BF", "2015-06-01", "F1", pr=4.0, vol=16000.0),
        _row("BF", "2015-06-02", "F2", pr=100.0, vol=9999.0),
        # G — Jostova screens (commission, data-entry)
        _row("BG", "2010-07-01", "G1", pr=96.0, vol=17000.0, cmsn="Y"),
        _row("BG", "2015-07-02", "G2", pr=-1.0, vol=17500.0),
        # H — interdealer B/S same-key pair (E1 envelope discriminator)
        _row("BH", "2015-08-03", "H1", pr=95.0, vol=18000.0, side="S"),
        _row("BH", "2015-08-03", "H2", pr=95.0, vol=18000.0, side="B"),
        # I — asof 'D' row (P5): dropped by every default; retained only by the
        # p5_asof_retain OFAT variant
        _row("BI", "2015-10-01", "I1", pr=94.0, vol=21000.0, asof="D"),
        # Z — clean row
        _row("BZ", "2015-09-01", "Z1", pr=102.0, vol=20000.0),
    ]
    pd.DataFrame(rows, columns=_COLUMNS).to_csv(path, index=False, compression="gzip")


@pytest.fixture
def raw_csv(tmp_path, monkeypatch):
    raw = tmp_path / "trace_enhanced_repull.csv.gz"
    _make_raw_csv(raw)
    monkeypatch.setattr(pt, "RAW_FILE", raw)
    return tmp_path


def _run(profile, tmp_path):
    dev = pd.read_parquet(profile.dev_out) if profile.dev_out.exists() else None
    return dev


def _run_profile(tmp_path, profile_id=None, *, dedup=True, monkeypatch=None):
    if profile_id is None:
        # raw path: redirect the module globals (as the existing harness does)
        monkeypatch.setattr(pt, "DEV_OUT", tmp_path / "dev_raw.parquet")
        monkeypatch.setattr(pt, "HOLD_OUT", tmp_path / "hold_raw.parquet")
        counts = run_pandas(_CFG)
        dev = pd.read_parquet(tmp_path / "dev_raw.parquet")
        return counts, dev
    tag = "on" if dedup else "off"
    profile = build_profile(
        profile_id, dedup=dedup,
        dev_out=tmp_path / f"dev_{profile_id}_{tag}.parquet",
        hold_out=tmp_path / f"hold_{profile_id}_{tag}.parquet",
        report_out=tmp_path / f"report_{profile_id}_{tag}.json",
    )
    counts = run_pandas(_CFG, profile)
    dev = pd.read_parquet(profile.dev_out)
    return counts, dev


def _msgs(dev: pd.DataFrame) -> set:
    # msg_seq_nb is not in the output schema; identify rows by (bond, price).
    return set(zip(dev["bond_id"], dev["rptd_pr"]))


def test_raw_as_built_comparator(raw_csv, monkeypatch):
    counts, dev = _run_profile(raw_csv, None, monkeypatch=monkeypatch)
    kept = _msgs(dev)
    # As-built: correction pairs deleted both; reversal ORIGINAL retained (no
    # netting); asof A dropped; wis dropped; no profile screens; dedup on.
    assert ("BC", 98.0) in kept                      # C1 retained (no net-out)
    assert ("BD", 99.0) not in kept                  # asof A dropped
    assert ("BA", 101.0) not in kept                 # C record dropped
    assert ("BF", 4.0) in kept and ("BG", -1.0) in kept   # no screens
    assert counts["dropped_reversal_netout"] == 0
    assert counts["dropped_profile_screens"] == 0
    assert counts["dropped_interdealer_duplicate"] == 1   # H pair deduped
    assert len(dev) == 7


def test_bbw_2019_profile(raw_csv, monkeypatch):
    counts, dev = _run_profile(raw_csv, "bbw_2019", dedup=True)
    kept = _msgs(dev)
    # Keep-replacement: the pre-2012 corrected value survives, original dropped.
    assert ("BA", 101.0) in kept and ("BA", 100.0) not in kept
    # Post-2012 C is a cancellation: both rows gone.
    assert not any(b == "BB" for b, _ in kept)
    # Reversal netted out; asof A retained; wis dropped; screens applied.
    assert not any(b == "BC" for b, _ in kept)       # C1 netted, C2 dropped
    assert ("BD", 99.0) in kept                      # P4 retain
    assert not any(b == "BE" for b, _ in kept)       # wis STATED
    assert ("BF", 4.0) not in kept                   # price floor
    assert ("BF", 100.0) not in kept                 # volume floor
    assert ("BG", 96.0) in kept                      # bbw keeps commission rows
    assert ("BG", -1.0) not in kept                  # price-range screen
    assert ("BH", 95.0) in kept                      # H1 survives...
    assert counts["dropped_interdealer_duplicate"] == 1   # ...H2 deduped
    assert counts["dropped_reversal_netout"] == 1
    assert counts["dropped_profile_screens"] == 3    # F1, F2, G2
    assert len(dev) == 5                             # A2, D1, G1, H1, Z1


def test_jostova_2013_profile_dedup_off(raw_csv, monkeypatch):
    counts, dev = _run_profile(raw_csv, "jostova_2013", dedup=False)
    kept = _msgs(dev)
    # Delete-both corrections (BKMX rung-2): all of A and B gone.
    assert not any(b in ("BA", "BB") for b, _ in kept)
    # Reversal netted; asof A retained; when-issued KEPT (R2 not applied).
    assert not any(b == "BC" for b, _ in kept)
    assert ("BD", 99.0) in kept
    assert ("BE", 97.0) in kept
    # No price range: the $4 print SURVIVES (FL-D21 D3); commission + negative
    # price dropped by the Jostova screens.
    assert ("BF", 4.0) in kept and ("BF", 100.0) in kept
    assert ("BG", 96.0) not in kept and ("BG", -1.0) not in kept
    # E1 dedup OFF: BOTH sides of the H pair kept.
    assert sum(1 for b, _ in zip(dev["bond_id"], dev["rptd_pr"]) if b == "BH") == 2
    assert counts["dropped_interdealer_duplicate"] == 0
    assert counts["dropped_reversal_netout"] == 1
    assert counts["dropped_profile_screens"] == 2    # G1, G2
    assert len(dev) == 7                             # D1, E1, F1, F2, H1, H2, Z1


def test_jostova_2013_profile_dedup_on(raw_csv, monkeypatch):
    counts, dev = _run_profile(raw_csv, "jostova_2013", dedup=True)
    # Same as dedup-off but the H buy is dropped: the E1 envelope differs by
    # exactly the dedup axis.
    assert counts["dropped_interdealer_duplicate"] == 1
    assert len(dev) == 6


def _run_variant(tmp_path, profile_id, variant):
    import dataclasses

    from preprocess_trace import build_variant_profile
    prof = build_variant_profile(profile_id, variant)
    prof = dataclasses.replace(
        prof,
        dev_out=tmp_path / f"dev_{profile_id}_{variant}.parquet",
        hold_out=tmp_path / f"hold_{profile_id}_{variant}.parquet",
        report_out=tmp_path / f"report_{profile_id}_{variant}.json",
    )
    counts = run_pandas(_CFG, prof)
    dev = pd.read_parquet(prof.dev_out)
    return counts, dev


def test_ofat_variants_flip_exactly_one_dimension(raw_csv, monkeypatch):
    # Baseline bbw keeps: A2, D1, G1, H1, Z1 (see test_bbw_2019_profile).
    # p4_asof_drop: ONLY the asof-'A' row (D1) disappears.
    _, dev = _run_variant(raw_csv, "bbw_2019", "p4_asof_drop")
    kept = _msgs(dev)
    assert ("BD", 99.0) not in kept
    assert ("BA", 101.0) in kept and ("BG", 96.0) in kept and ("BZ", 102.0) in kept
    assert len(dev) == 4

    # p5_asof_retain: ONLY the asof-'D' row (I1) appears.
    _, dev = _run_variant(raw_csv, "bbw_2019", "p5_asof_retain")
    kept = _msgs(dev)
    assert ("BI", 94.0) in kept
    assert len(dev) == 6

    # commission flip on bbw: ONLY the cmsn-'Y' row (G1) disappears.
    counts, dev = _run_variant(raw_csv, "bbw_2019", "commission")
    kept = _msgs(dev)
    assert ("BG", 96.0) not in kept
    assert len(dev) == 4
    assert counts["dropped_profile_screens"] == 4       # F1, F2, G2 + now G1

    # min_volume flip on jostova: the $9,999 row (F2) disappears; the $4 print
    # (F1) still SURVIVES (price range remains unapplied); dedup at ON reference
    # drops the H buy.
    counts, dev = _run_variant(raw_csv, "jostova_2013", "min_volume")
    kept = _msgs(dev)
    assert ("BF", 100.0) not in kept and ("BF", 4.0) in kept
    assert counts["dropped_interdealer_duplicate"] == 1
    assert len(dev) == 5                                # D1, E1, F1, H1, Z1


def test_unknown_profile_rejected():
    from preprocess_trace import build_variant_profile
    with pytest.raises(KeyError):
        build_profile("nope", dedup=True)
    with pytest.raises(KeyError):
        build_variant_profile("bbw_2019", "when_issued")   # not promoted
    with pytest.raises(KeyError):
        build_variant_profile("jostova_2013", "commission")  # bbw-only variant


def test_raw_profile_is_default_config():
    # The raw profile must encode the as-built posture exactly.
    assert RAW_PROFILE.correction_mode == "delete_both"
    assert RAW_PROFILE.retain_asof is False
    assert RAW_PROFILE.net_reversals is False
    assert RAW_PROFILE.apply_wis is True
    assert RAW_PROFILE.dedup is True
    assert RAW_PROFILE.screens is None
