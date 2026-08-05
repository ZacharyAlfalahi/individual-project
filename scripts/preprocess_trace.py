"""
One-time TRACE Dick-Nielsen cleaning and dev/holdout split.

Phase 1 of the bias-toggle registry pipeline. This script performs ONLY the
basic-cleaning filters that are common to both column families (raw and
corrected) per Section 2 of the registry spec and A1 of the amendments:

  1a. trc_st == "T"                 — drop cancelled/reversed/withdrawn records
  1b. msg_seq_nb not in cancelled   — drop T records cancelled by a later C/W/X/Y/R
  2.  asof_cd == "" (blank)         — keep only on-time original reports; drops A/R/D/X
  3.  wis_fl != "Y"                 — drop when-issued trades
  6.  interdealer dedup             — Dick-Nielsen Algorithm B2, on raw prices

What this script DOES NOT do (these are meas_err-gated and live downstream):

  4.  price plausibility            — relocated to apply_decimal_shift.py
  5.  decimal-shift correction      — relocated to apply_decimal_shift.py
  7.  bounce-back filter            — bounce_back_filter.py (corrected branch only)
  8.  distressed daily filters      — apply_distressed_filters.py (corrected daily branch)

The dedup runs on RAW prices for both families. Within-pair matches (S and B
of the same trade reporting the same raw price + volume + date) work
identically pre- or post-decimal-shift. The rare degenerate case where one
side has a decimal-slip and the other doesn't (raw 1500 vs raw 15) is
accepted as a known residual — pre-shift dedup leaves both in raw, and the
corrected family's `apply_decimal_shift` will drop the unresolvable side
while keeping the resolvable one. Both families inherit the same dedup
decisions on the same trades.

Output:
  data/development/trace_clean_raw.parquet  (RAW family final + meas_err=ON input)
  data/holdout/trace_clean_raw.parquet      (mechanical; not surfaced)
  data/development/cleaning_report.json     (DN counts; meas_err layers append later)

WRDS-MMN decimal-shift and DRR bounce-back live in their own scripts now.
Price plausibility is now meas_err-gated (raw family preserves junk bit-exact
per A1.7 — including wildly implausible prices like 1e-6 or 1e9).

NOTE on cusip_id: populated on ~99.97% of rows in the 2026-06-09 WRDS re-pull.
Retained for downstream FISD merging. Primary CUSIP-month keying happens in
build_monthly_panel.py.

Usage:
  python scripts/preprocess_trace.py
"""

import hashlib
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_):
        return it

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = REPO_ROOT / "data" / "trace_enhanced_repull.csv.gz"
DEV_OUT = REPO_ROOT / "data" / "development" / "trace_clean_raw.parquet"
HOLD_OUT = REPO_ROOT / "data" / "holdout" / "trace_clean_raw.parquet"
REPORT_OUT = REPO_ROOT / "data" / "development" / "cleaning_report.json"
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"

KEEP_COLUMNS = [
    "bond_sym_id",       # renamed → bond_id
    "cusip_id",
    "company_symbol",
    "trd_exctn_dt",
    "trd_exctn_tm",
    "rptd_pr",
    "entrd_vol_qt",
    "sub_prdct",
    "rpt_side_cd",
    "trdg_mkt_cd",
    "trd_mod_3",
    "bloomberg_identifier",
    "scrty_type_cd",
]

# ---------------------------------------------------------------------------
# Cleaning profiles (FL-D21a): ONE code path, a profile is a CONFIG.
#
# The as-published baseline profiles (bbw_2019, jostova_2013) and the existing
# raw family run through the same three-scan engine below; a profile selects
# which primitives apply. The "raw" profile reproduces the as-built behaviour
# EXACTLY (regression-pinned by tests/unit/test_preprocess_trace.py).
#
# Regime note (FL-D21 G3, fl_d21_gate_results.md): the trc_st vocabulary shifts
# at Feb-2012 — pre-2012 `C` is a CORRECTION record (its values are the
# corrected trade; keep-replacement keeps it), post-2012 `C` is a CANCELLATION
# (always dropped). ISO date strings compare lexicographically, so the regime
# split is a plain string comparison on trd_exctn_dt.
# ---------------------------------------------------------------------------

REGIME_SPLIT_ISO = "2012-02-06"   # DN-2014 Feb-2012 reporting change


@dataclass(frozen=True)
class CleaningProfile:
    """One cleaning configuration through the shared three-scan engine.

    correction_mode (P2, FL-D21c):
      'delete_both'              — drop correction records AND their originals
                                   (as-built; Jostova per the BKMX rung-2 read).
      'keep_replacement_pre2012' — pre-2012 keep the C record (it carries the
                                   corrected values) while its original is still
                                   dropped via 1b; post-2012 C is a cancellation
                                   and is dropped in every mode (BBW "adjust").
    retain_asof (P4, FL-D21h): keep asof_cd 'A' rows (genuine late-reported
      executions, execution-dated). Raw as-built drops them (blank-only keep).
    net_reversals (P3, FL-D21d): net out the ORIGINAL of an asof-'R' reversal
      record by value-matching (bond, pr_trd_dt, price, vol). The as-built raw
      path drops only the reversal RECORD and leaves the original in place.
    apply_wis: drop when-issued (wis_fl == 'Y'). STATED for BBW; UNKNOWN-default
      (not applied, FL-D21f R2) for Jostova.
    dedup (P6/P7, FL-D21g E1): interdealer dedup — the UNKNOWN envelope axis,
      run both ways per profile. Raw as-built: always on.
    screens: cleaning_primitives.PROFILE_SCREENS key for the profile-specific
      transaction screens (price range / volume floor / commission / data-entry),
      applied AFTER dedup, or None.
    """
    profile_id: str
    correction_mode: str = "delete_both"
    retain_asof: bool = False
    net_reversals: bool = False
    apply_wis: bool = True
    dedup: bool = True
    screens: str | None = None
    # OFAT envelope knobs (FL-D21e promoted primitives): retain_asof_dx flips
    # P5 ('D'/'X' rows retained instead of dropped); screen_names overrides the
    # profile's default screen list (None = PROFILE_SCREENS[screens]).
    retain_asof_dx: bool = False
    screen_names: "tuple | None" = None
    dev_out: "Path | None" = field(default=None)
    hold_out: "Path | None" = field(default=None)
    report_out: "Path | None" = field(default=None)

    def keep_1a_mask(self, chunk):
        """The pass-2 / scan-2 filter-1a keep mask under this profile's
        correction mode. Raw/delete_both: trc_st == 'T' (as-built)."""
        trc = chunk["trc_st"]
        if self.correction_mode == "keep_replacement_pre2012":
            pre2012 = chunk["trd_exctn_dt"].astype(str) < REGIME_SPLIT_ISO
            return (trc == "T") | ((trc == "C") & pre2012)
        return trc == "T"


RAW_PROFILE = CleaningProfile(profile_id="raw")

_PROFILE_SPECS = {
    # BBW 2019 (drf/crf): keep-replacement corrections ("adjust"), retain as-of,
    # net reversals, when-issued STATED, BBW transaction screens.
    "bbw_2019": dict(correction_mode="keep_replacement_pre2012", retain_asof=True,
                     net_reversals=True, apply_wis=True, screens="bbw_2019"),
    # Jostova 2013 (mom6): delete-both corrections (BKMX rung-2), retain as-of,
    # net reversals, when-issued NOT applied (R2 UNKNOWN-default), Jostova screens.
    "jostova_2013": dict(correction_mode="delete_both", retain_asof=True,
                         net_reversals=True, apply_wis=False, screens="jostova_2013"),
}


# OFAT envelope variants (FL-D21e R1 promotions, user decision 2026-08-05):
# each flips ONE promoted primitive against the profile baseline, all other
# knobs at default, dedup held at the declared OFAT reference arm (ON — the
# as-built DN posture). `when_issued` did NOT promote (default stands).
_OFAT_VARIANTS = {
    "bbw_2019": ("p4_asof_drop", "p5_asof_retain", "commission"),
    "jostova_2013": ("p4_asof_drop", "p5_asof_retain", "price_range",
                     "min_volume", "locked_in", "special_sales", "settlement"),
}


def build_variant_profile(profile_id: str, variant: str) -> CleaningProfile:
    """The OFAT envelope build for one promoted primitive: the profile baseline
    with exactly that primitive flipped (dedup at the ON reference arm)."""
    if variant not in _OFAT_VARIANTS.get(profile_id, ()):
        raise KeyError(
            f"unknown OFAT variant {variant!r} for {profile_id!r}; "
            f"promoted: {_OFAT_VARIANTS.get(profile_id, ())}"
        )
    spec = dict(_PROFILE_SPECS[profile_id])
    overrides: dict = {}
    if variant == "p4_asof_drop":
        overrides["retain_asof"] = False
    elif variant == "p5_asof_retain":
        overrides["retain_asof_dx"] = True
    else:
        # A screen flip: ADD the unstated screen to the profile's default list.
        from agents.quant.library.cleaning_primitives import profile_screen_names
        overrides["screen_names"] = (*profile_screen_names(profile_id), variant)
    stem = f"trace_clean_{profile_id}__var_{variant}"
    return CleaningProfile(
        profile_id=profile_id, dedup=True,
        dev_out=REPO_ROOT / "data" / "development" / f"{stem}.parquet",
        hold_out=REPO_ROOT / "data" / "holdout" / f"{stem}.parquet",
        report_out=REPO_ROOT / "data" / "development"
                   / f"cleaning_report_{profile_id}__var_{variant}.json",
        **{**spec, **overrides},
    )


def build_profile(profile_id: str, *, dedup: bool,
                  dev_out=None, hold_out=None, report_out=None) -> CleaningProfile:
    """Factory for the as-published profiles. `dedup` is the E1 envelope axis
    (both settings are built per FL-D21g). Default outputs are keyed by
    (profile, dedup) so the four family builds never collide."""
    if profile_id not in _PROFILE_SPECS:
        raise KeyError(f"unknown cleaning profile {profile_id!r}; "
                       f"known: {sorted(_PROFILE_SPECS)} (or 'raw')")
    tag = "on" if dedup else "off"
    stem = f"trace_clean_{profile_id}__dedup_{tag}"
    return CleaningProfile(
        profile_id=profile_id, dedup=dedup,
        dev_out=dev_out or (REPO_ROOT / "data" / "development" / f"{stem}.parquet"),
        hold_out=hold_out or (REPO_ROOT / "data" / "holdout" / f"{stem}.parquet"),
        report_out=report_out or (REPO_ROOT / "data" / "development"
                                  / f"cleaning_report_{profile_id}__dedup_{tag}.json"),
        **_PROFILE_SPECS[profile_id],
    )


def load_thresholds() -> dict:
    with open(THRESHOLDS_FILE) as f:
        cfg = yaml.safe_load(f)
    return cfg["trace_cleaning"]


def thresholds_sha256() -> str:
    return hashlib.sha256(THRESHOLDS_FILE.read_bytes()).hexdigest()


def _csv_reader(chunk_size: int = 500_000):
    """Chunked CSV reader with the dtype spec required for all three scans."""
    import pandas as pd
    return pd.read_csv(
        RAW_FILE,
        chunksize=chunk_size,
        dtype={
            "msg_seq_nb": str,
            "orig_msg_seq_nb": str,
            "rptd_pr": float,
            "entrd_vol_qt": float,
            "trc_st": str,
            "asof_cd": str,
            "wis_fl": str,
            "rpt_side_cd": str,
            "cusip_id": str,
            "bond_sym_id": str,
            # Pin to str so the composite cancellation/dedup keys hash an
            # IDENTICAL string in every scan. Left to inference, a chunk with a
            # blank date infers float64 ("20200101.0") while an all-valid chunk
            # infers int64 ("20200101") — the two-scan split makes that mismatch
            # silently un-drop a cancelled trade or split a dedup key.
            "trd_exctn_dt": str,
            # Profile-path columns (FL-D21): the reversal value-match key uses
            # pr_trd_dt (same string-identity argument as trd_exctn_dt), and the
            # Jostova commission screen reads cmsn_trd. Pinning unused dtype keys
            # is a no-op for the raw path.
            "pr_trd_dt": str,
            "cmsn_trd": str,
        },
        on_bad_lines="skip",
    )


def _composite_hash(*parts) -> int:
    """64-bit deterministic hash of a tuple of stringy/numeric parts.

    SHA-256 → first 8 bytes → big-endian unsigned int. Python's built-in hash()
    is session-randomized (PYTHONHASHSEED), so it cannot be used here.
    """
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], "big")


def _asof_keep_mask(chunk, asof_keep: str, retain_asof: bool,
                    retain_dx: bool = False):
    """Filter-2 keep mask. As-built (raw): blank/NaN asof only. Profiles
    (FL-D21h P4): additionally retain 'A' (as-of/late executions, execution-
    dated). 'R' records are dropped here in every mode (the reversal RECORD is
    not a trade; profiles net out its ORIGINAL separately); 'D'/'X' (P5) take
    the faithful-to-silence default (dropped) unless `retain_dx` (the promoted
    p5_asof_retain OFAT variant) flips them to retained."""
    keep = chunk["asof_cd"].isna() | (chunk["asof_cd"] == asof_keep)
    if retain_asof:
        keep = keep | (chunk["asof_cd"] == "A")
    if retain_dx:
        keep = keep | chunk["asof_cd"].isin(["D", "X"])
    return keep


def _apply_reversal_netout(chunk, rev_counter: Counter):
    """P3 net-out (FL-D21d): drop surviving originals value-matched by a
    reversal record — key (bond, trd_exctn_dt, price, vol) against the pool
    keyed on (bond, pr_trd_dt, price, vol). Consumes one pool count per drop,
    in file order (deterministic across scan 2 / pass 2, each on its own copy).
    Non-finite rows can never match (the pool excludes them). Returns
    (chunk, n_dropped)."""
    if not rev_counter or chunk.empty:
        return chunk, 0
    finite = (
        np.isfinite(chunk["rptd_pr"].to_numpy())
        & np.isfinite(chunk["entrd_vol_qt"].to_numpy())
    )
    idx = chunk.index[finite]
    sub = chunk.loc[idx]
    drop = []
    for i, b, d, pr, v in zip(
        idx,
        sub["bond_sym_id"].astype(str).tolist(),
        sub["trd_exctn_dt"].astype(str).tolist(),
        sub["rptd_pr"].tolist(),
        sub["entrd_vol_qt"].tolist(),
    ):
        h = _composite_hash(b, d, pr, v)
        if rev_counter.get(h, 0) > 0:
            rev_counter[h] -= 1
            drop.append(i)
    if drop:
        chunk = chunk.drop(index=drop)
    return chunk, len(drop)


def _collect_sets(cfg: dict, chunk_size: int = 500_000,
                  profile: CleaningProfile = RAW_PROFILE):
    """Build cross-chunk matching state in TWO scans of the raw file.

    Returns (cancelled_keys, sell_counts):
      - cancelled_keys: frozenset[int] of 64-bit hashes of
        (bond_sym_id, trd_exctn_dt, orig_msg_seq_nb) for every non-T record
        with a non-null orig_msg_seq_nb (Dick-Nielsen B1).
      - sell_counts: Counter[int] of (bond_sym_id, trd_exctn_dt, RAW price,
        entrd_vol_qt) for sell-side records surviving filters 1a/1b/2/3.
        RAW prices (no decimal-shift) — the dedup basis for both families.

    Why TWO scans: filter 1b (drop a T record cancelled by a later
    C/W/X/Y/R) needs the COMPLETE cancelled_keys set, because a cancellation
    can appear in a later chunk than its original. The sell pool therefore
    cannot apply 1b in the same pass that builds cancelled_keys. Scan 1 builds
    cancelled_keys; scan 2 — with cancelled_keys frozen — counts sells applying
    filters in the SAME order as Pass 2 (1a → 1b → 2 → 3 → side==S), so the
    dedup pool equals Pass 2's surviving sells exactly. A one-pass alternative
    that retained every sell's msg_seq_nb to subtract afterwards would be
    O(n_sells) memory; two scans keep it at O(distinct keys).

    Rows with non-finite RAW price or volume are EXCLUDED from the pool:
    _composite_hash uses repr(), and repr(nan)=='nan' (repr(inf)=='inf'), so
    unrelated non-finite rows would otherwise collide into one dedup key and a
    genuine buy could drop against a phantom NaN sell. Such rows are still
    WRITTEN unchanged downstream (A1.7 raw-junk preservation); they are only
    barred from dedup matching.
    """
    asof_keep = cfg["asof_cd_keep"]

    # --- Scan 1/3: cancellation keys (+ profile reversal pool) ---------------
    cancelled: set = set()
    reversal_pool: Counter = Counter()
    print("Scan 1/3: collecting cancellation keys...")
    for chunk in tqdm(_csv_reader(chunk_size), desc="scan1", unit="chunk"):
        non_t = chunk[chunk["trc_st"] != "T"]
        if not non_t.empty:
            origs = non_t["orig_msg_seq_nb"]
            mask = origs.notna() & (origs.astype(str) != "")
            valid = non_t[mask]
            for b, d, m in zip(
                valid["bond_sym_id"].astype(str).tolist(),
                valid["trd_exctn_dt"].astype(str).tolist(),
                valid["orig_msg_seq_nb"].astype(str).tolist(),
            ):
                cancelled.add(_composite_hash(b, d, m))
        # P3 reversal value-match pool (profiles only, FL-D21d): an asof-'R'
        # record references its original by (bond, prior-trade date, price,
        # volume). The original is netted out downstream; the reversal RECORD
        # itself is dropped by the asof policy. Non-finite price/vol and blank
        # pr_trd_dt records cannot match and are skipped (counted nowhere —
        # the unmatched-reversal count comes out of pass 2's bookkeeping).
        if profile.net_reversals and "pr_trd_dt" in chunk.columns:
            rev = chunk[chunk["asof_cd"] == "R"]
            if not rev.empty:
                prd = rev["pr_trd_dt"]
                ok = (
                    prd.notna() & (prd.astype(str) != "")
                    & np.isfinite(rev["rptd_pr"].to_numpy())
                    & np.isfinite(rev["entrd_vol_qt"].to_numpy())
                )
                rev = rev[ok]
                for b, d, pr, v in zip(
                    rev["bond_sym_id"].astype(str).tolist(),
                    rev["pr_trd_dt"].astype(str).tolist(),
                    rev["rptd_pr"].tolist(),
                    rev["entrd_vol_qt"].tolist(),
                ):
                    reversal_pool[_composite_hash(b, d, pr, v)] += 1
    cancelled_keys = frozenset(cancelled)

    if not profile.dedup:
        # E1 dedup OFF: no sell pool — pass 2 skips the dedup step entirely.
        print(f"  Scan 1 done: {len(cancelled_keys):,} cancellation keys; "
              f"dedup OFF for profile {profile.profile_id!r} (scan 2 skipped).")
        return cancelled_keys, Counter(), reversal_pool

    # --- Scan 2/3: interdealer sell-side dedup pool (post-1b) ----------------
    # cancelled_keys is now complete, so 1b can be applied. Filter order
    # mirrors Pass 2 exactly so the pool == Pass 2's surviving sells — including
    # the profile's correction mode, asof policy, wis policy, and reversal
    # net-out (each consuming its OWN copy of the reversal pool; same file
    # order ⇒ identical outcomes).
    sell_counts: Counter = Counter()
    scan2_reversals = Counter(reversal_pool)
    print("Scan 2/3: collecting interdealer sell-side keys (post-1b)...")
    for chunk in tqdm(_csv_reader(chunk_size), desc="scan2", unit="chunk"):
        # 1a (profile-aware: keep-replacement keeps pre-2012 C records)
        chunk = chunk[profile.keep_1a_mask(chunk)]
        if chunk.empty:
            continue
        # 1b — drop T records cancelled by a later C/W/X/Y/R record
        if cancelled_keys:
            is_cancelled = np.fromiter(
                (
                    _composite_hash(b, d, m) in cancelled_keys
                    for b, d, m in zip(
                        chunk["bond_sym_id"].astype(str).tolist(),
                        chunk["trd_exctn_dt"].astype(str).tolist(),
                        chunk["msg_seq_nb"].astype(str).tolist(),
                    )
                ),
                dtype=bool,
                count=len(chunk),
            )
            chunk = chunk[~is_cancelled]
            if chunk.empty:
                continue
        # 2 (profile-aware asof policy)
        chunk = chunk[_asof_keep_mask(chunk, asof_keep, profile.retain_asof, profile.retain_asof_dx)]
        if chunk.empty:
            continue
        # 3 (skipped when the profile does not state when-issued removal)
        if profile.apply_wis:
            chunk = chunk[chunk["wis_fl"] != "Y"]
            if chunk.empty:
                continue
        # P3 reversal net-out (profiles; scan-2 copy of the pool)
        if profile.net_reversals:
            chunk, _ = _apply_reversal_netout(chunk, scan2_reversals)
            if chunk.empty:
                continue

        sells = chunk[chunk["rpt_side_cd"] == "S"]
        if sells.empty:
            continue
        # Exclude non-finite price/volume from the dedup pool (see docstring).
        finite = (
            np.isfinite(sells["rptd_pr"].to_numpy())
            & np.isfinite(sells["entrd_vol_qt"].to_numpy())
        )
        if not finite.any():
            continue
        sells = sells[finite]
        for b, d, pr, v in zip(
            sells["bond_sym_id"].astype(str).tolist(),
            sells["trd_exctn_dt"].astype(str).tolist(),
            sells["rptd_pr"].tolist(),
            sells["entrd_vol_qt"].tolist(),
        ):
            sell_counts[_composite_hash(b, d, pr, v)] += 1

    total_sells = sum(sell_counts.values())
    print(f"  Scans 1-2 done: {len(cancelled_keys):,} cancellation keys, "
          f"{total_sells:,} sell-side records "
          f"({len(sell_counts):,} distinct keys).")
    return cancelled_keys, sell_counts, reversal_pool


def run_pandas(cfg: dict, profile: CleaningProfile = RAW_PROFILE) -> dict:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    asof_keep = cfg["asof_cd_keep"]
    holdout_year = cfg["holdout_start_year"]
    holdout_end_year = cfg["holdout_end_year"]
    chunk_size = 500_000

    # Profile output paths default to the module globals (the raw family), so
    # the test harness's monkeypatched paths keep working unchanged.
    dev_out = profile.dev_out or DEV_OUT
    hold_out = profile.hold_out or HOLD_OUT

    cancelled_keys, sell_counts, reversal_pool = _collect_sets(cfg, chunk_size, profile)
    pass2_reversals = Counter(reversal_pool)  # pass-2's own copy (scan 2 had its own)

    # Explicit output schema: prevents schema drift when pandas infers mixed-type
    # columns differently across chunks.
    OUTPUT_SCHEMA = pa.schema([
        pa.field("bond_id",              pa.string()),
        pa.field("cusip_id",             pa.string()),
        pa.field("company_symbol",       pa.string()),
        pa.field("trd_exctn_dt",         pa.timestamp("us")),
        pa.field("trd_exctn_tm",         pa.string()),
        pa.field("rptd_pr",              pa.float64()),
        pa.field("entrd_vol_qt",         pa.float64()),
        pa.field("sub_prdct",            pa.string()),
        pa.field("rpt_side_cd",          pa.string()),
        pa.field("trdg_mkt_cd",          pa.string()),
        pa.field("trd_mod_3",            pa.string()),
        pa.field("bloomberg_identifier", pa.string()),
        pa.field("scrty_type_cd",        pa.string()),
    ])

    print("Scan 3/3: applying filters and writing output...")

    dev_writer = None
    hold_writer = None
    counts = {
        "raw_total": 0,
        "after_trc_st_T": 0,
        "dropped_cancelled_original": 0,
        "after_asof_cd_blank": 0,
        "after_wis_fl": 0,
        "dropped_interdealer_duplicate": 0,
        "after_interdealer_dedup": 0,
        "dropped_invalid_date": 0,
        "development_rows": 0,
        "holdout_rows": 0,
        # Counted on rows that survive Pass-2 filters across dev+holdout.
        # NAMING: this is post-Dick-Nielsen, pre-bounce-back (bounce-back is
        # meas_err-gated and runs only in the corrected branch). The bounce-
        # back filter no longer overwrites this file; it reads from
        # apply_decimal_shift's output, so "pre-bounce" describes the
        # logical pipeline position not the literal file.
        "cusip_populated_post_dn": 0,
        "cusip_blank_post_dn": 0,
        # Profile-path counters (always present; identically zero on the raw
        # profile, whose path skips both steps).
        "dropped_reversal_netout": 0,
        "dropped_profile_screens": 0,
    }

    try:
        for chunk in tqdm(_csv_reader(chunk_size), desc="pass2", unit="chunk"):
            counts["raw_total"] += len(chunk)

            # Filter 1a: trade status (profile-aware — keep-replacement mode
            # additionally keeps pre-2012 C correction records, FL-D21c)
            chunk = chunk[profile.keep_1a_mask(chunk)]

            # Filter 1b: drop T records cancelled by a later C/W/X/Y/R record.
            if cancelled_keys and not chunk.empty:
                before = len(chunk)
                is_cancelled = np.fromiter(
                    (
                        _composite_hash(b, d, m) in cancelled_keys
                        for b, d, m in zip(
                            chunk["bond_sym_id"].astype(str).tolist(),
                            chunk["trd_exctn_dt"].astype(str).tolist(),
                            chunk["msg_seq_nb"].astype(str).tolist(),
                        )
                    ),
                    dtype=bool,
                    count=len(chunk),
                )
                chunk = chunk[~is_cancelled]
                counts["dropped_cancelled_original"] += before - len(chunk)
            counts["after_trc_st_T"] += len(chunk)
            if chunk.empty:
                continue

            # Filter 2: asof policy — as-built keeps blank only (Dick-Nielsen
            # 2009 Table 1 Panel B); profiles additionally retain 'A' (P4,
            # FL-D21h). Reversal RECORDS ('R') are dropped in every mode.
            chunk = chunk[_asof_keep_mask(chunk, asof_keep, profile.retain_asof, profile.retain_asof_dx)]
            counts["after_asof_cd_blank"] += len(chunk)
            if chunk.empty:
                continue

            # Filter 3: when-issued trades (skipped when the profile does not
            # state the step — Jostova R2 UNKNOWN-default, FL-D21f)
            if profile.apply_wis:
                chunk = chunk[chunk["wis_fl"] != "Y"]
            counts["after_wis_fl"] += len(chunk)
            if chunk.empty:
                continue

            # P3 reversal net-out (profiles, FL-D21d): drop the ORIGINAL a
            # reversal record value-matches; pass 2 consumes its own pool copy
            # in the same file order as scan 2, so the two stay in lockstep.
            if profile.net_reversals:
                chunk, n_rev = _apply_reversal_netout(chunk, pass2_reversals)
                counts["dropped_reversal_netout"] += n_rev
                if chunk.empty:
                    continue

            # Filter 6: interdealer dedup on RAW prices (Dick-Nielsen B2).
            # The dedup price basis is raw — within-pair matches (same trade
            # reported by both sides) hold pre-shift and post-shift. Per the
            # registry spec Section 2, dedup is basic cleaning shared by both
            # families.
            if sell_counts:
                buys_mask = (chunk["rpt_side_cd"] == "B").to_numpy()
                # Non-finite buys need no special handling here: _collect_sets
                # (scan 2) excludes non-finite price/volume from sell_counts, so
                # no NaN/inf key exists to match against. A non-finite buy hashes
                # to a repr(nan)/repr(inf) key, finds no sell, and is kept —
                # preserving raw junk bit-exact (A1.7). NaN handling lives in one
                # place (the sell pool), not duplicated on the buy side.
                if buys_mask.any():
                    buys = chunk[buys_mask]
                    buy_hashes = [
                        _composite_hash(b, d, pr, v)
                        for b, d, pr, v in zip(
                            buys["bond_sym_id"].astype(str).tolist(),
                            buys["trd_exctn_dt"].astype(str).tolist(),
                            buys["rptd_pr"].tolist(),
                            buys["entrd_vol_qt"].tolist(),
                        )
                    ]
                    is_dup = np.zeros(len(buy_hashes), dtype=bool)
                    for i, h in enumerate(buy_hashes):
                        if sell_counts.get(h, 0) > 0:
                            is_dup[i] = True
                            sell_counts[h] -= 1
                    drop_idx = buys.index[is_dup]
                    if len(drop_idx):
                        chunk = chunk.drop(index=drop_idx)
                        counts["dropped_interdealer_duplicate"] += int(len(drop_idx))
            counts["after_interdealer_dedup"] += len(chunk)
            if chunk.empty:
                continue

            # Profile transaction screens (STATED steps: BBW price range /
            # volume floor / when-issued; Jostova commission / data-entry) —
            # composable primitives from cleaning_primitives (FL-D21a).
            if profile.screens is not None:
                try:
                    from agents.quant.library.cleaning_primitives import (
                        apply_profile_screens,
                    )
                except ImportError:  # `python scripts/...` puts scripts/ on path
                    sys.path.insert(0, str(REPO_ROOT))
                    from agents.quant.library.cleaning_primitives import (
                        apply_profile_screens,
                    )
                before_screens = len(chunk)
                if profile.screen_names is not None:
                    from agents.quant.library.cleaning_primitives import apply_screens
                    chunk = chunk[apply_screens(chunk, profile.screen_names)]
                else:
                    chunk = chunk[apply_profile_screens(chunk, profile.screens)]
                counts["dropped_profile_screens"] += before_screens - len(chunk)
                if chunk.empty:
                    continue

            # Date parse — track rows lost to unparseable dates explicitly
            before_date_drop = len(chunk)
            chunk["trd_exctn_dt"] = pd.to_datetime(chunk["trd_exctn_dt"], errors="coerce")
            chunk = chunk.dropna(subset=["trd_exctn_dt"])
            counts["dropped_invalid_date"] += before_date_drop - len(chunk)
            if chunk.empty:
                continue

            chunk = chunk[KEEP_COLUMNS].rename(columns={"bond_sym_id": "bond_id"})

            # CUSIP populated-rate audit (counted on rows that survived all filters)
            cusip_col = chunk["cusip_id"]
            blank_mask = cusip_col.isna() | (cusip_col.astype(str).str.strip() == "")
            counts["cusip_blank_post_dn"] += int(blank_mask.sum())
            counts["cusip_populated_post_dn"] += int(len(chunk) - blank_mask.sum())

            # Dev/holdout split. Window is the closed interval
            # [holdout_year, holdout_end_year] = 2022–2025. The upper bound is
            # EXPLICIT: raise on any row beyond it (e.g. a future re-pull) so
            # the holdout window cannot silently creep — extending it must be a
            # conscious thresholds.yaml + docs edit, not a side effect of new
            # data landing in the file. NOTE: this fires mid-loop, so earlier
            # chunks may already be written — abort leaves partial parquet, which
            # a corrected re-run overwrites (ParquetWriter truncates on open).
            year = chunk["trd_exctn_dt"].dt.year
            beyond = year > holdout_end_year
            if beyond.any():
                raise ValueError(
                    f"{int(beyond.sum()):,} row(s) dated beyond holdout_end_year"
                    f"={holdout_end_year} (max year {int(year.max())}). The "
                    f"dev/holdout window is {holdout_year}–{holdout_end_year}; "
                    "raise holdout_end_year in docs/thresholds.yaml (and update "
                    "the docs) before ingesting newer data."
                )
            dev_chunk = chunk[year < holdout_year]
            hold_chunk = chunk[year >= holdout_year]

            if not dev_chunk.empty:
                table = pa.Table.from_pandas(dev_chunk, schema=OUTPUT_SCHEMA, preserve_index=False)
                if dev_writer is None:
                    dev_out.parent.mkdir(parents=True, exist_ok=True)
                    dev_writer = pq.ParquetWriter(str(dev_out), OUTPUT_SCHEMA)
                dev_writer.write_table(table)
                counts["development_rows"] += len(dev_chunk)

            if not hold_chunk.empty:
                table = pa.Table.from_pandas(hold_chunk, schema=OUTPUT_SCHEMA, preserve_index=False)
                if hold_writer is None:
                    hold_out.parent.mkdir(parents=True, exist_ok=True)
                    hold_writer = pq.ParquetWriter(str(hold_out), OUTPUT_SCHEMA)
                hold_writer.write_table(table)
                counts["holdout_rows"] += len(hold_chunk)

    finally:
        if dev_writer:
            dev_writer.close()
        if hold_writer:
            hold_writer.close()

    counts["final_clean_total"] = counts["development_rows"] + counts["holdout_rows"]
    # raw → after_trc_st_T includes both filter 1a and 1b. Split for the
    # audit trail.
    counts["dropped_trc_st"] = (
        counts["raw_total"] - counts["after_trc_st_T"] - counts["dropped_cancelled_original"]
    )
    counts["dropped_asof_cd"] = counts["after_trc_st_T"] - counts["after_asof_cd_blank"]
    counts["dropped_wis_fl"] = counts["after_asof_cd_blank"] - counts["after_wis_fl"]

    # Verify written parquet row counts match accumulators
    if dev_out.exists():
        actual = pq.read_metadata(str(dev_out)).num_rows
        assert actual == counts["development_rows"], (
            f"DEV parquet row count {actual:,} != counter {counts['development_rows']:,}"
        )
    if hold_out.exists():
        actual = pq.read_metadata(str(hold_out)).num_rows
        assert actual == counts["holdout_rows"], (
            f"HOLD parquet row count {actual:,} != counter {counts['holdout_rows']:,}"
        )
    counts["parquet_row_count_verified"] = True

    print(f"  Development rows: {counts['development_rows']:,}")
    print(f"  Holdout rows:     {counts['holdout_rows']:,}")
    return counts


def write_report(row_counts: dict, cfg: dict, git_commit: str,
                 profile: CleaningProfile = RAW_PROFILE) -> None:
    dev_out = profile.dev_out or DEV_OUT
    hold_out = profile.hold_out or HOLD_OUT
    report_out = profile.report_out or REPORT_OUT
    report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "source_file": str(RAW_FILE.relative_to(REPO_ROOT)) if RAW_FILE.is_relative_to(REPO_ROOT) else str(RAW_FILE),
        "thresholds_file": str(THRESHOLDS_FILE.relative_to(REPO_ROOT)) if THRESHOLDS_FILE.is_relative_to(REPO_ROOT) else str(THRESHOLDS_FILE),
        "thresholds_sha256": thresholds_sha256(),
        "thresholds_used": cfg,
        "rows": row_counts,
        "stage": ("dick_nielsen_only" if profile.profile_id == "raw"
                  else f"as_published_profile_{profile.profile_id}"
                       f"__dedup_{'on' if profile.dedup else 'off'}"),
        "cleaning_profile": {
            "profile_id": profile.profile_id,
            "correction_mode": profile.correction_mode,
            "retain_asof": profile.retain_asof,
            "net_reversals": profile.net_reversals,
            "apply_wis": profile.apply_wis,
            "dedup": profile.dedup,
            "screens": profile.screens,
        },
        "outputs": {
            "dev": str(dev_out.relative_to(REPO_ROOT)) if dev_out.is_relative_to(REPO_ROOT) else str(dev_out),
            "holdout": str(hold_out.relative_to(REPO_ROOT)) if hold_out.is_relative_to(REPO_ROOT) else str(hold_out),
        },
        "downstream_pipeline": (
            "trace_clean_raw.parquet feeds (a) the raw column family directly "
            "and (b) apply_decimal_shift.py → bounce_back_filter.py → "
            "trace_clean_corr.parquet for the corrected column family. "
            "Bias-toggle registry meas_err = OFF reads trace_clean_raw.parquet; "
            "meas_err = ON reads trace_clean_corr.parquet."
        ),
        "output_columns": [c if c != "bond_sym_id" else "bond_id" for c in KEEP_COLUMNS],
        "git_commit": git_commit,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    tmp = report_out.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, report_out)
    print(f"  Report written: {report_out}")


def get_git_commit() -> str:
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--profile", choices=["raw", *sorted(_PROFILE_SPECS)], default="raw",
        help="Cleaning profile to emit (FL-D21a). 'raw' reproduces the as-built "
             "Dick-Nielsen family; the as-published profiles emit "
             "trace_clean_<profile>__dedup_<on|off>.parquet.",
    )
    parser.add_argument(
        "--dedup", choices=["on", "off"], default="on",
        help="E1 interdealer-dedup envelope setting (profiles only; FL-D21g). "
             "The raw family always dedups (as-built).",
    )
    parser.add_argument(
        "--variant", default=None,
        help="OFAT envelope variant: flip ONE promoted primitive against the "
             "profile baseline (FL-D21e; dedup held at the ON reference arm). "
             "Mutually exclusive with --dedup off.",
    )
    args = parser.parse_args()

    if not RAW_FILE.exists():
        print(f"ERROR: Raw file not found: {RAW_FILE}", file=sys.stderr)
        sys.exit(1)

    if args.profile == "raw":
        if args.variant:
            print("ERROR: --variant requires a profile", file=sys.stderr)
            sys.exit(1)
        profile = RAW_PROFILE
    elif args.variant:
        if args.dedup == "off":
            print("ERROR: --variant holds dedup at the ON reference arm", file=sys.stderr)
            sys.exit(1)
        profile = build_variant_profile(args.profile, args.variant)
    else:
        profile = build_profile(args.profile, dedup=(args.dedup == "on"))

    cfg = load_thresholds()
    print(f"Thresholds loaded from {THRESHOLDS_FILE}")
    print(f"  asof_cd_keep={cfg['asof_cd_keep']!r}, "
          f"holdout window={cfg['holdout_start_year']}–{cfg['holdout_end_year']}")
    if profile.profile_id == "raw":
        print("Stage: Dick-Nielsen filters only (no price plausibility, no decimal-shift).")
        print("       Those are meas_err-gated and live in apply_decimal_shift.py downstream.")
    else:
        print(f"Stage: as-published cleaning profile {profile.profile_id!r} "
              f"(dedup {'ON' if profile.dedup else 'OFF'}) — FL-D21a fork from raw TRACE.")

    git_commit = get_git_commit()
    row_counts = run_pandas(cfg, profile)
    write_report(row_counts, cfg, git_commit, profile)

    dev_out = profile.dev_out or DEV_OUT
    hold_out = profile.hold_out or HOLD_OUT
    report_out = profile.report_out or REPORT_OUT
    print("\nDone.")
    print(f"  Development: {row_counts['development_rows']:,} rows → {dev_out}")
    print(f"  Holdout:     {row_counts['holdout_rows']:,} rows → {hold_out}")
    print(f"  Report:      {report_out}")


if __name__ == "__main__":
    main()
