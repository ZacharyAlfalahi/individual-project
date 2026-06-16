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
        },
        on_bad_lines="skip",
    )


def _composite_hash(*parts) -> int:
    """64-bit deterministic hash of a tuple of stringy/numeric parts.

    SHA-256 → first 8 bytes → big-endian unsigned int. Python's built-in hash()
    is session-randomized (PYTHONHASHSEED), so it cannot be used here.
    """
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], "big")


def _collect_sets(cfg: dict, chunk_size: int = 500_000):
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

    # --- Scan 1/3: cancellation keys only -----------------------------------
    cancelled: set = set()
    print("Scan 1/3: collecting cancellation keys...")
    for chunk in tqdm(_csv_reader(chunk_size), desc="scan1", unit="chunk"):
        non_t = chunk[chunk["trc_st"] != "T"]
        if non_t.empty:
            continue
        origs = non_t["orig_msg_seq_nb"]
        mask = origs.notna() & (origs.astype(str) != "")
        valid = non_t[mask]
        if valid.empty:
            continue
        for b, d, m in zip(
            valid["bond_sym_id"].astype(str).tolist(),
            valid["trd_exctn_dt"].astype(str).tolist(),
            valid["orig_msg_seq_nb"].astype(str).tolist(),
        ):
            cancelled.add(_composite_hash(b, d, m))
    cancelled_keys = frozenset(cancelled)

    # --- Scan 2/3: interdealer sell-side dedup pool (post-1b) ----------------
    # cancelled_keys is now complete, so 1b can be applied. Filter order
    # mirrors Pass 2 exactly so the pool == Pass 2's surviving sells.
    sell_counts: Counter = Counter()
    print("Scan 2/3: collecting interdealer sell-side keys (post-1b)...")
    for chunk in tqdm(_csv_reader(chunk_size), desc="scan2", unit="chunk"):
        # 1a
        chunk = chunk[chunk["trc_st"] == "T"]
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
        # 2
        chunk = chunk[chunk["asof_cd"].isna() | (chunk["asof_cd"] == asof_keep)]
        if chunk.empty:
            continue
        # 3
        chunk = chunk[chunk["wis_fl"] != "Y"]
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
    return cancelled_keys, sell_counts


def run_pandas(cfg: dict) -> dict:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    asof_keep = cfg["asof_cd_keep"]
    holdout_year = cfg["holdout_start_year"]
    holdout_end_year = cfg["holdout_end_year"]
    chunk_size = 500_000

    cancelled_keys, sell_counts = _collect_sets(cfg, chunk_size)

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
    }

    try:
        for chunk in tqdm(_csv_reader(chunk_size), desc="pass2", unit="chunk"):
            counts["raw_total"] += len(chunk)

            # Filter 1a: trade status
            chunk = chunk[chunk["trc_st"] == "T"]

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

            # Filter 2: keep only blank asof_cd — Dick-Nielsen (2009) Table 1 Panel B
            chunk = chunk[chunk["asof_cd"].isna() | (chunk["asof_cd"] == asof_keep)]
            counts["after_asof_cd_blank"] += len(chunk)
            if chunk.empty:
                continue

            # Filter 3: when-issued trades
            chunk = chunk[chunk["wis_fl"] != "Y"]
            counts["after_wis_fl"] += len(chunk)
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
                    DEV_OUT.parent.mkdir(parents=True, exist_ok=True)
                    dev_writer = pq.ParquetWriter(str(DEV_OUT), OUTPUT_SCHEMA)
                dev_writer.write_table(table)
                counts["development_rows"] += len(dev_chunk)

            if not hold_chunk.empty:
                table = pa.Table.from_pandas(hold_chunk, schema=OUTPUT_SCHEMA, preserve_index=False)
                if hold_writer is None:
                    HOLD_OUT.parent.mkdir(parents=True, exist_ok=True)
                    hold_writer = pq.ParquetWriter(str(HOLD_OUT), OUTPUT_SCHEMA)
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
    if DEV_OUT.exists():
        actual = pq.read_metadata(str(DEV_OUT)).num_rows
        assert actual == counts["development_rows"], (
            f"DEV parquet row count {actual:,} != counter {counts['development_rows']:,}"
        )
    if HOLD_OUT.exists():
        actual = pq.read_metadata(str(HOLD_OUT)).num_rows
        assert actual == counts["holdout_rows"], (
            f"HOLD parquet row count {actual:,} != counter {counts['holdout_rows']:,}"
        )
    counts["parquet_row_count_verified"] = True

    print(f"  Development rows: {counts['development_rows']:,}")
    print(f"  Holdout rows:     {counts['holdout_rows']:,}")
    return counts


def write_report(row_counts: dict, cfg: dict, git_commit: str) -> None:
    report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "source_file": str(RAW_FILE.relative_to(REPO_ROOT)) if RAW_FILE.is_relative_to(REPO_ROOT) else str(RAW_FILE),
        "thresholds_file": str(THRESHOLDS_FILE.relative_to(REPO_ROOT)) if THRESHOLDS_FILE.is_relative_to(REPO_ROOT) else str(THRESHOLDS_FILE),
        "thresholds_sha256": thresholds_sha256(),
        "thresholds_used": cfg,
        "rows": row_counts,
        "stage": "dick_nielsen_only",
        "outputs": {
            "dev": str(DEV_OUT.relative_to(REPO_ROOT)),
            "holdout": str(HOLD_OUT.relative_to(REPO_ROOT)),
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
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = REPORT_OUT.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, REPORT_OUT)
    print(f"  Report written: {REPORT_OUT}")


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
    if not RAW_FILE.exists():
        print(f"ERROR: Raw file not found: {RAW_FILE}", file=sys.stderr)
        sys.exit(1)

    cfg = load_thresholds()
    print(f"Thresholds loaded from {THRESHOLDS_FILE}")
    print(f"  asof_cd_keep={cfg['asof_cd_keep']!r}, "
          f"holdout window={cfg['holdout_start_year']}–{cfg['holdout_end_year']}")
    print("Stage: Dick-Nielsen filters only (no price plausibility, no decimal-shift).")
    print("       Those are meas_err-gated and live in apply_decimal_shift.py downstream.")

    git_commit = get_git_commit()
    row_counts = run_pandas(cfg)
    write_report(row_counts, cfg, git_commit)

    print("\nDone.")
    print(f"  Development: {row_counts['development_rows']:,} rows → {DEV_OUT}")
    print(f"  Holdout:     {row_counts['holdout_rows']:,} rows → {HOLD_OUT}")
    print(f"  Report:      {REPORT_OUT}")


if __name__ == "__main__":
    main()
