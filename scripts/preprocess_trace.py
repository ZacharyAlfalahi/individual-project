"""
One-time TRACE data cleaning and dev/holdout split.

Two-pass architecture:
  Pass 1 (_collect_sets): scan the raw file once to gather two cross-chunk sets
    - cancelled_originals: msg_seq_nb of T records later cancelled/corrected
      (Dick-Nielsen 2009 two-pass matching)
    - sell_key_hashes: 64-bit hashes of (bond_id, dt, price, vol) for sell-side
      records, used for interdealer-pair dedup (Dick-Nielsen 2009 Algorithm B2)
  Pass 2 (run_pandas main loop): apply all filters and write output.

Filters applied in Pass 2:
  1a. trc_st == "T"                 — drop cancelled/reversed/withdrawn records
  1b. msg_seq_nb not in cancelled   — drop T records cancelled by a later C/W/X/Y/R
  2.  asof_cd == "" (blank)         — keep only on-time original reports; drops A/R/D/X
  3.  wis_fl != "Y"                 — drop when-issued trades
  4.  price in (price_floor, 30000] — drop implausible prices (pre-correction ceiling)
  5.  decimal-shift correction      — see below
  6.  interdealer dedup             — drop B-side row when matched S-side row exists

WRDS-MMN decimal-shift correction (Dickerson, Robotti, Rossetti 2025):
  prices in (300, 3000]   → divide by 10
  prices in (3000, 30000] → divide by 100
  prices in (0, 300]      → keep as-is

NOTE on cusip_id: blank on every row in this dataset. bond_sym_id is used as the primary
bond identifier and renamed to bond_id in the output. No CUSIP mapping attempted; a WRDS
crosswalk would be required and is not available in this repo.

NOTE on PyBondLab: PyBondLab has no clean_trace() function — it is a portfolio formation
library. PyBondLab Filter (price, bounce, trim, winsorise) is applied to the monthly return
panel in the Quant agent's return-aggregation step, not here.

Usage:
  python scripts/preprocess_trace.py

Requires: Python >=3.10, <3.13 (Numba/PyBondLab constraint when building Quant agent)
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
RAW_FILE = REPO_ROOT / "data" / "trace_enhanced_raw.csv.gz"
DEV_OUT = REPO_ROOT / "data" / "development" / "trace_clean.parquet"
HOLD_OUT = REPO_ROOT / "data" / "holdout" / "trace_clean.parquet"
REPORT_OUT = REPO_ROOT / "data" / "development" / "cleaning_report.json"
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"

KEEP_COLUMNS = [
    "bond_sym_id",       # renamed → bond_id
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


def decimal_shift(price: float, floor: float, ceiling: float):
    """
    WRDS-MMN decimal-shift correction (Dickerson, Robotti, Rossetti 2025).

    Assumes no legitimate corporate bond trades above `ceiling` (300 by default).
    Any price exceeding the ceiling is treated as a decimal-point reporting error.
    Applies the minimal divisor (10 or 100) that brings the price into (floor, ceiling].
    Returns None if no shift restores a valid price — caller should drop that row.
    """
    if floor < price <= ceiling:
        return price
    if floor < price / 10 <= ceiling:
        return price / 10
    if floor < price / 100 <= ceiling:
        return price / 100
    return None


def _csv_reader(chunk_size: int = 500_000):
    """Chunked CSV reader with the dtype spec required for both passes."""
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
        },
        on_bad_lines="skip",
        # NOTE: malformed rows (123 observed) are skipped silently by pandas;
        # there is no hook to count them without a custom reader. Known gap —
        # see docs/trace_preprocessing.md.
    )


def _composite_hash(*parts) -> int:
    """64-bit deterministic hash of a tuple of stringy/numeric parts.

    SHA-256 → first 8 bytes → big-endian unsigned int. Python's built-in hash()
    is session-randomized (PYTHONHASHSEED), so it cannot be used here.
    Collision probability at ~50M entries is ~10⁻⁷ — safe for this use.

    The tuple is serialised via repr() so each part is quoted/escaped; this
    avoids ambiguity when a delimiter character would otherwise appear inside
    a string field.
    """
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], "big")


def _apply_decimal_shift(chunk, floor: float, ceiling: float):
    """Vectorized decimal-shift correction; drops unresolvable rows."""
    chunk = chunk.copy()
    p = chunk["rptd_pr"].to_numpy(dtype=float)
    corrected = np.where(
        (p > floor) & (p <= ceiling), p,
        np.where(
            (p / 10 > floor) & (p / 10 <= ceiling), p / 10,
            np.where(
                (p / 100 > floor) & (p / 100 <= ceiling), p / 100,
                np.nan,
            ),
        ),
    )
    chunk["rptd_pr"] = corrected
    return chunk.dropna(subset=["rptd_pr"])


def _collect_sets(cfg: dict, chunk_size: int = 500_000):
    """Pass 1: scan the raw file once to build cross-chunk matching state.

    Returns (cancelled_keys, sell_counts):
      - cancelled_keys: frozenset[int] of 64-bit hashes of
        (bond_sym_id, trd_exctn_dt, orig_msg_seq_nb) for every non-T record
        with a non-null orig_msg_seq_nb. The composite key is required because
        TRACE's msg_seq_nb is a per-dealer-per-day counter, not globally
        unique — matching on msg_seq_nb alone would drop unrelated T rows
        across the 20-year file (Dick-Nielsen 2009 B1).
      - sell_counts: collections.Counter[int] mapping
        hash(bond_sym_id, trd_exctn_dt, corrected_price, entrd_vol_qt) → count
        of sell-side ('S') records sharing that key. Counter (not set) is
        required because Dick-Nielsen B2 is a 1-to-1 pairing: N sells should
        consume at most N buys. Pass 2 decrements this Counter as it walks
        B records so over-matching can never drop legitimate client trades.

    Memory: cancelled keys ≲500 MB; sell_counts Counter ~4–7 GB at full scale
    (50M Python ints × ~28 B + dict overhead). Measure on a sample first.
    """
    floor = cfg["price_floor"]
    ceiling = cfg["price_ceiling"]
    pre_ceiling = cfg["pre_correction_ceiling"]
    asof_keep = cfg["asof_cd_keep"]

    cancelled_keys: set = set()
    sell_counts: Counter = Counter()

    print("Pass 1/2: scanning for cancellation keys and interdealer sell-side keys...")
    for chunk in tqdm(_csv_reader(chunk_size), desc="pass1", unit="chunk"):
        # Cancellation keys — composite (bond_sym_id, trd_exctn_dt, orig_msg_seq_nb)
        # collected from non-T records that reference a prior original trade.
        non_t = chunk[chunk["trc_st"] != "T"]
        if not non_t.empty:
            origs = non_t["orig_msg_seq_nb"]
            mask = origs.notna() & (origs.astype(str) != "")
            valid = non_t[mask]
            if not valid.empty:
                for b, d, m in zip(
                    valid["bond_sym_id"].astype(str).tolist(),
                    valid["trd_exctn_dt"].astype(str).tolist(),
                    valid["orig_msg_seq_nb"].astype(str).tolist(),
                ):
                    cancelled_keys.add(_composite_hash(b, d, m))

        # Replicate Pass 2 filters 1-4 + decimal-shift to identify sell-side survivors
        chunk = chunk[chunk["trc_st"] == "T"]
        if chunk.empty:
            continue
        chunk = chunk[chunk["asof_cd"].isna() | (chunk["asof_cd"] == asof_keep)]
        if chunk.empty:
            continue
        chunk = chunk[chunk["wis_fl"] != "Y"]
        if chunk.empty:
            continue
        chunk = chunk[(chunk["rptd_pr"] > floor) & (chunk["rptd_pr"] <= pre_ceiling)]
        if chunk.empty:
            continue
        chunk = _apply_decimal_shift(chunk, floor, ceiling)
        if chunk.empty:
            continue

        sells = chunk[chunk["rpt_side_cd"] == "S"]
        if sells.empty:
            continue
        for b, d, pr, v in zip(
            sells["bond_sym_id"].astype(str).tolist(),
            sells["trd_exctn_dt"].astype(str).tolist(),
            sells["rptd_pr"].tolist(),
            sells["entrd_vol_qt"].tolist(),
        ):
            sell_counts[_composite_hash(b, d, pr, v)] += 1

    total_sells = sum(sell_counts.values())
    print(f"  Pass 1 done: {len(cancelled_keys):,} cancellation keys, "
          f"{total_sells:,} sell-side records "
          f"({len(sell_counts):,} distinct keys).")
    return frozenset(cancelled_keys), sell_counts


def run_pandas(cfg: dict) -> dict:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    floor = cfg["price_floor"]
    ceiling = cfg["price_ceiling"]
    pre_ceiling = cfg["pre_correction_ceiling"]
    asof_keep = cfg["asof_cd_keep"]
    holdout_year = cfg["holdout_start_year"]
    chunk_size = 500_000

    cancelled_keys, sell_counts = _collect_sets(cfg, chunk_size)

    # Explicit output schema: prevents schema drift when pandas infers mixed-type columns
    # differently across chunks (e.g. scrty_type_cd is numeric in some chunks, string in others).
    OUTPUT_SCHEMA = pa.schema([
        pa.field("bond_id",              pa.string()),
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

    print("Pass 2/2: applying filters and writing output...")

    dev_writer = None
    hold_writer = None
    counts = {
        "raw_total": 0,
        "after_trc_st_T": 0,
        "dropped_cancelled_original": 0,
        "after_asof_cd_blank": 0,
        "after_wis_fl": 0,
        "after_price_plausibility": 0,
        "after_decimal_shift_correction": 0,
        "dropped_interdealer_duplicate": 0,
        "after_interdealer_dedup": 0,
        "dropped_invalid_date": 0,
        "development_rows": 0,
        "holdout_rows": 0,
    }

    try:
        for chunk in tqdm(_csv_reader(chunk_size), desc="pass2", unit="chunk"):
            counts["raw_total"] += len(chunk)

            # Filter 1a: trade status
            chunk = chunk[chunk["trc_st"] == "T"]

            # Filter 1b: drop T records cancelled by a later C/W/X/Y/R record.
            # Match on composite key (bond_sym_id, trd_exctn_dt, msg_seq_nb)
            # per Dick-Nielsen 2009 B1 — msg_seq_nb alone is not globally
            # unique (per-dealer-per-day counter).
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
            # isna() is required: pandas reads blank CSV cells as NaN, not "".
            chunk = chunk[chunk["asof_cd"].isna() | (chunk["asof_cd"] == asof_keep)]
            counts["after_asof_cd_blank"] += len(chunk)
            if chunk.empty:
                continue

            # Filter 3: when-issued trades
            chunk = chunk[chunk["wis_fl"] != "Y"]
            counts["after_wis_fl"] += len(chunk)
            if chunk.empty:
                continue

            # Filter 4: price plausibility (pre-correction ceiling preserves decimal-shift candidates)
            chunk = chunk[(chunk["rptd_pr"] > floor) & (chunk["rptd_pr"] <= pre_ceiling)]
            counts["after_price_plausibility"] += len(chunk)
            if chunk.empty:
                continue

            # WRDS-MMN decimal-shift correction — vectorized via np.where for performance
            chunk = _apply_decimal_shift(chunk, floor, ceiling)
            counts["after_decimal_shift_correction"] += len(chunk)
            if chunk.empty:
                continue

            # Interdealer dedup (Dick-Nielsen 2009 Algorithm B2): drop B-side rows
            # whose (bond_id, dt, price, vol) matches a sell-side record AS A PAIR.
            # sell_counts is a Counter: each B match decrements the count for that
            # key, so N sells can pair with at most N buys. A market-maker that
            # legitimately buys from one client and sells to another at the same
            # price/vol/day produces 1 S + 2 B; only 1 B is dropped here, the
            # second B (no remaining S to pair with) is kept.
            if sell_counts:
                buys_mask = (chunk["rpt_side_cd"] == "B").to_numpy()
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

            dev_chunk = chunk[chunk["trd_exctn_dt"].dt.year < holdout_year]
            hold_chunk = chunk[chunk["trd_exctn_dt"].dt.year >= holdout_year]

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
    # raw → after_trc_st_T includes both filter 1a (trc_st != "T") and 1b
    # (cancelled originals). Separate them so the audit trail stays readable.
    counts["dropped_trc_st"] = (
        counts["raw_total"] - counts["after_trc_st_T"] - counts["dropped_cancelled_original"]
    )
    counts["dropped_asof_cd"] = counts["after_trc_st_T"] - counts["after_asof_cd_blank"]
    counts["dropped_wis_fl"] = counts["after_asof_cd_blank"] - counts["after_wis_fl"]
    counts["dropped_price_plausibility"] = counts["after_wis_fl"] - counts["after_price_plausibility"]
    counts["decimal_shift_unresolvable_dropped"] = (
        counts["after_price_plausibility"] - counts["after_decimal_shift_correction"]
    )

    # Verify written parquet row counts match accumulators — loud failure beats silent corruption
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
        "identifier_decision": (
            "bond_sym_id used as primary identifier; cusip_id is blank on all rows in this dataset"
        ),
        "output_columns": [c if c != "bond_sym_id" else "bond_id" for c in KEEP_COLUMNS],
        "git_commit": git_commit,
    }
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = REPORT_OUT.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, REPORT_OUT)  # atomic on POSIX and Windows (Python 3.3+)
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
    print(f"  price_floor={cfg['price_floor']}, price_ceiling={cfg['price_ceiling']}, "
          f"holdout_start_year={cfg['holdout_start_year']}")

    git_commit = get_git_commit()
    row_counts = run_pandas(cfg)
    write_report(row_counts, cfg, git_commit)

    print("\nDone.")
    print(f"  Development: {row_counts['development_rows']:,} rows → {DEV_OUT}")
    print(f"  Holdout:     {row_counts['holdout_rows']:,} rows → {HOLD_OUT}")
    print(f"  Report:      {REPORT_OUT}")


if __name__ == "__main__":
    main()
