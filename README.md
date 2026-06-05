# Multi-agent LLM for Corporate Bond Strategy Replication & Bias Detection

Six-agent LLM pipeline: PDF ingestion → strategy extraction → backtesting → bias auditing → extension proposals → research notes.

> Detailed specifications, decision registers, validation records, and licensed data are maintained internally by intention and are not part of this public repository; in-code references to `docs/…` are provenance labels for that internal record.

---

## Setup

Requires Python 3.12 (PyBondLab/Numba constraint).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r scripts/requirements_preprocess.txt -r scripts/requirements_parser.txt
```

---

## What's Built

### 1. Data preprocessing (Stage 1)
Cleans raw TRACE Enhanced data and splits into dev/holdout.

```bash
python scripts/preprocess_trace.py
```

Reads `data/trace_enhanced_raw.csv.gz` (~435M rows). Applies Dick-Nielsen (2009/2014) filters (including blank `asof_cd` only) + WRDS-MMN decimal-shift correction. Writes:

- `data/development/trace_clean.parquet` — 2002–2021 transactions, safe to use
- `data/holdout/trace_clean.parquet` — 2022–2024, **never read during development**
- `data/development/cleaning_report.json` — per-filter row counts with verified parquet integrity

See `docs/trace_preprocessing.md` for full methodology.

### 2. Bounce-back filter (Stage 2)
Transaction-level removal of erroneous price spikes that survive Stage 1's decimal-shift correction.

```bash
python scripts/bounce_back_filter.py
```

Implements the DRR (Dickerson, Robotti, Rossetti 2026) bounce-back filter (Table A.2): uses each bond's own rolling trailing median as the reference and removes a flagged trade only when a future trade within 5 transactions shows recovery to the bond's actual level. Required because Stage 1's decimal-shift only catches clean ×10/×100 errors — it does not catch typos like "106 → 1.87" that distort the monthly VWAP and produce 50×+ returns the following month.

Reads and atomically overwrites `data/development/trace_clean.parquet` and `data/holdout/trace_clean.parquet` (holdout is treated as a mechanical transformation; parameters are frozen in `thresholds.yaml` before any run). Updates `data/development/cleaning_report.json` with two new drop counters and additivity-verified row totals. Uses DuckDB for the partition-wide external sort (polars `sink_parquet` OOMs on the 200M-row dev partition).

See `docs/bounce_back_filter_spec.md` for the full algorithm and `docs/trace_preprocessing.md` §5 for how it fits in the pipeline.

### 3. Risk-free rate
Downloads 3-month T-bill rate (TB3MS) from FRED.

```bash
python scripts/download_rf_rate.py
```

Writes `data/development/rf_rate.parquet` (monthly, 1934–present).

### 4. Monthly bond return panel
Aggregates transaction data to bond × month panel with excess returns.

```bash
python scripts/build_monthly_panel.py
```

Requires: `trace_clean.parquet` (after Stage 1 + Stage 2) + `rf_rate.parquet`. Writes:

- `data/development/monthly_panel.parquet` — bond_id, year_month, price_eom, ret, xret, n_trades, total_vol
- `data/development/monthly_panel_report.json`

Filters: CORP + pre-2012 null sub_prdct; institutional trades ≥100k par. Returns computed as `(P_t−P_{t-1})/P_{t-1}`; excess return = ret − rf_monthly (TB3MS/12/100).

---

## Tests

```bash
pytest tests/ -v
```
