# Output contract

Your script MUST write exactly one CSV file to the path in the environment
variable `OUTPUT_PATH`, with exactly these two columns:

    date,portfolio_return

- `date`: the month of the realised portfolio return, ISO format (YYYY-MM-DD,
  month-end).
- `portfolio_return`: the strategy's long-short monthly return, decimal
  (0.01 = 1%).
- One row per month, no duplicate months, no other columns, no index column.

Anything else written anywhere is discarded. A missing or malformed output
file is recorded as WONT_RUN. The script runs with no network access, reads
the panel from the parquet at `PANEL_PATH`, and must finish within the wall
clock limit.
