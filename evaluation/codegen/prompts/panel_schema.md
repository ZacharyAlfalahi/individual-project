# Panel schema (columns and dtypes only — no data)

The panel at `PANEL_PATH` is a parquet file of bond-month observations on the
development window. One row per (cusip, month). Columns:

| column     | dtype          | meaning                                                        |
|------------|----------------|----------------------------------------------------------------|
| `cusip`    | str            | bond identifier                                                |
| `date`     | datetime64[ns] | month-end stamp of the observation month                       |
| `ret`      | float64        | total monthly return (decimal; 0.01 = 1%)                      |
| `xret`     | float64        | monthly return in excess of the risk-free rate (decimal)       |
| `size`     | float64        | bond amount outstanding (value-weighting weight)               |
| `rating`   | float64        | numeric credit rating (higher = worse credit)                  |
| `var_5pct` | float64        | 5% Value-at-Risk downside-risk signal (precomputed)            |
| `gamma`    | float64        | illiquidity signal (precomputed)                               |
| `mom6`     | float64        | six-month momentum signal (precomputed)                        |

Signals are point-in-time as of `date`. NaN means the signal is not defined
for that bond-month (insufficient history); such rows are not sortable on
that signal.
