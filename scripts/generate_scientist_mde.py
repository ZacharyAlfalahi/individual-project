"""
Generate the Scientist minimum-detectable-effect (MDE) table programmatically (spec §4.1).

The BH step-up rejects up to i* = max{i : p_(i) <= i·q/m}, so the evidential bar is a RANGE
over ranks 1..m, not a point. For each rank i:

    required_p(i)   = i · q / m                       (the BH threshold at that rank)
    |t|(i)          = Phi^{-1}(1 - required_p/2)      (two-sided, normal/independence approx)
    ann_sharpe(i)   = |t|(i) · sqrt(12 / T)           (T = primary-inference months)

Parameters come from docs/scientist_protocol.yaml (q = inference.bh_fdr_q,
T = windows.primary_inference.n_months, m from generation.proposals_per_strategy) so the table
is reproducible from the pre-registration and never hard-coded (spec §4.1, prohibition 1). CLI
flags override for ad-hoc checks.

The normal approximation matches the spec footnote ("independence approximation"); the audited
NW-HAC test requires stronger evidence, and positive residual autocorrelation raises the bar.
Six extensions of one parent are positively correlated (PRDS), the regime under which BH stays
valid, so the easier (rank-m) end is realistic.

Usage:
  python scripts/generate_scientist_mde.py                 # read protocol, write the table
  python scripts/generate_scientist_mde.py --m 6 --q 0.10 --T 209 --stdout
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import NormalDist

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_FILE = REPO_ROOT / "docs" / "scientist_protocol.yaml"
OUT_FILE = REPO_ROOT / "docs" / "scientist" / "mde_table.md"

_MONTHS_PER_YEAR = 12
_NORMAL = NormalDist()


def required_p(rank: int, q: float, m: int) -> float:
    """BH threshold at a given rank within a family of size m."""
    return rank * q / m


def two_sided_abs_t(p: float) -> float:
    """|t| whose two-sided normal tail area equals p."""
    return _NORMAL.inv_cdf(1.0 - p / 2.0)


def annualised_sharpe(abs_t: float, n_months: int) -> float:
    """Annualised Sharpe implied by a t-stat on the mean over n_months (independence approx)."""
    return abs_t * math.sqrt(_MONTHS_PER_YEAR / n_months)


def ladder(m: int, q: float, n_months: int) -> list[dict]:
    """The full rank-1..m MDE ladder for one family size."""
    rows = []
    for i in range(1, m + 1):
        p = required_p(i, q, m)
        t = two_sided_abs_t(p)
        rows.append(
            {
                "rank": i,
                "required_p": p,
                "abs_t": t,
                "ann_sharpe": annualised_sharpe(t, n_months),
            }
        )
    return rows


def _load_protocol_params() -> tuple[float, int, list[int]]:
    """(q, T, distinct_m_values) from the pre-registration protocol. Fail-loud on absence —
    the MDE must derive from committed parameters, never a default (prohibition 1)."""
    if not PROTOCOL_FILE.exists():
        raise SystemExit(
            f"{PROTOCOL_FILE} not found — commit the pre-registration protocol first, "
            f"or pass --m/--q/--T explicitly."
        )
    proto = yaml.safe_load(PROTOCOL_FILE.read_text())
    q = float(proto["inference"]["bh_fdr_q"])
    n_months = int(proto["windows"]["primary_inference"]["n_months"])
    pps = proto["generation"]["proposals_per_strategy"]
    m_values = {int(pps.get("default"))}
    m_values.update(int(v) for v in (pps.get("per_strategy") or {}).values())
    return q, n_months, sorted(m_values)


def render(m_values: list[int], q: float, n_months: int) -> str:
    lines = [
        "# Scientist MDE table (generated — do not edit by hand)",
        "",
        "Source: `scripts/generate_scientist_mde.py` from `docs/scientist_protocol.yaml`.",
        f"Parameters: q = {q}, T = {n_months} primary-inference months "
        f"(2004-08…2021-12, DRF/CRF-bound). Normal/independence approximation; the audited "
        f"NW-HAC bar is stronger.",
        "",
    ]
    for m in m_values:
        rows = ladder(m, q, n_months)
        lo, hi = rows[0], rows[-1]
        lines += [
            f"## m = {m}",
            "",
            f"Bounds: rank 1 (one genuine among {m - 1} nulls) requires "
            f"|t| ≈ {lo['abs_t']:.2f} → annualised Sharpe ≈ {lo['ann_sharpe']:.2f}; "
            f"rank {m} (all share a real effect) requires "
            f"|t| ≈ {hi['abs_t']:.2f} → annualised Sharpe ≈ {hi['ann_sharpe']:.2f}.",
            "",
            "| rank | required p | approx \\|t\\| | approx annualised Sharpe |",
            "| --- | --- | --- | --- |",
        ]
        for r in rows:
            lines.append(
                f"| {r['rank']} | {r['required_p']:.4f} | "
                f"{r['abs_t']:.2f} | {r['ann_sharpe']:.2f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m", type=int, help="family size (single int; use with --q and --T)")
    ap.add_argument("--q", type=float, help="BH-FDR level")
    ap.add_argument("--T", type=int, help="primary-inference months")
    ap.add_argument("--stdout", action="store_true", help="print only; do not write the file")
    args = ap.parse_args()

    if args.m or args.q or args.T:
        if not (args.m and args.q and args.T):
            raise SystemExit("--m, --q, --T must be given together when overriding the protocol")
        q, n_months, m_values = args.q, args.T, [args.m]
    else:
        q, n_months, m_values = _load_protocol_params()

    text = render(m_values, q, n_months)
    print(text)
    if not args.stdout:
        OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        OUT_FILE.write_text(text + "\n")
        print(f"\nWritten: {OUT_FILE}")


if __name__ == "__main__":
    main()
