"""
signal_recompute.py — per-panel-state recompute of the return-history signals (decision b).

The signal propagation recomputes the signals that are
trailing functionals of the monthly returns from each panel state's OWN (toggle-applied) returns, so
the stale_price / survivorship intervention propagates into the characteristics rather than being
truncated. This is category-2 deterministic recomputation (D-A49) — closed-form, point-in-time — and
it reuses the CANONICAL compute functions from the build scripts (no reimplementation):

  * var_5pct : BBW-2019 trailing 5% VaR of `ret`          (build_var_5pct.compute_var_5pct)
  * bond_vol : KPP 24-month std of `xret`                 (build_bond_vol.compute_bond_vol)
  * mom6     : JNPS 6-month cumulative return of `ret`     (build_mom6_signal.compute_mom6_signal)

**gamma_illiq is INVARIANT under both panel-view toggles by construction** (a spike finding, now
asserted once in `tests/synthetic/test_ipca_gamma_invariance.py`): the code shows it is a *daily-sourced*
within-month autocovariance (from the trace daily panels), whose source data lies OUTSIDE both toggles'
registered primitives (monthly price_eom for stale_price; monthly terminal rows for survivorship). So its
per-state recompute is identically equal to the shared value — nothing the toggles reach feeds it, hence
nothing is truncated. Not recomputing it is therefore a reclassification, NOT a membership-only fallback
(that earlier phrasing is withdrawn as understating the invariance).

The existing on-disk signal parquets embody the UNMASKED maximal panel (the spike's defect finding);
recomputing on the view panel's returns fixes the P_N signal/return inconsistency.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

_REPO = Path(__file__).resolve().parents[3]
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from build_bond_vol import compute_bond_vol            # noqa: E402  canonical KPP TOTAL_VOL
from build_mom6_signal import compute_mom6_signal      # noqa: E402  canonical JNPS momentum signal
from build_var_5pct import compute_var_5pct            # noqa: E402  canonical BBW-2019 VaR

_THRESHOLDS = _REPO / "docs" / "thresholds.yaml"

# The return-history signals recomputed per panel state, and the return column each consumes.
RECOMPUTED_SIGNALS: tuple[str, ...] = ("var_5pct", "bond_vol", "mom6")
# gamma_illiq is daily-sourced → invariant under both toggles by construction → not recomputed (see docstring).


def _signals_cfg(thresholds_path: str | Path | None = None) -> dict:
    return yaml.safe_load(Path(thresholds_path or _THRESHOLDS).read_text())["signals"]


def recompute_signals(view_panel: pd.DataFrame, *, thresholds_path: str | Path | None = None) -> pd.DataFrame:
    """Return a copy of ``view_panel`` with var_5pct, bond_vol, mom6 overwritten by values recomputed
    from the panel's own (toggle-applied) returns via the canonical compute functions. gamma_illiq,
    rating and everything else pass through unchanged."""
    s = _signals_cfg(thresholds_path)
    ret = view_panel[["cusip", "date", "ret"]]
    xret = view_panel[["cusip", "date", "xret"]].rename(columns={"xret": "ret"})

    v5 = compute_var_5pct(
        ret, window=int(s["var_5pct"]["window"]), min_obs=int(s["var_5pct"]["min_obs"]),
        rank=int(s["var_5pct"]["rank"]), multiplier=float(s["var_5pct"]["multiplier"]),
    )
    bv = compute_bond_vol(
        xret, window=int(s["bond_vol"]["window"]), min_obs=int(s["bond_vol"]["min_obs"]),
    )
    m6 = compute_mom6_signal(
        ret, formation_months=int(s["mom6"]["formation_months"]), min_obs=int(s["mom6"]["min_obs"]),
    )

    out = view_panel.drop(columns=list(RECOMPUTED_SIGNALS), errors="ignore")
    for df, col in ((v5, "var_5pct"), (bv, "bond_vol"), (m6, "mom6")):
        out = out.merge(df[["cusip", "date", col]], on=["cusip", "date"], how="left")
    return out
