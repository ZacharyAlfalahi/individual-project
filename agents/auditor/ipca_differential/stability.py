"""
stability.py — the coupling stability diagnostic (spec §5.4, D-A46).

R' ≈ 20-30 outer refits on blocked resamples of the REAL panel — the SAME block scheme as the §5.3
bootstrap (one resampling universe, not two) — each refit under the production rule §5.2 (i.e. both
arms are RE-FIT, unlike the conditional bootstrap which holds the fitted states fixed). It reports
whether the **sign and order of magnitude of the interaction bracket I survive refitting**, per bias.

Field ``coupling_stability_diagnostic``; pre-registered (R', scheme, scope); **quoted nowhere as an
interval** — its pre-registration exists precisely so it cannot become a post-hoc rescue. This is the
uncertainty the conditional bootstrap deliberately does NOT estimate (re-fitting Θ under resampling).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from agents.auditor.checks.bootstrap import (
    BootstrapError,
    block_length,
    circular_block_indices,
    effective_blocks,
)
from agents.auditor.thresholds import IPCALambda, IPCAProjectionGate, IPCAStabilityConfig
from agents.quant.library.ipca import ContractViolation
from agents.quant.library.ipca_feed import IPCAFeed

from .differential import differential_from_feeds

_SCHEME = "synchronised fixed-block moving resample of the panel; each refit under production rule §5.2"


@dataclass(frozen=True, eq=False)
class StabilityDiagnostic:
    """Sign/magnitude survival of I under refitting. NEVER an interval (§5.4)."""

    bias: str
    anchor: str
    r_prime: int
    block_length: int
    effective_blocks: int
    t_common: int
    i_obs: float
    sign_survival: float               # fraction of usable refits with sign(I_r) == sign(I_obs)
    magnitude_survival: float          # fraction with |I_r| within [1/f, f] * |I_obs|
    order_of_magnitude_factor: float
    n_usable_refits: int               # refits that produced a finite I_r
    i_draws: tuple[float, ...]         # the refit brackets (diagnostic transparency; NOT a CI)
    scheme: str = _SCHEME

    def to_dict(self) -> dict:
        return {
            "coupling_stability_diagnostic": {
                "bias": self.bias,
                "anchor": self.anchor,
                "r_prime": self.r_prime,
                "block_length": self.block_length,
                "effective_blocks": self.effective_blocks,
                "t_common": self.t_common,
                "i_obs": self.i_obs,
                "sign_survival": self.sign_survival,
                "magnitude_survival": self.magnitude_survival,
                "order_of_magnitude_factor": self.order_of_magnitude_factor,
                "n_usable_refits": self.n_usable_refits,
                "scheme": self.scheme,
                "is_interval": False,   # explicit: this is a diagnostic, never an interval
            }
        }


def _resample_feed(feed: IPCAFeed, pos: np.ndarray) -> IPCAFeed:
    """Reindex a feed by the resample positions, renumbering months positionally (1..T) so the
    duplicated months remain valid ascending labels for the estimator."""
    t = len(pos)
    return feed._replace(
        Z=[feed.Z[i] for i in pos],
        R=[feed.R[i] for i in pos],
        months=np.arange(1, t + 1, dtype=np.int64),
        asof=np.arange(0, t, dtype=np.int64),
        vol_scaler=[feed.vol_scaler[i] for i in pos],
        cusips=[feed.cusips[i] for i in pos],
    )


def _resample_anchor(anchor_arr: np.ndarray, pos: np.ndarray) -> pd.Series:
    """Anchor return series aligned to the resampled positions, indexed by the renumbered periods."""
    per = pd.PeriodIndex([pd.Period(ordinal=i + 1, freq="M") for i in range(len(pos))])
    return pd.Series(anchor_arr[pos], index=per)


def _restrict_feed(feed: IPCAFeed, common_months: list[int]) -> IPCAFeed:
    """Restrict a feed to ``common_months`` (a subset of feed.months), in the given order, keeping
    the original month labels. Used to align the two arms onto their shared calendar support before
    resampling, so one resample sequence refers to the SAME calendar month in both arms."""
    pos_of = {int(m): i for i, m in enumerate(feed.months)}
    idx = [pos_of[int(m)] for m in common_months]
    return feed._replace(
        Z=[feed.Z[i] for i in idx],
        R=[feed.R[i] for i in idx],
        months=np.asarray([int(feed.months[i]) for i in idx], dtype=np.int64),
        asof=np.asarray([int(feed.asof[i]) for i in idx], dtype=np.int64),
        vol_scaler=[feed.vol_scaler[i] for i in idx],
        cusips=[feed.cusips[i] for i in idx],
    )


def stability_diagnostic(
    bias: str,
    anchor_name: str,
    feed_n: IPCAFeed,
    feed_b: IPCAFeed,
    anchor: pd.Series,
    lam: IPCALambda,
    gate: IPCAProjectionGate,
    cfg: IPCAStabilityConfig,
    *,
    i_obs: float | None = None,
    seed: int = 0,
) -> StabilityDiagnostic:
    """Run R' blocked-resample refits and report sign/order-of-magnitude survival of I.

    Raises ``BootstrapError`` if the block length is incompatible with the common support."""
    # ONE resampling universe = the return months SHARED by both arms. P_N and P_{N\b} are
    # different panels (survivorship reintroduction, stale masking) and generally differ in their
    # surviving months, so we must resample on the intersection — otherwise a positional index would
    # refer to different calendar months in the two arms (or overrun the shorter arm).
    common = sorted(set(int(m) for m in feed_n.months) & set(int(m) for m in feed_b.months))
    if not common:
        raise BootstrapError("no common return-month support between the two arms — cannot resample")
    feed_n_c = _restrict_feed(feed_n, common)
    feed_b_c = _restrict_feed(feed_b, common)

    t = len(common)
    ell = block_length(cfg.holding_period_default, cfg.block_length_months)
    if ell >= t:
        raise BootstrapError(f"block length ℓ={ell} >= T_common={t} (§6.2 refusal)")
    eff = effective_blocks(t, ell)
    if eff < cfg.min_effective_blocks:
        raise BootstrapError(
            f"only {eff} effective blocks (ℓ={ell}, T_common={t}); need >= {cfg.min_effective_blocks}"
        )

    # Anchor aligned to the common return-month support (the resampling universe).
    anchor_arr = np.array(
        [float(anchor.loc[pd.Period(ordinal=m, freq="M")]) for m in common], dtype=np.float64
    )

    if i_obs is None:
        obs = differential_from_feeds(bias, anchor_name, feed_n_c, feed_b_c, anchor, lam, gate)
        i_obs = obs.interaction_bracket_raw.value

    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(cfg.r_prime):
        pos = circular_block_indices(t, ell, rng)          # one common sequence over the shared support
        try:
            res_r = differential_from_feeds(
                bias, anchor_name,
                _resample_feed(feed_n_c, pos), _resample_feed(feed_b_c, pos),
                _resample_anchor(anchor_arr, pos), lam, gate,
            )
            draws.append(res_r.interaction_bracket_raw.value)
        except (ContractViolation, BootstrapError, np.linalg.LinAlgError):
            draws.append(float("nan"))

    finite = [x for x in draws if x == x]                  # drop NaN (degenerate refits)
    n = len(finite)
    if n == 0 or i_obs == 0 or i_obs != i_obs:
        sign_survival = float("nan")
        magnitude_survival = float("nan")
    else:
        sign_survival = float(np.mean([np.sign(x) == np.sign(i_obs) for x in finite]))
        f = cfg.order_of_magnitude_factor
        lo, hi = abs(i_obs) / f, abs(i_obs) * f
        magnitude_survival = float(np.mean([lo <= abs(x) <= hi for x in finite]))

    return StabilityDiagnostic(
        bias=bias,
        anchor=anchor_name,
        r_prime=cfg.r_prime,
        block_length=ell,
        effective_blocks=eff,
        t_common=t,
        i_obs=float(i_obs),
        sign_survival=sign_survival,
        magnitude_survival=magnitude_survival,
        order_of_magnitude_factor=cfg.order_of_magnitude_factor,
        n_usable_refits=n,
        i_draws=tuple(draws),
    )
