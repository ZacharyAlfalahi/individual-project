"""
Unit tests for views.view() — the canonical engine-shape panel materialiser
for the bias-toggle registry.

Covers:
  - Family selection (raw vs corr) renames *_<family> to unsuffixed
  - Purity invariant (same inputs → byte-identical output)
  - Stale-price mask: gap > θ masks price_eom(t), ret(t), ret(t+1)
  - Stale-price mask coexists with the existing ret-adjacency rule (A3.4)
  - terminal-row toggle is a documented no-op pre-FISD (exit_reason all NaN)
  - A9 cross-family raise: signals with only the wrong family → ValueError
  - Signals without family suffix pass through
"""

import numpy as np
import pandas as pd
import pytest

from agents.quant.library.run_config import (
    ConstructionConfig,
    EvaluationConfig,
    PanelViewConfig,
    RunConfig,
)
from agents.quant.library.views import view


# ---------------------------------------------------------------------------
# Synthetic maximal panel
# ---------------------------------------------------------------------------

def _me(s: str) -> pd.Timestamp:
    return pd.Timestamp(s) + pd.offsets.MonthEnd(0)


def _make_maximal(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]) + pd.offsets.MonthEnd(0)
    if "last_trade_date_raw" in df.columns:
        df["last_trade_date_raw"] = pd.to_datetime(df["last_trade_date_raw"])
    if "last_trade_date_corr" in df.columns:
        df["last_trade_date_corr"] = pd.to_datetime(df["last_trade_date_corr"])
    return df


def _maximal_row(cusip, date, *, p_raw, p_corr, r_raw, r_corr,
                 ltd_raw=None, ltd_corr=None, exit_reason=None):
    return {
        "cusip": cusip,
        "date": date,
        "size": 1.0,
        "rf_monthly": 0.001,
        "exit_reason": exit_reason,
        "price_eom_raw": p_raw,
        "price_eom_corr": p_corr,
        "ret_raw": r_raw,
        "ret_corr": r_corr,
        "xret_raw": (r_raw - 0.001) if r_raw is not None and not (isinstance(r_raw, float) and np.isnan(r_raw)) else r_raw,
        "xret_corr": (r_corr - 0.001) if r_corr is not None and not (isinstance(r_corr, float) and np.isnan(r_corr)) else r_corr,
        "n_trades_raw": 5,
        "n_trades_corr": 5,
        "total_vol_raw": 1e6,
        "total_vol_corr": 1e6,
        "last_trade_date_raw": ltd_raw if ltd_raw else date,
        "last_trade_date_corr": ltd_corr if ltd_corr else date,
    }


def _cfg(family, stale_mask=False, include_terminal_rows=False,
         signal_lag=0, expost_trim="none"):
    return RunConfig(
        panel_view=PanelViewConfig(
            price_family=family,
            stale_mask=stale_mask,
            include_terminal_rows=include_terminal_rows,
        ),
        construction=ConstructionConfig(
            signal_lag=signal_lag, expost_trim=expost_trim,
        ),
        evaluation=EvaluationConfig(),
    )


# ---------------------------------------------------------------------------
# Family selection
# ---------------------------------------------------------------------------

class TestFamilySelection:
    def test_raw_family_selects_raw_columns(self):
        panel = _make_maximal([
            _maximal_row("A", "2010-01", p_raw=100.0, p_corr=99.0,
                         r_raw=0.05, r_corr=0.04),
        ])
        out = view(panel, _cfg("raw"))
        assert out.iloc[0]["price_eom"] == 100.0
        assert out.iloc[0]["ret"] == 0.05
        assert "price_eom_raw" not in out.columns
        assert "price_eom_corr" not in out.columns

    def test_corr_family_selects_corr_columns(self):
        panel = _make_maximal([
            _maximal_row("A", "2010-01", p_raw=100.0, p_corr=99.0,
                         r_raw=0.05, r_corr=0.04),
        ])
        out = view(panel, _cfg("corr"))
        assert out.iloc[0]["price_eom"] == 99.0
        assert out.iloc[0]["ret"] == 0.04

    def test_engine_contract_columns_present(self):
        panel = _make_maximal([
            _maximal_row("A", "2010-01", p_raw=100.0, p_corr=99.0,
                         r_raw=0.05, r_corr=0.04),
        ])
        out = view(panel, _cfg("raw"))
        for col in ("cusip", "date", "size", "ret", "xret", "price_eom",
                    "last_trade_date", "n_trades", "total_vol", "rf_monthly"):
            assert col in out.columns, f"missing {col}"

    def test_exit_reason_dropped_from_output(self):
        panel = _make_maximal([
            _maximal_row("A", "2010-01", p_raw=100.0, p_corr=99.0,
                         r_raw=0.05, r_corr=0.04),
        ])
        out = view(panel, _cfg("raw"))
        assert "exit_reason" not in out.columns


# ---------------------------------------------------------------------------
# Purity invariant
# ---------------------------------------------------------------------------

class TestPurity:
    def test_same_inputs_byte_identical_output(self):
        panel = _make_maximal([
            _maximal_row("A", "2010-01", p_raw=100.0, p_corr=99.0,
                         r_raw=0.05, r_corr=0.04),
            _maximal_row("B", "2010-01", p_raw=200.0, p_corr=198.0,
                         r_raw=0.02, r_corr=0.01),
        ])
        out1 = view(panel, _cfg("corr"))
        out2 = view(panel, _cfg("corr"))
        pd.testing.assert_frame_equal(out1, out2)

    def test_input_not_mutated(self):
        panel = _make_maximal([
            _maximal_row("A", "2010-01", p_raw=100.0, p_corr=99.0,
                         r_raw=0.05, r_corr=0.04),
        ])
        snapshot = panel.copy()
        view(panel, _cfg("corr"))
        pd.testing.assert_frame_equal(panel, snapshot)


# ---------------------------------------------------------------------------
# Stale-price mask
# ---------------------------------------------------------------------------

class TestStaleMask:
    def test_no_stale_mask_when_toggle_off(self):
        panel = _make_maximal([
            _maximal_row("A", "2010-01", p_raw=100.0, p_corr=99.0,
                         r_raw=0.05, r_corr=0.04,
                         ltd_corr="2009-11-01"),  # 60-day gap
        ])
        out = view(panel, _cfg("corr", stale_mask=False))
        # No mask applied — ret/price preserved.
        assert out.iloc[0]["ret"] == 0.04
        assert out.iloc[0]["price_eom"] == 99.0

    def test_gap_above_theta_masks_price_and_ret(self):
        # 60-day gap at theta=30 → masked
        panel = _make_maximal([
            _maximal_row("A", "2010-01", p_raw=100.0, p_corr=99.0,
                         r_raw=0.05, r_corr=0.04,
                         ltd_corr="2009-11-01"),
        ])
        out = view(panel, _cfg("corr", stale_mask=True), stale_threshold_days=30)
        assert pd.isna(out.iloc[0]["price_eom"])
        assert pd.isna(out.iloc[0]["ret"])
        assert pd.isna(out.iloc[0]["xret"])

    def test_gap_below_theta_unaffected(self):
        # 5-day gap at theta=30 → kept
        panel = _make_maximal([
            _maximal_row("A", "2010-01", p_raw=100.0, p_corr=99.0,
                         r_raw=0.05, r_corr=0.04,
                         ltd_corr="2010-01-26"),
        ])
        out = view(panel, _cfg("corr", stale_mask=True), stale_threshold_days=30)
        assert out.iloc[0]["price_eom"] == 99.0
        assert out.iloc[0]["ret"] == 0.04

    def test_stale_propagates_to_next_month_ret(self):
        # Cusip A: Jan is stale (long gap), Feb has fresh trade.
        # Per A3.2: ret(Feb) depends on price(Jan), so when price(Jan) is
        # masked, ret(Feb) must also become NaN.
        panel = _make_maximal([
            _maximal_row("A", "2010-01", p_raw=100.0, p_corr=99.0,
                         r_raw=0.05, r_corr=0.04,
                         ltd_corr="2009-11-01"),     # Jan stale (60-day gap)
            _maximal_row("A", "2010-02", p_raw=102.0, p_corr=101.0,
                         r_raw=0.02, r_corr=0.02,
                         ltd_corr="2010-02-25"),     # Feb fresh
        ])
        out = view(panel, _cfg("corr", stale_mask=True), stale_threshold_days=30)
        jan = out[out["date"] == _me("2010-01")].iloc[0]
        feb = out[out["date"] == _me("2010-02")].iloc[0]
        assert pd.isna(jan["price_eom"]) and pd.isna(jan["ret"])
        # Feb's own price stays (its last_trade_date is fresh)...
        assert feb["price_eom"] == 101.0
        # ...but its ret is masked by propagation from Jan being stale.
        assert pd.isna(feb["ret"])
        assert pd.isna(feb["xret"])

    def test_stale_does_not_propagate_across_calendar_gap(self):
        # Cusip A: Jan is stale; the NEXT panel row is Mar (Feb absent). ret(Mar)
        # does not derive from price(Jan) — they are not calendar-adjacent — so
        # the stale mask must NOT propagate to Mar. A positional shift(1) would
        # wrongly mask it. Mar carries a real (non-NaN) return and a fresh price;
        # the maximal panel is built directly so build-time adjacency does not
        # pre-NaN it, isolating the view-layer propagation logic.
        panel = _make_maximal([
            _maximal_row("A", "2010-01", p_raw=100.0, p_corr=99.0,
                         r_raw=0.05, r_corr=0.04,
                         ltd_corr="2009-11-01"),     # Jan stale (60-day gap)
            _maximal_row("A", "2010-03", p_raw=103.0, p_corr=102.0,
                         r_raw=0.03, r_corr=0.03,
                         ltd_corr="2010-03-26"),     # Mar fresh, real return
        ])
        out = view(panel, _cfg("corr", stale_mask=True), stale_threshold_days=30)
        jan = out[out["date"] == _me("2010-01")].iloc[0]
        mar = out[out["date"] == _me("2010-03")].iloc[0]
        assert pd.isna(jan["ret"])            # Jan self-masked (stale)
        assert mar["price_eom"] == 102.0      # Mar fresh — own price kept
        assert mar["ret"] == 0.03             # NOT masked: Jan→Mar is a gap
        assert pd.notna(mar["xret"])

    def test_stale_mask_distinct_from_ret_adjacency(self):
        """A3.4 coexistence: stale-mask masks because of price freshness;
        the existing ret-adjacency rule (baked into ret_corr at the
        monthly-panel build) masks because of month-to-month gaps. Both
        rules can fire independently; the view layer doesn't deduplicate
        them."""
        # Cusip with fresh price but ret already NaN from adjacency
        panel = _make_maximal([
            _maximal_row("A", "2010-01", p_raw=100.0, p_corr=99.0,
                         r_raw=float("nan"), r_corr=float("nan"),
                         ltd_corr="2010-01-28"),  # 3-day gap (fresh)
        ])
        out = view(panel, _cfg("corr", stale_mask=True), stale_threshold_days=30)
        # ret stays NaN (it was NaN from adjacency); price NOT masked.
        assert pd.isna(out.iloc[0]["ret"])
        assert out.iloc[0]["price_eom"] == 99.0


# ---------------------------------------------------------------------------
# Terminal rows (pre-FISD no-op)
# ---------------------------------------------------------------------------

class TestTerminalRows:
    def test_no_op_pre_fisd(self):
        """When exit_reason is NaN everywhere (pre-FISD reality), the
        include_terminal_rows toggle has no observable effect on rows kept."""
        panel = _make_maximal([
            _maximal_row("A", "2010-01", p_raw=100.0, p_corr=99.0,
                         r_raw=0.05, r_corr=0.04),
            _maximal_row("B", "2010-01", p_raw=200.0, p_corr=198.0,
                         r_raw=0.02, r_corr=0.01),
        ])
        out_off = view(panel, _cfg("corr", include_terminal_rows=False))
        out_on = view(panel, _cfg("corr", include_terminal_rows=True))
        pd.testing.assert_frame_equal(out_off, out_on)

    def test_distress_rows_toggled_others_kept_in_both(self):
        """The survivorship toggle acts ONLY on distress (default) exits: a
        defaulted row is dropped in the as-published view and kept in the
        corrected view, while matured/defeased rows are kept in BOTH (they
        cancel out of the survivorship differential)."""
        panel = _make_maximal([
            _maximal_row("A", "2010-01", p_raw=100.0, p_corr=99.0,
                         r_raw=0.05, r_corr=0.04, exit_reason=None),
            _maximal_row("D", "2010-01", p_raw=80.0, p_corr=79.0,
                         r_raw=-0.5, r_corr=-0.5, exit_reason="defaulted"),
            _maximal_row("M", "2010-01", p_raw=200.0, p_corr=198.0,
                         r_raw=0.02, r_corr=0.01, exit_reason="matured"),
            _maximal_row("F", "2010-01", p_raw=150.0, p_corr=149.0,
                         r_raw=0.0, r_corr=0.0, exit_reason="defeased"),
        ])
        out_off = view(panel, _cfg("corr", include_terminal_rows=False))
        out_on = view(panel, _cfg("corr", include_terminal_rows=True))
        # As-published drops ONLY the defaulted bond; maturity/defeased survive.
        assert set(out_off["cusip"]) == {"A", "M", "F"}
        # Corrected keeps everything.
        assert set(out_on["cusip"]) == {"A", "D", "M", "F"}
        # The survivorship differential is exactly the distress row.
        assert set(out_on["cusip"]) - set(out_off["cusip"]) == {"D"}

    def test_distress_exits_override(self):
        """The distress_exits argument overrides the thresholds default."""
        panel = _make_maximal([
            _maximal_row("M", "2010-01", p_raw=200.0, p_corr=198.0,
                         r_raw=0.02, r_corr=0.01, exit_reason="matured"),
        ])
        # Treat 'matured' as distress for this call → dropped when toggle off.
        out = view(panel, _cfg("corr", include_terminal_rows=False),
                   distress_exits=["matured"])
        assert out.empty


# ---------------------------------------------------------------------------
# Universe restriction (FISD) — applied across all views when the flag exists
# ---------------------------------------------------------------------------

class TestUniverseRestriction:
    @staticmethod
    def _row(cusip, eligible):
        r = _maximal_row(cusip, "2010-01", p_raw=100.0, p_corr=99.0,
                         r_raw=0.05, r_corr=0.04)
        r["universe_eligible"] = eligible
        return r

    def test_drops_ineligible_and_removes_column(self):
        panel = _make_maximal([self._row("A", True), self._row("B", False)])
        for fam in ("raw", "corr"):
            out = view(panel, _cfg(fam))
            assert set(out["cusip"]) == {"A"}, f"{fam}: ineligible bond not dropped"
            assert "universe_eligible" not in out.columns

    def test_no_op_when_column_absent(self):
        # Panels without the flag (pre-FISD) keep all rows.
        panel = _make_maximal([
            _maximal_row("A", "2010-01", p_raw=100.0, p_corr=99.0, r_raw=0.05, r_corr=0.04),
            _maximal_row("B", "2010-01", p_raw=200.0, p_corr=198.0, r_raw=0.02, r_corr=0.01),
        ])
        out = view(panel, _cfg("corr"))
        assert set(out["cusip"]) == {"A", "B"}


# ---------------------------------------------------------------------------
# A9 cross-family raise
# ---------------------------------------------------------------------------

class TestA9SignalResolution:
    def _panel(self):
        return _make_maximal([
            _maximal_row("A", "2010-01", p_raw=100.0, p_corr=99.0,
                         r_raw=0.05, r_corr=0.04),
        ])

    def test_raw_signal_selected_for_raw_family(self):
        sig = pd.DataFrame({
            "cusip": ["A"], "date": [_me("2010-01")],
            "var_5pct_raw": [0.1], "var_5pct_corr": [0.08],
        })
        out = view(self._panel(), _cfg("raw"), signals=sig)
        assert "var_5pct" in out.columns
        assert "var_5pct_raw" not in out.columns
        assert "var_5pct_corr" not in out.columns
        assert out.iloc[0]["var_5pct"] == pytest.approx(0.1)

    def test_corr_signal_selected_for_corr_family(self):
        sig = pd.DataFrame({
            "cusip": ["A"], "date": [_me("2010-01")],
            "var_5pct_raw": [0.1], "var_5pct_corr": [0.08],
        })
        out = view(self._panel(), _cfg("corr"), signals=sig)
        assert out.iloc[0]["var_5pct"] == pytest.approx(0.08)

    def test_cross_family_raise_when_only_wrong_family_provided(self):
        # Configured for raw but only var_5pct_corr present → chimera, raise.
        sig = pd.DataFrame({
            "cusip": ["A"], "date": [_me("2010-01")],
            "var_5pct_corr": [0.08],
        })
        with pytest.raises(ValueError, match="A9"):
            view(self._panel(), _cfg("raw"), signals=sig)

    def test_signal_without_family_suffix_passes_through(self):
        sig = pd.DataFrame({
            "cusip": ["A"], "date": [_me("2010-01")],
            "rating": ["BBB"],
        })
        out = view(self._panel(), _cfg("raw"), signals=sig)
        assert "rating" in out.columns
        assert out.iloc[0]["rating"] == "BBB"

    def test_duplicate_signal_keys_raise(self):
        """A signals frame with duplicate (cusip, date) rows would silently
        fan out panel rows through the left-merge — must raise instead."""
        sig = pd.DataFrame({
            "cusip": ["A", "A"], "date": [_me("2010-01"), _me("2010-01")],
            "var_5pct_raw": [0.1, 0.2],
        })
        with pytest.raises(ValueError, match="duplicate"):
            view(self._panel(), _cfg("raw"), signals=sig)


# ---------------------------------------------------------------------------
# Dtype hygiene
# ---------------------------------------------------------------------------

class TestNoFutureWarnings:
    def test_stale_mask_emits_no_future_warning(self):
        """Pins the shift(1, fill_value=False) fix: the bool stale-mask
        propagation must not rely on deprecated object-dtype downcasting."""
        import warnings

        panel = _make_maximal([
            _maximal_row("A", "2010-01", p_raw=100.0, p_corr=99.0,
                         r_raw=0.05, r_corr=0.04,
                         ltd_corr="2009-11-01"),
            _maximal_row("A", "2010-02", p_raw=102.0, p_corr=101.0,
                         r_raw=0.02, r_corr=0.02,
                         ltd_corr="2010-02-25"),
        ])
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            view(panel, _cfg("corr", stale_mask=True), stale_threshold_days=30)
