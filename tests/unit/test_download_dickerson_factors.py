"""
Unit tests for scripts/download_dickerson_factors.py.

Covers:
  - Format discovery: bare CSV, zip with the named member, single-member
    fallback, ambiguous zips, __MACOSX junk, xlsx-is-a-zip disambiguation,
    HTML error pages.
  - Date handling: date strings, yyyymm ints/floats, mixed-case headers,
    unparseable rows dropped (never a "NaT" string in year_month).
  - Window discipline: the dev-boundary derivation and the truncation
    tripwires — post-boundary rows can never survive to the written parquet.

No network: fetch_bytes is never called (monkeypatched in the end-to-end test).
"""

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import download_dickerson_factors as ddf  # noqa: E402


def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _factor_frame(dates, **cols) -> pd.DataFrame:
    base = {"date": dates,
            "str*": [0.01] * len(dates),
            "mom6_1": [0.002] * len(dates)}
    base.update(cols)
    return pd.DataFrame(base)


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, raw in members.items():
            zf.writestr(name, raw)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# parse_payload — format discovery
# ---------------------------------------------------------------------------

def test_parse_payload_csv_roundtrip():
    df = _factor_frame(["2004-08-31", "2004-09-30"])
    out = ddf.parse_payload(_csv_bytes(df))
    assert list(out.columns) == ["date", "str*", "mom6_1"]
    assert out["str*"].tolist() == [0.01, 0.01]


def test_parse_payload_lowercases_and_drops_unnamed():
    df = pd.DataFrame({"Unnamed: 0": [0, 1], "Date": ["2004-08-31", "2004-09-30"],
                       "STR*": [0.01, 0.02]})
    out = ddf.parse_payload(_csv_bytes(df))
    assert list(out.columns) == ["date", "str*"]


def test_parse_payload_zip_selects_named_member():
    """The real distribution: six CSVs + a README — the exact ZIP_MEMBER wins."""
    wanted = _factor_frame(["2004-08-31"])
    decoy = _factor_frame(["1999-01-31"], **{"str*": [9.9]})
    raw = _zip_bytes({
        "README_factor_time_series.txt": b"readme",
        "single_sort_exc_ig.csv": _csv_bytes(decoy),
        ddf.ZIP_MEMBER: _csv_bytes(wanted),
        "single_sort_dur_all.csv": _csv_bytes(decoy),
    })
    out = ddf.parse_payload(raw)
    assert out["str*"].tolist() == [0.01]


def test_parse_payload_zip_single_member_fallback():
    raw = _zip_bytes({"other_name.csv": _csv_bytes(_factor_frame(["2004-08-31"]))})
    out = ddf.parse_payload(raw)
    assert out["mom6_1"].tolist() == [0.002]


def test_parse_payload_zip_macosx_junk_ignored():
    raw = _zip_bytes({
        "__MACOSX/._junk.csv": b"junk",
        "only.csv": _csv_bytes(_factor_frame(["2004-08-31"])),
    })
    assert len(ddf.parse_payload(raw)) == 1


def test_parse_payload_zip_ambiguous_members_raises():
    raw = _zip_bytes({
        "a.csv": _csv_bytes(_factor_frame(["2004-08-31"])),
        "b.csv": _csv_bytes(_factor_frame(["2004-08-31"])),
    })
    with pytest.raises(ValueError, match="exactly one data member"):
        ddf.parse_payload(raw)


def test_parse_payload_xlsx_not_treated_as_zip():
    """An xlsx is itself a zip — the [Content_Types].xml branch must win."""
    pytest.importorskip("openpyxl")
    buf = io.BytesIO()
    _factor_frame(["2004-08-31", "2004-09-30"]).to_excel(buf, index=False)
    out = ddf.parse_payload(buf.getvalue())
    assert "mom6_1" in out.columns
    assert len(out) == 2


def test_parse_payload_html_error_page_raises():
    html = b"<!DOCTYPE html><html><head><title>404</title></head><body>gone</body></html>"
    with pytest.raises(ValueError):
        ddf.parse_payload(html)


def test_parse_payload_duplicate_columns_raise():
    raw = b"date,STR*,str*\n2004-08-31,0.1,0.2\n"
    with pytest.raises(ValueError, match="duplicate column"):
        ddf.parse_payload(raw)


# ---------------------------------------------------------------------------
# _to_year_month — date discovery
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dates", [
    ["2004-08-31", "2004-09-30"],            # date strings
    [200408, 200409],                        # yyyymm ints
    [200408.0, 200409.0],                    # yyyymm floats
])
def test_to_year_month_variants(dates):
    out = ddf._to_year_month(pd.DataFrame({"date": dates, "x": [1.0, 2.0]}))
    assert out["year_month"].tolist() == ["2004-08", "2004-09"]
    assert list(out.columns) == ["year_month", "x"]


def test_to_year_month_unparseable_rows_dropped_not_nat_strings():
    out = ddf._to_year_month(
        pd.DataFrame({"date": ["2004-08-31", "not-a-date"], "x": [1.0, 2.0]})
    )
    assert out["year_month"].tolist() == ["2004-08"]
    assert "NaT" not in set(out["year_month"])


def test_to_year_month_no_date_column_raises():
    with pytest.raises(ValueError, match="no date column"):
        ddf._to_year_month(pd.DataFrame({"a": [1], "b": [2]}))


# ---------------------------------------------------------------------------
# Boundary + truncation tripwires
# ---------------------------------------------------------------------------

def test_dev_boundary_period_is_2021_12():
    assert ddf._dev_boundary_period() == pd.Period("2021-12", freq="M")


def test_truncate_fencepost():
    """TRIPWIRE — exact boundary semantics: 2021-12 survives, 2022-01 dropped."""
    df = pd.DataFrame({"year_month": ["2021-11", "2021-12", "2022-01"],
                       "x": [1.0, 2.0, 3.0]})
    out = ddf.truncate_to_dev(df, pd.Period("2021-12", freq="M"))
    assert out["year_month"].tolist() == ["2021-11", "2021-12"]


def test_truncate_post_boundary_rows_cannot_survive():
    """TRIPWIRE — post-boundary rows can never survive truncation, even for
    adversarial unsorted/duplicated input spanning deep into the holdout."""
    months = ["2023-05", "2021-10", "2025-09", "2021-12", "2022-01",
              "2021-12", "2021-11", "2024-03"]
    df = pd.DataFrame({"year_month": months, "x": range(len(months))})
    out = ddf.truncate_to_dev(df, pd.Period("2021-12", freq="M"))
    periods = pd.PeriodIndex(out["year_month"], freq="M")
    assert (periods <= pd.Period("2021-12", freq="M")).all()
    assert not any(m.startswith(("2022", "2023", "2024", "2025"))
                   for m in out["year_month"])
    assert out["year_month"].is_monotonic_increasing
    assert not out["year_month"].duplicated().any()


def test_truncate_all_post_boundary_raises():
    df = pd.DataFrame({"year_month": ["2022-01", "2023-06"], "x": [1.0, 2.0]})
    with pytest.raises(ValueError, match="zero rows"):
        ddf.truncate_to_dev(df, pd.Period("2021-12", freq="M"))


def test_truncate_duplicate_months_deduped():
    df = pd.DataFrame({"year_month": ["2020-01", "2020-01", "2020-02"],
                       "x": [1.0, 9.0, 2.0]})
    out = ddf.truncate_to_dev(df, pd.Period("2021-12", freq="M"))
    assert out["year_month"].tolist() == ["2020-01", "2020-02"]
    assert out["x"].tolist() == [1.0, 2.0]  # keep="first"


def test_build_parquet_end_to_end_never_writes_holdout(tmp_path, monkeypatch):
    """TRIPWIRE — the at-rest guarantee: a payload containing holdout-era rows
    yields a parquet with zero post-boundary rows and no .tmp residue."""
    out_file = tmp_path / "dickerson_factor_returns.parquet"
    monkeypatch.setattr(ddf, "OUT_FILE", out_file)
    dates = ["2020-12-31", "2021-12-31", "2022-01-31", "2024-06-30", "2025-09-30"]
    n = ddf.build_parquet(_csv_bytes(_factor_frame(dates)))
    assert n == 2
    written = pd.read_parquet(out_file)
    assert written["year_month"].tolist() == ["2020-12", "2021-12"]
    assert not (pd.PeriodIndex(written["year_month"], freq="M")
                > pd.Period("2021-12", freq="M")).any()
    assert list(tmp_path.glob("*.tmp")) == []


# ---------------------------------------------------------------------------
# Real-artifact smoke (skipped until the downloader has run)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not ddf.OUT_FILE.exists(),
                    reason="dickerson_factor_returns.parquet not downloaded")
def test_real_artifact_is_dev_only():
    """TRIPWIRE — the on-disk artifact contains no holdout-era rows."""
    df = pd.read_parquet(ddf.OUT_FILE)
    assert df["year_month"].max() <= "2021-12"
    assert "str*" in df.columns and "mom6_1" in df.columns
