"""
Global pytest fixtures.
"""
import pathlib

import pytest


_CANONICAL_TEXTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "evaluation" / "canonical_texts"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """A public clone does not ship evaluation/canonical_texts/ (the frozen paper
    texts are copyrighted, gitignored — see README). When that directory is absent,
    reclassify any "missing canonical text" failure/error as a SKIP so the suite is
    all-pass/skip on a clone without the licensed inputs. Gated on the directory being
    absent, so a genuinely missing single file when the set IS present still fails.
    Covers every read path (Path.open FileNotFoundError and the loader's own
    LibrarianSchemaError both carry the canonical_texts path in their message).
    """
    outcome = yield
    report = outcome.get_result()
    if report.when in ("setup", "call") and report.failed and not _CANONICAL_TEXTS_DIR.exists():
        exc = getattr(call, "excinfo", None)
        if exc is not None and "canonical_texts" in str(exc.value):
            report.outcome = "skipped"
            report.longrepr = (
                str(item.location[0]),
                item.location[1] or 0,
                "Skipped: requires gitignored canonical texts "
                "(evaluation/canonical_texts/; copyrighted, not shipped — see README)",
            )


@pytest.fixture(autouse=True)
def block_holdout_reads(monkeypatch):
    """
    Prevent accidental reads from data/holdout/ during tests.
    Holdout is written once by preprocess_trace.py and never read during development.
    """
    _real_open = open

    def guarded_open(path, *args, **kwargs):
        if "holdout" in str(path):
            raise AssertionError(
                f"Attempted to read from holdout path: {path}\n"
                "Holdout data must not be accessed during development or testing."
            )
        return _real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", guarded_open)

    try:
        import pandas as pd

        _real_pd_read = pd.read_parquet

        def guarded_pd_read(path, *args, **kwargs):
            if "holdout" in str(path):
                raise AssertionError(f"Attempted pd.read_parquet on holdout: {path}")
            return _real_pd_read(path, *args, **kwargs)

        monkeypatch.setattr(pd, "read_parquet", guarded_pd_read)
    except ImportError:
        pass

    try:
        import polars as pl

        _real_pl_read = pl.read_parquet

        def guarded_pl_read(path, *args, **kwargs):
            if "holdout" in str(path):
                raise AssertionError(f"Attempted pl.read_parquet on holdout: {path}")
            return _real_pl_read(path, *args, **kwargs)

        monkeypatch.setattr(pl, "read_parquet", guarded_pl_read)

        _real_pl_scan_parquet = pl.scan_parquet

        def guarded_pl_scan_parquet(path, *args, **kwargs):
            if "holdout" in str(path):
                raise AssertionError(f"Attempted pl.scan_parquet on holdout: {path}")
            return _real_pl_scan_parquet(path, *args, **kwargs)

        monkeypatch.setattr(pl, "scan_parquet", guarded_pl_scan_parquet)

        _real_pl_scan_csv = pl.scan_csv

        def guarded_pl_scan_csv(path, *args, **kwargs):
            if "holdout" in str(path):
                raise AssertionError(f"Attempted pl.scan_csv on holdout: {path}")
            return _real_pl_scan_csv(path, *args, **kwargs)

        monkeypatch.setattr(pl, "scan_csv", guarded_pl_scan_csv)
    except ImportError:
        pass

    try:
        # pyarrow.parquet.ParquetFile is the only data-bearing read entry
        # point we need to guard — bounce_back_filter.py uses it via
        # iter_batches to stream row data. pq.read_metadata and pq.read_schema
        # are header-only (no row data crosses the boundary) and are used
        # legitimately by preprocess_trace.py for row-count verification of
        # its own write to the holdout partition, so they remain allowed.
        import pyarrow.parquet as _pq

        _real_pq_parquetfile = _pq.ParquetFile

        def _guarded_parquetfile(path, *args, **kwargs):
            if "holdout" in str(path):
                raise AssertionError(
                    f"Attempted pq.ParquetFile on holdout: {path}"
                )
            return _real_pq_parquetfile(path, *args, **kwargs)

        monkeypatch.setattr(_pq, "ParquetFile", _guarded_parquetfile)
    except ImportError:
        pass
