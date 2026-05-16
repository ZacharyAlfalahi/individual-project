"""
Global pytest fixtures.
"""
import pytest


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
