"""
Offline unit tests for the P1 codegen-ablation sandbox (WS-C).

Every test passes under whichever mechanism ``detect_mechanism()`` selects on
this machine. The only guarantee that genuinely differs between mechanisms is
write confinement (seatbelt denies writes outside the jail; the shim does
not), so only that assertion is skipped under the shim. Network denial holds
under both -- kernel-enforced under seatbelt, socket-patch-enforced under the
shim -- and a plain control subprocess connecting to the same loopback
listener proves the denier is the sandbox, not the environment. No test
touches the network beyond 127.0.0.1.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest  # noqa: E402

from evaluation.codegen.sandbox import (  # noqa: E402
    SandboxSpec,
    detect_mechanism,
    parse_output_csv,
    run_sandboxed,
)


def _spec(jail_dir: Path, **overrides) -> SandboxSpec:
    return SandboxSpec(python_bin=Path(sys.executable), jail_dir=jail_dir, **overrides)


def test_passthrough_succeeds(tmp_path: Path) -> None:
    code = textwrap.dedent(
        """
        import csv
        import os

        with open(os.environ["OUTPUT_PATH"], "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["date", "portfolio_return"])
            writer.writerows(
                [
                    ["2024-01-31", 0.012],
                    ["2024-02-29", -0.004],
                    ["2024-03-31", 0.021],
                ]
            )
        """
    )
    result = run_sandboxed(code, _spec(tmp_path / "jail"))
    assert result.status == "ok", (result.reason, result.stderr_tail)
    assert result.reason is None
    assert result.exit_code == 0
    assert result.output_sha256 is not None
    assert len(result.output_sha256) == 64
    assert result.output_path is not None
    series = parse_output_csv(result.output_path)
    assert len(series) == 3
    assert series.name == "portfolio_return"


def test_denies_network(tmp_path: Path) -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        # "CONNECTED" lands in the date field only on a successful connect,
        # making that outcome observable either as a malformed-output verdict
        # or as a 1.0 flag -- both are asserted against below.
        code = textwrap.dedent(
            f"""
            import csv
            import os
            import socket

            row = ["2024-01-31", 0.0]
            try:
                connection = socket.create_connection(("127.0.0.1", {port}), timeout=2)
                connection.close()
                row = ["CONNECTED", 1.0]
            except Exception as exc:
                print("connect denied:", type(exc).__name__, exc)
            with open(os.environ["OUTPUT_PATH"], "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["date", "portfolio_return"])
                writer.writerow(row)
            """
        )
        result = run_sandboxed(code, _spec(tmp_path / "jail"))
        assert result.status == "ok", (result.reason, result.stderr_tail)
        assert parse_output_csv(result.output_path).iloc[0] == 0.0, (
            "sandboxed code reached the loopback listener"
        )

        control = subprocess.run(
            [
                sys.executable,
                "-c",
                f'import socket; socket.create_connection(("127.0.0.1", {port}), timeout=5).close()',
            ],
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert control.returncode == 0, control.stderr
    finally:
        listener.close()


def test_denies_write_outside_jail(tmp_path: Path) -> None:
    jail = tmp_path / "jail"
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    escape_path = outside_dir / "escape.txt"

    code = textwrap.dedent(
        f"""
        import csv
        import os

        flag = 0.0
        try:
            with open({str(escape_path)!r}, "w") as handle:
                handle.write("escaped the jail")
            flag = 1.0
        except Exception as exc:
            print("outside write denied:", type(exc).__name__, exc)
        with open(os.environ["OUTPUT_PATH"], "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["date", "portfolio_return"])
            writer.writerow(["2024-01-31", flag])
        """
    )
    result = run_sandboxed(code, _spec(jail))
    assert result.status == "ok", (result.reason, result.stderr_tail)
    if detect_mechanism() == "shim":
        pytest.skip(
            "shim mechanism does not confine file writes; "
            "outside-write denial is a seatbelt-only guarantee"
        )
    assert not escape_path.exists(), "sandboxed code wrote outside the jail"
    assert parse_output_csv(result.output_path).iloc[0] == 0.0


def test_output_missing_is_wont_run(tmp_path: Path) -> None:
    result = run_sandboxed("print('no output written')\n", _spec(tmp_path / "jail"))
    assert result.status == "wont_run"
    assert result.reason == "output_missing"
    assert result.exit_code == 0
    assert result.output_path is None
    assert result.output_sha256 is None


def test_output_malformed_is_wont_run(tmp_path: Path) -> None:
    code = textwrap.dedent(
        """
        import csv
        import os

        with open(os.environ["OUTPUT_PATH"], "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["month", "alpha"])
            writer.writerow(["2024-01-31", 0.01])
        """
    )
    result = run_sandboxed(code, _spec(tmp_path / "jail"))
    assert result.status == "wont_run"
    assert result.reason == "output_malformed"
    assert result.exit_code == 0


def test_nonzero_exit_is_wont_run(tmp_path: Path) -> None:
    result = run_sandboxed('raise RuntimeError("strategy blew up")\n', _spec(tmp_path / "jail"))
    assert result.status == "wont_run"
    assert result.reason == "nonzero_exit"
    assert result.exit_code not in (0, None)
    assert "strategy blew up" in result.stderr_tail


def test_timeout_kills(tmp_path: Path) -> None:
    code = "import time\ntime.sleep(30)\n"
    result = run_sandboxed(code, _spec(tmp_path / "jail", wall_clock_s=2))
    assert result.status == "wont_run"
    assert result.reason == "timeout"
    assert result.exit_code is None
    assert result.elapsed_s < 15


def test_parse_output_csv_breaches(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text(
        "date,portfolio_return\n2024-01-31,0.01\n2024-01-31,0.02\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate dates"):
        parse_output_csv(duplicate)

    extra = tmp_path / "extra.csv"
    extra.write_text("date,portfolio_return,leakage\n2024-01-31,0.01,x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly the columns"):
        parse_output_csv(extra)

    unparseable = tmp_path / "unparseable.csv"
    unparseable.write_text("date,portfolio_return\nnot-a-date,0.01\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dates failed to parse"):
        parse_output_csv(unparseable)


def test_no_api_keys_in_child_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOO_API_KEY", "super-secret-value")
    code = textwrap.dedent(
        """
        import csv
        import os

        keys = sorted(os.environ)
        print("child environment keys:", keys)
        leaked = sum(1 for key in keys if "API_KEY" in key)
        with open(os.environ["OUTPUT_PATH"], "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["date", "portfolio_return"])
            writer.writerow(["2024-01-31", float(leaked)])
        """
    )
    result = run_sandboxed(code, _spec(tmp_path / "jail"))
    assert result.status == "ok", (result.reason, result.stderr_tail)
    assert parse_output_csv(result.output_path).iloc[0] == 0.0, "an *_API_KEY variable leaked"
    assert "FOO_API_KEY" not in result.stdout_tail
    assert "super-secret-value" not in result.stdout_tail
