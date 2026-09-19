"""
Sandboxed execution of LLM-generated strategy code -- P1 codegen ablation (WS-C).

Generated code runs under deny-by-default constraints: network denied, file
writes confined to a per-run jail directory, wall clock bounded. The output
contract is a single CSV (``date,portfolio_return``) at ``$OUTPUT_PATH``;
anything else is a typed ``wont_run``. Generated code never executes outside
this sandbox (invariant I7 of the extensions build spec).

Two mechanisms, auto-selected per machine:

**seatbelt** (primary, macOS): ``/usr/bin/sandbox-exec`` with a per-run profile
written into the jail -- ``(deny network*)`` and ``(deny file-write*)`` with
allow carve-outs for the jail subtree and ``/dev/null``. Kernel-enforced.

**shim** (fallback, auto-selected when ``sandbox-exec`` is absent or its probe
fails): a plain subprocess with a jail-local ``sitecustomize.py`` injected via
``PYTHONPATH`` that replaces ``socket.socket`` / ``socket.create_connection``
with functions raising ``RuntimeError("network disabled in codegen sandbox")``
and installs a ``sys.addaudithook`` denying every ``socket.`` event. The shim
is best-effort: adversarial code could bypass it (generated strategy code is
not adversarial), and unlike seatbelt it does not confine file writes. The
residual risk is recorded in the P1 mini-contract.

The child environment is always built from scratch -- the parent environment
(and hence anything matching ``*_API_KEY``) is never passed through.
"""

from __future__ import annotations

import hashlib
import resource
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")

_TAIL_CHARS = 2000

_REQUIRED_COLUMNS = ("date", "portfolio_return")

_SEATBELT_PROFILE = """\
(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write* (subpath "{jail}"))
(allow file-write* (literal "/dev/null"))
"""

# Appended once per ``SandboxSpec.deny_read_paths`` entry. Seatbelt applies the last
# matching rule, so these override the profile's ``(allow default)``.
_SEATBELT_DENY_READ = '(deny file-read* (subpath "{path}"))\n'

_SHIM_SITECUSTOMIZE = '''\
"""Jail-local network shim for the codegen sandbox (best-effort; see sandbox.py)."""

import socket
import sys

_MESSAGE = "network disabled in codegen sandbox"


def _denied(*args, **kwargs):
    raise RuntimeError(_MESSAGE)


socket.socket = _denied
socket.create_connection = _denied


def _deny_socket_events(event, args):
    if event.startswith("socket."):
        raise RuntimeError(_MESSAGE)


sys.addaudithook(_deny_socket_events)
'''


@dataclass(frozen=True)
class SandboxSpec:
    """What to run and under which bounds; ``mechanism`` is normally ``"auto"``.

    ``deny_read_paths`` lists directory trees the generated code must not read (for example
    the holdout partition or the oracle factor files). Only seatbelt can enforce it, so a
    non-empty list under the shim raises rather than silently running unconfined."""

    python_bin: Path
    jail_dir: Path
    panel_path: Path | None = None
    output_name: str = "out/portfolio_returns.csv"
    wall_clock_s: int = 600
    memory_mb: int = 8192
    mechanism: str = "auto"
    deny_read_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class SandboxResult:
    """Typed outcome of one sandboxed run; ``status`` is ``"ok"`` or ``"wont_run"``."""

    status: str
    reason: str | None
    exit_code: int | None
    stdout_tail: str
    stderr_tail: str
    elapsed_s: float
    mechanism: str
    output_path: Path | None
    output_sha256: str | None


_detected_mechanism: str | None = None


def detect_mechanism() -> str:
    """Return ``"seatbelt"`` iff ``sandbox-exec`` exists and passes a trivial probe.

    Otherwise return ``"shim"``. The probe result is cached at module level:
    the mechanism is a property of the machine, not of the run.
    """
    global _detected_mechanism
    if _detected_mechanism is None:
        _detected_mechanism = "seatbelt" if _seatbelt_probe() else "shim"
    return _detected_mechanism


def _seatbelt_probe() -> bool:
    if not _SANDBOX_EXEC.exists():
        return False
    with tempfile.TemporaryDirectory() as tmp:
        profile = Path(tmp) / "probe.sb"
        profile.write_text("(version 1)\n(allow default)\n", encoding="utf-8")
        try:
            proc = subprocess.run(
                [str(_SANDBOX_EXEC), "-f", str(profile), "/usr/bin/true"],
                capture_output=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return proc.returncode == 0


def parse_output_csv(path: Path) -> pd.Series:
    """Validate the single-CSV output contract and return the portfolio returns.

    Requires exactly the columns ``date`` and ``portfolio_return`` (any order,
    nothing extra), parseable dates, float-coercible returns, no duplicate
    dates, and at least one row. Returns a float Series named
    ``portfolio_return`` indexed by the parsed dates sorted ascending. Raises
    ``ValueError`` naming the specific breach otherwise.
    """
    frame = pd.read_csv(path)
    columns = sorted(frame.columns)
    if columns != sorted(_REQUIRED_COLUMNS):
        raise ValueError(
            f"output CSV must contain exactly the columns "
            f"{sorted(_REQUIRED_COLUMNS)}; found {columns}"
        )
    if len(frame) == 0:
        raise ValueError("output CSV must contain at least one row")
    try:
        dates = pd.to_datetime(frame["date"], errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"output CSV dates failed to parse: {exc}") from exc
    try:
        values = frame["portfolio_return"].astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"output CSV returns are not float-coercible: {exc}") from exc
    if dates.duplicated().any():
        offenders = dates[dates.duplicated()].unique().tolist()
        raise ValueError(f"output CSV contains duplicate dates: {offenders}")
    series = pd.Series(
        values.to_numpy(dtype=float),
        index=pd.DatetimeIndex(dates, name="date"),
        name="portfolio_return",
    )
    return series.sort_index()


def seatbelt_profile(jail: Path, deny_read_paths: tuple[Path, ...] = ()) -> str:
    """The per-run seatbelt profile. With no deny paths it is the base P1 profile;
    each deny path is resolved first, because seatbelt subpath rules match real paths only."""
    profile = _SEATBELT_PROFILE.format(jail=jail)
    for path in deny_read_paths:
        profile += _SEATBELT_DENY_READ.format(path=Path(path).resolve())
    return profile


def run_sandboxed(code: str, spec: SandboxSpec) -> SandboxResult:
    """Execute ``code`` inside the jail and enforce the single-CSV contract.

    The child environment is built from scratch (never inherited, so nothing
    matching ``*_API_KEY`` can leak) and holds exactly ``PANEL_PATH``,
    ``OUTPUT_PATH``, ``TMPDIR``, ``HOME``, ``PYTHONDONTWRITEBYTECODE`` and
    ``PATH`` -- plus ``PYTHONPATH`` (the jail) under the shim mechanism.
    """
    mechanism = detect_mechanism() if spec.mechanism == "auto" else spec.mechanism
    if mechanism not in ("seatbelt", "shim"):
        raise ValueError(f"unknown sandbox mechanism {spec.mechanism!r}")
    if spec.deny_read_paths and mechanism != "seatbelt":
        raise ValueError("deny_read_paths needs the seatbelt mechanism; the shim cannot deny reads")

    # Resolve to the real path: on darwin /tmp and /var are symlinks into
    # /private, and seatbelt subpath rules match real paths only.
    jail = spec.jail_dir.resolve()
    jail.mkdir(parents=True, exist_ok=True)
    (jail / "out").mkdir(parents=True, exist_ok=True)
    (jail / "tmp").mkdir(parents=True, exist_ok=True)
    output_path = (jail / spec.output_name).resolve()
    if not output_path.is_relative_to(jail):
        raise ValueError(f"output_name {spec.output_name!r} escapes the jail")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    job_path = jail / "job.py"
    job_path.write_text(code, encoding="utf-8")

    env = {
        "PANEL_PATH": "" if spec.panel_path is None else str(spec.panel_path),
        "OUTPUT_PATH": str(output_path),
        "TMPDIR": str(jail / "tmp"),
        "HOME": str(jail),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PATH": "/usr/bin:/bin",
    }
    if mechanism == "shim":
        (jail / "sitecustomize.py").write_text(_SHIM_SITECUSTOMIZE, encoding="utf-8")
        env["PYTHONPATH"] = str(jail)
        argv = [str(spec.python_bin), str(job_path)]
    else:
        profile_path = jail / "profile.sb"
        profile_path.write_text(seatbelt_profile(jail, spec.deny_read_paths), encoding="utf-8")
        argv = [
            str(_SANDBOX_EXEC),
            "-f",
            str(profile_path),
            str(spec.python_bin),
            str(job_path),
        ]

    def _apply_rlimits() -> None:
        # Best-effort only: darwin's RLIMIT_AS is largely unenforced (and
        # RLIMIT_DATA only partially honoured), so the subprocess timeout
        # below is the enforced backstop.
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (spec.wall_clock_s, spec.wall_clock_s))
        except (ValueError, OSError):
            pass
        try:
            limit_bytes = spec.memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_DATA, (limit_bytes, limit_bytes))
        except (ValueError, OSError):
            pass

    start = time.monotonic()

    def _wont_run(reason: str, exit_code: int | None, stdout: str, stderr: str) -> SandboxResult:
        return SandboxResult(
            status="wont_run",
            reason=reason,
            exit_code=exit_code,
            stdout_tail=stdout,
            stderr_tail=stderr,
            elapsed_s=time.monotonic() - start,
            mechanism=mechanism,
            output_path=None,
            output_sha256=None,
        )

    try:
        proc = subprocess.run(
            argv,
            cwd=str(jail),
            env=env,
            capture_output=True,
            timeout=spec.wall_clock_s,
            preexec_fn=_apply_rlimits,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return _wont_run("timeout", None, _tail(exc.stdout), _tail(exc.stderr))
    except OSError as exc:
        return _wont_run("spawn_error", None, "", str(exc)[-_TAIL_CHARS:])

    stdout_tail = _tail(proc.stdout)
    stderr_tail = _tail(proc.stderr)
    if proc.returncode != 0:
        return _wont_run("nonzero_exit", proc.returncode, stdout_tail, stderr_tail)
    if not output_path.is_file():
        return _wont_run("output_missing", 0, stdout_tail, stderr_tail)
    try:
        parse_output_csv(output_path)
    except ValueError:
        return _wont_run("output_malformed", 0, stdout_tail, stderr_tail)
    return SandboxResult(
        status="ok",
        reason=None,
        exit_code=0,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        elapsed_s=time.monotonic() - start,
        mechanism=mechanism,
        output_path=output_path,
        output_sha256=hashlib.sha256(output_path.read_bytes()).hexdigest(),
    )


def _tail(stream: bytes | str | None) -> str:
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        stream = stream.decode("utf-8", errors="replace")
    return stream[-_TAIL_CHARS:]
