"""
Consistency guard between the TRACKED pre-registration register (prereg/scientist_prereg.yaml)
and the UNTRACKED working copies it mirrors (docs/scientist_protocol.yaml,
docs/extension_1_config.yaml, the docs/thresholds.yaml `scientist:` block).

Because docs/ is gitignored, the frozen constants must also live in version control (approval
decision: "also track the values"). This test asserts the two agree on every load-bearing
number, so the tracked record and the working docs cannot silently drift — a change to one
without the other fails here, which is exactly the tamper/propagation tripwire we want.
"""

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shared.handoff.scientist_case import load_entry_rule_params  # noqa: E402

REGISTER = REPO_ROOT / "prereg" / "scientist_prereg.yaml"
PROTOCOL = REPO_ROOT / "docs" / "scientist_protocol.yaml"
EXTENSION_1 = REPO_ROOT / "docs" / "extension_1_config.yaml"


def _load(p: Path) -> dict:
    if not p.exists():
        pytest.skip(f"{p} absent (gitignored working copy not present in this checkout)")
    return yaml.safe_load(p.read_text())


@pytest.fixture(scope="module")
def reg() -> dict:
    return _load(REGISTER)


@pytest.fixture(scope="module")
def proto() -> dict:
    return _load(PROTOCOL)


@pytest.fixture(scope="module")
def ext1() -> dict:
    return _load(EXTENSION_1)


def test_windows_match(reg, proto):
    for key in ("panel_development", "primary_inference", "evaluation_holdout"):
        assert reg["windows"][key]["n_months"] == proto["windows"][key]["n_months"], key
    assert reg["windows"]["primary_inference"]["n_months"] == 209  # T, DRF/CRF-bound


def test_entry_rule_resolves_to_register(reg):
    # The register's resolved theta/q must equal what the real thresholds.yaml block resolves.
    params = load_entry_rule_params()
    assert params.theta == pytest.approx(reg["entry_rule"]["resolved_theta"])
    assert params.q == pytest.approx(reg["entry_rule"]["resolved_q"])


def test_generation_and_inference_match(reg, proto):
    assert reg["generation"]["seeds_k"] == proto["generation"]["seeds"]["k"] == 5
    assert (
        reg["generation"]["proposals_per_strategy"]["default"]
        == proto["generation"]["proposals_per_strategy"]["default"]
    )
    assert reg["inference"]["bh_fdr_q"] == proto["inference"]["bh_fdr_q"] == 0.10


def test_cpcv_embargo_is_months(reg, proto):
    # A4 — embargo is months, not blocks; the register and protocol must agree.
    assert reg["cpcv"]["embargo"]["rule"] == proto["cpcv"]["embargo"]["rule"] == "holding_period_months"
    assert reg["cpcv"]["embargo"]["floor_months"] == proto["cpcv"]["embargo"]["floor_months"] == 1
    assert "embargo_blocks" not in proto["cpcv"]  # the superseded spec key must be gone


def test_holdout_cap_and_gate(reg, proto):
    assert reg["holdout"]["cap"] == proto["holdout"]["cap"] == 3
    # A5 — no self-opening date; the protocol carries artefact open_conditions, not `opens:`.
    assert "opens" not in proto["holdout"]
    assert "open_conditions" in proto["holdout"]


def test_costs_source_deferred(reg, proto):
    assert reg["costs"]["source"] == proto["costs"]["source"] == "UNRESOLVED_pending_citation"


def test_extension_1_median_matches(reg, ext1):
    assert reg["extension_1"]["median_baa_aaa_spread_pp"] == pytest.approx(
        ext1["median_split"]["median_baa_aaa_spread_pp"]
    ) == pytest.approx(0.935)


def test_mde_bounds_reproduce(reg):
    # The recorded MDE bounds must equal what the generator computes from (m, q, T).
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from generate_scientist_mde import ladder  # noqa: E402

    rows = ladder(reg["mde"]["m"], reg["mde"]["q"], reg["mde"]["T"])
    assert round(rows[0]["ann_sharpe"], 2) == reg["mde"]["rank_1_ann_sharpe"] == 0.57
    assert round(rows[-1]["ann_sharpe"], 2) == reg["mde"]["rank_m_ann_sharpe"] == 0.39
