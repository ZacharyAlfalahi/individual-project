"""
Template-reachability census for the Scientist (spec §6 eligibility census, run BEFORE the
mechanism library — the census does not need the library; the library is authored backwards
from it so every entry survives the eligibility filter by construction).

METHOD (per the build directive):
  * Inputs: the four §7 templates (prereg/templates/), the monthly_panel_corrected column list,
    data/development/signals/, and docs/reporting_delays.yaml (the macro conditioning universe).
  * Cross-product each template's permitted_fields enums (conditioning_variable × lag × form).
  * Drop any conditioning_variable not available (not a panel column, not a signals/ file, not a
    reporting_delays macro series) or not timing-feasible (lag below the variable's floor:
    reporting_delays.lag_months for macro, 1 month for bond-level per INVARIANT 3.2).
  * Group the surviving executable tuples by the strategy families each template supports.

OUTPUT: prereg/reachability_census.md — the executable (template, conditioning_variable, lag,
form) space per family, plus the FINDING-5 coverage analysis (which available conditioning
variables reach NO template enum — an artificially small reachable space would make any m
reduction wrong). This script REPORTS; it does not resolve enum decisions.

The raw cross-product is an UPPER BOUND on the reachable space: some (variable, form) pairs are
semantically invalid (e.g. rating × a liquidity-tercile restriction) and are rejected downstream
by the eligibility filter, not here.
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from shared.licensed_inputs import require_licensed_input  # noqa: E402
TEMPLATES_DIR = REPO_ROOT / "prereg" / "templates"
PANEL = REPO_ROOT / "data" / "development" / "monthly_panel_corrected.parquet"
SIGNALS_DIR = REPO_ROOT / "data" / "development" / "signals"
REPORTING_DELAYS = REPO_ROOT / "docs" / "reporting_delays.yaml"
OUT_FILE = REPO_ROOT / "prereg" / "reachability_census.md"

# Curated conditioning-candidate universe (identifiers / returns / prices are not conditioning
# characteristics and are excluded): macro regime series + bond-level characteristics.
MACRO_VARS = ["baa_aaa_spread", "vix", "term_spread"]
BOND_LEVEL_VARS = [
    "rating", "investment_grade", "var_5pct", "gamma_illiq",
    "bond_vol", "size", "time_to_maturity", "total_vol",
]


def load_templates() -> list[dict]:
    return [yaml.safe_load(p.read_text()) for p in sorted(TEMPLATES_DIR.glob("*.yaml"))]


def available_variables() -> tuple[set[str], dict[str, int]]:
    """Available conditioning variables and their timing floor (months)."""
    panel_cols = set(pd.read_parquet(require_licensed_input(PANEL, "corrected monthly panel"), columns=None).columns)
    signal_stems = {p.stem for p in SIGNALS_DIR.glob("*.parquet")}
    delays = yaml.safe_load(REPORTING_DELAYS.read_text())["reporting_delays"]
    macro = set(delays)

    available = set(panel_cols) | signal_stems | macro
    floor: dict[str, int] = {}
    for v in available:
        floor[v] = int(delays[v]["lag_months"]) if v in delays else 1  # bond-level floor = 1
    return available, floor


def executable_tuples(template: dict, available: set[str], floor: dict[str, int]) -> list[dict]:
    pf = template["permitted_fields"]
    cvars = pf["conditioning_variable"]["allowed"]
    lags = pf["conditioning_lag_months"]["allowed"]
    forms = pf["interaction_form"]["allowed"]
    out = []
    for cvar, lag, form in product(cvars, lags, forms):
        if cvar not in available:
            continue                       # dropped: variable not present in the data
        if lag < floor.get(cvar, 1):
            continue                       # dropped: timing-infeasible
        out.append({"conditioning_variable": cvar, "lag": lag, "form": form})
    return out


def render(templates, available, floor) -> str:
    lines = ["# Scientist reachability census (generated — do not edit by hand)", ""]
    lines.append(
        "Source: `scripts/scientist_reachability_census.py` over `prereg/templates/`, "
        "`monthly_panel_corrected`, `data/development/signals/`, `docs/reporting_delays.yaml`."
    )
    lines.append("")

    per_family: dict[str, int] = {}
    per_template_counts = []
    enum_membership: dict[str, list[str]] = {v: [] for v in MACRO_VARS + BOND_LEVEL_VARS}

    for tmpl in templates:
        tid = tmpl["template_id"]
        tuples = executable_tuples(tmpl, available, floor)
        per_template_counts.append((tid, tmpl["supported_families"], len(tuples)))
        for fam in tmpl["supported_families"]:
            per_family[fam] = per_family.get(fam, 0) + len(tuples)
        for v in tmpl["permitted_fields"]["conditioning_variable"]["allowed"]:
            if v in enum_membership:
                enum_membership[v].append(tid.replace("_v1", ""))

    lines.append("## Executable (template, conditioning_variable, lag, form) tuples")
    lines.append("")
    lines.append("| template | supported families | executable tuples (upper bound) |")
    lines.append("| --- | --- | --- |")
    for tid, fams, n in per_template_counts:
        lines.append(f"| {tid} | {', '.join(fams)} | {n} |")
    lines.append("")
    lines.append("## Reachable space per strategy family")
    lines.append("")
    lines.append("| strategy family | executable tuples across all templates |")
    lines.append("| --- | --- |")
    for fam, n in sorted(per_family.items()):
        lines.append(f"| {fam} | {n} |")
    lines.append("")

    # ---- FINDING F5: enum coverage vs the available conditioning universe -------------------
    lines.append("## Finding F5 — enum coverage vs the available conditioning universe")
    lines.append("")
    lines.append(
        "REPORT ONLY — not resolved here. Each available conditioning-candidate variable and "
        "the templates whose `conditioning_variable` enum reaches it. A variable reachable by NO "
        "template is an omission that would shrink the reachable space artificially."
    )
    lines.append("")
    lines.append("| variable | kind | available? | reached by templates |")
    lines.append("| --- | --- | --- | --- |")
    for v in MACRO_VARS + BOND_LEVEL_VARS:
        kind = "macro" if v in MACRO_VARS else "bond-level"
        avail = "yes" if v in available else "NO"
        reach = ", ".join(enum_membership[v]) if enum_membership[v] else "**NONE**"
        lines.append(f"| {v} | {kind} | {avail} | {reach} |")
    lines.append("")
    unreached = [v for v in MACRO_VARS + BOND_LEVEL_VARS
                 if v in available and not enum_membership[v]]
    lines.append(
        f"**Available but reached by no template enum:** "
        f"{', '.join(unreached) if unreached else '(none)'}."
    )
    lines.append("")
    lines.append(
        "**`total_vol` exclusion (recorded, corrected reason):** `total_vol` in the panel is "
        "**dollar volume, a trading-activity METADATA column** (characteristic_registry_spec.md; "
        "listed among metadata columns beside `n_trades` in bias_toggle_registry_spec.md) — NOT a "
        "volatility measure, so it is NOT redundant with `bond_vol` (the 24-month return vol, "
        "computed fresh from returns). It was never in the characteristic registry; liquidity is "
        "represented by `gamma_illiq`. Secondary reason to keep it out: raw dollar volume is "
        "confounded with issue size and mechanically correlated with the stale-price toggle."
    )
    lines.append("")
    lines.append(
        "**F5 open design decisions (not resolved here):** (a) template 1 "
        "`lagged_binary_regime_interaction_v1` is macro-only per §7 — the bond-level "
        "characteristics gamma_illiq/var_5pct/bond_vol/rating are excluded from it (they are "
        "reached via templates 3/4). Confirm this is deliberate. (b) Templates 2–4 enums are "
        "AUTHORED here, not spec-given — confirm them; in particular whether template 2 "
        "(continuous interaction) should also admit bond-level continuous characteristics, which "
        "would enlarge the reachable space. (c) Confirm any 'NONE' row above is an intended "
        "exclusion, not an omission."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    templates = load_templates()
    available, floor = available_variables()
    text = render(templates, available, floor)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(text)
    print(text)
    print(f"Written: {OUT_FILE}")


if __name__ == "__main__":
    main()
