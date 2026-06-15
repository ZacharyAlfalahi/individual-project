#!/usr/bin/env bash
# Single source of truth for the licensed-data commit ban (FISD + TRACE).
#
# Scans the git index (git ls-files, which reflects staged additions) and refuses
# if any tracked path is licensed/redistribution-restricted data. Called by:
#   - .githooks/pre-commit   (refuses the commit)
#   - .githooks/pre-push     (refuses the push, so data never reaches GitHub)
#   - .github/workflows/data-governance.yml   (server-side backstop)
#   - tests/unit/test_data_governance.py       (verifiable invariant)
#
# Edit the FORBIDDEN/ALLOW patterns HERE only — every enforcement point reuses them.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

# Forbidden paths (POSIX ERE, relative to repo root):
#   - anything under data/fisd/              (FISD reference tables, Mergent-licensed)
#   - any data/trace_enhanced_* raw dump     (FINRA TRACE Enhanced, WRDS-licensed)
#   - any .parquet/.csv/.csv.gz under data/  (TRACE-derived panels, signals, headlines)
FORBIDDEN='^data/(fisd/|trace_enhanced_)|^data/.*\.(parquet|csv|csv\.gz)$'
# Allowlisted even inside data/: markdown notes and .gitkeep placeholders.
ALLOW='(/\.gitkeep$|\.md$)'

offenders=$(git ls-files | grep -E "$FORBIDDEN" | grep -Ev "$ALLOW" || true)

if [ -n "$offenders" ]; then
  echo "BLOCKED: licensed/raw data (FISD/TRACE) must never be committed or pushed." >&2
  echo "Offending tracked files:" >&2
  printf '  %s\n' $offenders >&2
  echo "Unstage with:  git rm --cached <file>   (the file stays on disk)" >&2
  exit 1
fi

echo "data-governance: no licensed/raw data tracked."
