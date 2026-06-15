#!/usr/bin/env bash
# One-time per clone: point git at the tracked .githooks/ directory and make the
# hook + guard scripts executable. Idempotent — safe to re-run.
set -euo pipefail

root=$(git rev-parse --show-toplevel)
git -C "$root" config core.hooksPath .githooks
chmod +x "$root"/.githooks/pre-commit "$root"/.githooks/pre-push \
         "$root"/scripts/check_no_licensed_data.sh

echo "Hooks activated (core.hooksPath=.githooks)."
echo "pre-commit + pre-push now refuse any FISD/TRACE licensed data."
