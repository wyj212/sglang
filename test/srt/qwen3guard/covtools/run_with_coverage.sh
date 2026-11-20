#!/usr/bin/env bash
# Run the Qwen3Guard-Stream guard tests under the self-contained line tracer,
# then report changed-line coverage for the adaptation's twelve commits.
#
# Usage:  bash test/srt/qwen3guard/covtools/run_with_coverage.sh [pytest args]
#
# The tracer lives in this directory and is picked up through sitecustomize, so
# no coverage package needs to be installed.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../../../.." && pwd)}"
OUT="${QGUARD_COV_OUT:-/tmp/qguard_cov}"

export PYTHONPATH="$HERE:$REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export QGUARD_COV=1
export QGUARD_COV_ROOT="${QGUARD_COV_ROOT:-$REPO/python/sglang}"
export QGUARD_COV_OUT="$OUT"

rm -rf "$OUT"
mkdir -p "$OUT"

cd "$REPO"
python -m pytest test/srt/qwen3guard/ -q "$@"

# Reporting must not itself be traced.
unset QGUARD_COV
python "$HERE/changed_line_coverage.py" \
    --repo "$REPO" \
    --dumps "$OUT" \
    --json-out "$OUT/report.json"
