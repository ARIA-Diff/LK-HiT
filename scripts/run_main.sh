#!/usr/bin/env bash
# All main models × 4 tasks × 5 seeds (Table 2 / Table 3).
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/run_sweep.py --suite main "$@"
