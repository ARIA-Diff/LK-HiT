#!/usr/bin/env bash
# Seven LK-HiT ablations × 3 tasks × 5 seeds.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/run_sweep.py --suite ablations "$@"
