#!/usr/bin/env bash
# Fully connected 48×24 lattice: Gaussian geometry only, thickness U(0.5, 1.5) per half-edge.
# Tune --pool and --workers for your machine; 48×24 is ~2× slower per solve than 40×20.
set -euo pipefail
cd "$(dirname "$0")"

OUT="${1:-}"
EXTRA=()
if [[ -n "$OUT" ]]; then
  EXTRA=(--out-dir "$OUT")
fi

exec python3 lattice_thickness_picker.py \
  --w 48 \
  --h 24 \
  --thick-low 0.5 \
  --thick-high 1.5 \
  --pool 800 \
  --workers 8 \
  --export-n 0 \
  --max-pool-seconds 28800 \
  --max-total-seconds 28800 \
  "${EXTRA[@]}"
