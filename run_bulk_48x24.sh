#!/usr/bin/env bash
# Massive 48×24 dataset: Gaussian geometry, thickness U(0.5,1.5). No E_eff/mass filtering.
#
# Default --pool is large; RAM scales with N (each sample holds a (4,24,48) float32 in workers).
# Rough guide (8 workers, this machine class): ~6–8 samples/s → 100k samples ≈ 4–5 h wall.
# Tune POOL / WORKERS / SEED below or pass a custom out dir as first argument.
set -euo pipefail
cd "$(dirname "$0")"

OUT="${1:-}"
EXTRA=()
[[ -n "$OUT" ]] && EXTRA=(--out-dir "$OUT")

POOL="${POOL:-100000}"
WORKERS="${WORKERS:-8}"
SEED="${SEED:-2026}"

exec python3 lattice_thickness_picker.py \
  --bulk \
  --w 48 \
  --h 24 \
  --thick-low 0.5 \
  --thick-high 1.5 \
  --pool "$POOL" \
  --master-seed "$SEED" \
  --workers "$WORKERS" \
  --export-n 0 \
  --max-pool-seconds 28800 \
  --max-total-seconds 28800 \
  "${EXTRA[@]}"
