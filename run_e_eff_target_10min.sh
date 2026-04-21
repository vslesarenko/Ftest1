#!/usr/bin/env bash
# ~10 min: keep samples with E_eff in [E_MIN, E_MAX] (default 6000–8000). Stress/mass saved.
# Sparse bonds: SPARSE_FRACTION=0.55 BOND_KEEP_P=0.82 (random missing struts) to fill E band.
# Equivalence width for analysis defaults to 50 (--e-bin-width); override: E_BIN_W=100
# Narrow goal mode: GOAL=7000 ./run_e_eff_target_10min.sh  → goal±TOL (default TOL=50)
# Analyze: python3 analyze_e_eff_target_run.py pic/e_eff_target_*
set -euo pipefail
cd "$(dirname "$0")"

E_MIN="${E_MIN:-6000}"
E_MAX="${E_MAX:-8000}"
E_BIN_W="${E_BIN_W:-50}"
TOL="${TOL:-50}"
MAX_SECONDS="${MAX_SECONDS:-580}"
WORKERS="${WORKERS:-6}"
SPARSE_FRACTION="${SPARSE_FRACTION:-0.55}"
BOND_KEEP_P="${BOND_KEEP_P:-0.82}"
GOAL="${GOAL:-}"
OUT="${1:-}"

EXTRA=()
[[ -n "$OUT" ]] && EXTRA=(--out-dir "$OUT")

if [[ -n "$GOAL" ]]; then
  MODE_ARGS=(--goal "$GOAL" --tol "$TOL" --e-bin-width "$E_BIN_W")
else
  MODE_ARGS=(--e-min "$E_MIN" --e-max "$E_MAX" --e-bin-width "$E_BIN_W")
fi

exec python3 lattice_e_eff_target.py \
  "${MODE_ARGS[@]}" \
  --w 48 \
  --h 24 \
  --thick-low 0.5 \
  --thick-high 1.5 \
  --sparse-fraction "$SPARSE_FRACTION" \
  --bond-keep-p "$BOND_KEEP_P" \
  --max-seconds "$MAX_SECONDS" \
  --workers "$WORKERS" \
  --master-seed 2026 \
  "${EXTRA[@]}"
