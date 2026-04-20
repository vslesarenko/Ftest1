#!/usr/bin/env python3
"""
Sample random geometries + connectivity with **uniform beam weights** (same thickness),
solve mechanics, then pick lattices with **similar E_eff and mass** but **widely varying σ_max**.

Uses only:
  • partial sparse lattices (soft bonds → mask τ → w=1 on survivors)
  • full grids with Gaussian geometric perturbation (all w=1)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from tensor_lattice import (
    H_DEFAULT,
    W_DEFAULT,
    SolveConfig,
    fully_connected_gaussian_tensor,
    partial_grid_uniform_beams,
    solve_tensor_mechanics,
    compute_global_scalars,
    tensor_to_geometry,
    visualize_tensor,
)

# --- generation -----------------------------------------------------------

def sample_lattice(rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    """One random structure; all active struts end up with weight 1 where applicable."""
    if rng.random() < 0.58:
        tau = float(rng.uniform(0.15, 0.37))
        gs = float(rng.uniform(0.48, 0.98))
        s = int(rng.integers(0, 2**31))
        t = partial_grid_uniform_beams(
            W_DEFAULT,
            H_DEFAULT,
            geom_scale=gs,
            bond_threshold=tau,
            seed=s,
        )
        meta = {"family": "sparse", "tau": tau, "geom_scale": gs, "lat_seed": s}
    else:
        p = float(rng.uniform(0.12, 1.15))
        s = int(rng.integers(0, 2**31))
        t = fully_connected_gaussian_tensor(
            W_DEFAULT, H_DEFAULT, perturb=p, seed=s
        )
        meta = {"family": "full_gaussian", "perturb": p, "lat_seed": s}
    return t, meta


def _work(job: tuple[int, float, float]) -> dict[str, Any]:
    job_id, pool_seed, e_scale = job
    rng = np.random.default_rng(int(pool_seed))
    t, meta = sample_lattice(rng)
    cfg = SolveConfig(bond_threshold=0.0, connect_all=False, e_scale=float(e_scale), delta=0.01)
    ms = solve_tensor_mechanics(t, W_DEFAULT, H_DEFAULT, cfg)
    r = ms.result
    row: dict[str, Any] = {
        "job_id": job_id,
        "pool_seed": int(pool_seed),
        "ok": r.ok,
        "error": r.error or "",
        "E_eff": float(r.E_eff) if r.ok else float("nan"),
        "strut_mass_metric": float(r.strut_mass_metric) if r.ok else float("nan"),
        "sigma_max": float(r.sigma_max) if r.ok else float("nan"),
        "strut_length_sum": float(r.strut_length_sum) if r.ok else float("nan"),
    }
    row["meta"] = json.dumps(meta)
    return row


def _rebuild_tensor(meta: dict[str, Any]) -> np.ndarray:
    if meta["family"] == "sparse":
        return partial_grid_uniform_beams(
            W_DEFAULT,
            H_DEFAULT,
            geom_scale=float(meta["geom_scale"]),
            bond_threshold=float(meta["tau"]),
            seed=int(meta["lat_seed"]),
        )
    return fully_connected_gaussian_tensor(
        W_DEFAULT,
        H_DEFAULT,
        perturb=float(meta["perturb"]),
        seed=int(meta["lat_seed"]),
    )


def select_similar_E_M_diverse_sigma(
    rows: list[dict[str, Any]],
    *,
    n_pick: int,
    distance_quantile: float = 0.18,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """
    Keep candidates with small Mahalanobis-like distance in (log E_eff, log mass)
    to the pooled median, then choose ``n_pick`` with σ_max spread across quantiles.
    """
    ok = [r for r in rows if r.get("ok") is True]
    if len(ok) < n_pick:
        raise SystemExit(f"Only {len(ok)} ok runs; increase --pool.")

    E = np.array([float(r["E_eff"]) for r in ok], dtype=float)
    M = np.array([float(r["strut_mass_metric"]) for r in ok], dtype=float)
    S = np.array([float(r["sigma_max"]) for r in ok], dtype=float)

    logE = np.log(np.clip(E, 1e-30, None))
    logM = np.log(np.clip(M, 1e-30, None))
    med_E = float(np.median(logE))
    med_M = float(np.median(logM))
    # robust scale (MAD)
    def _mad(x: np.ndarray, med: float) -> float:
        return float(np.median(np.abs(x - med))) + 1e-30

    sE = _mad(logE, med_E)
    sM = _mad(logM, med_M)
    d = np.sqrt(((logE - med_E) / sE) ** 2 + ((logM - med_M) / sM) ** 2)

    q = float(np.quantile(d, distance_quantile))
    cand_idx = np.where(d <= q)[0]
    if len(cand_idx) < max(n_pick + 10, 40):
        q = float(np.quantile(d, min(0.42, distance_quantile + 0.15)))
        cand_idx = np.where(d <= q)[0]

    cand = [ok[i] for i in cand_idx]
    if len(cand) < n_pick:
        cand = ok

    order = np.argsort([float(r["sigma_max"]) for r in cand])
    n_c = len(order)
    if n_c <= n_pick:
        idx_into_sorted = np.arange(n_c)
    else:
        idx_into_sorted = np.unique(
            np.round(np.linspace(0, n_c - 1, n_pick)).astype(int)
        )
    sel = [order[j] for j in idx_into_sorted]
    if len(sel) < n_pick:
        for j in range(n_c):
            if order[j] not in sel:
                sel.append(order[j])
            if len(sel) >= n_pick:
                break
    picked = [cand[i] for i in sel[:n_pick]]

    stats = {
        "median_log_E": med_E,
        "median_log_mass": med_M,
        "distance_cut_q": q,
        "n_candidates": float(len(cand)),
        "sigma_max_min_picked": float(min(float(r["sigma_max"]) for r in picked)),
        "sigma_max_max_picked": float(max(float(r["sigma_max"]) for r in picked)),
    }
    return picked, stats


def export_picked(out_dir: Path, picked: list[dict[str, Any]]) -> None:
    ex = out_dir / "picked_lattices"
    ex.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(picked):
        meta = json.loads(row["meta"])
        t = _rebuild_tensor(meta)
        xy, edges, _ = tensor_to_geometry(t, w=W_DEFAULT, h=H_DEFAULT, bond_threshold=0.5)
        gl = compute_global_scalars(t, xy, edges, W_DEFAULT, H_DEFAULT)
        tag = (
            f"pick_{i:02d}_E{float(row['E_eff']):.4g}_M{float(row['strut_mass_metric']):.4g}"
            f"_smax{float(row['sigma_max']):.4g}"
        )
        np.savez_compressed(ex / f"{tag}.npz", tensor=t, w=W_DEFAULT, h=H_DEFAULT, meta=meta)
        visualize_tensor(
            t,
            ex / f"{tag}.png",
            globals_=gl,
            note=f"{tag} | uniform w | similar E,mass — σ_max spread",
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pick lattices: similar E_eff & mass, diverse σ_max (uniform beams)"
    )
    ap.add_argument("--pool", type=int, default=3200, help="Number of random solves")
    ap.add_argument("--pick", type=int, default=32, help="How many to export (≥30)")
    ap.add_argument("--master-seed", type=int, default=4242)
    ap.add_argument("--e-scale", type=float, default=10000.0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="")
    ap.add_argument(
        "--dist-q",
        type=float,
        default=0.18,
        help="Quantile cutoff on (log E, log M) distance (smaller = tighter match)",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else root / "pic" / f"picker_E_M_sigma_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ms = int(args.master_seed)
    jobs = [
        (i, int((ms * 100003 + i * 7919) % (2**31)), float(args.e_scale))
        for i in range(int(args.pool))
    ]
    n_workers = int(args.workers) if args.workers > 0 else min(len(jobs), os.cpu_count() or 4)

    rows: list[dict[str, Any]] = []
    if n_workers <= 1:
        for j in jobs:
            rows.append(_work(j))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_work, j): j for j in jobs}
            for fut in as_completed(futs):
                rows.append(fut.result())

    rows.sort(key=lambda r: int(r["job_id"]))
    pool_csv = out_dir / "pool_results.csv"
    with pool_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    picked, st = select_similar_E_M_diverse_sigma(
        rows, n_pick=max(30, int(args.pick)), distance_quantile=float(args.dist_q)
    )

    pick_csv = out_dir / "picked_similar_E_M_diverse_sigma.csv"
    with pick_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(picked[0].keys()))
        w.writeheader()
        w.writerows(picked)

    (out_dir / "selection_stats.json").write_text(json.dumps(st, indent=2))
    export_picked(out_dir, picked)

    (out_dir / "README.txt").write_text(
        "Uniform beam weights on active edges (sparse: mask τ then w=1; full: w=1).\n"
        f"Pool N={args.pool}, picked N={len(picked)} with similar (E_eff, mass) in log-space, "
        f"σ_max spread [{st['sigma_max_min_picked']:.5g}, {st['sigma_max_max_picked']:.5g}].\n"
        f"See picked_lattices/ for npz+png.\n"
    )
    print(out_dir)


if __name__ == "__main__":
    main()
