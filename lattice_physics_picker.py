#!/usr/bin/env python3
"""
Sample random geometries + connectivity with **uniform beam weights** (same thickness),
solve mechanics, then pick lattices with **very similar E_eff and mass** but **diverse σ_max**.

Includes tensor PNGs with footer stats, mechanics ``sol_*.png``, and summary figures.
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
    compute_global_scalars,
    fully_connected_gaussian_tensor,
    partial_grid_uniform_beams,
    solve_tensor_mechanics,
    tensor_to_geometry,
    visualize_mechanics,
    visualize_tensor,
)

# --- generation (looser connectivity: higher τ on sparse branch) ----------------


def sample_lattice(rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    """Random structure; active struts uniform w=1 after mask (sparse) or full grid."""
    if rng.random() < 0.62:
        tau = float(rng.uniform(0.12, 0.54))
        gs = float(rng.uniform(0.32, 0.99))
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
        p = float(rng.uniform(0.1, 1.2))
        s = int(rng.integers(0, 2**31))
        t = fully_connected_gaussian_tensor(
            W_DEFAULT, H_DEFAULT, perturb=p, seed=s
        )
        meta = {"family": "full_gaussian", "perturb": p, "lat_seed": s}
    return t, meta


def _work(job: tuple[int, float, float, float]) -> dict[str, Any]:
    job_id, pool_seed, e_scale, delta = job
    rng = np.random.default_rng(int(pool_seed))
    t, meta = sample_lattice(rng)
    cfg = SolveConfig(
        bond_threshold=0.0,
        connect_all=False,
        e_scale=float(e_scale),
        delta=float(delta),
    )
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


def _footer_line(row: dict[str, Any], meta: dict[str, Any]) -> str:
    m = float(row["strut_mass_metric"])
    ee = float(row["E_eff"])
    sm = float(row["sigma_max"])
    if meta["family"] == "sparse":
        return (
            f"M={m:.6g}  E_eff={ee:.6g}  σ_max={sm:.6g}  τ={float(meta['tau']):.6g}"
        )
    return (
        f"M={m:.6g}  E_eff={ee:.6g}  σ_max={sm:.6g}  p_geom={float(meta['perturb']):.6g}  τ=n/a"
    )


def select_tight_E_M_diverse_sigma(
    ok: list[dict[str, Any]],
    *,
    n_pick: int,
    rel_e_start: float,
    rel_m_start: float,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """
    Tighten relative bands on E_eff and mass around medians until enough candidates,
    then subsample along σ_max quantiles.
    """
    if len(ok) < n_pick:
        raise SystemExit(f"Only {len(ok)} ok runs; increase --pool.")

    E = np.array([float(r["E_eff"]) for r in ok], dtype=float)
    M = np.array([float(r["strut_mass_metric"]) for r in ok], dtype=float)

    medE = float(np.median(E))
    medM = float(np.median(M))
    re, rm = float(rel_e_start), float(rel_m_start)
    rel_cap = 0.012

    cand_idx: np.ndarray | None = None
    for _ in range(40):
        mask = (np.abs(E - medE) <= re * max(abs(medE), 1e-30)) & (
            np.abs(M - medM) <= rm * max(abs(medM), 1e-30)
        )
        cand_idx = np.where(mask)[0]
        if len(cand_idx) >= max(n_pick * 4, 120):
            break
        re = min(rel_cap, re * 1.09)
        rm = min(rel_cap, rm * 1.09)

    assert cand_idx is not None
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

    S = np.array([float(r["sigma_max"]) for r in picked], dtype=float)
    stats = {
        "median_E_eff": medE,
        "median_mass": medM,
        "rel_band_E": float(re),
        "rel_band_M": float(rm),
        "rel_cap": rel_cap,
        "n_candidates": float(len(cand)),
        "sigma_max_min_picked": float(np.min(S)),
        "sigma_max_max_picked": float(np.max(S)),
        "sigma_max_std_picked": float(np.std(S, ddof=1)) if len(S) > 1 else 0.0,
    }
    return picked, stats


def _plot_diagnostics(
    out_dir: Path,
    picked: list[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    E = np.array([float(r["E_eff"]) for r in picked], dtype=float)
    M = np.array([float(r["strut_mass_metric"]) for r in picked], dtype=float)
    S = np.array([float(r["sigma_max"]) for r in picked], dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    sc = ax.scatter(E, M, c=S, cmap="turbo", s=55, alpha=0.9, edgecolors="k", linewidths=0.3)
    plt.colorbar(sc, ax=ax, label="σ_max (scaled)")
    ax.set_xlabel("E_eff (homogenized)")
    ax.set_ylabel("strut mass Σ(L·w)")
    ax.set_title("Picked lattices: E vs M, color = σ_max")
    fig.tight_layout()
    fig.savefig(out_dir / "picked_E_vs_M_sigma_color.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Filled contours on convex hull / triangulation
    try:
        import matplotlib.tri as mtri

        if len(E) >= 4:
            triang = mtri.Triangulation(E, M)
            fig2, ax2 = plt.subplots(figsize=(7.5, 6.2))
            tcf = ax2.tricontourf(triang, S, levels=24, cmap="turbo")
            ax2.scatter(E, M, c="k", s=12, alpha=0.5)
            plt.colorbar(tcf, ax=ax2, label="σ_max (scaled)")
            ax2.set_xlabel("E_eff")
            ax2.set_ylabel("mass Σ(L·w)")
            ax2.set_title("Triangulation contour: σ_max(E, M)")
            fig2.tight_layout()
            fig2.savefig(out_dir / "picked_sigma_contourf.png", dpi=150, bbox_inches="tight")
            plt.close(fig2)
    except Exception:
        pass

    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.hist(S, bins=min(35, max(10, len(S) // 3)), color="steelblue", edgecolor="k", alpha=0.85)
    ax3.set_xlabel("σ_max (scaled)")
    ax3.set_ylabel("count")
    ax3.set_title("Picked set: distribution of σ_max")
    fig3.tight_layout()
    fig3.savefig(out_dir / "picked_sigma_max_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)


def export_picked(
    out_dir: Path,
    picked: list[dict[str, Any]],
    *,
    e_scale: float,
    delta: float,
) -> None:
    ex = out_dir / "picked_lattices"
    ex.mkdir(parents=True, exist_ok=True)
    cfg = SolveConfig(
        bond_threshold=0.0,
        connect_all=False,
        e_scale=float(e_scale),
        delta=float(delta),
    )
    for i, row in enumerate(picked):
        meta = json.loads(row["meta"])
        t = _rebuild_tensor(meta)
        xy, edges, _ = tensor_to_geometry(t, w=W_DEFAULT, h=H_DEFAULT, bond_threshold=0.5)
        gl = compute_global_scalars(t, xy, edges, W_DEFAULT, H_DEFAULT)
        tag = (
            f"pick_{i:03d}_E{float(row['E_eff']):.4g}_M{float(row['strut_mass_metric']):.4g}"
            f"_smax{float(row['sigma_max']):.4g}"
        )
        footer = _footer_line(row, meta)
        np.savez_compressed(ex / f"{tag}.npz", tensor=t, w=W_DEFAULT, h=H_DEFAULT, meta=meta)
        visualize_tensor(
            t,
            ex / f"{tag}.png",
            globals_=gl,
            note=f"{tag} | uniform beams",
            footer=footer,
        )

        ms = solve_tensor_mechanics(t, W_DEFAULT, H_DEFAULT, cfg)
        if not ms.result.ok or ms.frame is None or ms.sigma is None:
            continue
        r = ms.result
        eff_plot = {
            "E_eff": float(r.E_eff),
            "sigma_macro_end": float(r.sigma_macro_end),
            "eps_macro": float(r.eps_macro),
            "K_sec": float(r.K_sec),
        }
        sol_name = f"sol_{tag}.png"
        visualize_mechanics(
            ms.frame,
            ms.sigma,
            ex / sol_name,
            disp_scale=30.0,
            title=f"e_scale={e_scale}, δ={delta}",
            eff=eff_plot,
            stress_units="scaled",
            footer=footer,
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pick lattices: tight E_eff & mass, diverse σ_max (uniform beams)"
    )
    ap.add_argument("--pool", type=int, default=16000, help="Random solves in pool")
    ap.add_argument("--pick", type=int, default=100, help="How many to export")
    ap.add_argument("--master-seed", type=int, default=4242)
    ap.add_argument("--e-scale", type=float, default=10000.0)
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="")
    ap.add_argument(
        "--rel-e",
        type=float,
        default=0.007,
        help="Starting relative half-width on E_eff around median (widens if needed)",
    )
    ap.add_argument(
        "--rel-m",
        type=float,
        default=0.007,
        help="Starting relative half-width on mass around median",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else root / "pic" / f"picker_E_M_sigma_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    ms = int(args.master_seed)
    jobs = [
        (
            i,
            int((ms * 100003 + i * 7919) % (2**31)),
            float(args.e_scale),
            float(args.delta),
        )
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

    ok = [r for r in rows if r.get("ok") is True]
    picked, st = select_tight_E_M_diverse_sigma(
        ok,
        n_pick=int(args.pick),
        rel_e_start=float(args.rel_e),
        rel_m_start=float(args.rel_m),
    )

    pick_csv = out_dir / "picked_similar_E_M_diverse_sigma.csv"
    with pick_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(picked[0].keys()))
        w.writeheader()
        w.writerows(picked)

    (out_dir / "selection_stats.json").write_text(json.dumps(st, indent=2))
    export_picked(out_dir, picked, e_scale=float(args.e_scale), delta=float(args.delta))
    _plot_diagnostics(out_dir, picked)

    (out_dir / "README.txt").write_text(
        "Uniform beam weights on active edges. Sparse branch uses higher τ (looser graphs).\n"
        f"Pool N={args.pool}, picked N={len(picked)}. Footer on PNGs: M, E_eff, σ_max, τ or p_geom.\n"
        f"Mechanics: sol_*.png. Plots: picked_E_vs_M_sigma_color.png, picked_sigma_contourf.png, "
        f"picked_sigma_max_histogram.png\n"
    )
    print(out_dir)


if __name__ == "__main__":
    main()
