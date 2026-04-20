#!/usr/bin/env python3
"""
Ensemble study: full-grid lattices with Gaussian (or uniform) geometric perturbation.
Runs many realizations, logs CSV, plots mean ± std of E_eff and σ_max vs perturbation level.
Uses ``tensor_lattice`` as a library with optional parallel workers.
"""
from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from tensor_lattice import (
    H_DEFAULT,
    W_DEFAULT,
    MechanicsResult,
    SolveConfig,
    fully_connected_gaussian_tensor,
    fully_connected_perturbed_tensor,
    solve_tensor_mechanics,
)

PertKind = Literal["gaussian", "uniform"]


def _make_tensor(kind: PertKind, perturb: float, seed: int) -> np.ndarray:
    if kind == "gaussian":
        return fully_connected_gaussian_tensor(
            W_DEFAULT, H_DEFAULT, perturb=float(perturb), seed=int(seed)
        )
    return fully_connected_perturbed_tensor(
        W_DEFAULT, H_DEFAULT, perturb=float(perturb), seed=int(seed)
    )


def _run_single(job: tuple[PertKind, float, int, float]) -> dict[str, Any]:
    """Worker entry point (must be picklable for ProcessPoolExecutor)."""
    kind, perturb, seed, e_scale = job
    t = _make_tensor(kind, perturb, seed)
    cfg = SolveConfig(bond_threshold=0.0, connect_all=False, e_scale=float(e_scale))
    ms = solve_tensor_mechanics(t, W_DEFAULT, H_DEFAULT, cfg)
    r: MechanicsResult = ms.result
    row: dict[str, Any] = {
        "perturb": float(perturb),
        "seed": int(seed),
        "kind": kind,
        "ok": r.ok,
        "error": r.error or "",
        "E_eff": r.E_eff,
        "sigma_max": r.sigma_max,
        "sigma_min": r.sigma_min,
        "sigma_macro_end": r.sigma_macro_end,
        "rms_disp": r.rms_disp,
        "n_edges_reduced": r.n_edges_reduced,
        "min_node_distance": r.min_node_distance,
    }
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _aggregate(rows: list[dict[str, Any]]) -> tuple[np.ndarray, dict[float, dict[str, np.ndarray]]]:
    """Group by perturb level; return sorted levels and stats per level (ok rows only)."""
    by_p: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        if not row.get("ok"):
            continue
        p = float(row["perturb"])
        by_p.setdefault(p, []).append(row)
    levels = np.array(sorted(by_p.keys()))
    stats: dict[float, dict[str, np.ndarray]] = {}
    for p in levels:
        sub = by_p[float(p)]
        e = np.array([x["E_eff"] for x in sub], dtype=float)
        s = np.array([x["sigma_max"] for x in sub], dtype=float)
        stats[float(p)] = {
            "E_mean": np.mean(e),
            "E_std": np.std(e, ddof=1) if len(e) > 1 else 0.0,
            "sigma_mean": np.mean(s),
            "sigma_std": np.std(s, ddof=1) if len(s) > 1 else 0.0,
            "n": len(sub),
        }
    return levels, stats


def _matrix_sorted_by_column(
    rows: list[dict[str, Any]],
    levels: np.ndarray,
    repeats: int,
    field: str,
) -> np.ndarray:
    """
    Build a (repeats × n_levels) matrix: for each perturbation column, take all ok
    values for `field`, sort ascending (low→high), and place along rows (short
    columns padded with NaN). Good for a 2D “map” of how the response band widens.
    """
    by_p: dict[float, list[float]] = {}
    for row in rows:
        if not row.get("ok"):
            continue
        p = float(row["perturb"])
        by_p.setdefault(p, []).append(float(row[field]))

    z = np.full((repeats, len(levels)), np.nan, dtype=float)
    for j, p in enumerate(levels):
        vals = sorted(by_p.get(float(p), []))
        n = min(len(vals), repeats)
        z[:n, j] = vals[:n]
    return z


def _plot_heatmap_maps(
    levels: np.ndarray,
    z_E: np.ndarray,
    z_S: np.ndarray,
    out_png: Path,
    title: str,
) -> None:
    """
    2D heatmap: x = perturbation p, y = row index after sorting values within
    each column (row 0 = smallest response in that column). Shows how the band
    of outcomes widens with p.
    """
    import matplotlib.pyplot as plt

    n_r, n_c = z_E.shape
    dp = (float(levels[-1]) - float(levels[0])) / max(n_c - 1, 1) if n_c > 1 else 1.0
    extent = [
        float(levels[0]) - 0.5 * dp,
        float(levels[-1]) + 0.5 * dp,
        -0.5,
        float(n_r) - 0.5,
    ]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    m0 = ax0.imshow(
        z_E,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
        interpolation="nearest",
    )
    ax0.set_xlabel("perturbation level p")
    ax0.set_ylabel("rank slot (low → high within column)")
    ax0.set_title(f"{title}\nE_eff")
    fig.colorbar(m0, ax=ax0, label="E_eff", fraction=0.046)

    m1 = ax1.imshow(
        z_S,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="inferno",
        interpolation="nearest",
    )
    ax1.set_xlabel("perturbation level p")
    ax1.set_title("σ_max (edge)")
    fig.colorbar(m1, ax=ax1, label="σ_max", fraction=0.046)
    fig.suptitle("2D map: sorted response bands vs p", fontsize=10, y=1.02)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_hexbin_phase(
    rows: list[dict[str, Any]],
    out_png: Path,
    title: str,
) -> None:
    """2D density: hexbin of (p, E_eff) and (p, σ_max) as side-by-side panels."""
    import matplotlib.pyplot as plt

    p_arr: list[float] = []
    e_arr: list[float] = []
    s_arr: list[float] = []
    for row in rows:
        if not row.get("ok"):
            continue
        p_arr.append(float(row["perturb"]))
        e_arr.append(float(row["E_eff"]))
        s_arr.append(float(row["sigma_max"]))
    if len(p_arr) < 5:
        return

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.5))
    hb0 = ax0.hexbin(p_arr, e_arr, gridsize=24, cmap="Blues", mincnt=1)
    ax0.set_xlabel("perturbation p")
    ax0.set_ylabel("E_eff")
    ax0.set_title("count density (p, E_eff)")
    fig.colorbar(hb0, ax=ax0, label="count")

    hb1 = ax1.hexbin(p_arr, s_arr, gridsize=24, cmap="Oranges", mincnt=1)
    ax1.set_xlabel("perturbation p")
    ax1.set_ylabel("σ_max")
    ax1.set_title("count density (p, σ_max)")
    fig.colorbar(hb1, ax=ax1, label="count")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_summary(
    levels: np.ndarray,
    stats: dict[float, dict[str, np.ndarray]],
    out_png: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    Em = np.array([stats[float(p)]["E_mean"] for p in levels])
    Es = np.array([stats[float(p)]["E_std"] for p in levels])
    Sm = np.array([stats[float(p)]["sigma_mean"] for p in levels])
    Ss = np.array([stats[float(p)]["sigma_std"] for p in levels])

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax0.errorbar(levels, Em, yerr=Es, fmt="o-", capsize=3, markersize=5)
    ax0.set_ylabel("E_eff (homogenized)")
    ax0.set_title(title)
    ax0.grid(True, alpha=0.3)

    ax1.errorbar(levels, Sm, yerr=Ss, fmt="s-", capsize=3, color="darkorange", markersize=5)
    ax1.set_xlabel("perturbation level p")
    ax1.set_ylabel("σ_max (edge, scaled units)")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Ensemble lattice study vs perturbation level")
    ap.add_argument(
        "--kind",
        choices=("gaussian", "uniform"),
        default="gaussian",
        help="Gaussian: N(0,(p·DX/3)²) offsets; uniform: p scales U(-DX_MAX,DX_MAX) (same as --full-grid)",
    )
    ap.add_argument("--levels", type=int, default=16, help="Number of p values from 0 to --p-max")
    ap.add_argument("--p-max", type=float, default=1.0, help="Maximum perturbation level")
    ap.add_argument("--repeats", type=int, default=48, help="Random realizations per level")
    ap.add_argument("--master-seed", type=int, default=0, help="Base seed for SeedSequence")
    ap.add_argument("--e-scale", type=float, default=10000.0, help="Young modulus scale (same as solve)")
    ap.add_argument("--workers", type=int, default=0, help="Parallel workers (0 = CPU count)")
    ap.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory (default: pic/ensemble_<timestamp>)",
    )
    ap.add_argument(
        "--no-2d-maps",
        action="store_true",
        help="Skip heatmap and hexbin figures",
    )
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else script_dir / "pic" / f"ensemble_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    p_levels = np.linspace(0.0, float(args.p_max), int(args.levels))
    rng = np.random.default_rng(int(args.master_seed))
    jobs: list[tuple[PertKind, float, int, float]] = []
    for p in p_levels:
        for _ in range(int(args.repeats)):
            seed = int(rng.integers(0, 2**31, endpoint=False))
            jobs.append((args.kind, float(p), seed, float(args.e_scale)))

    n_workers = int(args.workers) if args.workers > 0 else min(len(jobs), os.cpu_count() or 4)

    rows: list[dict[str, Any]] = []
    if n_workers <= 1:
        for j in jobs:
            rows.append(_run_single(j))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_run_single, j): j for j in jobs}
            for fut in as_completed(futs):
                rows.append(fut.result())

    csv_path = out_dir / "ensemble_results.csv"
    _write_csv(csv_path, sorted(rows, key=lambda r: (r["perturb"], r["seed"])))

    ok_n = sum(1 for r in rows if r.get("ok"))
    readme = out_dir / "README.txt"
    extra = ""
    if not args.no_2d_maps:
        extra = (
            "figures: ensemble_Eeff_sigma.png  ensemble_map_2d_Eeff_sigma.png  "
            "ensemble_hexbin_phase.png\n"
        )
    readme.write_text(
        f"kind={args.kind}  levels={args.levels}  p_max={args.p_max}  repeats={args.repeats}\n"
        f"master_seed={args.master_seed}  e_scale={args.e_scale}  workers={n_workers}\n"
        f"total jobs={len(rows)}  ok={ok_n}  failed={len(rows) - ok_n}\n"
        f"csv: {csv_path.name}\n"
        f"{extra}"
    )

    levels, stats = _aggregate(rows)
    if len(levels) == 0:
        print("No successful runs — check CSV for errors.")
        return

    title = f"Full grid, {args.kind} perturbation — mean±std over {args.repeats} runs/level"
    _plot_summary(levels, stats, out_dir / "ensemble_Eeff_sigma.png", title)
    print(f"Wrote {csv_path}")
    print(f"Wrote {out_dir / 'ensemble_Eeff_sigma.png'}")

    if not args.no_2d_maps:
        z_E = _matrix_sorted_by_column(rows, levels, int(args.repeats), "E_eff")
        z_S = _matrix_sorted_by_column(rows, levels, int(args.repeats), "sigma_max")
        map_path = out_dir / "ensemble_map_2d_Eeff_sigma.png"
        _plot_heatmap_maps(levels, z_E, z_S, map_path, title)
        hex_path = out_dir / "ensemble_hexbin_phase.png"
        _plot_hexbin_phase(rows, hex_path, f"{args.kind} — density in (p, response) space")
        print(f"Wrote {map_path}")
        print(f"Wrote {hex_path}")


if __name__ == "__main__":
    main()
