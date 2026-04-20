#!/usr/bin/env python3
"""
Ensemble study: full-grid lattices with Gaussian (or uniform) geometric perturbation.
Runs many realizations, logs CSV, and plots E_eff, σ_max, strut mass Σ(L·w), and stiffness/mass vs perturbation level.
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
    compute_global_scalars,
    fully_connected_gaussian_tensor,
    fully_connected_perturbed_tensor,
    partial_grid_uniform_beams,
    randomize_bond_stiffness_inplace,
    solve_tensor_mechanics,
    tensor_to_geometry,
    visualize_tensor,
)

PertKind = Literal["gaussian", "uniform", "sparse"]


def _is_ok_row(row: dict[str, Any]) -> bool:
    return row.get("ok") in (True, "True", "true", 1, "1")


def _make_tensor(kind: PertKind, perturb: float, seed: int) -> np.ndarray:
    if kind == "gaussian":
        return fully_connected_gaussian_tensor(
            W_DEFAULT, H_DEFAULT, perturb=float(perturb), seed=int(seed)
        )
    if kind == "uniform":
        return fully_connected_perturbed_tensor(
            W_DEFAULT, H_DEFAULT, perturb=float(perturb), seed=int(seed)
        )
    raise ValueError(f"_make_tensor: use partial_grid for kind={kind!r}")


def _run_single(
    job: tuple[PertKind, float, int, float, float, float | None, float],
) -> dict[str, Any]:
    """Worker entry point (must be picklable for ProcessPoolExecutor)."""
    kind, perturb, seed, e_scale, delta, bond_hetero_low, geom_scale = job
    if kind == "sparse":
        gs = float(geom_scale) if np.isfinite(geom_scale) else 0.8
        t = partial_grid_uniform_beams(
            W_DEFAULT,
            H_DEFAULT,
            geom_scale=gs,
            bond_threshold=float(perturb),
            seed=int(seed),
        )
    else:
        t = _make_tensor(kind, perturb, seed)
        if bond_hetero_low is not None:
            hseed = int((int(seed) * 1103515245 + 12345) % (2**31))
            randomize_bond_stiffness_inplace(
                t, hseed, low=float(bond_hetero_low), high=1.0
            )
    cfg = SolveConfig(
        bond_threshold=0.0,
        connect_all=False,
        e_scale=float(e_scale),
        delta=float(delta),
    )
    ms = solve_tensor_mechanics(t, W_DEFAULT, H_DEFAULT, cfg)
    r: MechanicsResult = ms.result
    mm = float(r.strut_mass_metric) if r.ok else float("nan")
    e_over_m = (
        float(r.E_eff) / mm
        if r.ok and mm > 1e-30 and np.isfinite(mm)
        else float("nan")
    )
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
        "strut_mass_metric": mm,
        "strut_length_sum": float(r.strut_length_sum) if r.ok else float("nan"),
        "E_eff_over_mass": e_over_m,
        "delta_applied": float(delta),
        "bond_hetero_low": float(bond_hetero_low)
        if bond_hetero_low is not None
        else float("nan"),
        "geom_scale": float(geom_scale)
        if kind == "sparse" and np.isfinite(geom_scale)
        else float("nan"),
    }
    return row


def _tensor_from_result_row(
    row: dict[str, Any], *, geom_scale_fallback: float
) -> np.ndarray:
    """Rebuild lattice tensor from a CSV row (for figures only)."""
    kind = str(row["kind"])
    p = float(row["perturb"])
    seed = int(row["seed"])
    g_raw = row.get("geom_scale", "")
    try:
        gs = float(g_raw)
    except (TypeError, ValueError):
        gs = float("nan")
    if not np.isfinite(gs):
        gs = geom_scale_fallback
    if kind == "sparse":
        return partial_grid_uniform_beams(
            W_DEFAULT,
            H_DEFAULT,
            geom_scale=gs,
            bond_threshold=p,
            seed=seed,
        )
    if kind == "gaussian":
        t = fully_connected_gaussian_tensor(
            W_DEFAULT, H_DEFAULT, perturb=p, seed=seed
        )
    else:
        t = fully_connected_perturbed_tensor(
            W_DEFAULT, H_DEFAULT, perturb=p, seed=seed
        )
    bh = row.get("bond_hetero_low", "")
    try:
        hlo = float(bh)
    except (TypeError, ValueError):
        hlo = float("nan")
    if np.isfinite(hlo):
        hseed = int((seed * 1103515245 + 12345) % (2**31))
        randomize_bond_stiffness_inplace(t, hseed, low=hlo, high=1.0)
    return t


def export_examples_folder(
    out_dir: Path,
    rows: list[dict[str, Any]],
    *,
    max_examples: int,
    geom_scale_fallback: float,
    bond_threshold_display: float = 0.5,
) -> None:
    """Save ``examples/`` with tensor PNG + npz for stratified successful runs."""
    ok_rows = [r for r in rows if _is_ok_row(r)]
    if not ok_rows or max_examples <= 0:
        return
    ex = out_dir / "examples"
    ex.mkdir(parents=True, exist_ok=True)

    by_p: dict[float, list[dict[str, Any]]] = {}
    for r in ok_rows:
        by_p.setdefault(float(r["perturb"]), []).append(r)
    for k in by_p:
        by_p[k].sort(key=lambda x: int(x["seed"]))
    levels = sorted(by_p.keys())
    nlev = max(len(levels), 1)
    per = max(1, max_examples // nlev)

    chosen: list[dict[str, Any]] = []
    for lev in levels:
        chosen.extend(by_p[lev][:per])
    if len(chosen) < max_examples:
        rest = [r for r in ok_rows if r not in chosen]
        chosen.extend(rest[: max_examples - len(chosen)])
    chosen = chosen[:max_examples]

    for i, row in enumerate(chosen):
        t = _tensor_from_result_row(row, geom_scale_fallback=geom_scale_fallback)
        xy, edges, _ = tensor_to_geometry(
            t, w=W_DEFAULT, h=H_DEFAULT, bond_threshold=bond_threshold_display
        )
        gl = compute_global_scalars(t, xy, edges, W_DEFAULT, H_DEFAULT)
        tag = f"ex_{i:02d}_k{row['kind']}_p{float(row['perturb']):.4f}_s{int(row['seed'])}"
        np.savez_compressed(ex / f"{tag}.npz", tensor=t, w=W_DEFAULT, h=H_DEFAULT)
        note = f"{tag} | uniform w on active bonds where applicable"
        outp = ex / f"{tag}.png"
        visualize_tensor(t, outp, globals_=gl, note=note)

    (ex / "README.txt").write_text(
        "Stratified snapshots (tensor channels + geometry). "
        "Bond overlay uses display threshold 0.5; mechanics uses all w>0.\n"
    )


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
        if not _is_ok_row(row):
            continue
        p = float(row["perturb"])
        by_p.setdefault(p, []).append(row)
    levels = np.array(sorted(by_p.keys()))
    stats: dict[float, dict[str, np.ndarray]] = {}
    for p in levels:
        sub = by_p[float(p)]
        e = np.array([x["E_eff"] for x in sub], dtype=float)
        s = np.array([x["sigma_max"] for x in sub], dtype=float)
        m = np.array([x["strut_mass_metric"] for x in sub], dtype=float)
        eom = np.array([x["E_eff_over_mass"] for x in sub], dtype=float)
        stats[float(p)] = {
            "E_mean": np.mean(e),
            "E_std": np.std(e, ddof=1) if len(e) > 1 else 0.0,
            "sigma_mean": np.mean(s),
            "sigma_std": np.std(s, ddof=1) if len(s) > 1 else 0.0,
            "mass_mean": np.mean(m),
            "mass_std": np.std(m, ddof=1) if len(m) > 1 else 0.0,
            "E_over_mass_mean": float(np.nanmean(eom)),
            "E_over_mass_std": float(np.nanstd(eom, ddof=1)) if np.sum(np.isfinite(eom)) > 1 else 0.0,
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
        if not _is_ok_row(row):
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
    z_M: np.ndarray,
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

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 5.0), sharey=True)
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

    m2 = ax2.imshow(
        z_M,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="cividis",
        interpolation="nearest",
    )
    ax2.set_xlabel("perturbation level p")
    ax2.set_title("Σ(L·w) mass proxy")
    fig.colorbar(m2, ax=ax2, label="Σ(L·w)", fraction=0.046)
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
    """2D density: hexbin of (p, E_eff), (p, σ_max), (p, mass), (mass, E_eff)."""
    import matplotlib.pyplot as plt

    p_arr: list[float] = []
    e_arr: list[float] = []
    s_arr: list[float] = []
    m_arr: list[float] = []
    for row in rows:
        if not _is_ok_row(row):
            continue
        p_arr.append(float(row["perturb"]))
        e_arr.append(float(row["E_eff"]))
        s_arr.append(float(row["sigma_max"]))
        m_arr.append(float(row["strut_mass_metric"]))
    if len(p_arr) < 5:
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    ax00, ax01, ax10, ax11 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    hb0 = ax00.hexbin(p_arr, e_arr, gridsize=22, cmap="Blues", mincnt=1)
    ax00.set_xlabel("perturbation p")
    ax00.set_ylabel("E_eff (homogenized)")
    ax00.set_title("(p, E_eff)")
    fig.colorbar(hb0, ax=ax00, label="count")

    hb1 = ax01.hexbin(p_arr, s_arr, gridsize=22, cmap="Oranges", mincnt=1)
    ax01.set_xlabel("perturbation p")
    ax01.set_ylabel("σ_max")
    ax01.set_title("(p, σ_max)")
    fig.colorbar(hb1, ax=ax01, label="count")

    hb2 = ax10.hexbin(p_arr, m_arr, gridsize=22, cmap="Greens", mincnt=1)
    ax10.set_xlabel("perturbation p")
    ax10.set_ylabel("Σ(L·w) mass proxy")
    ax10.set_title("(p, strut mass proxy)")
    fig.colorbar(hb2, ax=ax10, label="count")

    hb3 = ax11.hexbin(m_arr, e_arr, gridsize=22, cmap="Purples", mincnt=1)
    ax11.set_xlabel("Σ(L·w) mass proxy")
    ax11.set_ylabel("E_eff")
    ax11.set_title("(mass proxy, E_eff)")
    fig.colorbar(hb3, ax=ax11, label="count")

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
    *,
    x_label: str = "perturbation level p",
) -> None:
    import matplotlib.pyplot as plt

    Em = np.array([stats[float(p)]["E_mean"] for p in levels])
    Es = np.array([stats[float(p)]["E_std"] for p in levels])
    Sm = np.array([stats[float(p)]["sigma_mean"] for p in levels])
    Ss = np.array([stats[float(p)]["sigma_std"] for p in levels])
    Mm = np.array([stats[float(p)]["mass_mean"] for p in levels])
    Ms = np.array([stats[float(p)]["mass_std"] for p in levels])
    Rom = np.array([stats[float(p)]["E_over_mass_mean"] for p in levels])
    Ros = np.array([stats[float(p)]["E_over_mass_std"] for p in levels])

    fig, axes = plt.subplots(4, 1, figsize=(8, 11), sharex=True)
    ax0, ax1, ax2, ax3 = axes
    ax0.errorbar(levels, Em, yerr=Es, fmt="o-", capsize=3, markersize=5)
    ax0.set_ylabel("E_eff (homogenized)")
    ax0.set_title(title)
    ax0.grid(True, alpha=0.3)

    ax1.errorbar(levels, Sm, yerr=Ss, fmt="s-", capsize=3, color="darkorange", markersize=5)
    ax1.set_ylabel("σ_max (edge, scaled)")
    ax1.grid(True, alpha=0.3)

    ax2.errorbar(levels, Mm, yerr=Ms, fmt="^-", capsize=3, color="seagreen", markersize=5)
    ax2.set_ylabel("Σ(L·w) mass proxy")
    ax2.grid(True, alpha=0.3)

    ax3.errorbar(levels, Rom, yerr=Ros, fmt="D-", capsize=3, color="purple", markersize=4)
    ax3.set_xlabel(x_label)
    ax3.set_ylabel("E_eff / Σ(L·w)\n(stiffness per mass proxy)")
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Ensemble lattice study vs perturbation level")
    ap.add_argument(
        "--kind",
        choices=("gaussian", "uniform", "sparse"),
        default="gaussian",
        help="gaussian/uniform: full grid; sparse: partial connectivity, τ on soft bonds, uniform w=1 active",
    )
    ap.add_argument("--levels", type=int, default=16, help="Number of p values from 0 to --p-max")
    ap.add_argument("--p-max", type=float, default=1.0, help="Maximum perturbation level (ignored for sparse)")
    ap.add_argument(
        "--tau-min",
        type=float,
        default=0.14,
        help="sparse kind: min bond threshold τ (avoid full grid)",
    )
    ap.add_argument(
        "--tau-max",
        type=float,
        default=0.38,
        help="sparse kind: max τ (avoid overly loose lattices)",
    )
    ap.add_argument(
        "--geom-scale",
        type=float,
        default=0.82,
        help="sparse kind: geometry disorder amplitude for node offsets",
    )
    ap.add_argument("--repeats", type=int, default=48, help="Random realizations per level")
    ap.add_argument("--master-seed", type=int, default=0, help="Base seed for SeedSequence")
    ap.add_argument("--e-scale", type=float, default=10000.0, help="Young modulus scale (same as solve)")
    ap.add_argument(
        "--delta",
        type=float,
        default=0.01,
        help="Prescribed u_x on right column; peak stresses scale ~linearly with δ at fixed E",
    )
    ap.add_argument(
        "--bond-hetero-low",
        type=float,
        default=float("nan"),
        help="If set (e.g. 0.15–0.35), scale each strut stiffness by U(low,1) for load concentration",
    )
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
    ap.add_argument(
        "--examples",
        type=int,
        default=0,
        help="If >0, save this many stratified tensor PNG+npz under examples/",
    )
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else script_dir / "pic" / f"ensemble_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.kind == "sparse":
        p_levels = np.linspace(float(args.tau_min), float(args.tau_max), int(args.levels))
    else:
        p_levels = np.linspace(0.0, float(args.p_max), int(args.levels))
    rng = np.random.default_rng(int(args.master_seed))
    hetero = (
        None
        if (args.bond_hetero_low is None or np.isnan(float(args.bond_hetero_low)))
        else float(args.bond_hetero_low)
    )
    if args.kind == "sparse":
        hetero = None
    geom_scale_job = float(args.geom_scale) if args.kind == "sparse" else float("nan")

    jobs: list[tuple[PertKind, float, int, float, float, float | None, float]] = []
    for p in p_levels:
        for _ in range(int(args.repeats)):
            seed = int(rng.integers(0, 2**31, endpoint=False))
            jobs.append(
                (
                    args.kind,
                    float(p),
                    seed,
                    float(args.e_scale),
                    float(args.delta),
                    hetero,
                    geom_scale_job,
                )
            )

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

    ok_n = sum(1 for r in rows if _is_ok_row(r))
    readme = out_dir / "README.txt"
    extra = ""
    if not args.no_2d_maps:
        extra = (
            "figures: ensemble_Eeff_sigma.png  ensemble_map_2d_Eeff_sigma.png  "
            "ensemble_hexbin_phase.png  (heatmaps: E_eff, σ_max, mass; hexbins include p–mass and mass–E_eff)\n"
        )
    sparse_line = ""
    if args.kind == "sparse":
        sparse_line = f"tau_min={args.tau_min}  tau_max={args.tau_max}  geom_scale={args.geom_scale}\n"
    readme.write_text(
        f"kind={args.kind}  levels={args.levels}  p_max={args.p_max}  repeats={args.repeats}\n"
        f"{sparse_line}"
        f"master_seed={args.master_seed}  e_scale={args.e_scale}  workers={n_workers}\n"
        f"total jobs={len(rows)}  ok={ok_n}  failed={len(rows) - ok_n}\n"
        f"csv: {csv_path.name}\n"
        f"{extra}"
    )

    levels, stats = _aggregate(rows)
    if len(levels) == 0:
        print("No successful runs — check CSV for errors.")
        return

    x_label = (
        "bond threshold τ (mask on soft bonds; then w=1 on survivors)"
        if args.kind == "sparse"
        else "perturbation level p"
    )
    title = f"{args.kind} lattice — mean±std over {args.repeats} runs/level"
    _plot_summary(
        levels,
        stats,
        out_dir / "ensemble_Eeff_sigma.png",
        title,
        x_label=x_label,
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {out_dir / 'ensemble_Eeff_sigma.png'}")

    if int(args.examples) > 0:
        export_examples_folder(
            out_dir,
            rows,
            max_examples=int(args.examples),
            geom_scale_fallback=float(args.geom_scale),
        )
        print(f"Wrote {out_dir / 'examples'} (up to {int(args.examples)} snapshots)")

    if not args.no_2d_maps:
        z_E = _matrix_sorted_by_column(rows, levels, int(args.repeats), "E_eff")
        z_S = _matrix_sorted_by_column(rows, levels, int(args.repeats), "sigma_max")
        z_M = _matrix_sorted_by_column(rows, levels, int(args.repeats), "strut_mass_metric")
        map_path = out_dir / "ensemble_map_2d_Eeff_sigma.png"
        _plot_heatmap_maps(levels, z_E, z_S, z_M, map_path, title)
        hex_path = out_dir / "ensemble_hexbin_phase.png"
        _plot_hexbin_phase(rows, hex_path, f"{args.kind} — density in (p, response) space")
        print(f"Wrote {map_path}")
        print(f"Wrote {hex_path}")


if __name__ == "__main__":
    main()
