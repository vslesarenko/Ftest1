#!/usr/bin/env python3
"""Plot lattice graphs from targeted_lattice_tensors.npz.

Default: stress → edge color (from mechanics solve), thickness factor → line width.
Optional ``--color-by thickness`` restores the older turbo-on-thickness style.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _pick_indices_stratified_by_E(E: np.ndarray, n_show: int, seed: int) -> np.ndarray:
    """Spread picks across the E_eff range (ordered bins)."""
    n = len(E)
    if n == 0:
        return np.array([], dtype=np.int64)
    if n <= n_show:
        return np.arange(n, dtype=np.int64)
    order = np.argsort(E)
    edges = np.linspace(0, n, n_show + 1)
    picks: list[int] = []
    rng = np.random.default_rng(seed)
    for k in range(n_show):
        lo = int(edges[k])
        hi = int(edges[k + 1])
        if hi <= lo:
            hi = lo + 1
        j = int(rng.integers(lo, hi))
        picks.append(int(order[j]))
    return np.array(picks, dtype=np.int64)


def _thickness_to_linewidth(
    w_arr: np.ndarray,
    wmin: float,
    wmax: float,
    lw_min: float,
    lw_max: float,
    *,
    compress: float,
) -> np.ndarray:
    """Map thickness → stroke width. compress>1 pulls down the upper range (less ‘sausage’)."""
    t = (w_arr - wmin) / (wmax - wmin + 1e-12)
    t = np.clip(t, 0.0, 1.0)
    if compress != 1.0:
        t = np.power(t, float(compress))
    return lw_min + t * (lw_max - lw_min)


def _indices_for_job_ids(job_id_arr: np.ndarray, ids: list[int]) -> list[int]:
    out: list[int] = []
    for jid in ids:
        hit = np.where(job_id_arr == int(jid))[0]
        if hit.size == 0:
            raise SystemExit(f"job_id {jid} not found in NPZ")
        out.append(int(hit[0]))
    return out


def _solve_config_from_run(
    run_dir: Path,
    *,
    bond_threshold: float,
    e_scale_override: float | None,
    delta_override: float | None,
) -> tuple[float, float]:
    """Return (e_scale, delta) from run_config.json with optional CLI overrides."""
    cfg_path = run_dir / "run_config.json"
    e_scale, delta = 10000.0, 0.01
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text())
        e_scale = float(cfg.get("e_scale", e_scale))
        delta = float(cfg.get("delta", delta))
    if e_scale_override is not None:
        e_scale = float(e_scale_override)
    if delta_override is not None:
        delta = float(delta_override)
    return e_scale, delta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "run_dir",
        type=Path,
        help="Folder with targeted_lattice_tensors.npz (e.g. pic/e_eff_target_*)",
    )
    ap.add_argument("--n", type=int, default=20, help="How many lattices to show")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument(
        "--job-ids",
        type=str,
        default=None,
        help="Comma-separated job ids (e.g. 796,527): 1×N comparison strip, stress colors + thickness linewidth",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output image path (default: run_dir/lattice_gallery/...)",
    )
    ap.add_argument(
        "--bond-threshold",
        type=float,
        default=0.0,
        help="Min thickness to keep a bond in geometry (0 = match lattice_e_eff_target)",
    )
    ap.add_argument(
        "--color-by",
        choices=("stress", "thickness"),
        default="stress",
        help="Edge color: mechanics σ estimate or thickness factor (legacy gallery look)",
    )
    ap.add_argument(
        "--stress-cmap",
        type=str,
        default="inferno",
        help="Colormap for per-edge stress (when --color-by stress)",
    )
    ap.add_argument("--dpi", type=int, default=320, help="Figure export resolution")
    ap.add_argument(
        "--lw-min",
        type=float,
        default=0.32,
        help="Line width (pt) at minimum thickness factor",
    )
    ap.add_argument(
        "--lw-max",
        type=float,
        default=2.35,
        help="Line width (pt) at maximum thickness factor (after compress)",
    )
    ap.add_argument(
        "--lw-compress",
        type=float,
        default=1.45,
        help="Exponent on normalized thickness for linewidth only (>1 = softer thick struts; 1 = linear)",
    )
    ap.add_argument(
        "--e-scale",
        type=float,
        default=None,
        help="Override SolveConfig e_scale (default from run_config.json)",
    )
    ap.add_argument(
        "--delta",
        type=float,
        default=None,
        help="Override applied displacement δ (default from run_config.json)",
    )
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()
    npz_path = run_dir / "targeted_lattice_tensors.npz"
    if not npz_path.is_file():
        raise SystemExit(f"Missing {npz_path}")

    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    from tensor_lattice import SolveConfig, solve_tensor_mechanics, tensor_to_geometry

    data = np.load(npz_path)
    X = np.asarray(data["X"], dtype=np.float64)
    E_eff = np.asarray(data["E_eff"], dtype=np.float64)
    S = np.asarray(data["sigma_max"], dtype=np.float64)
    if "strut_mass_metric" in data.files:
        M = np.asarray(data["strut_mass_metric"], dtype=np.float64)
    else:
        M = np.full(E_eff.shape, np.nan, dtype=np.float64)
    job_id = np.asarray(data["job_id"], dtype=np.int64)
    w = int(np.asarray(data["w"]).reshape(()))
    h = int(np.asarray(data["h"]).reshape(()))

    n_tot = X.shape[0]
    job_ids_list: list[int] | None = None
    if args.job_ids:
        job_ids_list = [int(s.strip()) for s in args.job_ids.split(",") if s.strip()]
        idx = np.array(_indices_for_job_ids(job_id, job_ids_list), dtype=np.int64)
        color_by = "stress"
    else:
        idx = _pick_indices_stratified_by_E(E_eff, min(args.n, n_tot), args.seed)
        color_by = str(args.color_by)

    e_scale, delta = _solve_config_from_run(
        run_dir,
        bond_threshold=float(args.bond_threshold),
        e_scale_override=args.e_scale,
        delta_override=args.delta,
    )
    solve_cfg = SolveConfig(
        bond_threshold=float(args.bond_threshold),
        connect_all=False,
        e_scale=e_scale,
        delta=delta,
    )

    # Build per-panel geometry + scalar per edge for coloring
    geoms: list[
        tuple[np.ndarray, list[tuple[int, int]], np.ndarray, np.ndarray, bool, str]
    ] = []
    # tuple: xy, edges, color_arr (sigma or thickness), w_arr, ok, err_msg

    all_color_vals: list[float] = []
    all_w: list[float] = []

    for k in idx:
        t = np.array(X[k], copy=True, dtype=np.float64)
        if color_by == "stress":
            ms = solve_tensor_mechanics(t, w, h, solve_cfg)
            if not ms.result.ok or ms.sigma is None or ms.xy is None or ms.edges is None:
                geoms.append(
                    (
                        np.zeros((0, 2)),
                        [],
                        np.array([], dtype=np.float64),
                        np.array([], dtype=np.float64),
                        False,
                        ms.result.error or "solve failed",
                    )
                )
                continue
            w_arr = np.asarray(ms.edge_weights, dtype=np.float64)
            sig = np.asarray(ms.sigma, dtype=np.float64)
            geoms.append((ms.xy, ms.edges, sig, w_arr, True, ""))
            all_color_vals.extend(sig.tolist())
            all_w.extend(w_arr.tolist())
        else:
            xy, edges, weights = tensor_to_geometry(
                t,
                w=w,
                h=h,
                bond_threshold=args.bond_threshold,
                clip_bonds=False,
            )
            w_arr = np.asarray(weights, dtype=np.float64)
            geoms.append((xy, edges, w_arr, w_arr, True, ""))
            all_color_vals.extend(w_arr.tolist())
            all_w.extend(w_arr.tolist())

    if not all_w:
        raise SystemExit("No edges to plot (empty geometry).")

    wmin = float(np.min(all_w))
    wmax = float(np.max(all_w))
    if wmax - wmin < 1e-9:
        wmin -= 0.05
        wmax += 0.05

    if color_by == "stress":
        cmin = float(np.min(all_color_vals))
        cmax = float(np.max(all_color_vals))
        if cmax - cmin < 1e-12:
            cmax = cmin + 1e-12
        color_norm = plt.Normalize(cmin, cmax)
        cmap = args.stress_cmap
    else:
        color_norm = plt.Normalize(wmin, wmax)
        cmap = "turbo"

    n_plot = len(idx)
    ncols = len(idx) if job_ids_list else 5
    nrows = int(np.ceil(n_plot / ncols)) if not job_ids_list else 1
    if job_ids_list:
        ncols = len(job_ids_list)

    left_m, right_m, bot_m, top_m = 0.012, 0.874, 0.012, 0.968
    u_w = right_m - left_m
    u_h = top_m - bot_m
    data_ar = (max(w, 2) - 1) / max(h - 1, 1)
    fig_ratio = data_ar * (ncols / max(nrows, 1)) * (u_h / u_w)
    fig_h = 1.18 * max(nrows, 1) if not job_ids_list else 5.2
    fig_w = fig_h * fig_ratio
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )

    for ax in axes.ravel():
        ax.set_visible(False)

    for plot_i, k in enumerate(idx):
        r, c = divmod(plot_i, ncols)
        ax = axes[r, c]
        ax.set_visible(True)
        xy, edges, c_arr, w_arr, ok, err = geoms[plot_i]
        if not ok or len(edges) == 0:
            ax.text(0.5, 0.5, err or "no geometry", ha="center", va="center", fontsize=8)
            ax.set_title(f"id {int(job_id[k])} (failed)", fontsize=7)
            continue
        segs = np.stack([xy[[a, b]] for a, b in edges], axis=0)
        lws = _thickness_to_linewidth(
            w_arr,
            wmin,
            wmax,
            float(args.lw_min),
            float(args.lw_max),
            compress=float(args.lw_compress),
        )
        lc = LineCollection(
            segs,
            array=c_arr,
            cmap=cmap,
            norm=color_norm,
            linewidths=lws,
        )
        ax.add_collection(lc)
        ax.scatter(xy[:, 0], xy[:, 1], s=1.2, c="#1a1a1a", alpha=0.28, linewidths=0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.margins(0.01)
        ax.set_title(
            f"id {int(job_id[k])}\n"
            rf"$E_{{\mathrm{{eff}}}}$={E_eff[k]:.0f}  $M$={M[k]:.0f}  $\sigma$={S[k]:.2f}",
            fontsize=6.5 if not job_ids_list else 10,
            pad=1,
        )

    fig.subplots_adjust(
        left=left_m,
        right=right_m,
        top=top_m,
        bottom=bot_m,
        wspace=0.01,
        hspace=0.035,
    )
    cax = fig.add_axes([0.888, 0.06, 0.011, 0.88])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=color_norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    if color_by == "stress":
        cb.set_label(r"$\sigma$ (fiber estimate, same units as $E$)", fontsize=9)
    else:
        cb.set_label("Thickness factor", fontsize=9)

    if job_ids_list:
        fig.suptitle(
            r"Stress color + linewidth $\propto$ thickness factor "
            r"(same $\sigma$ color scale for both panels)",
            fontsize=11,
            y=0.98,
        )
        default_out = (
            run_dir / "lattice_gallery" / f"compare_jobs_{'_'.join(str(j) for j in job_ids_list)}_stress.png"
        )
    else:
        if color_by == "stress":
            fig.suptitle(
                f"Lattices ({n_plot} of {n_tot}) — stress → color, thickness → line width",
                fontsize=10,
                y=0.995,
            )
            default_out = run_dir / "lattice_gallery" / "lattices_stress.png"
        else:
            fig.suptitle(
                f"Lattices ({n_plot} of {n_tot}) — thickness → color & line width (legacy)",
                fontsize=10,
                y=0.995,
            )
            default_out = run_dir / "lattice_gallery" / "lattices_thickness.png"

    out = args.out or default_out
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(args.dpi), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
