#!/usr/bin/env python3
"""
Fully connected Gaussian geometry (geometric perturb only) + independent strut thickness
factors U(low, high) on half-edges.

Pool random solves, select the **central band** in (E_eff, mass) (interquartile overlap),
report σ_max range. Optional PNG exports. Progress bar + time budget for the pool phase.

Grid size is configurable (--w, --h); default matches tensor_lattice W_DEFAULT×H_DEFAULT.

Use ``--bulk`` for large ML datasets: skips E_eff/mass “adequate band” stats, histogram,
and PNG exports; writes ``summary.json`` only.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    # Minimal stub: supports ``for x in tqdm(xs)`` and ``with tqdm(total=N) as pbar: pbar.update()``.
    class _ManualPbar:
        __slots__ = ()

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

        def update(self, _n: int = 1) -> None:
            pass

    def tqdm(iterable=None, **_kwargs):  # type: ignore
        if iterable is not None:
            return iterable
        return _ManualPbar()


from tensor_lattice import (
    H_DEFAULT,
    W_DEFAULT,
    SolveConfig,
    apply_independent_thickness_inplace,
    clip_displacements_only_inplace,
    compute_global_scalars,
    fully_connected_gaussian_tensor,
    solve_tensor_mechanics,
    tensor_to_geometry,
    visualize_mechanics,
    visualize_tensor,
)

# Job tuple: jid, geom_seed, thick_seed, p_geom, e_scale, delta, w, h, t_lo, t_hi, bond_max
_WorkJob = tuple[int, float, float, float, float, float, int, int, float, float, float]


def _work(job: _WorkJob) -> dict[str, Any]:
    (
        jid,
        geom_seed,
        thick_seed,
        p_geom,
        e_scale,
        delta,
        w,
        h,
        t_lo,
        t_hi,
        bond_max,
    ) = job
    t = fully_connected_gaussian_tensor(
        int(w), int(h), perturb=float(p_geom), seed=int(geom_seed)
    )
    apply_independent_thickness_inplace(
        t, int(thick_seed), low=float(t_lo), high=float(t_hi), bond_max=float(bond_max)
    )
    clip_displacements_only_inplace(t)
    cfg = SolveConfig(
        bond_threshold=0.0,
        connect_all=False,
        e_scale=float(e_scale),
        delta=float(delta),
    )
    ms = solve_tensor_mechanics(t, int(w), int(h), cfg)
    r = ms.result
    meta = {
        "w": int(w),
        "h": int(h),
        "geom_seed": int(geom_seed),
        "thick_seed": int(thick_seed),
        "perturb": float(p_geom),
        "thick_low": float(t_lo),
        "thick_high": float(t_hi),
        "bond_max": float(bond_max),
    }
    return {
        "job_id": jid,
        "ok": r.ok,
        "error": r.error or "",
        "E_eff": float(r.E_eff) if r.ok else float("nan"),
        "strut_mass_metric": float(r.strut_mass_metric) if r.ok else float("nan"),
        "sigma_max": float(r.sigma_max) if r.ok else float("nan"),
        "meta": json.dumps(meta),
        # Full lattice tensor (ch0=dx, ch1=dy, ch2=horiz, ch3=vert); not written to CSV.
        "channels": np.ascontiguousarray(t, dtype=np.float32),
    }


def _is_ok(row: dict[str, Any]) -> bool:
    return row.get("ok") in (True, "True", True)


def select_central_band(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Intersection of E_eff and mass interquartile bands (adequate middle)."""
    ok = [r for r in rows if _is_ok(r)]
    if not ok:
        return [], {}
    E = np.array([float(r["E_eff"]) for r in ok], dtype=float)
    M = np.array([float(r["strut_mass_metric"]) for r in ok], dtype=float)
    qe_lo, qe_hi = np.percentile(E, [25, 75])
    qm_lo, qm_hi = np.percentile(M, [25, 75])
    mask = (E >= qe_lo) & (E <= qe_hi) & (M >= qm_lo) & (M <= qm_hi)
    ade = [ok[i] for i in np.where(mask)[0]]
    band = "IQR25_75"
    if len(ade) < 10:
        qe_lo, qe_hi = np.percentile(E, [10, 90])
        qm_lo, qm_hi = np.percentile(M, [10, 90])
        mask = (E >= qe_lo) & (E <= qe_hi) & (M >= qm_lo) & (M <= qm_hi)
        ade = [ok[i] for i in np.where(mask)[0]]
        band = "P10_90"
    S = np.array([float(r["sigma_max"]) for r in ade], dtype=float)
    st = {
        "band_method": band,
        "n_pool_ok": float(len(ok)),
        "n_adequate": float(len(ade)),
        "E_band_lo": float(qe_lo),
        "E_band_hi": float(qe_hi),
        "mass_band_lo": float(qm_lo),
        "mass_band_hi": float(qm_hi),
        "sigma_min": float(np.min(S)) if len(S) else float("nan"),
        "sigma_max": float(np.max(S)) if len(S) else float("nan"),
        "sigma_mean": float(np.mean(S)) if len(S) else float("nan"),
        "sigma_std": float(np.std(S, ddof=1)) if len(S) > 1 else 0.0,
    }
    return ade, st


def _export_one(
    row: dict[str, Any],
    out_dir: Path,
    tag: str,
    e_scale: float,
    delta: float,
) -> None:
    meta = json.loads(row["meta"])
    w = int(meta.get("w", W_DEFAULT))
    h = int(meta.get("h", H_DEFAULT))
    t_lo = float(meta.get("thick_low", 0.7))
    t_hi = float(meta.get("thick_high", 1.3))
    bmx = float(meta.get("bond_max", max(t_lo, t_hi)))
    t = fully_connected_gaussian_tensor(
        w,
        h,
        perturb=float(meta["perturb"]),
        seed=int(meta["geom_seed"]),
    )
    apply_independent_thickness_inplace(
        t, int(meta["thick_seed"]), low=t_lo, high=t_hi, bond_max=bmx
    )
    clip_displacements_only_inplace(t)
    xy, edges, _ = tensor_to_geometry(
        t,
        w=w,
        h=h,
        bond_threshold=1e-12,
        clip_bonds=False,
    )
    gl = compute_global_scalars(t, xy, edges, w, h)
    footer = (
        f"M={float(row['strut_mass_metric']):.5g}  E_eff={float(row['E_eff']):.5g}  "
        f"σ_max={float(row['sigma_max']):.5g}  {w}×{h}  thickness∈[{t_lo},{t_hi}]"
    )
    visualize_tensor(
        t, out_dir / f"{tag}.png", globals_=gl, note=tag, footer=footer
    )
    cfg = SolveConfig(
        bond_threshold=0.0, connect_all=False, e_scale=e_scale, delta=delta
    )
    ms = solve_tensor_mechanics(t, w, h, cfg)
    if ms.result.ok and ms.frame is not None and ms.sigma is not None:
        eff = {
            "E_eff": float(ms.result.E_eff),
            "sigma_macro_end": float(ms.result.sigma_macro_end),
            "eps_macro": float(ms.result.eps_macro),
            "K_sec": float(ms.result.K_sec),
        }
        visualize_mechanics(
            ms.frame,
            ms.sigma,
            out_dir / f"sol_{tag}.png",
            disp_scale=30.0,
            title=f"e_scale={e_scale}, δ={delta}",
            eff=eff,
            stress_units="scaled",
            footer=footer,
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Full Gaussian grid + U(thick_low, thick_high) half-edge thickness; optional bulk mode"
    )
    ap.add_argument(
        "--w",
        type=int,
        default=W_DEFAULT,
        help="Lattice width (nodes in x); tensor shape (4, H, W)",
    )
    ap.add_argument(
        "--h",
        type=int,
        default=H_DEFAULT,
        help="Lattice height (nodes in y)",
    )
    ap.add_argument(
        "--thick-low",
        type=float,
        default=0.5,
        help="Min relative thickness factor (half-edge U)",
    )
    ap.add_argument(
        "--thick-high",
        type=float,
        default=1.5,
        help="Max relative thickness factor",
    )
    ap.add_argument(
        "--bond-max",
        type=float,
        default=None,
        help="Upper clip for bond weights after thickness (default: thick_high)",
    )
    ap.add_argument("--pool", type=int, default=1600, help="Monte Carlo samples (keep run <~10 min)")
    ap.add_argument("--master-seed", type=int, default=2026)
    ap.add_argument("--e-scale", type=float, default=10000.0)
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", type=str, default="")
    ap.add_argument(
        "--max-pool-seconds",
        type=float,
        default=595.0,
        help="Stop submitting new pool work after this wall time; in-flight jobs still finish",
    )
    ap.add_argument(
        "--max-total-seconds",
        type=float,
        default=600.0,
        help="Skip PNG export if elapsed time already exceeds this (keeps full run under cap)",
    )
    ap.add_argument("--export-n", type=int, default=10, help="Example PNGs (0=skip)")
    ap.add_argument(
        "--save-tensors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write lattice_tensors.npz with X (N,4,H,W) float32 + labels (default: on)",
    )
    ap.add_argument(
        "--tensors-name",
        type=str,
        default="lattice_tensors.npz",
        help="Output array file inside out-dir (NumPy compressed npz)",
    )
    ap.add_argument(
        "--bulk",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Dataset-only: skip E_eff/mass band stats, histogram, PNG exports; write summary.json",
    )
    args = ap.parse_args()

    w = int(args.w)
    h = int(args.h)
    if w < 2 or h < 2:
        raise SystemExit("--w and --h must be at least 2")
    t_lo = float(args.thick_low)
    t_hi = float(args.thick_high)
    if t_lo <= 0 or t_hi <= 0 or t_lo > t_hi:
        raise SystemExit("require 0 < thick_low <= thick_high")
    bond_max = float(args.bond_max) if args.bond_max is not None else float(t_hi)

    root = Path(__file__).resolve().parent
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else root / "pic" / f"thickness_picker_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = {
        "w": w,
        "h": h,
        "thick_low": t_lo,
        "thick_high": t_hi,
        "bond_max": bond_max,
        "pool": int(args.pool),
        "master_seed": int(args.master_seed),
        "e_scale": float(args.e_scale),
        "delta": float(args.delta),
        "workers": int(args.workers),
        "geom_perturb_range": [0.08, 1.22],
        "save_tensors": bool(args.save_tensors),
        "tensors_name": str(args.tensors_name),
        "bulk": bool(args.bulk),
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2))

    rng = np.random.default_rng(int(args.master_seed))
    t0 = time.monotonic()
    jobs: list[_WorkJob] = [
        (
            jid,
            int(rng.integers(0, 2**31)),
            int(rng.integers(0, 2**31)),
            float(rng.uniform(0.08, 1.22)),
            float(args.e_scale),
            float(args.delta),
            w,
            h,
            t_lo,
            t_hi,
            bond_max,
        )
        for jid in range(int(args.pool))
    ]

    n_workers = max(1, int(args.workers))
    rows: list[dict[str, Any]] = []
    deadline = t0 + float(args.max_pool_seconds)

    if n_workers <= 1:
        for j in tqdm(jobs, desc="pool", unit="solve"):
            if time.monotonic() >= deadline:
                break
            rows.append(_work(j))
    else:
        job_it = iter(jobs)
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            pending: dict[Future, int] = {}

            def pump() -> None:
                while len(pending) < n_workers:
                    if time.monotonic() >= deadline:
                        return
                    try:
                        j = next(job_it)
                    except StopIteration:
                        return
                    fut = ex.submit(_work, j)
                    pending[fut] = int(j[0])

            pump()
            with tqdm(total=len(jobs), desc="pool", unit="solve") as pbar:
                while pending:
                    done, _ = wait(
                        pending.keys(),
                        timeout=0.35,
                        return_when=FIRST_COMPLETED,
                    )
                    for fut in done:
                        pending.pop(fut)
                        rows.append(fut.result())
                        pbar.update(1)
                    pump()

    rows.sort(key=lambda r: int(r["job_id"]))

    tensor_path = out_dir / str(args.tensors_name)
    if rows and args.save_tensors:
        X = np.stack([r["channels"] for r in rows], axis=0)
        np.savez_compressed(
            tensor_path,
            X=X,
            job_id=np.array([int(r["job_id"]) for r in rows], dtype=np.int64),
            ok=np.array([bool(r["ok"]) for r in rows], dtype=np.bool_),
            E_eff=np.array([float(r["E_eff"]) for r in rows], dtype=np.float64),
            strut_mass_metric=np.array(
                [float(r["strut_mass_metric"]) for r in rows], dtype=np.float64
            ),
            sigma_max=np.array([float(r["sigma_max"]) for r in rows], dtype=np.float64),
            w=np.int32(w),
            h=np.int32(h),
        )

    csv_rows = [{k: v for k, v in r.items() if k != "channels"} for r in rows]
    if csv_rows:
        with (out_dir / "pool_results.csv").open("w", newline="") as f:
            csv_w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            csv_w.writeheader()
            csv_w.writerows(csv_rows)

    wall = time.monotonic() - t0
    n_done = len(csv_rows)
    n_ok = sum(1 for r in csv_rows if _is_ok(r)) if csv_rows else 0
    n_fail = n_done - n_ok

    if args.bulk:
        summary = {
            "mode": "bulk",
            "lattice": f"{w}x{h}",
            "thick_range": [t_lo, t_hi],
            "planned_jobs": int(len(jobs)),
            "completed_samples": n_done,
            "n_solve_ok": n_ok,
            "n_solve_fail": n_fail,
            "wall_seconds": float(wall),
            "max_pool_seconds": float(args.max_pool_seconds),
            "workers": int(args.workers),
            "files": {
                "tensors": str(args.tensors_name) if args.save_tensors else None,
                "csv": "pool_results.csv",
                "run_config": "run_config.json",
            },
        }
        if n_done < len(jobs):
            summary["note"] = (
                "Fewer completed than planned (deadline hit or early stop); raise "
                "--max-pool-seconds or lower --workers contention."
            )
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        readme = (
            f"Bulk dataset: {w}×{h}, thickness U({t_lo},{t_hi}).\n"
            f"Completed {n_done}/{len(jobs)} samples ({n_ok} ok, {n_fail} fail). "
            f"Wall ~{wall:.1f}s.\n"
            f"See summary.json, {tensor_path.name if args.save_tensors else 'no npz'}, pool_results.csv.\n"
        )
        if rows and args.save_tensors:
            readme += (
                f"X: (N,4,{h},{w}) float32. PyTorch: "
                "d=np.load(...); x=torch.from_numpy(d['X'])\n"
            )
    else:
        adequate, st = select_central_band(csv_rows)
        if not st:
            st = {
                "band_method": "none",
                "n_pool_ok": 0.0,
                "n_adequate": 0.0,
                "E_band_lo": float("nan"),
                "E_band_hi": float("nan"),
                "mass_band_lo": float("nan"),
                "mass_band_hi": float("nan"),
                "sigma_min": float("nan"),
                "sigma_max": float("nan"),
                "sigma_mean": float("nan"),
                "sigma_std": 0.0,
            }
        (out_dir / "adequate_band_stats.json").write_text(json.dumps(st, indent=2))

        try:
            import matplotlib.pyplot as plt

            if adequate:
                S = np.array([float(r["sigma_max"]) for r in adequate], dtype=np.float64)
                fig, ax = plt.subplots(figsize=(6.5, 3.8))
                ax.hist(
                    S,
                    bins=min(28, max(8, len(S) // 4)),
                    color="teal",
                    edgecolor="k",
                    alpha=0.85,
                )
                ax.set_xlabel("σ_max (scaled)")
                ax.set_ylabel("count")
                ax.set_title("Adequate band (IQR E & M): σ_max distribution")
                fig.tight_layout()
                fig.savefig(out_dir / "adequate_sigma_histogram.png", dpi=140)
                plt.close(fig)
        except Exception:
            pass

        t_after_stats = time.monotonic()
        if (
            int(args.export_n) > 0
            and adequate
            and (t_after_stats - t0) < float(args.max_total_seconds)
        ):
            order = np.argsort([float(r["sigma_max"]) for r in adequate])
            n = len(order)
            pos = np.unique(
                np.round(np.linspace(0, n - 1, min(int(args.export_n), n))).astype(int)
            )
            chosen = [adequate[order[i]] for i in pos]
            exdir = out_dir / "examples"
            exdir.mkdir(exist_ok=True)
            for k, row in enumerate(tqdm(chosen, desc="export", unit="fig")):
                _export_one(row, exdir, f"ex_{k:02d}", float(args.e_scale), float(args.delta))

        readme = (
            f"Fully connected Gaussian geometry ({w}×{h}); each strut stiffness × U({t_lo},{t_hi}).\n"
            f"Pool jobs={len(jobs)}, ok={int(st['n_pool_ok'])}, adequate (IQR∩) n={int(st['n_adequate'])}.\n"
            f"σ_max in adequate band: min={st['sigma_min']:.6g} max={st['sigma_max']:.6g} "
            f"mean={st['sigma_mean']:.6g} std={st['sigma_std']:.6g}\n"
            f"Wall time ~{wall:.1f}s (submit cap {args.max_pool_seconds}s, total cap {args.max_total_seconds}s).\n"
        )
        if rows and args.save_tensors:
            readme += (
                f"\nTensors: {tensor_path.name} — X shape (N,4,H,W) float32; "
                "channels ch0=dx, ch1=dy, ch2=→(i+1,j), ch3=→(i,j+1); "
                "parallel arrays job_id, ok, E_eff, strut_mass_metric, sigma_max; attrs w, h.\n"
                "PyTorch: d=np.load(path); x=torch.from_numpy(d['X'])\n"
            )

    (out_dir / "README.txt").write_text(readme)
    print(readme)
    print(out_dir)


if __name__ == "__main__":
    main()
