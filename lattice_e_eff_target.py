#!/usr/bin/env python3
"""
Monte Carlo lattice solves; keep samples whose homogenized E_eff lies in an
acceptable band. Default: inclusive [6000, 8000]. Optional: --goal with --tol
for a narrow band (goal ± tol). Mass and σ_max are saved for accepted samples.

A fraction of samples use **sparse bond masks** (random missing half-edges after
Gaussian geometry) to soften the structure and help fill lower E_eff in the band.

Runs until --max-seconds wall time (default ~10 min) or optional --max-attempts.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    class _ManualPbar:
        __slots__ = ()

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

        def set_postfix(self, *_a, **_k):
            pass

        def update(self, _n: int = 1) -> None:
            pass

    def tqdm(*_a, **kwargs):  # type: ignore
        return _ManualPbar()


from tensor_lattice import (
    CH_RIGHT,
    CH_TOP,
    SolveConfig,
    apply_independent_thickness_inplace,
    clip_displacements_only_inplace,
    fully_connected_gaussian_tensor,
    solve_tensor_mechanics,
)


def _in_E_interval(e: float, e_lo: float, e_hi: float) -> bool:
    return e_lo <= e <= e_hi


# Columns written to CSV (same as _work row keys except `channels`).
_CSV_FIELDS = (
    "job_id",
    "ok",
    "error",
    "E_eff",
    "strut_mass_metric",
    "sigma_max",
    "meta",
)

# Worker job: … bond_mask_seed, use_sparse (0/1), bond_keep_p
_EffiJob = tuple[Any, ...]


def _apply_random_bond_drops(
    t: np.ndarray, bond_mask_seed: int, bond_keep_p: float
) -> None:
    """In-place: each positive half-edge survives independently with prob bond_keep_p."""
    br = np.random.default_rng(int(bond_mask_seed))
    p = float(np.clip(bond_keep_p, 0.0, 1.0))
    for ch in (CH_RIGHT, CH_TOP):
        active = t[ch] > 0.0
        surv = br.random(t[ch].shape) < p
        t[ch] = np.where(active & surv, t[ch], 0.0)


def _work_effi(job: _EffiJob) -> dict[str, Any]:
    (
        jid,
        geom_seed,
        thick_seed,
        bond_mask_seed,
        p_geom,
        e_scale,
        delta,
        w,
        h,
        t_lo,
        t_hi,
        bond_max,
        use_sparse,
        edge_keep_p,
    ) = job
    t = fully_connected_gaussian_tensor(
        int(w), int(h), perturb=float(p_geom), seed=int(geom_seed)
    )
    if int(use_sparse):
        _apply_random_bond_drops(t, int(bond_mask_seed), float(edge_keep_p))
    apply_independent_thickness_inplace(
        t,
        int(thick_seed),
        low=float(t_lo),
        high=float(t_hi),
        bond_max=float(bond_max),
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
        "topology": "sparse" if int(use_sparse) else "full",
        "bond_keep_p": float(edge_keep_p) if int(use_sparse) else None,
    }
    return {
        "job_id": jid,
        "ok": r.ok,
        "error": r.error or "",
        "E_eff": float(r.E_eff) if r.ok else float("nan"),
        "strut_mass_metric": float(r.strut_mass_metric) if r.ok else float("nan"),
        "sigma_max": float(r.sigma_max) if r.ok else float("nan"),
        "meta": json.dumps(meta),
        "channels": np.ascontiguousarray(t, dtype=np.float32),
    }


def _make_job(
    jid: int,
    rng: np.random.Generator,
    *,
    e_scale: float,
    delta: float,
    w: int,
    h: int,
    t_lo: float,
    t_hi: float,
    bond_max: float,
    sparse_fraction: float,
    bond_keep_p: float,
) -> _EffiJob:
    use_sparse = 1 if rng.random() < float(sparse_fraction) else 0
    # Per sparse job: random keep-p around the CLI anchor so E_eff spans the band better.
    if use_sparse:
        spread = 0.14
        lo = max(0.62, float(bond_keep_p) - spread)
        hi = min(0.98, float(bond_keep_p) + spread)
        eff_keep = float(rng.uniform(lo, hi))
    else:
        eff_keep = 0.0
    return (
        jid,
        int(rng.integers(0, 2**31)),
        int(rng.integers(0, 2**31)),
        int(rng.integers(0, 2**31)),
        float(rng.uniform(0.08, 1.22)),
        float(e_scale),
        float(delta),
        int(w),
        int(h),
        float(t_lo),
        float(t_hi),
        float(bond_max),
        int(use_sparse),
        eff_keep,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Collect lattices with E_eff in [e-min,e-max] or goal±tol; time-limited pool"
    )
    ap.add_argument(
        "--e-min",
        type=float,
        default=6000.0,
        help="Lower edge of accepted E_eff (used if --goal is not set)",
    )
    ap.add_argument(
        "--e-max",
        type=float,
        default=8000.0,
        help="Upper edge of accepted E_eff (used if --goal is not set)",
    )
    ap.add_argument(
        "--goal",
        type=float,
        default=None,
        help="If set, accept only [goal−tol, goal+tol] (overrides --e-min/--e-max)",
    )
    ap.add_argument(
        "--tol",
        type=float,
        default=50.0,
        help="Half-width with --goal; also default E-equivalence width for analysis",
    )
    ap.add_argument(
        "--e-bin-width",
        type=float,
        default=50.0,
        help="Bin width (MPa) for grouping similar E_eff in saved config / analysis",
    )
    ap.add_argument("--w", type=int, default=48)
    ap.add_argument("--h", type=int, default=24)
    ap.add_argument("--thick-low", type=float, default=0.5)
    ap.add_argument("--thick-high", type=float, default=1.5)
    ap.add_argument("--bond-max", type=float, default=None)
    ap.add_argument(
        "--sparse-fraction",
        type=float,
        default=0.55,
        help="Fraction of samples that get random bond removal before thickness (0–1)",
    )
    ap.add_argument(
        "--bond-keep-p",
        type=float,
        default=0.82,
        help="Per-edge survival probability when sparse (Bernoulli; only missing bonds)",
    )
    ap.add_argument("--e-scale", type=float, default=10000.0)
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--master-seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--max-seconds",
        type=float,
        default=580.0,
        help="Stop submitting new jobs after this wall time (in-flight still finish)",
    )
    ap.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="Optional hard cap on solve attempts (0 = no cap, use time only)",
    )
    ap.add_argument("--out-dir", type=str, default="")
    ap.add_argument(
        "--save-tensors",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument("--tensors-name", type=str, default="targeted_lattice_tensors.npz")
    ap.add_argument("--csv-name", type=str, default="targeted_pool_results.csv")
    args = ap.parse_args()

    w, h = int(args.w), int(args.h)
    if w < 2 or h < 2:
        raise SystemExit("--w and --h must be at least 2")
    tol = float(args.tol)
    if tol < 0:
        raise SystemExit("--tol must be non-negative")
    e_bin_w = float(args.e_bin_width)
    if e_bin_w <= 0:
        raise SystemExit("--e-bin-width must be positive")
    if args.goal is not None:
        goal = float(args.goal)
        e_lo, e_hi = goal - tol, goal + tol
        accept_mode = "goal"
    else:
        goal = None
        e_lo, e_hi = float(args.e_min), float(args.e_max)
        if e_lo >= e_hi:
            raise SystemExit("--e-min must be < --e-max")
        accept_mode = "interval"
    t_lo = float(args.thick_low)
    t_hi = float(args.thick_high)
    if t_lo <= 0 or t_hi <= 0 or t_lo > t_hi:
        raise SystemExit("invalid thickness range")
    bond_max = float(args.bond_max) if args.bond_max is not None else float(t_hi)
    sparse_fraction = float(args.sparse_fraction)
    if not 0.0 <= sparse_fraction <= 1.0:
        raise SystemExit("--sparse-fraction must be in [0, 1]")
    bond_keep_p = float(args.bond_keep_p)
    if not 0.0 < bond_keep_p <= 1.0:
        raise SystemExit("--bond-keep-p must be in (0, 1]")

    root = Path(__file__).resolve().parent
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else root / "pic" / f"e_eff_target_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = {
        "accept_mode": accept_mode,
        "goal_E_eff": goal,
        "tolerance": tol if goal is not None else None,
        "E_band": [e_lo, e_hi],
        "E_equivalence_width": e_bin_w,
        "w": w,
        "h": h,
        "thick_low": t_lo,
        "thick_high": t_hi,
        "bond_max": bond_max,
        "e_scale": float(args.e_scale),
        "delta": float(args.delta),
        "master_seed": int(args.master_seed),
        "workers": int(args.workers),
        "max_seconds": float(args.max_seconds),
        "max_attempts": int(args.max_attempts),
        "geom_perturb_range": [0.08, 1.22],
        "sparse_fraction": sparse_fraction,
        "bond_keep_p": bond_keep_p,
        "save_tensors": bool(args.save_tensors),
        "tensors_name": str(args.tensors_name),
        "csv_name": str(args.csv_name),
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2))

    rng = np.random.default_rng(int(args.master_seed))
    t0 = time.monotonic()
    deadline = t0 + float(args.max_seconds)
    max_attempts = int(args.max_attempts)

    all_rows: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    next_jid = 0
    n_workers = max(1, int(args.workers))

    def gen_job() -> _EffiJob | None:
        nonlocal next_jid
        if max_attempts > 0 and next_jid >= max_attempts:
            return None
        if time.monotonic() >= deadline:
            return None
        j = _make_job(
            next_jid,
            rng,
            e_scale=float(args.e_scale),
            delta=float(args.delta),
            w=w,
            h=h,
            t_lo=t_lo,
            t_hi=t_hi,
            bond_max=bond_max,
            sparse_fraction=sparse_fraction,
            bond_keep_p=bond_keep_p,
        )
        next_jid += 1
        return j

    if n_workers <= 1:
        pbar = tqdm(desc="attempts", unit="try", dynamic_ncols=True)
        while True:
            if time.monotonic() >= deadline:
                break
            job = gen_job()
            if job is None:
                break
            row = _work_effi(job)
            all_rows.append(row)
            pbar.set_postfix(
                acc=len(accepted),
                tries=len(all_rows),
                ok=sum(1 for r in all_rows if r.get("ok")),
            )
            pbar.update(1)
            if row.get("ok") and _in_E_interval(float(row["E_eff"]), e_lo, e_hi):
                accepted.append(row)
        pbar.close()
    else:
        pool_kw: dict[str, Any] = {"max_workers": n_workers}
        if sys.version_info >= (3, 11):
            pool_kw["max_tasks_per_child"] = 64
        with ProcessPoolExecutor(**pool_kw) as ex:
            pending: dict[Future, int] = {}

            def pump() -> None:
                while len(pending) < n_workers:
                    job = gen_job()
                    if job is None:
                        return
                    fut = ex.submit(_work_effi, job)
                    pending[fut] = int(job[0])

            pump()
            pbar = tqdm(desc="attempts", unit="try", dynamic_ncols=True)
            while pending:
                done, _ = wait(
                    pending.keys(),
                    timeout=0.35,
                    return_when=FIRST_COMPLETED,
                )
                for fut in done:
                    pending.pop(fut)
                    row = fut.result()
                    all_rows.append(row)
                    pbar.set_postfix(
                        acc=len(accepted),
                        tries=len(all_rows),
                        ok=sum(1 for r in all_rows if r.get("ok")),
                    )
                    pbar.update(1)
                    if row.get("ok") and _in_E_interval(float(row["E_eff"]), e_lo, e_hi):
                        accepted.append(row)
                    pump()
            pbar.close()

    wall = time.monotonic() - t0
    n_att = len(all_rows)
    n_ok = sum(1 for r in all_rows if r.get("ok"))
    n_acc = len(accepted)
    n_acc_sparse = sum(
        1
        for r in accepted
        if json.loads(r["meta"]).get("topology") == "sparse"
    )

    tensor_path = out_dir / str(args.tensors_name)
    if accepted and args.save_tensors:
        X = np.stack([r["channels"] for r in accepted], axis=0)
        np.savez_compressed(
            tensor_path,
            X=X,
            job_id=np.array([int(r["job_id"]) for r in accepted], dtype=np.int64),
            ok=np.array([bool(r["ok"]) for r in accepted], dtype=np.bool_),
            E_eff=np.array([float(r["E_eff"]) for r in accepted], dtype=np.float64),
            strut_mass_metric=np.array(
                [float(r["strut_mass_metric"]) for r in accepted], dtype=np.float64
            ),
            sigma_max=np.array([float(r["sigma_max"]) for r in accepted], dtype=np.float64),
            E_band_lo=np.float64(e_lo),
            E_band_hi=np.float64(e_hi),
            E_equivalence_width=np.float64(e_bin_w),
            w=np.int32(w),
            h=np.int32(h),
        )

    csv_rows = [
        {k: r[k] for k in _CSV_FIELDS}
        for r in accepted
    ]
    with (out_dir / str(args.csv_name)).open("w", newline="") as f:
        csv_w = csv.DictWriter(f, fieldnames=list(_CSV_FIELDS))
        csv_w.writeheader()
        csv_w.writerows(csv_rows)

    summary = {
        "accept_mode": accept_mode,
        "goal_E_eff": goal,
        "tolerance": tol if goal is not None else None,
        "E_band": [e_lo, e_hi],
        "E_equivalence_width": e_bin_w,
        "sparse_fraction": sparse_fraction,
        "bond_keep_p": bond_keep_p,
        "n_attempts": n_att,
        "n_solve_ok": n_ok,
        "n_accepted_in_band": n_acc,
        "n_accepted_sparse_topology": n_acc_sparse,
        "acceptance_rate_of_ok": float(n_acc / max(n_ok, 1)),
        "acceptance_rate_of_all": float(n_acc / max(n_att, 1)),
        "wall_seconds": float(wall),
        "max_seconds_config": float(args.max_seconds),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if accept_mode == "goal":
        band_desc = f"E_eff goal {goal:g} ± {tol:g} → [{e_lo:g}, {e_hi:g}]"
    else:
        band_desc = f"E_eff band [{e_lo:g}, {e_hi:g}] (equiv. bin width {e_bin_w:g})"
    readme = (
        f"{band_desc}.\n"
        f"Topology mix: sparse fraction **{sparse_fraction:g}**, bond-keep-p **{bond_keep_p:g}** "
        f"→ accepted sparse **{n_acc_sparse}** / {n_acc}.\n"
        f"Attempts: {n_att}, solves ok: {n_ok}, **accepted**: {n_acc}.\n"
        f"Wall ~{wall:.1f}s (submit cap {args.max_seconds:g}s).\n"
    )
    if accepted and args.save_tensors:
        readme += f"Tensors: {tensor_path.name} — X shape ({n_acc}, 4, {h}, {w}).\n"
    if not accepted:
        readme += "No accepted samples; widen band, raise time, or change goal/physics.\n"
    (out_dir / "README.txt").write_text(readme)
    print(readme)
    print(out_dir)


if __name__ == "__main__":
    main()
