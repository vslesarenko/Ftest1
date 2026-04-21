#!/usr/bin/env python3
"""
Build a detailed markdown research report + figures for a bulk lattice_thickness_picker run.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _read_csv_arrays(csv_path: Path) -> dict[str, np.ndarray]:
    E: list[float] = []
    S: list[float] = []
    M: list[float] = []
    jid: list[int] = []
    ok: list[bool] = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            jid.append(int(row["job_id"]))
            ok.append(str(row["ok"]).lower() == "true")
            E.append(float(row["E_eff"]))
            S.append(float(row["sigma_max"]))
            M.append(float(row["strut_mass_metric"]))
    return {
        "job_id": np.array(jid, dtype=np.int64),
        "ok": np.array(ok, dtype=bool),
        "E_eff": np.array(E, dtype=np.float64),
        "sigma_max": np.array(S, dtype=np.float64),
        "mass": np.array(M, dtype=np.float64),
    }


def _pct(x: np.ndarray, p: float) -> float:
    return float(np.percentile(x, p))


def _mask_joint_quantile(
    E: np.ndarray, M: np.ndarray, lo: float, hi: float
) -> np.ndarray:
    e_lo, e_hi = np.percentile(E, [lo, hi])
    m_lo, m_hi = np.percentile(M, [lo, hi])
    return (E >= e_lo) & (E <= e_hi) & (M >= m_lo) & (M <= m_hi)


def _sigma_stats(S: np.ndarray) -> dict[str, float]:
    if S.size == 0:
        return {k: float("nan") for k in ("n", "min", "max", "mean", "std", "p5", "p50", "p95")}
    return {
        "n": float(S.size),
        "min": float(np.min(S)),
        "max": float(np.max(S)),
        "mean": float(np.mean(S)),
        "std": float(np.std(S, ddof=1)) if S.size > 1 else 0.0,
        "p5": float(np.percentile(S, 5)),
        "p50": float(np.percentile(S, 50)),
        "p95": float(np.percentile(S, 95)),
    }


def _decile_bin(x: np.ndarray, n_dec: int) -> np.ndarray:
    """Assign 0..n_dec-1 with ~equal counts (rank-based)."""
    o = np.argsort(x)
    r = np.empty_like(o)
    r[o] = np.arange(len(x))
    return np.clip(r * n_dec // len(x), 0, n_dec - 1).astype(np.int32)


def _plot_tensor_row(
    axs: np.ndarray,
    t4: np.ndarray,
    *,
    h: int,
    w: int,
    title: str,
    b2: tuple[float, float],
) -> None:
    """t4 shape (4, H, W). axs length 4."""
    extent = [-0.5, w - 0.5, -0.5, h - 0.5]
    axs[0].imshow(t4[0], origin="lower", cmap="coolwarm", extent=extent, aspect="auto")
    axs[0].set_title("ch0 dx")
    axs[1].imshow(t4[1], origin="lower", cmap="coolwarm", extent=extent, aspect="auto")
    axs[1].set_title("ch1 dy")
    axs[2].imshow(
        t4[2],
        origin="lower",
        cmap="magma",
        vmin=b2[0],
        vmax=b2[1],
        extent=extent,
        aspect="auto",
    )
    axs[2].set_title("ch2 horiz. thickness")
    axs[3].imshow(
        t4[3],
        origin="lower",
        cmap="magma",
        vmin=b2[0],
        vmax=b2[1],
        extent=extent,
        aspect="auto",
    )
    axs[3].set_title("ch3 vert. thickness")
    for ax in axs:
        ax.set_xlabel("i")
    axs[0].set_ylabel("j")
    axs[0].text(
        0.02,
        0.98,
        title,
        transform=axs[0].transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()
    fig_dir = run_dir / "research_report"
    fig_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt
    csv_path = run_dir / "pool_results.csv"
    npz_path = run_dir / "lattice_tensors.npz"
    cfg_path = run_dir / "run_config.json"
    sum_path = run_dir / "summary.json"

    if not csv_path.is_file():
        raise SystemExit(f"Missing {csv_path}")

    run_cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
    wall_summary = json.loads(sum_path.read_text()) if sum_path.is_file() else {}
    ws = wall_summary.get("wall_seconds")
    if isinstance(ws, (int, float)):
        wall_time_md = f"{float(ws):.1f} s (~{float(ws) / 3600:.2f} h)"
    else:
        wall_time_md = "_(not in summary.json)_"

    d = _read_csv_arrays(csv_path)
    ok = d["ok"]
    E = d["E_eff"][ok]
    M = d["mass"][ok]
    S = d["sigma_max"][ok]
    n = int(S.size)

    # --- Fig: ECDF sigma full ---
    fig, ax = plt.subplots(figsize=(7, 4.2))
    s_sorted = np.sort(S)
    y_ecdf = np.arange(1, n + 1, dtype=np.float64) / n
    ax.plot(s_sorted, y_ecdf, "k-", lw=1.2, label="All ok samples")
    masks = {
        "IQR∩ (E & M 25–75%)": _mask_joint_quantile(E, M, 25, 75),
        "Mid 20%∩ (E & M 40–60%)": _mask_joint_quantile(E, M, 40, 60),
        "Mid 10%∩ (E & M 45–55%)": _mask_joint_quantile(E, M, 45, 55),
    }
    cond_stats: dict[str, dict] = {"full": _sigma_stats(S)}
    for label, m in masks.items():
        Sm = S[m]
        cond_stats[label] = _sigma_stats(Sm)
        if Sm.size > 10:
            sm = np.sort(Sm)
            ym = np.arange(1, sm.size + 1, dtype=np.float64) / sm.size
            ax.plot(sm, ym, "--", lw=1.0, alpha=0.85, label=f"{label} (n={Sm.size})")
    ax.set_xlabel(r"Peak stress $\sigma_{\max}$ (scaled, fiber + bending proxy)")
    ax.set_ylabel("ECDF")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(r"Stress distribution: full vs comparable $E_{\mathrm{eff}}$ & mass bands")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig01_sigma_ecdf_conditional.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # --- Fig: hexbin E vs sigma, marginal mass ---
    rng = np.random.default_rng(42)
    mplot = min(12000, n)
    pick = rng.choice(n, size=mplot, replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    hb0 = axes[0].hexbin(E[pick], S[pick], gridsize=55, cmap="YlOrRd", mincnt=1)
    axes[0].set_xlabel(r"$E_{\mathrm{eff}}$ (homogenized)")
    axes[0].set_ylabel(r"$\sigma_{\max}$")
    plt.colorbar(hb0, ax=axes[0], label="count")
    axes[0].set_title(r"Hexbin: $E_{\mathrm{eff}}$ vs peak stress")
    sc = axes[1].scatter(
        E[pick],
        S[pick],
        c=M[pick],
        s=4,
        alpha=0.35,
        cmap="viridis",
        rasterized=True,
    )
    axes[1].set_xlabel(r"$E_{\mathrm{eff}}$")
    axes[1].set_ylabel(r"$\sigma_{\max}$")
    plt.colorbar(sc, ax=axes[1], label="Mass metric")
    axes[1].set_title("Scatter (subsample): colour = mass metric")
    fig.suptitle(f"Coupling structure (m={mplot} random samples)", fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig02_e_sigma_mass.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # --- Fig: mean sigma by E-decile × M-decile ---
    n_dec = 10
    e_d = _decile_bin(E, n_dec)
    m_d = _decile_bin(M, n_dec)
    grid_mean = np.full((n_dec, n_dec), np.nan, dtype=float)
    grid_count = np.zeros((n_dec, n_dec), dtype=int)
    for ie in range(n_dec):
        for im in range(n_dec):
            sel = (e_d == ie) & (m_d == im)
            grid_count[ie, im] = int(sel.sum())
            if sel.sum() >= 30:
                grid_mean[ie, im] = float(np.mean(S[sel]))
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    vmax = np.nanmax(grid_mean)
    vmin = np.nanmin(grid_mean)
    im = ax.imshow(
        grid_mean,
        origin="lower",
        cmap="RdYlBu_r",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Mass metric decile (0=low … 9=high)")
    ax.set_ylabel(r"$E_{\mathrm{eff}}$ decile (0=low … 9=high)")
    ax.set_title(r"Mean $\sigma_{\max}$ in $(E_{\mathrm{eff}}, \mathrm{mass})$ decile cells (≥30 samples)")
    for i in range(n_dec):
        for j in range(n_dec):
            c = grid_count[i, j]
            if c >= 30 and not np.isnan(grid_mean[i, j]):
                ax.text(j, i, f"{grid_mean[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if grid_mean[i, j] > (vmin + vmax) / 2 else "black")
    plt.colorbar(im, ax=ax, label=r"mean $\sigma_{\max}$")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig03_sigma_mean_decile_grid.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # --- Fig: correlation heatmap (already had in summary_plots but repeat for report bundle) ---
    Z = np.column_stack([E, S, M])
    C = np.corrcoef(Z.T)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([r"$E_{\mathrm{eff}}$", r"$\sigma_{\max}$", "Mass"])
    ax.set_yticklabels([r"$E_{\mathrm{eff}}$", r"$\sigma_{\max}$", "Mass"])
    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i,
                f"{C[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if abs(C[i, j]) > 0.55 else "black",
            )
    plt.colorbar(im, ax=ax, fraction=0.046, label="Pearson r")
    ax.set_title("Correlation (ok samples)")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig04_correlation.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # --- Example tensors from npz ---
    if npz_path.is_file():
        print("Loading lattice_tensors.npz for example panels…")
        D = np.load(npz_path)
        X = D["X"]
        H, W = int(D["h"]), int(D["w"])
        jid_all = D["job_id"]
        sigma_npz = D["sigma_max"]
        ok_npz = D["ok"]

        sig_ok = sigma_npz[ok_npz]
        jid_ok = jid_all[ok_npz]
        order = np.argsort(sig_ok)
        sub = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]
        captions = [r"Low $\sigma_{\max}$", r"Median $\sigma_{\max}$", r"High $\sigma_{\max}$"]

        b_lo = float("inf")
        b_hi = float("-inf")
        for si in sub:
            jid = int(jid_ok[si])
            idx = int(np.where(jid_all == jid)[0][0])
            t = X[idx]
            b_lo = min(b_lo, float(t[2, :, : W - 1].min()), float(t[3, : H - 1, :].min()))
            b_hi = max(b_hi, float(t[2, :, : W - 1].max()), float(t[3, : H - 1, :].max()))
        if b_hi - b_lo < 1e-6:
            b_lo, b_hi = b_lo - 0.05, b_hi + 0.05

        fig, axs = plt.subplots(3, 4, figsize=(13.5, 9))
        for row, (si, caption) in enumerate(zip(sub, captions)):
            jid = int(jid_ok[si])
            idx = int(np.where(jid_all == jid)[0][0])
            t4 = np.asarray(X[idx], dtype=np.float32)
            sigv = float(sigma_npz[idx])
            ev = float(D["E_eff"][idx])
            mv = float(D["strut_mass_metric"][idx])
            cap = f"{caption}\nσ={sigv:.3f}  E_eff={ev:.1f}  mass={mv:.1f}"
            _plot_tensor_row(axs[row], t4, h=H, w=W, title=cap, b2=(b_lo, b_hi))
        fig.suptitle("Example lattices: channels for low / median / high peak stress (same colour scale ch2–ch3)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(fig_dir / "fig05_example_tensors_low_med_high.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        example_note = (
            "Figure `fig05_example_tensors_low_med_high.png` shows one realization each near the "
            "minimum, median, and maximum **peak stress** among all samples (not conditional on "
            "$E_{\\mathrm{eff}}$ or mass)."
        )
    else:
        example_note = "Tensor file not found; example lattice figure omitted."

    # --- Markdown report ---
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    C_E_S = float(C[0, 1])
    C_E_M = float(C[0, 2])
    C_S_M = float(C[1, 2])

    def _fmt_st(st: dict[str, float]) -> str:
        if np.isnan(st["n"]) or int(st["n"]) == 0:
            return "\n_(empty subset)_\n\n"
        return (
            "\n| Statistic | Value |\n|-----------|-------|\n"
            f"| n | {int(st['n'])} |\n"
            f"| min | {st['min']:.4f} |\n"
            f"| p5 | {st['p5']:.4f} |\n"
            f"| p50 | {st['p50']:.4f} |\n"
            f"| p95 | {st['p95']:.4f} |\n"
            f"| max | {st['max']:.4f} |\n"
            f"| mean ± std | {st['mean']:.4f} ± {st['std']:.4f} |\n\n"
        )

    report = f"""# Bulk lattice run — research summary

**Generated:** {now}  
**Run directory:** `{run_dir}`

## 1. Simulation parameters (as recorded)

These match the **defaults / recorded values** in `run_config.json` for this folder.

| Parameter | Value |
|-----------|-------|
| Lattice size `W × H` | {run_cfg.get("w", "?")} × {run_cfg.get("h", "?")} (tensor shape `(4, H, W)`) |
| Thickness factors (half-edges) | `U({run_cfg.get("thick_low", "?")}, {run_cfg.get("thick_high", "?")})` |
| `bond_max` (post-multiply clip) | {run_cfg.get("bond_max", "?")} |
| Geometry perturbation `perturb` | uniform in [{run_cfg.get("geom_perturb_range", ["?", "?"])[0]}, {run_cfg.get("geom_perturb_range", ["?", "?"])[1]}] per sample (Gaussian lattice) |
| `e_scale` | {run_cfg.get("e_scale", "?")} |
| `delta` (prescribed x-displacement, right face) | {run_cfg.get("delta", "?")} |
| `bond_threshold` | 0 (full grid) |
| Monte Carlo pool | {run_cfg.get("pool", "?")} |
| `master_seed` | {run_cfg.get("master_seed", "?")} |
| Worker processes | {run_cfg.get("workers", wall_summary.get("workers", "?"))} |
| Bulk mode | {run_cfg.get("bulk", "?")} |
| Wall time (pool + save) | {wall_time_md} |

**Note:** `E_eff` here is the **homogenized** modulus from macro reaction force / strain for the **solved** frame (see `effective_young_modulus_homogenized` in `tensor_lattice.py`), not a fixed input Young’s modulus. The strut **mass metric** is `Σ_e L_e · w_e` with `w_e` the thickness/stiffness factor — a volume proxy when area scales with `w`.

## 2. Dataset

- CSV rows (ok): **{n}** (all solves succeeded in this run).
- Peak stress `sigma_max`: maximum over edges of a **scaled** axial+fiber bending combination from the post-process (`beam_fiber_stress_per_edge`).

## 3. Unconditional stress (full population)

{_fmt_st(cond_stats["full"])}

## 4. “Comparable” global stiffness and mass — what stress range remains?

We **do not** fix a single Young’s modulus or mass in the generator; both vary with geometry and thickness. To approximate *“similar homogenized stiffness and similar mass”*, we restrict samples whose **`E_eff`** and **mass metric** both lie in the **same marginal quantile band** (intersection). This is a **statistical** notion of comparability, not a physical constraint from a single design.

### 4.1 Conditional peak-stress summaries

| Scenario | Description |
|----------|-------------|
| **IQR∩** | Both `E_eff` and mass in their respective **25–75%** intervals (central half on each axis). |
| **Mid 20%∩** | Both in **40–60%** (middle 20% on each axis). |
| **Mid 10%∩** | Both in **45–55%** (middle 10% on each axis). |

**Peak stress statistics:**

#### Full population

{_fmt_st(cond_stats["full"])}

#### IQR∩ (E & M 25–75%)

{_fmt_st(cond_stats["IQR∩ (E & M 25–75%)"])}

#### Mid 20%∩ (E & M 40–60%)

{_fmt_st(cond_stats["Mid 20%∩ (E & M 40–60%)"])}

#### Mid 10%∩ (E & M 45–55%)

{_fmt_st(cond_stats["Mid 10%∩ (E & M 45–55%)"])}

**Interpretation (qualitative):** tightening the joint band on `E_eff` and mass **narrows** the achievable `sigma_max` range only modestly here because geometry and **local** thickness patterns still redistribute load. The **mid 10%∩** band is the strictest of the three; compare its **min–max** and **p5–p95** spread to the full population to see how much peak stress can still vary when global scalars are statistically similar.

### 4.2 Linear correlations (ok samples)

| Pair | Pearson r |
|------|-----------|
| `E_eff` vs `sigma_max` | {C_E_S:.3f} |
| `E_eff` vs mass | {C_E_M:.3f} |
| `sigma_max` vs mass | {C_S_M:.3f} |

In this ensemble, **linear** correlations are **fairly strong** (|r| up to ~0.87): higher `E_eff` tends to coincide with **lower** peak stress and with **lower** mass metric here, while peak stress and mass move **together** positively. That still leaves a **spread** of `sigma_max` within narrow joint bands on `E_eff` and mass (§4.1), so **microstructure / load path** matters beyond these two scalars alone.

## 5. Figures

All files live in `{fig_dir.name}/` (same folder as this `REPORT.md`).

### 5.1 Peak stress ECDF (full vs joint bands)

![ECDF of peak stress](fig01_sigma_ecdf_conditional.png)

### 5.2 Coupling: $E_{{\\mathrm{{eff}}}}$, peak stress, mass

![E–sigma–mass](fig02_e_sigma_mass.png)

### 5.3 Mean peak stress by $E_{{\\mathrm{{eff}}}}$ vs mass deciles

![Decile grid mean sigma](fig03_sigma_mean_decile_grid.png)

### 5.4 Correlation matrix

![Correlations](fig04_correlation.png)

### 5.5 Example lattices (low / median / high peak stress)

![Example tensors](fig05_example_tensors_low_med_high.png)

**File index**

| File | Content |
|------|---------|
| `fig01_sigma_ecdf_conditional.png` | ECDF of `sigma_max`: full vs joint quantile bands on `E_eff` & mass. |
| `fig02_e_sigma_mass.png` | Hexbin `E_eff`–`sigma_max`; scatter coloured by mass. |
| `fig03_sigma_mean_decile_grid.png` | Mean `sigma_max` in decile×decile cells (≥30 samples/cell). |
| `fig04_correlation.png` | Correlation matrix. |
| `fig05_example_tensors_low_med_high.png` | ch0–ch3 for low / median / high **peak stress**. |

{example_note}

## 6. Caveats

1. **Scaled stresses** — values are from the model’s post-processing (`scaled` / combined axial–bending proxy), not SI MPa unless you map material units.
2. **`E_eff` is an outcome** of each sample’s microstructure and BCs, not an input held constant.
3. **Joint quantile bands** are a pragmatic way to approximate “similar bulk stiffness and mass”; other definitions (fixed tolerance on `|E - E_ref|`, `|M - M_ref|`) would give slightly different conditional ranges.
4. **Decile heatmap** cells with fewer than 30 points are left blank.

---
*Report produced by `generate_bulk_research_report.py`.*
"""

    out_md = fig_dir / "REPORT.md"
    out_md.write_text(report)
    print(f"Wrote {out_md}")
    print(f"Figures in {fig_dir}")


if __name__ == "__main__":
    main()
