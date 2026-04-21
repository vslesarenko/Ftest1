#!/usr/bin/env python3
"""Analyze lattice_e_eff_target.py runs: E_eff band, stress spread vs Young's modulus bins (±50), figures + REPORT.md."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _load_band_and_width(
    cfg: dict, summary: dict
) -> tuple[float, float, float, str | None]:
    """Return (e_lo, e_hi, equiv_width, goal_or_none)."""
    band = cfg.get("E_band") or summary.get("E_band")
    if band and len(band) == 2:
        lo, hi = float(band[0]), float(band[1])
    else:
        g = float(cfg.get("goal_E_eff", summary.get("goal_E_eff", float("nan"))))
        tol = float(cfg.get("tolerance", summary.get("tolerance", 50.0)))
        lo, hi = g - tol, g + tol
    w = float(cfg.get("E_equivalence_width", summary.get("E_equivalence_width", 50.0)))
    goal = cfg.get("goal_E_eff", summary.get("goal_E_eff"))
    if goal is None:
        g_out = None
    else:
        g_out = float(goal)
    return lo, hi, w, g_out


def _assign_bins(
    E: np.ndarray, e_lo: float, e_hi: float, bin_w: float
) -> tuple[np.ndarray, np.ndarray, int]:
    n_bin = max(1, int(round((e_hi - e_lo) / bin_w)))
    idx = np.floor((E - e_lo) / bin_w).astype(np.int64)
    idx = np.clip(idx, 0, n_bin - 1)
    centers = e_lo + (np.arange(n_bin) + 0.5) * bin_w
    return idx, centers, n_bin


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()
    cfg_path = run_dir / "run_config.json"
    sum_path = run_dir / "summary.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
    summary = json.loads(sum_path.read_text()) if sum_path.is_file() else {}
    csv_name = str(cfg.get("csv_name", "targeted_pool_results.csv"))
    csv_path = run_dir / csv_name
    npz_path = run_dir / str(cfg.get("tensors_name", "targeted_lattice_tensors.npz"))

    if not csv_path.is_file():
        raise SystemExit(
            f"Missing {csv_path} (run lattice_e_eff_target.py first; empty runs still write a header-only CSV)"
        )

    import matplotlib.pyplot as plt

    lo, hi, equiv_w, goal = _load_band_and_width(cfg, summary)
    mode = str(cfg.get("accept_mode", summary.get("accept_mode", "")))

    E, M, S = [], [], []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("job_id"):
                continue
            E.append(float(row["E_eff"]))
            M.append(float(row["strut_mass_metric"]))
            S.append(float(row["sigma_max"]))
    E, M, S = np.array(E), np.array(M), np.array(S)
    n = len(E)

    in_band = (E >= lo) & (E <= hi)
    n_bad = int(np.sum(~in_band))

    out = run_dir / "analysis_target"
    out.mkdir(exist_ok=True)

    # --- Histograms ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    nb = min(50, max(10, n // 3)) if n else 10
    axes[0].hist(E, bins=nb, range=(lo, hi) if n else None, color="#3182bd", edgecolor="none", alpha=0.9)
    axes[0].axvline(lo, color="red", ls="--", lw=1)
    axes[0].axvline(hi, color="red", ls="--", lw=1)
    axes[0].set_title(r"$E_{\mathrm{eff}}$")
    axes[0].set_ylabel("count")
    axes[1].hist(S, bins=min(40, max(8, n // 5)) if n else 8, color="#e6550d", edgecolor="none", alpha=0.9)
    axes[1].set_title(r"$\sigma_{\max}$")
    axes[2].hist(M, bins=min(40, max(8, n // 5)) if n else 8, color="#31a354", edgecolor="none", alpha=0.9)
    axes[2].set_title("Strut mass metric")
    title_goal = f"goal={goal:g}±" if goal is not None else ""
    band_title = f"[{lo:g}, {hi:g}]" if goal is None else f"{title_goal} → [{lo:g}, {hi:g}]"
    fig.suptitle(
        f"E_eff run (n={n}, band {band_title}, outside band: {n_bad}; equiv. bins {equiv_w:g})",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out / "fig_histograms.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    if n:
        hb = ax.hexbin(
            E, S, gridsize=min(35, max(10, n // 20)), cmap="YlOrRd", mincnt=1
        )
        plt.colorbar(hb, ax=ax, label="count")
    else:
        ax.text(0.5, 0.5, "No accepted samples", ha="center", va="center", transform=ax.transAxes)
    ax.axvline(lo, color="cyan", ls="--", lw=1, alpha=0.8)
    ax.axvline(hi, color="cyan", ls="--", lw=1, alpha=0.8)
    ax.set_xlim(lo, hi)
    ax.set_xlabel(r"$E_{\mathrm{eff}}$")
    ax.set_ylabel(r"$\sigma_{\max}$")
    fig.tight_layout()
    fig.savefig(out / "fig_e_sigma_hexbin.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    if n:
        sc = ax.scatter(E, M, c=S, s=12, alpha=0.5, cmap="inferno", rasterized=True)
        plt.colorbar(sc, ax=ax, label=r"$\sigma_{\max}$")
    else:
        ax.text(0.5, 0.5, "No accepted samples", ha="center", va="center", transform=ax.transAxes)
    ax.axvline(lo, color="cyan", ls="--", lw=1)
    ax.axvline(hi, color="cyan", ls="--", lw=1)
    ax.set_xlim(lo, hi)
    ax.set_xlabel(r"$E_{\mathrm{eff}}$")
    ax.set_ylabel("Mass metric")
    fig.tight_layout()
    fig.savefig(out / "fig_e_mass_sigma_colour.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # --- Per-bin stress variance (Young's modulus grouped in ±equiv_w / 50-wide strips) ---
    by_bin: list[dict[str, float | int]] = []
    if n:
        b_idx, centers, n_bin = _assign_bins(E, lo, hi, equiv_w)
        for i in range(n_bin):
            mask = b_idx == i
            k = int(np.sum(mask))
            s_sub = S[mask]
            if k == 0:
                by_bin.append(
                    {
                        "bin_index": i,
                        "E_center": float(centers[i]),
                        "n": 0,
                        "sigma_mean": float("nan"),
                        "sigma_std": float("nan"),
                        "sigma_cv": float("nan"),
                        "sigma_p10": float("nan"),
                        "sigma_p90": float("nan"),
                    }
                )
                continue
            sm = float(np.mean(s_sub))
            sd = float(np.std(s_sub, ddof=1)) if k > 1 else 0.0
            by_bin.append(
                {
                    "bin_index": i,
                    "E_center": float(centers[i]),
                    "n": k,
                    "sigma_mean": sm,
                    "sigma_std": sd,
                    "sigma_cv": float(sd / sm) if sm else float("nan"),
                    "sigma_p10": float(np.percentile(s_sub, 10)),
                    "sigma_p90": float(np.percentile(s_sub, 90)),
                }
            )

        # Boxplot: bins with at least 2 samples
        plot_bins = [b for b in by_bin if b["n"] >= 2]
        if plot_bins:
            fig, ax = plt.subplots(figsize=(max(10.0, 0.28 * len(plot_bins)), 4.5))
            data = [S[b_idx == b["bin_index"]] for b in plot_bins]
            pos = [b["E_center"] for b in plot_bins]
            ax.boxplot(data, positions=pos, widths=equiv_w * 0.65, manage_ticks=False)
            ax.set_xticks(pos)
            ax.set_xticklabels([f"{p:.0f}" for p in pos], rotation=75, ha="right", fontsize=7)
            ax.axvline(lo, color="gray", ls=":", lw=1, alpha=0.7)
            ax.axvline(hi, color="gray", ls=":", lw=1, alpha=0.7)
            ax.set_xlim(lo - equiv_w, hi + equiv_w)
            ax.set_xlabel(r"$E_{\mathrm{eff}}$ bin center (" + f"{equiv_w:g}-wide bins)")
            ax.set_ylabel(r"$\sigma_{\max}$ distribution per bin")
            ax.set_title(
                r"At similar $E_{\mathrm{eff}}$ (±"
                + f"{equiv_w / 2:g}"
                + r"), $\sigma_{\max}$ varies strongly across microstructures"
            )
            fig.tight_layout()
            fig.savefig(out / "fig_sigma_boxplot_by_E_bin.png", dpi=160, bbox_inches="tight")
            plt.close(fig)

        nonempty = [b for b in by_bin if b["n"] > 0]
        if nonempty:
            fig, ax = plt.subplots(figsize=(7.0, 4.2))
            cx = [b["E_center"] for b in nonempty]
            stds = [b["sigma_std"] for b in nonempty]
            ns = [b["n"] for b in nonempty]
            ax.plot(cx, stds, "o-", color="#3182bd", lw=1.2, ms=4, label=r"std($\sigma_{\max}$)")
            for c, s, nn in zip(cx, stds, ns):
                ax.annotate(
                    str(nn),
                    (c, s),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=6,
                    alpha=0.8,
                )
            ax.set_xlabel(r"$E_{\mathrm{eff}}$ bin center")
            ax.set_ylabel(r"std of $\sigma_{\max}$ in bin (annotated: n)")
            ax.set_title("Stress spread within each Young's-modulus bin")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out / "fig_sigma_std_per_E_bin.png", dpi=160, bbox_inches="tight")
            plt.close(fig)

    stats = {
        "n_samples": n,
        "accept_mode": mode,
        "E_goal": goal,
        "E_band": [lo, hi],
        "E_equivalence_width": equiv_w,
        "rows_outside_band": n_bad,
        "E_eff": {
            "min": float(E.min()) if n else float("nan"),
            "max": float(E.max()) if n else float("nan"),
            "mean": float(E.mean()) if n else float("nan"),
            "std": float(E.std(ddof=1)) if n > 1 else 0.0,
        },
        "sigma_max": {
            "min": float(S.min()) if n else float("nan"),
            "max": float(S.max()) if n else float("nan"),
            "mean": float(S.mean()) if n else float("nan"),
            "std": float(S.std(ddof=1)) if n > 1 else 0.0,
            "p10_p90": [float(np.percentile(S, 10)), float(np.percentile(S, 90))] if n else [float("nan"), float("nan")],
        },
        "strut_mass_metric": {
            "min": float(M.min()) if n else float("nan"),
            "max": float(M.max()) if n else float("nan"),
            "mean": float(M.mean()) if n else float("nan"),
            "std": float(M.std(ddof=1)) if n > 1 else 0.0,
        },
        "by_E_equivalence_bin": by_bin,
        "run_summary": summary,
    }
    (out / "analysis_stats.json").write_text(json.dumps(stats, indent=2))

    zs = float(np.corrcoef(E, S)[0, 1]) if n > 2 else float("nan")
    zm = float(np.corrcoef(E, M)[0, 1]) if n > 2 else float("nan")
    sm = float(np.corrcoef(S, M)[0, 1]) if n > 2 else float("nan")

    nonempty_bins = [b for b in by_bin if b.get("n", 0) > 0]
    mean_cv = (
        float(np.nanmean([b["sigma_cv"] for b in nonempty_bins if np.isfinite(b["sigma_cv"])]))
        if nonempty_bins
        else float("nan")
    )

    cfg_line = (
        f"- **Goal** mode: **{goal:g}** ± **{cfg.get('tolerance', '?')}** → **[{lo:g}, {hi:g}]**\n"
        if goal is not None
        else f"- **Band** [{lo:g}, {hi:g}] (all accepted samples in this Young's-modulus interval)\n"
    )
    bin_table_rows = ""
    for b in by_bin:
        if b["n"] == 0:
            continue
        bin_table_rows += (
            f"| {b['E_center']:.0f} | {b['n']} | {b['sigma_mean']:.4g} | {b['sigma_std']:.4g} | "
            f"{b['sigma_cv']:.4g} | [{b['sigma_p10']:.4g}, {b['sigma_p90']:.4g}] |\n"
        )

    report = f"""# E_eff run — stress variance vs Young's modulus

## Configuration (from `run_config.json`)

{cfg_line}- **Equivalence bin width** (treat similar \\(E\\)): **{equiv_w:g}** (samples grouped by strips of this width for stress statistics)
- Lattice: **{cfg.get("w", "?")}×{cfg.get("h", "?")}**
- Thickness: U({cfg.get("thick_low", "?")}, {cfg.get("thick_high", "?")})
- `e_scale`: {cfg.get("e_scale", "?")}, `delta`: {cfg.get("delta", "?")}

## Acceptance (from `summary.json`)

```json
{json.dumps(summary, indent=2)}
```

## Global sample summary (n = **{n}**)

| | \\(E_{{\\mathrm{{eff}}}}\\) | \\(\\sigma_{{\\max}}\\) | Mass metric |
|--|--|--|--|
| min | {stats["E_eff"]["min"]:.4g} | {stats["sigma_max"]["min"]:.4g} | {stats["strut_mass_metric"]["min"]:.4g} |
| max | {stats["E_eff"]["max"]:.4g} | {stats["sigma_max"]["max"]:.4g} | {stats["strut_mass_metric"]["max"]:.4g} |
| mean ± std | {stats["E_eff"]["mean"]:.4g} ± {stats["E_eff"]["std"]:.4g} | {stats["sigma_max"]["mean"]:.4g} ± {stats["sigma_max"]["std"]:.4g} | {stats["strut_mass_metric"]["mean"]:.4g} ± {stats["strut_mass_metric"]["std"]:.4g} |

- Rows with \\(E_{{\\mathrm{{eff}}}}\\) **outside** [{lo:g}, {hi:g}]: **{n_bad}** (should be 0 if filtering worked).

## Stress variability within each \\(E_{{\\mathrm{{eff}}}}\\) bin (width {equiv_w:g})

Across **[{lo:g}, {hi:g}]**, microstructures with **comparable** Young's modulus (same {equiv_w:g}-wide bin) can still show **large differences** in \\(\\sigma_{{\\max}}\\). Mean coefficient of variation of \\(\\sigma_{{\\max}}\\) across nonempty bins: **{mean_cv:.4g}**.

| Bin center | n | mean \\(\\sigma\\) | std \\(\\sigma\\) | CV | p10–p90 \\(\\sigma\\) |
|------------|---|------------------|-----------------|-----|---------------------|
{bin_table_rows if bin_table_rows else "| — | — | — | — | — | — |\n"}

## Correlations (Pearson, all accepted samples)

| Pair | r |
|------|---|
| \\(E_{{\\mathrm{{eff}}}}\\) vs \\(\\sigma_{{\\max}}\\) | {zs:.4f} |
| \\(E_{{\\mathrm{{eff}}}}\\) vs mass | {zm:.4f} |
| \\(\\sigma_{{\\max}}\\) vs mass | {sm:.4f} |

## Figures (in this folder)

- `fig_histograms.png` — \\(E_{{\\mathrm{{eff}}}}\\), \\(\\sigma_{{\\max}}\\), mass
- `fig_e_sigma_hexbin.png` — \\(E_{{\\mathrm{{eff}}}}\\) vs \\(\\sigma_{{\\max}}\\) (vertical band edges)
- `fig_e_mass_sigma_colour.png` — \\(E_{{\\mathrm{{eff}}}}\\) vs mass, colour = \\(\\sigma_{{\\max}}\\)
- `fig_sigma_boxplot_by_E_bin.png` — distribution of \\(\\sigma_{{\\max}}\\) per Young's-modulus bin (requires ≥2 samples per bin)
- `fig_sigma_std_per_E_bin.png` — std of \\(\\sigma_{{\\max}}\\) vs bin center (annotated sample counts)

## Tensor file

"""
    if npz_path.is_file():
        report += f"`{npz_path.name}` present — `X` shape aligns with n = {n}.\n"
    else:
        report += f"`{npz_path.name}` not found (tensors not saved or empty run).\n"

    (out / "REPORT.md").write_text(report)
    print(f"Wrote {out / 'REPORT.md'} and figures in {out}")


if __name__ == "__main__":
    main()
