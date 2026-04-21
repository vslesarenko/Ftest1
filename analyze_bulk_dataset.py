#!/usr/bin/env python3
"""Summary plots for lattice_thickness_picker bulk output (CSV + lattice_tensors.npz)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path, help="Folder with pool_results.csv and lattice_tensors.npz")
    ap.add_argument("--max-scatter", type=int, default=8000, help="Max points for scatter overlays")
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Not a directory: {run_dir}")

    import matplotlib.pyplot as plt

    csv_path = run_dir / "pool_results.csv"
    npz_path = run_dir / "lattice_tensors.npz"
    if not csv_path.is_file():
        raise SystemExit(f"Missing {csv_path}")

    # --- Scalars from CSV (DictReader: meta field contains commas) ---
    E: list[float] = []
    S: list[float] = []
    M: list[float] = []
    ok_list: list[bool] = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            ok_list.append(str(row["ok"]).lower() == "true")
            E.append(float(row["E_eff"]))
            S.append(float(row["sigma_max"]))
            M.append(float(row["strut_mass_metric"]))

    Ea = np.array(E, dtype=np.float64)
    Sa = np.array(S, dtype=np.float64)
    Ma = np.array(M, dtype=np.float64)
    ok = np.array(ok_list, dtype=bool)
    n = len(Ea)
    n_ok = int(ok.sum())

    fig_dir = run_dir / "summary_plots"
    fig_dir.mkdir(exist_ok=True)

    # --- Figure 1: 1D distributions ---
    fig1, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for ax, data, title, color in zip(
        axes,
        [Ea, Sa, Ma],
        [r"$E_{\mathrm{eff}}$", r"$\sigma_{\max}$ (scaled)", "Strut mass metric"],
        ["#2c7fb8", "#e6550d", "#31a354"],
    ):
        ax.hist(data[ok], bins=60, color=color, edgecolor="none", alpha=0.88)
        ax.set_title(title)
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.25)
    fig1.suptitle(f"Scalar distributions (n={n}, ok={n_ok})", fontsize=11, y=1.02)
    fig1.tight_layout()
    fig1.savefig(fig_dir / "01_scalar_histograms.png", dpi=160, bbox_inches="tight")
    plt.close(fig1)

    # --- Figure 2: 2D density ---
    rng = np.random.default_rng(0)
    m = min(int(args.max_scatter), n_ok)
    pick = rng.choice(np.where(ok)[0], size=m, replace=False)
    fig2, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    hb0 = axes[0].hexbin(Ea[pick], Sa[pick], gridsize=55, cmap="viridis", mincnt=1)
    axes[0].set_xlabel(r"$E_{\mathrm{eff}}$")
    axes[0].set_ylabel(r"$\sigma_{\max}$")
    axes[0].set_title("Density: " + r"$E_{\mathrm{eff}}$ vs $\sigma_{\max}$")
    plt.colorbar(hb0, ax=axes[0], label="count")
    hb1 = axes[1].hexbin(Ea[pick], Ma[pick], gridsize=55, cmap="magma", mincnt=1)
    axes[1].set_xlabel(r"$E_{\mathrm{eff}}$")
    axes[1].set_ylabel("Mass metric")
    axes[1].set_title(r"Density: $E_{\mathrm{eff}}$ vs mass")
    plt.colorbar(hb1, ax=axes[1], label="count")
    fig2.suptitle(f"Hexbin (m={m} random ok samples)", fontsize=10, y=1.02)
    fig2.tight_layout()
    fig2.savefig(fig_dir / "02_hexbin_couplings.png", dpi=160, bbox_inches="tight")
    plt.close(fig2)

    # --- Figure 3: correlation / scatter matrix style ---
    Z = np.column_stack([Ea[ok], Sa[ok], Ma[ok]])
    C = np.corrcoef(Z.T)
    fig3, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([r"$E_{\mathrm{eff}}$", r"$\sigma_{\max}$", "Mass"])
    ax.set_yticklabels([r"$E_{\mathrm{eff}}$", r"$\sigma_{\max}$", "Mass"])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", color="white" if abs(C[i, j]) > 0.5 else "black", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, label="Pearson r")
    ax.set_title("Correlation (ok samples only)")
    fig3.tight_layout()
    fig3.savefig(fig_dir / "03_correlation_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(fig3)

    tensor_stats: dict[str, float] = {}
    if npz_path.is_file():
        print("Loading npz (may take a minute, ~1–2 GB RAM)…")
        d = np.load(npz_path)
        X = d["X"]
        assert X.shape[1:] == (4, int(d["h"]), int(d["w"])), X.shape
        _, _, H, W = X.shape
        ch2 = X[:, 2, :, : W - 1]
        ch3 = X[:, 3, : H - 1, :]
        m2 = ch2.mean(axis=(1, 2))
        m3 = ch3.mean(axis=(1, 2))
        s2 = ch2.std(axis=(1, 2))
        s3 = ch3.std(axis=(1, 2))

        fig4, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        axes[0].hist(m2, bins=50, color="#6a51a3", alpha=0.85, label="ch2 (horiz.)")
        axes[0].hist(m3, bins=50, color="#2171b5", alpha=0.55, label="ch3 (vert.)")
        axes[0].set_xlabel("Sample mean thickness factor (active half-edges)")
        axes[0].set_ylabel("count")
        axes[0].legend()
        axes[0].set_title("Mean in-plane thickness (per sample)")
        axes[0].grid(True, alpha=0.25)
        axes[1].hist(s2, bins=50, color="#6a51a3", alpha=0.85, label="ch2 std")
        axes[1].hist(s3, bins=50, color="#2171b5", alpha=0.55, label="ch3 std")
        axes[1].set_xlabel("Within-sample std of thickness")
        axes[1].set_ylabel("count")
        axes[1].legend()
        axes[1].set_title("Spatial variability of thickness")
        axes[1].grid(True, alpha=0.25)
        fig4.suptitle("Channel statistics from X (all samples)", fontsize=10, y=1.02)
        fig4.tight_layout()
        fig4.savefig(fig_dir / "04_thickness_channels.png", dpi=160, bbox_inches="tight")
        plt.close(fig4)

        fig5, ax = plt.subplots(figsize=(5.5, 4.2))
        ax.scatter(m2[:: max(1, len(m2) // 5000)], m3[:: max(1, len(m3) // 5000)], s=3, alpha=0.25, c="teal")
        ax.set_xlabel("Mean ch2 (horizontal)")
        ax.set_ylabel("Mean ch3 (vertical)")
        ax.set_title("Horiz. vs vert. mean thickness (subsampled)")
        ax.grid(True, alpha=0.25)
        fig5.tight_layout()
        fig5.savefig(fig_dir / "05_mean_ch2_vs_ch3.png", dpi=160, bbox_inches="tight")
        plt.close(fig5)

        tensor_stats = {
            "mean_of_sample_mean_ch2": float(np.mean(m2)),
            "mean_of_sample_mean_ch3": float(np.mean(m3)),
            "mean_within_sample_std_ch2": float(np.mean(s2)),
            "mean_within_sample_std_ch3": float(np.mean(s3)),
        }
    else:
        print(f"No {npz_path.name}; skipped tensor channel plots.")

    # --- Text summary ---
    summary_txt = {
        "run_dir": str(run_dir),
        "n_rows_csv": n,
        "n_ok": n_ok,
        "E_eff": {"min": float(Ea[ok].min()), "max": float(Ea[ok].max()), "mean": float(Ea[ok].mean()), "std": float(Ea[ok].std())},
        "sigma_max": {"min": float(Sa[ok].min()), "max": float(Sa[ok].max()), "mean": float(Sa[ok].mean()), "std": float(Sa[ok].std())},
        "strut_mass_metric": {"min": float(Ma[ok].min()), "max": float(Ma[ok].max()), "mean": float(Ma[ok].mean()), "std": float(Ma[ok].std())},
        "tensor_channel_summaries": tensor_stats,
    }
    if (run_dir / "summary.json").is_file():
        summary_txt["run_summary_json"] = json.loads((run_dir / "summary.json").read_text())
    (fig_dir / "numeric_summary.json").write_text(json.dumps(summary_txt, indent=2))

    print(f"Wrote plots to {fig_dir}")
    for p in sorted(fig_dir.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
