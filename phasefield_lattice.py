#!/usr/bin/env python3
"""
Generate beam-lattice connectivity from a smooth phase field between two faces.

Left / right faces match frame_lattice.py:
  x = 0 and x = 6, nodes at y ∈ {0, 1, 2, 3}.

A smooth scalar field c(x, y) ∈ [0, 1] is built from anchored face disks plus
Gaussian “tubes” along random polylines between the faces, then lightly
Gaussian-filtered (phase-field–style diffuse interface). The field is
thresholded; a one-pixel-wide skeleton is extracted and compressed
to straight beam edges between junction / end points. Endpoints near the faces
snap to the fixed boundary node indices.

No mechanics — geometry / graph only. Optional PNG via matplotlib.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter

# --- Same geometry as frame_lattice.py -------------------------------------------------

LEFT_X = 0.0
RIGHT_X = 6.0
FACE_YS = (0.0, 1.0, 2.0, 3.0)
N_FACE = len(FACE_YS)


def boundary_node_coords() -> tuple[np.ndarray, np.ndarray]:
    """Return (left_xy, right_xy), each shape (4, 2)."""
    left = np.column_stack([np.full(N_FACE, LEFT_X, dtype=float), np.array(FACE_YS, dtype=float)])
    right = np.column_stack([np.full(N_FACE, RIGHT_X, dtype=float), np.array(FACE_YS, dtype=float)])
    return left, right


# --- Phase field -----------------------------------------------------------------------


def seed_disks(
    xx: np.ndarray,
    yy: np.ndarray,
    centers: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Union of disks (indicator 1 inside, 0 outside) in physical coordinates."""
    m = np.zeros_like(xx, dtype=float)
    for cx, cy in centers:
        m = np.maximum(m, ((xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2).astype(float))
    return m


def _dist2_point_to_segments(
    xx: np.ndarray, yy: np.ndarray, poly: np.ndarray
) -> np.ndarray:
    """Squared distance from each grid point to a polyline (N, 2)."""
    d2 = np.full(xx.shape, np.inf, dtype=np.float64)
    for k in range(len(poly) - 1):
        ax, ay = float(poly[k, 0]), float(poly[k, 1])
        bx, by = float(poly[k + 1, 0]), float(poly[k + 1, 1])
        vx, vy = bx - ax, by - ay
        den = vx * vx + vy * vy + 1e-18
        wx = xx - ax
        wy = yy - ay
        t = np.clip((wx * vx + wy * vy) / den, 0.0, 1.0)
        qx = ax + t * vx
        qy = ay + t * vy
        dd = (xx - qx) ** 2 + (yy - qy) ** 2
        d2 = np.minimum(d2, dd)
    return d2


def build_phase_field(
    xx: np.ndarray,
    yy: np.ndarray,
    *,
    seed: int,
    n_tubes: int = 18,
    tube_sigma: float = 0.11,
    seed_radius: float = 0.14,
    smooth_sigma: float = 1.15,
) -> np.ndarray:
    """
    Build a **smooth phase-like** scalar field c ∈ [0, 1] without a full PDE solve:

    * disks at the eight face nodes (anchors),
    * max of Gaussian **tubes** along random polylines that connect the left
      face to the right face (with an optional middle bend for non-straight bars),
    * light Gaussian filtering to mimic a diffuse interface (“phase-field” look).

    Pure Allen–Cahn on random IC tends to flatten the bulk to c ≈ ½, which
    thresholds to unusable masks; superposed tubes guarantee **filamentary**
    high-c regions between x = 0 and x = 6.
    """
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    left, right = boundary_node_coords()
    centers = np.vstack([left, right])
    c = seed_disks(xx, yy, centers, seed_radius)

    for _ in range(n_tubes):
        i0 = int(rng.integers(0, N_FACE))
        i1 = int(rng.integers(0, N_FACE))
        p0 = np.array([LEFT_X, float(FACE_YS[i0])])
        p1 = np.array([RIGHT_X, float(FACE_YS[i1])])
        xm = float(rng.uniform(1.8, 4.2))
        ym = float(rng.uniform(-1.0, 4.0))
        pm = np.array([xm, ym])
        if rng.random() < 0.55:
            poly = np.vstack([p0, p1])
        else:
            poly = np.vstack([p0, pm, p1])
        d2 = _dist2_point_to_segments(xx, yy, poly)
        tube = np.exp(-0.5 * d2 / (tube_sigma**2))
        c = np.maximum(c, tube)

    c = gaussian_filter(c, sigma=smooth_sigma, mode="nearest")
    c -= float(c.min())
    cmax = float(c.max())
    if cmax > 0:
        c /= cmax
    np.clip(c, 0.0, 1.0, out=c)
    return c


# --- Skeleton --------------------------------------------------------------------------


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Reduce a boolean foreground mask to a thin medial graph.
    Prefer scikit-image if installed; otherwise distance-transform ridge.
    """
    try:
        from skimage.morphology import skeletonize

        return skeletonize(mask)
    except ImportError:
        return _ridge_skeleton_dt(mask)


def _ridge_skeleton_dt(mask: np.ndarray) -> np.ndarray:
    """Fallback: local maxima of the interior distance transform (scipy only)."""
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    dt = distance_transform_edt(mask)
    mx = maximum_filter(dt, size=(3, 3))
    return ((dt > 0) & (dt >= mx - 1e-9) & mask)


# --- Graph from skeleton ----------------------------------------------------------------


def _neighbors8(iy: int, ix: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            jy, jx = iy + dy, ix + dx
            if 0 <= jy < h and 0 <= jx < w:
                out.append((jy, jx))
    return out


def skeleton_degrees(skel: np.ndarray) -> dict[tuple[int, int], int]:
    h, w = skel.shape
    deg: dict[tuple[int, int], int] = {}
    pts = np.argwhere(skel)
    for iy, ix in pts:
        d = 0
        for jy, jx in _neighbors8(int(iy), int(ix), h, w):
            if skel[jy, jx]:
                d += 1
        deg[(int(iy), int(ix))] = d
    return deg


def compress_skeleton_to_edges(
    skel: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    *,
    snap_tol: float,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Turn skeleton pixels into straight beam edges and global node coordinates.

    Returns:
      xy: (n_nodes, 2) with first 8 rows = fixed left (0–3) then right (4–7).
      edges: list of (i, j) with i < j, using global indices.
    """
    left, right = boundary_node_coords()
    xy_fix = np.vstack([left, right])
    h, w = skel.shape
    deg = skeleton_degrees(skel)
    special = [p for p, d in deg.items() if d != 2]
    special_set = set(special)

    def snap_face(iy: int, ix: int) -> int | None:
        x = float(xx[iy, ix])
        y = float(yy[iy, ix])
        if abs(x - LEFT_X) <= snap_tol:
            j = int(np.argmin(np.abs(np.array(FACE_YS) - y)))
            return j
        if abs(x - RIGHT_X) <= snap_tol:
            j = int(np.argmin(np.abs(np.array(FACE_YS) - y)))
            return 4 + j
        return None

    pid_to_nid: dict[tuple[int, int], int] = {}
    next_id = 8
    for p in special:
        sid = snap_face(*p)
        if sid is not None:
            pid_to_nid[p] = sid
        else:
            pid_to_nid[p] = next_id
            next_id += 1

    n_interior = next_id - 8
    xy = np.zeros((8 + n_interior, 2), dtype=float)
    xy[:8] = xy_fix
    for p, nid in pid_to_nid.items():
        if nid >= 8:
            iy, ix = p
            xy[nid] = (float(xx[iy, ix]), float(yy[iy, ix]))

    def neighbors_on_skel(iy: int, ix: int) -> list[tuple[int, int]]:
        return [(jy, jx) for jy, jx in _neighbors8(iy, ix, h, w) if skel[jy, jx]]

    edges_set: set[tuple[int, int]] = set()
    seen_undirected: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    for s in special:
        for nb in neighbors_on_skel(*s):
            a0, b0 = (s, nb) if s < nb else (nb, s)
            key = (a0, b0)
            if key in seen_undirected:
                continue
            seen_undirected.add(key)

            prev, cur = s, nb
            if cur in special_set and cur != s:
                ia, ib = pid_to_nid[s], pid_to_nid[cur]
                if ia != ib:
                    edges_set.add((min(ia, ib), max(ia, ib)))
                continue

            while True:
                if cur in special_set and cur != s:
                    ia, ib = pid_to_nid[s], pid_to_nid[cur]
                    if ia != ib:
                        edges_set.add((min(ia, ib), max(ia, ib)))
                    break
                nxt_cand = [p for p in neighbors_on_skel(*cur) if p != prev]
                if len(nxt_cand) != 1:
                    break
                prev, cur = cur, nxt_cand[0]
                if cur in special_set and cur != s:
                    ia, ib = pid_to_nid[s], pid_to_nid[cur]
                    if ia != ib:
                        edges_set.add((min(ia, ib), max(ia, ib)))
                    break

    return xy, sorted(edges_set)


@dataclass(frozen=True)
class PhaseLattice:
    """Outputs for inspection / later FEA coupling."""

    xx: np.ndarray
    yy: np.ndarray
    phase: np.ndarray
    mask: np.ndarray
    skeleton: np.ndarray
    xy_nodes: np.ndarray
    edges: list[tuple[int, int]]
    threshold: float


def generate_lattice(
    *,
    seed: int = 0,
    nx: int = 161,
    ny: int = 121,
    y_min: float = -1.5,
    y_max: float = 4.5,
    n_tubes: int = 18,
    tube_sigma: float = 0.11,
    smooth_sigma: float = 1.15,
    threshold: float = 0.35,
    snap_tol: float = 0.35,
) -> PhaseLattice:
    """Build phase-like field, threshold, skeletonize, build graph."""
    x1d = np.linspace(LEFT_X, RIGHT_X, nx, dtype=float)
    y1d = np.linspace(y_min, y_max, ny, dtype=float)
    xx, yy = np.meshgrid(x1d, y1d, indexing="xy")

    c = build_phase_field(
        xx,
        yy,
        seed=seed,
        n_tubes=n_tubes,
        tube_sigma=tube_sigma,
        smooth_sigma=smooth_sigma,
    )
    mask = c >= threshold
    skel = skeletonize_mask(mask)
    xy_nodes, edges = compress_skeleton_to_edges(skel, xx, yy, snap_tol=snap_tol)

    return PhaseLattice(
        xx=xx,
        yy=yy,
        phase=c,
        mask=mask,
        skeleton=skel,
        xy_nodes=xy_nodes,
        edges=edges,
        threshold=threshold,
    )


def try_plot(pl: PhaseLattice, out_path: Path | None) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax0, ax1, ax2, ax3 = axes.ravel()
    im0 = ax0.imshow(pl.phase, origin="lower", extent=[LEFT_X, RIGHT_X, pl.yy.min(), pl.yy.max()], aspect="auto")
    ax0.set_title("Phase field c(x, y)")
    fig.colorbar(im0, ax=ax0, fraction=0.046)
    ax1.imshow(pl.mask.astype(float), origin="lower", cmap="gray_r", extent=[LEFT_X, RIGHT_X, pl.yy.min(), pl.yy.max()], aspect="auto")
    ax1.set_title(f"Mask (c ≥ τ), τ={pl.threshold:.2f}")
    ax2.imshow(pl.skeleton.astype(float), origin="lower", cmap="gray", extent=[LEFT_X, RIGHT_X, pl.yy.min(), pl.yy.max()], aspect="auto")
    ax2.set_title("Skeleton")

    ax3.set_aspect("equal", adjustable="box")
    left, right = boundary_node_coords()
    ax3.plot(left[:, 0], left[:, 1], "ks", ms=8, label="left face")
    ax3.plot(right[:, 0], right[:, 1], "ks", ms=8, label="right face")
    for i, j in pl.edges:
        p0, p1 = pl.xy_nodes[i], pl.xy_nodes[j]
        ax3.plot([p0[0], p1[0]], [p0[1], p1[1]], "b-", lw=1.2, alpha=0.85)
    ax3.scatter(pl.xy_nodes[:, 0], pl.xy_nodes[:, 1], c="C1", s=22, zorder=3)
    ax3.set_xlim(LEFT_X - 0.2, RIGHT_X + 0.2)
    ax3.set_title("Beam graph (straight edges)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    pic_dir = script_dir / "pic"
    p = argparse.ArgumentParser(description="Phase-field lattice generator (geometry only)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nx", type=int, default=161)
    p.add_argument("--ny", type=int, default=121)
    p.add_argument(
        "--tubes",
        type=int,
        default=18,
        help="Number of random Gaussian tubes (polylines) between faces",
    )
    p.add_argument("--tube-sigma", type=float, default=0.11, help="Tube width (physical units)")
    p.add_argument(
        "--smooth-sigma",
        type=float,
        default=1.15,
        help="Gaussian blur σ in grid pixels (scipy.ndimage.gaussian_filter)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Mask where c ≥ τ (after field is normalized to [0, 1])",
    )
    p.add_argument("--snap-tol", type=float, default=0.35)
    p.add_argument(
        "--plot",
        type=str,
        default="",
        help="PNG path; default pic/phasefield_lattice_seed<N>.png",
    )
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    pl = generate_lattice(
        seed=args.seed,
        nx=args.nx,
        ny=args.ny,
        n_tubes=args.tubes,
        tube_sigma=args.tube_sigma,
        smooth_sigma=args.smooth_sigma,
        threshold=args.threshold,
        snap_tol=args.snap_tol,
    )

    n = pl.xy_nodes.shape[0]
    print(f"Nodes: {n}, edges: {len(pl.edges)}")
    print(f"Grid {args.ny}×{args.nx}, phase range [{pl.phase.min():.3f}, {pl.phase.max():.3f}]")

    if not args.no_plot:
        if args.plot:
            outp = Path(args.plot)
            if outp.is_absolute():
                out = outp
            elif outp.parts and outp.parts[0] == "pic":
                out = script_dir / outp
            else:
                out = pic_dir / outp
        else:
            out = pic_dir / f"phasefield_lattice_seed{args.seed}.png"
        try_plot(pl, out)
        print(f"Saved figure to {out}")


if __name__ == "__main__":
    main()
