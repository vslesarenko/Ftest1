#!/usr/bin/env python3
"""
2D plane frame (Euler–Bernoulli beams) on a line-element graph.

Geometry: left nodes x=0 at y=0,1,2,3; right nodes x=6 at same y;
three interior nodes at x=3 with configurable y in [-1, 4].

Connectivity: vertical chords on each side face only (no direct left–right
span) plus random beam connections from each middle node to nodes on the left
and right.

Loading / BC: left face clamped (u_x=u_y=theta=0); right face prescribed
horizontal extension u_x = delta (uniform stretch), u_y=theta=0.

This is the usual “beam lattice” reduction: each member is a 1D beam with
EA (axial) and EI (bending), not a 2D solid.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def euler_bernoulli_frame_local_k(
    ea: float, ei: float, length: float
) -> np.ndarray:
    """6x6 local stiffness for one 2D EB beam; dof order (u1,v1,t1,u2,v2,t2)."""
    if length <= 0:
        raise ValueError("Non-positive beam length")
    l = length
    l2 = l * l
    l3 = l2 * l
    ea_l = ea / l
    k = np.zeros((6, 6), dtype=float)
    # Axial
    k[0, 0] = ea_l
    k[0, 3] = -ea_l
    k[3, 0] = -ea_l
    k[3, 3] = ea_l
    # Bending (v, theta)
    b = np.array(
        [
            [12.0 / l3, 6.0 / l2, -12.0 / l3, 6.0 / l2],
            [6.0 / l2, 4.0 / l, -6.0 / l2, 2.0 / l],
            [-12.0 / l3, -6.0 / l2, 12.0 / l3, -6.0 / l2],
            [6.0 / l2, 2.0 / l, -6.0 / l2, 4.0 / l],
        ]
    )
    k[np.ix_([1, 2, 4, 5], [1, 2, 4, 5])] += ei * b
    # Symmetrize against FP noise
    k = (k + k.T) * 0.5
    return k


def rotation_matrix_2d_frame(c: float, s: float) -> np.ndarray:
    """3x3 R maps global (Ux,Uy,theta) -> local (axial, transverse, theta)."""
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def beam_global_k(
    xy_i: np.ndarray,
    xy_j: np.ndarray,
    ea: float,
    ei: float,
) -> np.ndarray:
    d = xy_j - xy_i
    l = float(np.linalg.norm(d))
    c, s = d[0] / l, d[1] / l
    kl = euler_bernoulli_frame_local_k(ea, ei, l)
    r = rotation_matrix_2d_frame(c, s)
    z = np.zeros((3, 3))
    t = np.block([[r, z], [z, r]])
    return t.T @ kl @ t


@dataclass(frozen=True)
class FrameResult:
    xy: np.ndarray  # (n,2) reference
    u: np.ndarray  # (n,3) ux, uy, theta_z per node
    edges: list[tuple[int, int]]
    reactions: np.ndarray  # (n,3) nodal reaction forces/moment (on constrained)


def element_local_displacement(
    u: np.ndarray, i: int, j: int, xy: np.ndarray
) -> tuple[np.ndarray, float, float, float]:
    """Return local 6-DOF displacement, length, cos, sin (beam i -> j)."""
    d = xy[j] - xy[i]
    l = float(np.linalg.norm(d))
    c, s = d[0] / l, d[1] / l
    r = rotation_matrix_2d_frame(c, s)
    ug1 = np.array([u[i, 0], u[i, 1], u[i, 2]])
    ug2 = np.array([u[j, 0], u[j, 1], u[j, 2]])
    d1 = r @ ug1
    d2 = r @ ug2
    return np.concatenate([d1, d2]), l, c, s


def beam_fiber_stress_max(
    u: np.ndarray,
    xy: np.ndarray,
    edges: list[tuple[int, int]],
    ea: float,
    ei: float,
    a: float,
    i_geom: float,
    y_extreme: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per beam: conservative estimate |σ_axial| + |σ_bend| at the extreme fiber.

    σ_ax = |N|/A with N = (EA/L)(u2_ax − u1_ax). σ_b = |M| y_max / I with M from
    k_loc @ u_loc at the nodes; |M| ≈ max(|M1|, |M2|) (M is linear between nodes).
    """
    sig = np.zeros(len(edges), dtype=float)
    n_ax = np.zeros(len(edges), dtype=float)
    m_max = np.zeros(len(edges), dtype=float)
    i_geom = max(float(i_geom), 1e-30)
    y_extreme = max(float(y_extreme), 1e-30)
    for k, (i, j) in enumerate(edges):
        d_loc, l, _, _ = element_local_displacement(u, i, j, xy)
        kl = euler_bernoulli_frame_local_k(ea, ei, l)
        fl = kl @ d_loc
        n_force = ea / l * (d_loc[3] - d_loc[0])
        m1 = float(fl[2])
        m2 = float(fl[5])
        m_w = max(abs(m1), abs(m2))
        sig_a = abs(n_force) / max(a, 1e-30)
        sig_b = m_w * y_extreme / i_geom
        sig[k] = sig_a + sig_b
        n_ax[k] = n_force
        m_max[k] = m_w
    return sig, n_ax, m_max


def assemble_and_solve(
    xy: np.ndarray,
    edges: list[tuple[int, int]],
    ea: float,
    ei: float,
    *,
    left_nodes: list[int],
    right_nodes: list[int],
    delta_x: float,
) -> FrameResult:
    n = xy.shape[0]
    ndof = 3 * n
    k = np.zeros((ndof, ndof), dtype=float)
    for i, j in edges:
        ke = beam_global_k(xy[i], xy[j], ea, ei)
        idx = np.array([3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 * j + 1, 3 * j + 2])
        k[np.ix_(idx, idx)] += ke

    fixed_mask = np.zeros(ndof, dtype=bool)
    prescribed = np.zeros(ndof, dtype=float)

    for nid in left_nodes:
        fixed_mask[3 * nid : 3 * nid + 3] = True
        prescribed[3 * nid : 3 * nid + 3] = 0.0

    for nid in right_nodes:
        fixed_mask[3 * nid] = True
        fixed_mask[3 * nid + 1] = True
        fixed_mask[3 * nid + 2] = True
        prescribed[3 * nid] = delta_x
        prescribed[3 * nid + 1] = 0.0
        prescribed[3 * nid + 2] = 0.0

    free = ~fixed_mask
    k_ff = k[np.ix_(free, free)]
    k_fp = k[np.ix_(free, fixed_mask)]
    u = np.zeros(ndof, dtype=float)
    u[fixed_mask] = prescribed[fixed_mask]
    rhs = -k_fp @ u[fixed_mask]
    u[free] = np.linalg.solve(k_ff, rhs)

    f_int = k @ u
    reactions = np.zeros((n, 3))
    for nid in left_nodes:
        reactions[nid] = f_int[3 * nid : 3 * nid + 3]
    return FrameResult(xy=xy, u=u.reshape(n, 3), edges=edges, reactions=reactions)


def build_default_geometry(
    rng: np.random.Generator,
    *,
    mid_y: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """11 nodes: 4 left, 4 right, 3 middle at x=3."""
    left = np.array([[0.0, y] for y in [0.0, 1.0, 2.0, 3.0]])
    right = np.array([[6.0, y] for y in [0.0, 1.0, 2.0, 3.0]])
    if mid_y is None:
        mid = np.column_stack(
            [np.full(3, 3.0), rng.uniform(-1.0, 4.0, size=3)]
        )
    else:
        mid = np.array([[3.0, mid_y[0]], [3.0, mid_y[1]], [3.0, mid_y[2]]])
    xy = np.vstack([left, right, mid])

    edges: set[tuple[int, int]] = set()
    # Vertical chords on left (0..3) and right (4..7)
    for base in (0, 4):
        for k in range(3):
            a, b = base + k, base + k + 1
            edges.add((a, b) if a < b else (b, a))

    left_ids = list(range(4))
    right_ids = list(range(4, 8))
    mid_ids = [8, 9, 10]
    for m in mid_ids:
        n_left = int(rng.integers(1, 4))  # 1–3 connections to left
        n_right = int(rng.integers(1, 4))
        for j in rng.choice(left_ids, size=n_left, replace=False):
            a, b = (m, j) if m < j else (j, m)
            edges.add((a, b))
        for j in rng.choice(right_ids, size=n_right, replace=False):
            a, b = (m, j) if m < j else (j, m)
            edges.add((a, b))

    # Light coupling between adjacent middle nodes (reduces mechanisms)
    edges.add((8, 9) if 8 < 9 else (9, 8))
    edges.add((9, 10) if 9 < 10 else (10, 9))

    return xy, sorted(edges)


def try_plot(
    result: FrameResult,
    scale: float,
    out_path: Path | str,
    sigma_edge: np.ndarray,
) -> None:
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
    except ImportError:
        return
    xy = result.xy
    disp = result.u[:, :2]
    xd = xy + scale * disp
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, j in result.edges:
        ax0.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], "k-", lw=0.8, alpha=0.4)
        ax0.plot([xd[i, 0], xd[j, 0]], [xd[i, 1], xd[j, 1]], "b-", lw=1.2)
    ax0.scatter(xy[:, 0], xy[:, 1], c="gray", s=25, zorder=3)
    ax0.scatter(xd[:, 0], xd[:, 1], c="C0", s=18, zorder=4)
    ax0.set_aspect("equal", adjustable="box")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.set_title(f"Undeformed (gray) vs deformed (blue), scale={scale:g}×")
    ax0.grid(True, alpha=0.3)

    segs = []
    for (i, j) in result.edges:
        segs.append([[xd[i, 0], xd[i, 1]], [xd[j, 0], xd[j, 1]]])
    segs = np.asarray(segs, dtype=float)
    smin, smax = float(np.min(sigma_edge)), float(np.max(sigma_edge))
    if smax <= smin:
        smax = smin + 1.0
    norm = mpl.colors.Normalize(vmin=smin, vmax=smax)
    cmap = mpl.colormaps["turbo"]
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=2.0, array=sigma_edge)
    ax1.add_collection(lc)
    ax1.scatter(xd[:, 0], xd[:, 1], c="k", s=12, zorder=3)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Estimated |σ_ax| + |σ_bend| on deformed shape (color)")
    ax1.autoscale()
    ax1.grid(True, alpha=0.3)
    cbar = fig.colorbar(lc, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Stress (model units)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    pic_dir = script_dir / "pic"
    p = argparse.ArgumentParser(description="2D beam lattice plane frame")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--delta", type=float, default=0.01, help="Prescribed u_x on right")
    p.add_argument("--E", type=float, default=1.0, help="Young's modulus (relative)")
    p.add_argument("--A", type=float, default=1.0, help="Cross-sectional area")
    p.add_argument("--I", type=float, default=1e-2, help="Second moment of area (geometry)")
    p.add_argument(
        "--y-ext",
        type=float,
        default=None,
        help="Distance from neutral axis to outer fiber (for bending stress); "
        "default sqrt(I/A)",
    )
    p.add_argument(
        "--plot-scale",
        type=float,
        default=50.0,
        help="Displacement amplification for visualization",
    )
    p.add_argument(
        "--plot",
        type=str,
        default="",
        help="PNG filename or path; default pic/frame_seed<N>.png next to this script",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip matplotlib output",
    )
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    ea = args.E * args.A
    ei = args.E * args.I
    y_ext = float(np.sqrt(args.I / max(args.A, 1e-30))) if args.y_ext is None else float(args.y_ext)

    xy, edges = build_default_geometry(rng)
    left_nodes = [0, 1, 2, 3]
    right_nodes = [4, 5, 6, 7]

    res = assemble_and_solve(
        xy,
        edges,
        ea,
        ei,
        left_nodes=left_nodes,
        right_nodes=right_nodes,
        delta_x=args.delta,
    )

    sigma, _, _ = beam_fiber_stress_max(
        res.u, xy, edges, ea, ei, args.A, args.I, y_ext
    )

    umax = float(np.max(np.linalg.norm(res.u[:, :2], axis=1)))
    print(f"Nodes: {xy.shape[0]}, beams: {len(edges)}")
    print(f"Max |planar disp| = {umax:.6g} (delta={args.delta})")
    print(f"Max estimated beam stress (|σ_ax|+|σ_bend|): {float(np.max(sigma)):.6g}")
    r_left = res.reactions[left_nodes].copy()
    rx = float(np.sum(r_left[:, 0]))
    ry = float(np.sum(r_left[:, 1]))
    mz = float(np.sum(r_left[:, 2]))
    print(f"Sum of left horizontal reactions (equilibrium with prescribed right shift): {rx:.6g}")
    print(f"Sum of left vertical reactions: {ry:.6g}")
    print(f"Sum of left nodal moments: {mz:.6g}")

    if not args.no_plot:
        pic_dir.mkdir(parents=True, exist_ok=True)
        if args.plot:
            p = Path(args.plot)
            out = p if p.is_absolute() else pic_dir / p
        else:
            out = pic_dir / f"frame_seed{args.seed}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        try_plot(res, args.plot_scale, out, sigma)
        print(f"Saved figure to {out}")


if __name__ == "__main__":
    main()
