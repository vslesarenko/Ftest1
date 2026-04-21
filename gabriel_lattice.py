#!/usr/bin/env python3
"""
Gabriel-graph beam lattice (geometry only — no FEA).

Boundary (same as frame_lattice.py):
  Left  x = 0,  y ∈ {0, 1, 2, 3}
  Right x = 6,  y ∈ {0, 1, 2, 3}

Interior: N points with random (x, y) in a strip between the faces.

Connectivity:
  • Start from the **Gabriel graph** on all boundary + interior points: edge (i, j)
    iff the **open** disk with diameter ij contains no other point.
  • **Always** include left/right **vertical face chords** between consecutive y
    levels (structural faces), in case the Gabriel test omits some.
  • If the graph is disconnected, **repair** by repeatedly adding the shortest
    available edge between two components until connected.

Ill-posed inverse designs can be ranked later (e.g. minimum peak stress) among
samples that share similar effective stiffness — this file only generates
candidates.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LEFT_X = 0.0
RIGHT_X = 6.0
FACE_YS = (0.0, 1.0, 2.0, 3.0)
N_FACE = len(FACE_YS)

LEFT_IDS = list(range(N_FACE))
RIGHT_IDS = list(range(N_FACE, 2 * N_FACE))


def face_coords() -> tuple[np.ndarray, np.ndarray]:
    left = np.column_stack([np.full(N_FACE, LEFT_X), np.array(FACE_YS, dtype=float)])
    right = np.column_stack([np.full(N_FACE, RIGHT_X), np.array(FACE_YS, dtype=float)])
    return left, right


def _dedupe_edges(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    s = {tuple(sorted((a, b))) for a, b in pairs if a != b}
    return sorted(s)


def _bbox_scale(xy: np.ndarray) -> float:
    span = float(np.ptp(xy, axis=0).max())
    return max(span, 1e-9)


def gabriel_edges(xy: np.ndarray, tol_scale: float = 1e-10) -> list[tuple[int, int]]:
    """Undirected Gabriel graph edges for 2D points xy, shape (n, 2)."""
    n = xy.shape[0]
    tol = tol_scale * _bbox_scale(xy)
    edges: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            pi, pj = xy[i], xy[j]
            d = float(np.linalg.norm(pi - pj))
            if d < tol:
                continue
            c = 0.5 * (pi + pj)
            r = 0.5 * d
            ok = True
            for k in range(n):
                if k == i or k == j:
                    continue
                if float(np.linalg.norm(xy[k] - c)) < r - tol:
                    ok = False
                    break
            if ok:
                edges.append((i, j))
    return edges


def face_vertical_edges() -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for base in (0, N_FACE):
        for j in range(N_FACE - 1):
            out.append((base + j, base + j + 1))
    return out


def _adjacency(n_nodes: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(n_nodes)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    return adj


def _components(n_nodes: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    adj = _adjacency(n_nodes, edges)
    seen = [False] * n_nodes
    comps: list[list[int]] = []
    for s in range(n_nodes):
        if seen[s]:
            continue
        stack = [s]
        seen[s] = True
        cur: list[int] = []
        while stack:
            u = stack.pop()
            cur.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(cur)
    return comps


def connect_components(
    xy: np.ndarray,
    edges: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], int]:
    """
    Add minimum-length edges between distinct connected components until one component.
    Returns (merged_edges, n_bridges_added).
    """
    n = xy.shape[0]
    e = list(edges)
    bridges = 0
    while True:
        comps = _components(n, e)
        if len(comps) <= 1:
            break
        best: tuple[float, int, int] | None = None
        for ci in range(len(comps)):
            for cj in range(ci + 1, len(comps)):
                for u in comps[ci]:
                    for v in comps[cj]:
                        d = float(np.sum((xy[u] - xy[v]) ** 2))
                        if best is None or d < best[0]:
                            best = (d, u, v)
        if best is None:
            break
        _, u, v = best
        e.append((min(u, v), max(u, v)))
        bridges += 1
    return _dedupe_edges(e), bridges


def build_gabriel_lattice(
    *,
    seed: int,
    n_interior: int,
    y_min: float = -1.0,
    y_max: float = 4.0,
    x_margin: float = 0.15,
) -> "GabrielLattice":
    """
    n_interior random nodes in [x_margin, 6-x_margin] × [y_min, y_max].
    """
    if n_interior < 0:
        raise ValueError("n_interior must be >= 0")
    if y_max <= y_min:
        raise ValueError("y_max must exceed y_min")
    if x_margin <= 0 or x_margin >= RIGHT_X / 2:
        raise ValueError("x_margin must be in (0, 3) for this strip")

    rng = np.random.default_rng(seed)
    left, right = face_coords()
    n_nodes = 2 * N_FACE + n_interior
    xy = np.zeros((n_nodes, 2), dtype=float)
    xy[LEFT_IDS] = left
    xy[RIGHT_IDS] = right

    for k in range(n_interior):
        nid = 2 * N_FACE + k
        x = float(rng.uniform(LEFT_X + x_margin, RIGHT_X - x_margin))
        y = float(rng.uniform(y_min, y_max))
        xy[nid] = (x, y)

    g_edges = gabriel_edges(xy)
    face_e = face_vertical_edges()
    edges0 = _dedupe_edges(g_edges + face_e)
    edges1, n_bridge = connect_components(xy, edges0)

    return GabrielLattice(
        xy=xy,
        edges=edges1,
        n_interior=n_interior,
        n_gabriel=len(g_edges),
        n_face_edges=len(face_e),
        n_bridge_edges=n_bridge,
    )


@dataclass(frozen=True)
class GabrielLattice:
    xy: np.ndarray
    edges: list[tuple[int, int]]
    n_interior: int
    n_gabriel: int
    n_face_edges: int
    n_bridge_edges: int


def try_plot(lat: GabrielLattice, out_path: Path | None) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    left, right = face_coords()
    ax.plot(left[:, 0], left[:, 1], "ks", ms=9, label="Left face", zorder=4)
    ax.plot(right[:, 0], right[:, 1], "ks", ms=9, label="Right face", zorder=4)
    for i, j in lat.edges:
        p0, p1 = lat.xy[i], lat.xy[j]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "b-", lw=1.15, alpha=0.85, zorder=2)
    ax.scatter(lat.xy[:, 0], lat.xy[:, 1], c="C1", s=26, zorder=3, label="Nodes")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(LEFT_X - 0.35, RIGHT_X + 0.35)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(
        f"Gabriel (+ face chords) + bridges | "
        f"|E|={len(lat.edges)}, interior={lat.n_interior}, bridges_added={lat.n_bridge_edges}"
    )
    fig.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    pic_dir = script_dir / "pic"
    p = argparse.ArgumentParser(description="Gabriel graph lattice generator")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-interior", type=int, default=12)
    p.add_argument("--y-min", type=float, default=-1.0)
    p.add_argument("--y-max", type=float, default=4.0)
    p.add_argument(
        "--x-margin",
        type=float,
        default=0.15,
        help="Keep interior x away from faces: (0+x_margin, 6-x_margin)",
    )
    p.add_argument(
        "--plot",
        type=str,
        default="",
        help="PNG path; default pic/gabriel_lattice_seed<N>.png",
    )
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    lat = build_gabriel_lattice(
        seed=args.seed,
        n_interior=args.n_interior,
        y_min=args.y_min,
        y_max=args.y_max,
        x_margin=args.x_margin,
    )

    print(
        f"Nodes: {lat.xy.shape[0]}, |E|={len(lat.edges)}, "
        f"Gabriel edges={lat.n_gabriel}, bridge_edges_added={lat.n_bridge_edges}"
    )

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
            out = pic_dir / f"gabriel_lattice_seed{args.seed}.png"
        try_plot(lat, out)
        print(f"Saved figure to {out}")


if __name__ == "__main__":
    main()
