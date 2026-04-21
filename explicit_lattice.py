#!/usr/bin/env python3
"""
Explicit parametric beam lattice (geometry only — no FEA).

Faces match frame_lattice.py:
  Left  x = 0,  y ∈ {0, 1, 2, 3}
  Right x = 6,  y ∈ {0, 1, 2, 3}

Internal structure:
  • Fixed count of **columns** between the faces (``n_layers`` vertical planes).
  • On each plane, a fixed number of **nodes** (``nodes_per_layer``) with
    **continuous** (x, y) — x is fixed per column, y is sampled in [y_min, y_max].

Connectivity (no phase fields, no skeletons):
  • Vertical chords on the left and right faces only.
  • **No** direct edges between the two faces.
  • Each internal node is linked to the **nearest** node on the **previous**
    column (toward left) and the **nearest** on the **next** column (toward right),
    plus **k_inter** nearest neighbors on each adjacent column for redundancy.
  • Optional **within-layer** chain: nodes in a column sorted by y are connected
    along that column (like a chord).

This is intended as a clean design space for ML / inverse design: the degrees of
freedom are the sampled y’s (and optionally column count / density), while the
graph topology rule stays deterministic given those positions.
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


def _nearest_indices(
    xy: np.ndarray,
    query_idx: int,
    candidate_indices: list[int],
    k: int,
) -> list[int]:
    if k <= 0 or not candidate_indices:
        return []
    q = xy[query_idx]
    d = [(float(np.sum((xy[j] - q) ** 2)), j) for j in candidate_indices]
    d.sort(key=lambda t: t[0])
    out: list[int] = []
    for _, j in d:
        if j not in out:
            out.append(j)
        if len(out) >= k:
            break
    return out


def _edges_between_columns(
    xy: np.ndarray,
    col_a: list[int],
    col_b: list[int],
    k: int,
) -> list[tuple[int, int]]:
    """Undirected edges between two disjoint node index sets (adjacent columns)."""
    if not col_a or not col_b or k <= 0:
        return []
    edges: list[tuple[int, int]] = []
    for ia in col_a:
        for jb in _nearest_indices(xy, ia, col_b, k):
            edges.append((ia, jb))
    for ib in col_b:
        for ja in _nearest_indices(xy, ib, col_a, k):
            edges.append((ib, ja))
    return _dedupe_edges(edges)


@dataclass(frozen=True)
class ExplicitLattice:
    xy: np.ndarray
    edges: list[tuple[int, int]]
    n_layers: int
    nodes_per_layer: int
    column_x: np.ndarray  # length n_layers


def build_explicit_lattice(
    *,
    seed: int,
    n_layers: int = 3,
    nodes_per_layer: int = 3,
    y_min: float = -1.0,
    y_max: float = 4.0,
    k_inter: int = 2,
    chord_within_layer: bool = True,
) -> ExplicitLattice:
    """
    Build node coordinates and edges.

    Node indices:
      0..N_FACE-1          left face
      N_FACE..2*N_FACE-1   right face
      ...                  internal, column-major: layer 0, layer 1, ...
    """
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")
    if nodes_per_layer < 1:
        raise ValueError("nodes_per_layer must be >= 1")
    if y_max <= y_min:
        raise ValueError("y_max must exceed y_min")

    rng = np.random.default_rng(seed)
    left, right = face_coords()

    # Column x positions strictly inside (0, 6), excluding direct L–R spans
    xs = np.array([(i + 1.0) / (n_layers + 1.0) * (RIGHT_X - LEFT_X) for i in range(n_layers)], dtype=float)

    n_int = n_layers * nodes_per_layer
    n_nodes = 2 * N_FACE + n_int
    xy = np.zeros((n_nodes, 2), dtype=float)
    xy[LEFT_IDS] = left
    xy[RIGHT_IDS] = right

    base = 2 * N_FACE
    internal_cols: list[list[int]] = []
    for ell in range(n_layers):
        col_ids = []
        for k in range(nodes_per_layer):
            nid = base + ell * nodes_per_layer + k
            yy = float(rng.uniform(y_min, y_max))
            xy[nid] = (float(xs[ell]), yy)
            col_ids.append(nid)
        internal_cols.append(col_ids)

    edges: list[tuple[int, int]] = []

    # Face vertical chords
    for base_id in (0, N_FACE):
        for j in range(N_FACE - 1):
            a, b = base_id + j, base_id + j + 1
            edges.append((a, b))

    # Within-layer chords (sorted by y)
    if chord_within_layer:
        for col in internal_cols:
            order = sorted(col, key=lambda i: float(xy[i, 1]))
            for a, b in zip(order[:-1], order[1:]):
                edges.append((a, b))

    # Columns: left face -> first internal -> ... -> last internal -> right face
    columns: list[list[int]] = [LEFT_IDS, *internal_cols, RIGHT_IDS]

    for i in range(len(columns) - 1):
        ea = _edges_between_columns(xy, columns[i], columns[i + 1], k_inter)
        edges.extend(ea)

    edges = _dedupe_edges(edges)
    return ExplicitLattice(
        xy=xy,
        edges=edges,
        n_layers=n_layers,
        nodes_per_layer=nodes_per_layer,
        column_x=xs,
    )


def try_plot(lat: ExplicitLattice, out_path: Path | None, title: str | None = None) -> None:
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
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "b-", lw=1.25, alpha=0.85, zorder=2)
    ax.scatter(lat.xy[:, 0], lat.xy[:, 1], c="C1", s=28, zorder=3, label="Nodes")
    # Column guides
    for xc in lat.column_x:
        ax.axvline(xc, color="0.85", ls="--", lw=0.8, zorder=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(LEFT_X - 0.35, RIGHT_X + 0.35)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right", fontsize=8)
    ttl = title or (
        f"Explicit lattice: {lat.n_layers} columns × {lat.nodes_per_layer} nodes/column, "
        f"|E|={len(lat.edges)}"
    )
    ax.set_title(ttl)
    fig.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    pic_dir = script_dir / "pic"
    p = argparse.ArgumentParser(description="Explicit parametric beam lattice (geometry)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-layers", type=int, default=3, help="Number of internal vertical planes")
    p.add_argument("--nodes-per-layer", type=int, default=3, help="Nodes on each plane")
    p.add_argument("--y-min", type=float, default=-1.0)
    p.add_argument("--y-max", type=float, default=4.0)
    p.add_argument(
        "--k-inter",
        type=int,
        default=2,
        help="Max neighbors per side between two adjacent columns",
    )
    p.add_argument(
        "--no-chord",
        action="store_true",
        help="Disable within-column chain (sorted by y)",
    )
    p.add_argument(
        "--plot",
        type=str,
        default="",
        help="PNG filename; default pic/explicit_lattice_seed<N>.png",
    )
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    lat = build_explicit_lattice(
        seed=args.seed,
        n_layers=args.n_layers,
        nodes_per_layer=args.nodes_per_layer,
        y_min=args.y_min,
        y_max=args.y_max,
        k_inter=args.k_inter,
        chord_within_layer=not args.no_chord,
    )

    print(
        f"Nodes: {lat.xy.shape[0]}, edges: {len(lat.edges)}, "
        f"columns: {lat.n_layers} × {lat.nodes_per_layer}"
    )
    print("Column x:", np.array2string(lat.column_x, precision=3))

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
            out = pic_dir / f"explicit_lattice_seed{args.seed}.png"
        try_plot(lat, out)
        print(f"Saved figure to {out}")


if __name__ == "__main__":
    main()
