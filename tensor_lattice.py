#!/usr/bin/env python3
"""
4-channel grid tensor → beam lattice geometry + optional 2D frame solve.

Grid layout (default 40×20, step 1 in x and y):
  • Node at logical (i, j) with i = 0..W-1 (left→right), j = 0..H-1 (bottom→top).
  • Ideal position: x0 = i * STEP, y0 = j * STEP.

Channels (shape (4, H, W), index order [channel, row j, col i]):
  ch0  dx  — x shift from ideal, clipped to [-DX_MAX, DX_MAX] **except on the left
       and right face columns (i=0 and i=W−1), where dx=0** so borders stay at
       x = 0 and x = (W−1)·STEP.
  ch1  dy  — y shift from ideal, clipped to [-DY_MAX, DY_MAX] **except on those
       same face columns, where dy=0** so face nodes sit at y = j·STEP.

  ch2  (horizontal half-edge) — **relative thickness / stiffness** factor for the
       **beam from node (i, j) to node (i+1, j)**. Often clipped to [0, 1] by
       ``clip_tensor_inplace``; thickness sweeps may keep factors in e.g. [0.7, 1.3].
       Stored at grid cell (j, i) for i = 0..W-2; column i = W-1 has no “right”
       partner (zeroed).

  ch3  (vertical half-edge) — same for **beam from node (i, j) to node (i, j+1)**
       (one step in **+j**, i.e. increasing y / “up” on the plot with origin=lower).
       Stored at (j, i) for j = 0..H-2; row j = H-1 is zeroed.

  Each undirected grid edge appears **once** (east or north from the lower-left
  endpoint), which avoids double-counting in CNN channels.

Global scalars (for logging / conditioning, not extra CNN channels):
  • bond_fill: average active bond weight (horizontal + vertical valid cells)
  • rms_disp: RMS of (dx, dy) over nodes
  • horizontal_span: (W-1)*STEP — used for normalized stretch metrics

Mechanics: Euler–Bernoulli frame (same as frame_lattice.py), left column clamped,
right column prescribed u_x = delta. Edge stiffness scales with bond weight.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from frame_lattice import (
    FrameResult,
    beam_global_k,
    euler_bernoulli_frame_local_k,
    element_local_displacement,
)

# --- Grid ---------------------------------------------------------------------------

W_DEFAULT = 40
H_DEFAULT = 20
STEP = 1.0
DX_MAX = 0.5
DY_MAX = 0.5

CH_DX = 0
CH_DY = 1
CH_RIGHT = 2
CH_TOP = 3
N_CH = 4


@dataclass
class SolveConfig:
    """Mechanical solve parameters (defaults match `solve` CLI)."""

    bond_threshold: float = 0.5
    connect_all: bool = False
    bond_floor: float = 1.0
    uniform_bonds: bool = False
    uniform_bond_value: float = 1.0
    delta: float = 0.01
    E: float = 1.0
    A: float = 1.0
    I: float = 1e-2
    y_ext: float | None = None
    e_scale: float = 10000.0


@dataclass
class MechanicsResult:
    """Scalar outputs from one mechanics solve (serializable for CSV / analysis)."""

    ok: bool
    error: str = ""
    E_eff: float = float("nan")
    sigma_max: float = float("nan")
    sigma_min: float = float("nan")
    sigma_macro_end: float = float("nan")
    eps_macro: float = float("nan")
    K_sec: float = float("nan")
    sum_rx_left: float = float("nan")
    mean_ux_left: float = float("nan")
    mean_ux_right: float = float("nan")
    strain_like_mean: float = float("nan")
    rms_disp: float = float("nan")
    bond_fill: float = float("nan")
    n_edges_full: int = 0
    n_nodes_reduced: int = 0
    n_edges_reduced: int = 0
    min_node_distance: float = float("nan")
    strut_mass_metric: float = float("nan")
    strut_length_sum: float = float("nan")


@dataclass
class MechanicsSolve:
    """Full solve bundle: metrics plus optional arrays for plotting."""

    result: MechanicsResult
    frame: FrameResult | None = None
    sigma: np.ndarray | None = None
    xy: np.ndarray | None = None
    edges: list[tuple[int, int]] | None = None
    edge_weights: list[float] | None = None
    connect_bridge: tuple[int, int] | None = None


def node_id(i: int, j: int, w: int) -> int:
    return j * w + i


def ij_from_id(n: int, w: int) -> tuple[int, int]:
    return n % w, n // w


def empty_tensor(w: int = W_DEFAULT, h: int = H_DEFAULT) -> np.ndarray:
    return np.zeros((N_CH, h, w), dtype=np.float64)


def fully_connected_perturbed_tensor(
    w: int = W_DEFAULT,
    h: int = H_DEFAULT,
    *,
    perturb: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """
    Full rectangular grid: every interior half-edge weight 1 (all bonds on).
    Node offsets (dx, dy) are perturb * U(-DX_MAX, DX_MAX) etc. with fixed RNG field;
    perturb in [0, 1] scales from ideal square (0) to maximum clipped offsets (1).
    """
    rng = np.random.default_rng(seed)
    t = empty_tensor(w, h)
    p = float(np.clip(perturb, 0.0, 1.0))
    t[CH_DX] = p * rng.uniform(-DX_MAX, DX_MAX, size=(h, w))
    t[CH_DY] = p * rng.uniform(-DY_MAX, DY_MAX, size=(h, w))
    t[CH_RIGHT] = 1.0
    t[CH_TOP] = 1.0
    t[CH_RIGHT][:, -1] = 0.0
    t[CH_TOP][-1, :] = 0.0
    return t


def fully_connected_gaussian_tensor(
    w: int = W_DEFAULT,
    h: int = H_DEFAULT,
    *,
    perturb: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """
    Full bond grid; dx, dy are i.i.d. Gaussian with std proportional to ``perturb``.
    Uses σ_x = perturb * (DX_MAX/3), σ_y = perturb * (DY_MAX/3) before clipping so
    typical magnitudes are comparable to the uniform ``fully_connected_perturbed_tensor``.
    ``perturb=0`` yields zero offsets (deterministic ideal lattice).
    """
    rng = np.random.default_rng(seed)
    p = max(0.0, float(perturb))
    sx = p * (DX_MAX / 3.0)
    sy = p * (DY_MAX / 3.0)
    t = empty_tensor(w, h)
    t[CH_DX] = rng.normal(0.0, sx, size=(h, w))
    t[CH_DY] = rng.normal(0.0, sy, size=(h, w))
    t[CH_RIGHT] = 1.0
    t[CH_TOP] = 1.0
    t[CH_RIGHT][:, -1] = 0.0
    t[CH_TOP][-1, :] = 0.0
    clip_tensor_inplace(t)
    return t


def random_tensor(
    w: int = W_DEFAULT,
    h: int = H_DEFAULT,
    *,
    seed: int = 0,
    bond_binary: bool = True,
) -> np.ndarray:
    """Random dx,dy and bonds; values clipped to valid ranges."""
    rng = np.random.default_rng(seed)
    t = empty_tensor(w, h)
    t[CH_DX] = rng.uniform(-DX_MAX, DX_MAX, size=(h, w))
    t[CH_DY] = rng.uniform(-DY_MAX, DY_MAX, size=(h, w))
    if bond_binary:
        # ~65% bonds on — keeps the grid largely connected left↔right (percolation)
        t[CH_RIGHT] = (rng.random(size=(h, w)) < 0.65).astype(np.float64)
        t[CH_TOP] = (rng.random(size=(h, w)) < 0.65).astype(np.float64)
    else:
        t[CH_RIGHT] = rng.uniform(0.15, 1.0, size=(h, w))
        t[CH_TOP] = rng.uniform(0.15, 1.0, size=(h, w))
    t[CH_RIGHT][:, -1] = 0.0
    t[CH_TOP][-1, :] = 0.0
    return t


def clip_tensor_inplace(t: np.ndarray) -> None:
    _, h, w = t.shape
    t[CH_DX] = np.clip(t[CH_DX], -DX_MAX, DX_MAX)
    t[CH_DY] = np.clip(t[CH_DY], -DY_MAX, DY_MAX)
    t[CH_RIGHT] = np.clip(t[CH_RIGHT], 0.0, 1.0)
    t[CH_TOP] = np.clip(t[CH_TOP], 0.0, 1.0)
    # Left / right platens at ideal x and y (no in-plane perturbation on faces)
    t[CH_DX, :, 0] = 0.0
    t[CH_DX, :, w - 1] = 0.0
    t[CH_DY, :, 0] = 0.0
    t[CH_DY, :, w - 1] = 0.0


def clip_displacements_only_inplace(t: np.ndarray) -> None:
    """Clip only dx, dy (faces fixed); leave bond channels unchanged."""
    _, h, w = t.shape
    t[CH_DX] = np.clip(t[CH_DX], -DX_MAX, DX_MAX)
    t[CH_DY] = np.clip(t[CH_DY], -DY_MAX, DY_MAX)
    t[CH_DX, :, 0] = 0.0
    t[CH_DX, :, w - 1] = 0.0
    t[CH_DY, :, 0] = 0.0
    t[CH_DY, :, w - 1] = 0.0


def apply_independent_thickness_inplace(
    t: np.ndarray,
    seed: int,
    *,
    low: float = 0.7,
    high: float = 1.3,
    bond_max: float = 1.35,
) -> None:
    """
    Multiply each positive half-edge weight by an independent U(low, high) factor
    (default relative thickness/stiffness), then clip bonds to ``bond_max``.
    """
    rng = np.random.default_rng(seed)
    lo = float(min(low, high))
    hi = float(max(low, high))
    bmx = float(max(bond_max, hi))
    for ch in (CH_RIGHT, CH_TOP):
        fac = rng.uniform(lo, hi, size=t[ch].shape)
        t[ch] = np.where(t[ch] > 0, np.clip(t[ch] * fac, 0.0, bmx), 0.0)


def mask_bonds_by_threshold_inplace(t: np.ndarray, tau: float) -> None:
    """Zero out bond weights with value ≤ τ (keep strict inequality as > τ active)."""
    t[CH_RIGHT] = np.where(t[CH_RIGHT] > tau, t[CH_RIGHT], 0.0)
    t[CH_TOP] = np.where(t[CH_TOP] > tau, t[CH_TOP], 0.0)


def uniform_bond_weights_inplace(t: np.ndarray, value: float = 1.0) -> None:
    """Set every positive bond half-edge to the same weight (topology unchanged)."""
    v = float(np.clip(value, 0.0, 1.0))
    t[CH_RIGHT] = np.where(t[CH_RIGHT] > 0.0, v, 0.0)
    t[CH_TOP] = np.where(t[CH_TOP] > 0.0, v, 0.0)


def partial_grid_uniform_beams(
    w: int = W_DEFAULT,
    h: int = H_DEFAULT,
    *,
    geom_scale: float = 0.8,
    bond_threshold: float = 0.32,
    seed: int = 0,
) -> np.ndarray:
    """
    Lattice that is **not** fully connected: random soft bonds are drawn, then bonds
    with value ≤ ``bond_threshold`` are removed, then **all surviving half-edges are
    set to weight 1** (uniform EA/EI / “same thickness” on active struts).

    ``geom_scale`` in [0, 1] scales random node offsets. Use moderate ``bond_threshold``
    (~0.15–0.40 on soft bonds in [0.2, 1]) to avoid trivial full grids and avoid
    excessive pruning.
    """
    rng = np.random.default_rng(seed)
    gs = float(np.clip(geom_scale, 0.0, 1.0))
    t = empty_tensor(w, h)
    t[CH_DX] = gs * rng.uniform(-DX_MAX, DX_MAX, size=(h, w))
    t[CH_DY] = gs * rng.uniform(-DY_MAX, DY_MAX, size=(h, w))
    t[CH_RIGHT] = rng.uniform(0.2, 1.0, size=(h, w))
    t[CH_TOP] = rng.uniform(0.2, 1.0, size=(h, w))
    t[CH_RIGHT][:, -1] = 0.0
    t[CH_TOP][-1, :] = 0.0
    clip_tensor_inplace(t)
    mask_bonds_by_threshold_inplace(t, float(bond_threshold))
    uniform_bond_weights_inplace(t, 1.0)
    return t


def randomize_bond_stiffness_inplace(
    t: np.ndarray,
    seed: int,
    *,
    low: float = 0.2,
    high: float = 1.0,
) -> None:
    """
    Multiply each positive half-edge weight by an independent U(low, high) factor
    (then clip). Same material E at the element level; relative stiffness varies,
    which redistributes internal forces and typically **raises peak beam stresses**
    vs a uniform lattice at fixed δ.
    """
    rng = np.random.default_rng(seed)
    lo = float(np.clip(low, 1e-6, 1.0))
    hi = float(np.clip(high, lo, 1.0))
    for ch in (CH_RIGHT, CH_TOP):
        fac = rng.uniform(lo, hi, size=t[ch].shape)
        t[ch] = np.where(t[ch] > 0.0, np.clip(t[ch] * fac, 0.0, 1.0), 0.0)


@dataclass(frozen=True)
class GlobalScalars:
    bond_fill: float
    rms_disp: float
    mean_abs_dx: float
    mean_abs_dy: float
    horizontal_span: float
    n_nodes: int
    n_edges: int
    min_node_distance: float


def compute_global_scalars(
    t: np.ndarray,
    xy: np.ndarray,
    edges: list[tuple[int, int]],
    w: int,
    h: int,
) -> GlobalScalars:
    h_w = t[CH_RIGHT][:, : w - 1]
    v_w = t[CH_TOP][: h - 1, :]
    bond_fill = float((h_w.sum() + v_w.sum()) / (h_w.size + v_w.size + 1e-30))
    dx, dy = t[CH_DX], t[CH_DY]
    rms = float(np.sqrt(np.mean(dx**2 + dy**2)))
    span = float((w - 1) * STEP)
    # nearest-node distance (4-neighbor grid adjacency in index space, not bonds)
    mind = _min_distance_neighbors(xy, w, h)
    return GlobalScalars(
        bond_fill=bond_fill,
        rms_disp=rms,
        mean_abs_dx=float(np.mean(np.abs(dx))),
        mean_abs_dy=float(np.mean(np.abs(dy))),
        horizontal_span=span,
        n_nodes=w * h,
        n_edges=len(edges),
        min_node_distance=mind,
    )


def _min_distance_neighbors(xy: np.ndarray, w: int, h: int) -> float:
    dmin = np.inf
    for j in range(h):
        for i in range(w):
            a = node_id(i, j, w)
            if i + 1 < w:
                b = node_id(i + 1, j, w)
                dmin = min(dmin, float(np.linalg.norm(xy[a] - xy[b])))
            if j + 1 < h:
                b = node_id(i, j + 1, w)
                dmin = min(dmin, float(np.linalg.norm(xy[a] - xy[b])))
    return float(dmin)


def tensor_to_geometry(
    t: np.ndarray,
    *,
    w: int | None = None,
    h: int | None = None,
    bond_threshold: float = 0.5,
    clip_bonds: bool = True,
) -> tuple[np.ndarray, list[tuple[int, int]], list[float]]:
    """
    Build node coordinates and edges. Edge list order: all horiz (i,j)→(i+1,j)
    then vert (i,j)→(i,j+1). Weights are bond values (> threshold kept).

    If ``clip_bonds`` is False, only displacements are clipped (same as
    ``clip_displacements_only_inplace``); bond channels are left as-is so
    thickness factors above 1 remain for visualization / weighted solves.
    """
    if t.ndim != 3 or t.shape[0] != N_CH:
        raise ValueError(f"Expected tensor ({N_CH}, H, W), got {t.shape}")
    _, h0, w0 = t.shape
    w = w or w0
    h = h or h0
    if w0 != w or h0 != h:
        raise ValueError("Shape mismatch")

    if clip_bonds:
        clip_tensor_inplace(t)
    else:
        clip_displacements_only_inplace(t)

    xy = np.zeros((w * h, 2), dtype=np.float64)
    for j in range(h):
        for i in range(w):
            n = node_id(i, j, w)
            xy[n, 0] = i * STEP + float(t[CH_DX, j, i])
            xy[n, 1] = j * STEP + float(t[CH_DY, j, i])

    edges: list[tuple[int, int]] = []
    weights: list[float] = []

    for j in range(h):
        for i in range(w - 1):
            wgt = float(t[CH_RIGHT, j, i])
            if wgt > bond_threshold:
                a = node_id(i, j, w)
                b = node_id(i + 1, j, w)
                edges.append((min(a, b), max(a, b)))
                weights.append(wgt)

    for j in range(h - 1):
        for i in range(w):
            wgt = float(t[CH_TOP, j, i])
            if wgt > bond_threshold:
                a = node_id(i, j, w)
                b = node_id(i, j + 1, w)
                edges.append((min(a, b), max(a, b)))
                weights.append(wgt)

    # dedupe (shouldn't duplicate if rules consistent)
    seen: dict[tuple[int, int], float] = {}
    for e, wg in zip(edges, weights):
        seen[e] = max(seen.get(e, 0.0), wg)
    edges_sorted = sorted(seen.keys())
    weights_out = [seen[e] for e in edges_sorted]
    return xy, edges_sorted, weights_out


def left_right_node_ids(w: int, h: int) -> tuple[list[int], list[int]]:
    left = [node_id(0, j, w) for j in range(h)]
    right = [node_id(w - 1, j, w) for j in range(h)]
    return left, right


def reachable_from_left(edges: list[tuple[int, int]], w: int, h: int) -> set[int]:
    """All node indices reachable from the left column along edges."""
    ntot = w * h
    adj: list[list[int]] = [[] for _ in range(ntot)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    start = [node_id(0, j, w) for j in range(h)]
    seen: set[int] = set(start)
    q = deque(start)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                q.append(v)
    return seen


def all_right_reachable_from_left(edges: list[tuple[int, int]], w: int, h: int) -> bool:
    """True if every node on the right face is in the same connected component as the left face."""
    r = reachable_from_left(edges, w, h)
    return all(node_id(w - 1, j, w) in r for j in range(h))


def connected_components_all_nodes(
    n_nodes: int, edges: list[tuple[int, int]]
) -> list[set[int]]:
    adj: list[list[int]] = [[] for _ in range(n_nodes)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    seen = [False] * n_nodes
    comps: list[set[int]] = []
    for s in range(n_nodes):
        if seen[s]:
            continue
        cur: set[int] = set()
        dq = deque([s])
        seen[s] = True
        while dq:
            u = dq.popleft()
            cur.add(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    dq.append(v)
        comps.append(cur)
    return comps


def grid_shortest_path_nodes(
    w: int, h: int, start: int, goal: int
) -> list[int]:
    """BFS shortest path on the full 4-connected grid (Manhattan moves)."""
    if start == goal:
        return [start]
    n = w * h
    adj_grid: list[list[int]] = [[] for _ in range(n)]
    for j in range(h):
        for i in range(w):
            u = node_id(i, j, w)
            if i + 1 < w:
                v = node_id(i + 1, j, w)
                adj_grid[u].append(v)
                adj_grid[v].append(u)
            if j + 1 < h:
                v = node_id(i, j + 1, w)
                adj_grid[u].append(v)
                adj_grid[v].append(u)
    parent = [-1] * n
    dq = deque([start])
    parent[start] = start
    while dq:
        u = dq.popleft()
        if u == goal:
            path: list[int] = []
            cur = goal
            while cur != start:
                path.append(cur)
                cur = parent[cur]
            path.append(start)
            path.reverse()
            return path
        for v in adj_grid[u]:
            if parent[v] == -1:
                parent[v] = u
                dq.append(v)
    return []


def _activate_edge_on_tensor(
    t: np.ndarray, a: int, b: int, w: int, h: int, floor: float
) -> bool:
    """Set ch2/ch3 so that nodes a–b are connected; return True if tensor changed."""
    ia, ja = ij_from_id(a, w)
    ib, jb = ij_from_id(b, w)
    changed = False
    if ib == ia + 1 and jb == ja:
        v = float(t[CH_RIGHT, ja, ia])
        nv = max(v, floor)
        if nv > v:
            t[CH_RIGHT, ja, ia] = nv
            changed = True
        return changed
    if ib == ia - 1 and jb == ja:
        v = float(t[CH_RIGHT, ja, ib])
        nv = max(v, floor)
        if nv > v:
            t[CH_RIGHT, ja, ib] = nv
            changed = True
        return changed
    if jb == ja + 1 and ib == ia:
        v = float(t[CH_TOP, ja, ia])
        nv = max(v, floor)
        if nv > v:
            t[CH_TOP, ja, ia] = nv
            changed = True
        return changed
    if jb == ja - 1 and ib == ia:
        v = float(t[CH_TOP, jb, ia])
        nv = max(v, floor)
        if nv > v:
            t[CH_TOP, jb, ia] = nv
            changed = True
        return changed
    return False


def connect_all_regions(
    t: np.ndarray,
    w: int,
    h: int,
    *,
    bond_floor: float = 1.0,
    bond_threshold: float = 1e-9,
) -> int:
    """
    Mutate tensor so the bond graph forms a **single connected component** on
    all W×H nodes by turning on grid-aligned half-edges along shortest paths
    between components. Returns the number of half-edges strengthened.
    """
    n_nodes = w * h
    added = 0
    max_iter = n_nodes * 8
    it = 0
    while it < max_iter:
        it += 1
        xy, edges, _ = tensor_to_geometry(t, w=w, h=h, bond_threshold=bond_threshold)
        comps = connected_components_all_nodes(n_nodes, edges)
        if len(comps) <= 1:
            break
        # Prefer merging into the component that touches the left column
        left_ids = {node_id(0, j, w) for j in range(h)}
        anchor = None
        for c in comps:
            if c & left_ids:
                anchor = c
                break
        if anchor is None:
            anchor = comps[0]
        others = [c for c in comps if c is not anchor]
        if not others:
            break
        target = others[0]
        best: tuple[int, list[int]] | None = None
        for ua in anchor:
            for vb in target:
                path = grid_shortest_path_nodes(w, h, ua, vb)
                if not path:
                    continue
                L = len(path) - 1
                if best is None or L < best[0]:
                    best = (L, path)
        if best is None:
            break
        _, path = best
        for u, v in zip(path[:-1], path[1:]):
            if _activate_edge_on_tensor(t, u, v, w, h, bond_floor):
                added += 1
    clip_tensor_inplace(t)
    return added


def ensure_left_to_all_nodes(
    t: np.ndarray,
    w: int,
    h: int,
    *,
    bond_floor: float = 1.0,
    bond_threshold: float = 1e-9,
) -> int:
    """
    After global connectivity, ensure every node is reachable from the **left
    column** along positive bond weights (add grid paths if needed).
    """
    n_nodes = w * h
    added = 0
    max_iter = n_nodes * 6
    it = 0
    while it < max_iter:
        it += 1
        xy, edges, _ = tensor_to_geometry(t, w=w, h=h, bond_threshold=bond_threshold)
        r = reachable_from_left(edges, w, h)
        if len(r) == n_nodes:
            break
        unreached = [nid for nid in range(n_nodes) if nid not in r]
        best: tuple[int, list[int]] | None = None
        for v in unreached:
            for u in r:
                path = grid_shortest_path_nodes(w, h, u, v)
                if not path:
                    continue
                L = len(path) - 1
                if best is None or L < best[0]:
                    best = (L, path)
        if best is None:
            break
        _, path = best
        for u, v in zip(path[:-1], path[1:]):
            if _activate_edge_on_tensor(t, u, v, w, h, bond_floor):
                added += 1
    clip_tensor_inplace(t)
    return added


def connect_tensor_full(
    t: np.ndarray,
    w: int,
    h: int,
    *,
    bond_floor: float = 1.0,
) -> tuple[int, int]:
    """Run global merge + left-span; returns (n_edges_activated_global, n_edges_activated_left)."""
    a = connect_all_regions(t, w, h, bond_floor=bond_floor)
    b = ensure_left_to_all_nodes(t, w, h, bond_floor=bond_floor)
    return a, b


def reduce_to_reachable_component(
    xy: np.ndarray,
    edges: list[tuple[int, int]],
    weights: list[float],
    w: int,
    h: int,
) -> tuple[np.ndarray, list[tuple[int, int]], list[float], list[int], list[int]]:
    """
    Keep only nodes reachable from the left column; renumber 0..n-1.
    Drops floating islands so the stiffness matrix is not singular.
    """
    seen = reachable_from_left(edges, w, h)
    right_all = {node_id(w - 1, j, w) for j in range(h)}
    if not right_all <= seen:
        raise ValueError("Not all right-face nodes are reachable from the left — check bonds.")

    old_ids = sorted(seen)
    remap = {o: k for k, o in enumerate(old_ids)}
    xy_n = xy[old_ids]
    edges_n: list[tuple[int, int]] = []
    weights_n: list[float] = []
    for (a, b), wg in zip(edges, weights):
        if a in seen and b in seen:
            aa, bb = remap[a], remap[b]
            edges_n.append((min(aa, bb), max(aa, bb)))
            weights_n.append(float(wg))
    # dedupe edges keeping max weight
    emap: dict[tuple[int, int], float] = {}
    for e, wg in zip(edges_n, weights_n):
        emap[e] = max(emap.get(e, 0.0), wg)
    edges_f = sorted(emap.keys())
    weights_f = [emap[e] for e in edges_f]

    left_n = [remap[node_id(0, j, w)] for j in range(h)]
    right_n = [remap[node_id(w - 1, j, w)] for j in range(h)]
    return xy_n, edges_f, weights_f, left_n, right_n


def strut_mass_and_length_sums(
    xy: np.ndarray,
    edges: list[tuple[int, int]],
    weights: list[float],
) -> tuple[float, float]:
    """
    Geometric strut totals on the reduced beam network.

    Returns (mass_metric, length_sum) where mass_metric = Σ_e L_e · w_e with L_e the
    deformed Euclidean length and w_e the bond weight (stiffness fraction). With
    unit reference density, this matches Σ (ρ A_ref w_e) L_e — a volume/mass proxy
    when cross-section area scales like w_e (same scaling as EA in the model).
    length_sum = Σ_e L_e ignores thickness weights.
    """
    mass_m = 0.0
    len_sum = 0.0
    for (i, j), wg in zip(edges, weights):
        L = float(np.linalg.norm(xy[j] - xy[i]))
        w = float(wg)
        len_sum += L
        mass_m += L * w
    return float(mass_m), float(len_sum)


def assemble_and_solve_weighted(
    xy: np.ndarray,
    edges: list[tuple[int, int]],
    edge_weights: list[float],
    ea_base: float,
    ei_base: float,
    *,
    left_nodes: list[int],
    right_nodes: list[int],
    delta_x: float,
) -> FrameResult:
    """Same BCs as frame_lattice.assemble_and_solve but EA/EI scaled per edge."""
    if len(edges) != len(edge_weights):
        raise ValueError("edges and edge_weights length mismatch")
    n = xy.shape[0]
    ndof = 3 * n
    k = np.zeros((ndof, ndof), dtype=float)
    for (i, j), wg in zip(edges, edge_weights):
        wgt = max(float(wg), 1e-12)
        ke = beam_global_k(xy[i], xy[j], ea_base * wgt, ei_base * wgt)
        idx = np.array([3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 * j + 1, 3 * j + 2])
        k[np.ix_(idx, idx)] += ke

    fixed_mask = np.zeros(ndof, dtype=bool)
    prescribed = np.zeros(ndof, dtype=float)
    for nid in left_nodes:
        fixed_mask[3 * nid : 3 * nid + 3] = True
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


def beam_fiber_stress_per_edge(
    u: np.ndarray,
    xy: np.ndarray,
    edges: list[tuple[int, int]],
    ea_edge: list[float],
    ei_edge: list[float],
    a: float,
    i_geom: float,
    y_extreme: float,
) -> np.ndarray:
    """Same stress estimate as frame_lattice, with per-edge EA/EI."""
    sig = np.zeros(len(edges), dtype=float)
    i_geom = max(float(i_geom), 1e-30)
    y_extreme = max(float(y_extreme), 1e-30)
    for k, (i, j) in enumerate(edges):
        d_loc, l, _, _ = element_local_displacement(u, i, j, xy)
        ea_e = float(ea_edge[k])
        ei_e = float(ei_edge[k])
        kl = euler_bernoulli_frame_local_k(ea_e, ei_e, l)
        fl = kl @ d_loc
        n_force = ea_e / l * (d_loc[3] - d_loc[0])
        m1 = float(fl[2])
        m2 = float(fl[5])
        m_w = max(abs(m1), abs(m2))
        sig_a = abs(n_force) / max(a, 1e-30)
        sig_b = m_w * y_extreme / i_geom
        sig[k] = sig_a + sig_b
    return sig


def effective_extension_metrics(
    res: FrameResult,
    left_nodes: list[int],
    right_nodes: list[int],
    horizontal_span: float,
    delta_x: float,
) -> dict[str, float]:
    """Rough scalar summaries for inverse / conditioning."""
    uxl = float(np.mean([res.u[i, 0] for i in left_nodes]))
    uxr = float(np.mean([res.u[i, 0] for i in right_nodes]))
    # Relative mean stretch of the bar network (informal)
    strain_like = (uxr - uxl) / max(horizontal_span, 1e-12)
    return {
        "mean_ux_left": uxl,
        "mean_ux_right": uxr,
        "delta_applied": float(delta_x),
        "strain_like_mean": float(strain_like),
    }


def effective_young_modulus_homogenized(
    res: FrameResult,
    left_nodes: list[int],
    *,
    span_x: float,
    height_y: float,
    delta_x: float,
) -> dict[str, float]:
    """
    Homogenized Young's modulus for macro uniaxial stretch in x:

      ε_macro = δ / Lx ,   σ_macro ≈ |Σ Fx_left| / A_end ,   A_end = Ly · (unit depth)

    E_eff = σ_macro / ε_macro  (same units as base E if lengths are consistent).
    """
    F_net = float(sum(res.reactions[i, 0] for i in left_nodes))
    A_end = max(float(height_y) * 1.0, 1e-30)
    sigma_macro = abs(F_net) / A_end
    eps_macro = float(delta_x) / max(float(span_x), 1e-30)
    E_eff = sigma_macro / max(eps_macro, 1e-30)
    K_sec = abs(F_net) / max(float(delta_x), 1e-30)
    return {
        "E_eff": float(E_eff),
        "sigma_macro_end": float(sigma_macro),
        "eps_macro": float(eps_macro),
        "K_sec": float(K_sec),
        "sum_Rx_left": float(F_net),
    }


def _half_edge_thickness_range(t: np.ndarray, w: int, h: int) -> tuple[float, float]:
    """Min/max stiffness over active half-edges (structural zeros at borders excluded from min)."""
    hr = t[CH_RIGHT][:, : w - 1]
    vt = t[CH_TOP][: h - 1, :]
    vals = np.concatenate([hr.ravel(), vt.ravel()])
    pos = vals[vals > 0]
    if pos.size == 0:
        return 0.0, 1.0
    lo, hi = float(pos.min()), float(pos.max())
    if hi - lo < 1e-12:
        lo -= 0.05
        hi += 0.05
    return lo, hi


def visualize_tensor(
    t: np.ndarray,
    out_path: Path | None,
    *,
    title: str = "",
    globals_: GlobalScalars | None = None,
    note: str = "",
    footer: str = "",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    _, h, w = t.shape
    # Copy so we do not clip bond/thickness channels in the caller's tensor; ch2/ch3
    # show actual relative thickness factors (may exceed 1).
    t_vis = np.asarray(t, dtype=np.float64, copy=True)
    xy, edges, _ = tensor_to_geometry(t_vis, w=w, h=h, clip_bonds=False)
    bmin, bmax = _half_edge_thickness_range(t_vis, w, h)

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15])

    extent = [-0.5, w - 0.5, -0.5, h - 0.5]
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(t_vis[CH_DX], origin="lower", cmap="coolwarm", extent=extent, aspect="auto")
    ax0.set_title("ch0: dx")
    fig.colorbar(im0, ax=ax0, fraction=0.046)

    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(t_vis[CH_DY], origin="lower", cmap="coolwarm", extent=extent, aspect="auto")
    ax1.set_title("ch1: dy")
    fig.colorbar(im1, ax=ax1, fraction=0.046)

    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(
        t_vis[CH_RIGHT],
        origin="lower",
        cmap="magma",
        vmin=bmin,
        vmax=bmax,
        extent=extent,
        aspect="auto",
    )
    ax2.set_title("ch2: →(i+1,j) horiz. thickness factor")
    fig.colorbar(im2, ax=ax2, fraction=0.046)

    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.imshow(
        t_vis[CH_TOP],
        origin="lower",
        cmap="magma",
        vmin=bmin,
        vmax=bmax,
        extent=extent,
        aspect="auto",
    )
    ax3.set_title("ch3: →(i,j+1) vert. thickness factor (+y)")
    fig.colorbar(im3, ax=ax3, fraction=0.046)

    ax4 = fig.add_subplot(gs[1, 1:])
    for i, j in edges:
        p0, p1 = xy[i], xy[j]
        ax4.plot([p0[0], p1[0]], [p0[1], p1[1]], "b-", lw=0.6, alpha=0.75)
    ax4.scatter(xy[:, 0], xy[:, 1], s=3, c="k", alpha=0.35)
    ax4.set_aspect("equal", adjustable="box")
    ax4.set_xlabel("x")
    ax4.set_ylabel("y")
    ax4.set_title("Extracted beam layout")
    ax4.grid(True, alpha=0.25)

    ttl = title or "Tensor lattice"
    if note:
        ttl = f"{note} — {ttl}"
    if globals_ is not None:
        ttl += (
            f" | bonds μ={globals_.bond_fill:.3f}, rms_disp={globals_.rms_disp:.3f}, "
            f"|E|={globals_.n_edges}, d_min={globals_.min_node_distance:.3f}"
        )
    fig.suptitle(ttl, fontsize=10, y=0.99)
    fig.tight_layout(rect=[0, 0.07 if footer else 0.05, 1, 0.93])
    if footer:
        fig.text(
            0.5,
            0.012,
            footer,
            ha="center",
            va="bottom",
            fontsize=8,
            family="monospace",
            transform=fig.transFigure,
        )
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_mechanics(
    res: FrameResult,
    sigma_edge: np.ndarray,
    out_path: Path | None,
    *,
    disp_scale: float,
    title: str = "",
    eff: dict[str, float] | None = None,
    stress_units: str = "model units",
    footer: str = "",
) -> None:
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
    except ImportError:
        return
    xy = res.xy
    disp = res.u[:, :2]
    xd = xy + disp_scale * disp
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
    for i, j in res.edges:
        ax0.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], "k-", lw=0.4, alpha=0.35)
        ax0.plot([xd[i, 0], xd[j, 0]], [xd[i, 1], xd[j, 1]], "b-", lw=0.7)
    ax0.set_aspect("equal", adjustable="box")
    ax0.set_title(f"Undeformed vs deformed (×{disp_scale:g})")
    ax0.grid(True, alpha=0.25)

    segs = [[[xd[i, 0], xd[i, 1]], [xd[j, 0], xd[j, 1]]] for i, j in res.edges]
    segs = np.asarray(segs, dtype=float)
    smin = float(np.min(sigma_edge))
    sig_max = float(np.max(sigma_edge))
    smax = sig_max
    if smax <= smin:
        smax = smin + 1.0
    norm = mpl.colors.Normalize(vmin=smin, vmax=smax)
    cmap = mpl.colormaps["turbo"]
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=1.2, array=sigma_edge)
    ax1.add_collection(lc)
    ax1.autoscale()
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_title(f"Beam stress — σ_max={sig_max:.5g} ({stress_units})")
    fig.colorbar(lc, ax=ax1, fraction=0.046, label=f"σ ({stress_units}), max={sig_max:.5g}")
    st2 = f"σ_max={sig_max:.5g} ({stress_units}) | " + title
    if eff is not None:
        st2 += (
            f" | E_eff≈{eff['E_eff']:.4g}  "
            f"K_sec={eff['K_sec']:.4g}  "
            f"σ_end={eff['sigma_macro_end']:.4g}  ε={eff['eps_macro']:.4g}"
        )
    fig.suptitle(st2, fontsize=9, y=0.99)
    fig.tight_layout(rect=[0, 0.08 if footer else 0.04, 1, 0.92])
    if footer:
        fig.text(
            0.5,
            0.015,
            footer,
            ha="center",
            va="bottom",
            fontsize=7.5,
            family="monospace",
            transform=fig.transFigure,
        )
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _resolve_output_path(script_dir: Path, path_str: str, default_name: str) -> Path:
    if not path_str:
        return script_dir / default_name
    p = Path(path_str)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "pic":
        return script_dir / p
    return script_dir / p


def _resolve_pic_path(script_dir: Path, pic_dir: Path, path_str: str, default_name: str) -> Path:
    if not path_str:
        return pic_dir / default_name
    p = Path(path_str)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "pic":
        return script_dir / p
    return pic_dir / p


def solve_tensor_mechanics(
    t: np.ndarray,
    w: int,
    h: int,
    cfg: SolveConfig | None = None,
) -> MechanicsSolve:
    """
    Run the same pipeline as CLI ``solve``: mask bonds, optional connect-all / uniform
    weights, reduce to left-reachable component, assemble and return homogenized metrics.
    """
    cfg = cfg if cfg is not None else SolveConfig()
    y_ext = cfg.y_ext if cfg.y_ext is not None else float(np.sqrt(cfg.I / max(cfg.A, 1e-30)))
    t = np.asarray(t, dtype=np.float64, copy=True)
    bridge: tuple[int, int] | None = None

    try:
        # Displacements must stay in range; bond channels are **not** clipped to [0,1]
        # here so relative thickness factors > 1 (stiffer than nominal) scale EA, EI
        # correctly — same material, stiffness ∝ thickness proxy in ch2/ch3.
        clip_displacements_only_inplace(t)
        mask_bonds_by_threshold_inplace(t, float(cfg.bond_threshold))

        if cfg.connect_all:
            ag, al = connect_tensor_full(t, w, h, bond_floor=cfg.bond_floor)
            bridge = (ag, al)

        if cfg.uniform_bonds:
            ub = float(np.clip(cfg.uniform_bond_value, 0.0, 1.0))
            uniform_bond_weights_inplace(t, ub)

        xy, edges, weights = tensor_to_geometry(
            t, w=w, h=h, bond_threshold=1e-12, clip_bonds=False
        )
        if len(edges) < 3:
            return MechanicsSolve(
                result=MechanicsResult(ok=False, error="Too few edges to form a frame."),
                connect_bridge=bridge,
            )
        if not all_right_reachable_from_left(edges, w, h):
            return MechanicsSolve(
                result=MechanicsResult(
                    ok=False,
                    error="Right face not fully connected to the left through bonds.",
                ),
                connect_bridge=bridge,
            )

        gl = compute_global_scalars(t, xy, edges, w, h)
        xy_r, edges_r, weights_r, left, right = reduce_to_reachable_component(
            xy, edges, weights, w, h
        )
        mass_m, len_sum = strut_mass_and_length_sums(xy_r, edges_r, weights_r)

        E_use = float(cfg.E) * float(cfg.e_scale)
        ea = E_use * cfg.A
        ei = E_use * cfg.I

        res = assemble_and_solve_weighted(
            xy_r,
            edges_r,
            weights_r,
            ea,
            ei,
            left_nodes=left,
            right_nodes=right,
            delta_x=cfg.delta,
        )

        ea_edge = [ea * wgt for wgt in weights_r]
        ei_edge = [ei * wgt for wgt in weights_r]
        sigma = beam_fiber_stress_per_edge(
            res.u, xy_r, edges_r, ea_edge, ei_edge, cfg.A, cfg.I, y_ext
        )
        span = float((w - 1) * STEP)
        height_span = float((h - 1) * STEP)
        eff = effective_young_modulus_homogenized(
            res, left, span_x=span, height_y=height_span, delta_x=cfg.delta
        )
        ext = effective_extension_metrics(res, left, right, span, cfg.delta)

        mr = MechanicsResult(
            ok=True,
            E_eff=float(eff["E_eff"]),
            sigma_max=float(np.max(sigma)),
            sigma_min=float(np.min(sigma)),
            sigma_macro_end=float(eff["sigma_macro_end"]),
            eps_macro=float(eff["eps_macro"]),
            K_sec=float(eff["K_sec"]),
            sum_rx_left=float(eff["sum_Rx_left"]),
            mean_ux_left=float(ext["mean_ux_left"]),
            mean_ux_right=float(ext["mean_ux_right"]),
            strain_like_mean=float(ext["strain_like_mean"]),
            rms_disp=gl.rms_disp,
            bond_fill=gl.bond_fill,
            n_edges_full=len(edges),
            n_nodes_reduced=int(xy_r.shape[0]),
            n_edges_reduced=len(edges_r),
            min_node_distance=gl.min_node_distance,
            strut_mass_metric=mass_m,
            strut_length_sum=len_sum,
        )
        return MechanicsSolve(
            result=mr,
            frame=res,
            sigma=sigma,
            xy=xy_r,
            edges=edges_r,
            edge_weights=list(weights_r),
            connect_bridge=bridge,
        )
    except (ValueError, np.linalg.LinAlgError) as e:
        return MechanicsSolve(
            result=MechanicsResult(ok=False, error=str(e)),
            connect_bridge=bridge,
        )


def cmd_random(args: argparse.Namespace) -> None:
    script_dir = Path(__file__).resolve().parent
    pic_dir = script_dir / "pic"
    if getattr(args, "full_grid", False):
        t = fully_connected_perturbed_tensor(
            W_DEFAULT,
            H_DEFAULT,
            perturb=float(args.perturb),
            seed=args.seed,
        )
    else:
        t = random_tensor(
            W_DEFAULT,
            H_DEFAULT,
            seed=args.seed,
            bond_binary=not args.soft_bonds,
        )
    clip_tensor_inplace(t)
    if args.connect_all:
        ag, al = connect_tensor_full(
            t, W_DEFAULT, H_DEFAULT, bond_floor=args.bond_floor
        )
        print(f"connect-all: +{ag} half-edges (merge components), +{al} (left reach)")
    xy, edges, _ = tensor_to_geometry(t, bond_threshold=args.bond_threshold)
    gl = compute_global_scalars(t, xy, edges, W_DEFAULT, H_DEFAULT)
    out_npz = _resolve_output_path(script_dir, args.out, "tensor_lattice_sample.npz")
    extra: dict = {"tensor": t, "w": W_DEFAULT, "h": H_DEFAULT}
    if getattr(args, "full_grid", False):
        extra["perturb"] = float(args.perturb)
    np.savez_compressed(out_npz, **extra)
    print(f"Saved tensor to {out_npz}")
    print(
        f"globals: bond_fill={gl.bond_fill:.4f}, rms_disp={gl.rms_disp:.4f}, "
        f"edges={gl.n_edges}, d_min={gl.min_node_distance:.4f}"
    )
    if gl.min_node_distance < 0.05:
        print("Warning: very small nearest-node spacing — consider smaller dx/dy range.")

    if not args.no_plot:
        outp = _resolve_pic_path(
            script_dir,
            pic_dir,
            args.plot,
            f"tensor_lattice_seed{args.seed}.png",
        )
        visualize_tensor(t, outp, globals_=gl, note=getattr(args, "note", "") or "")
        print(f"Saved figure to {outp}")


def cmd_visualize(args: argparse.Namespace) -> None:
    script_dir = Path(__file__).resolve().parent
    pic_dir = script_dir / "pic"
    data = np.load(args.load, allow_pickle=True)
    t = data["tensor"]
    w = int(data.get("w", t.shape[2]))
    h = int(data.get("h", t.shape[1]))
    xy, edges, _ = tensor_to_geometry(t, w=w, h=h, bond_threshold=args.bond_threshold)
    gl = compute_global_scalars(t, xy, edges, w, h)
    outp = _resolve_pic_path(script_dir, pic_dir, args.plot, "tensor_lattice_view.png")
    visualize_tensor(t, outp, globals_=gl, note=getattr(args, "note", "") or "")
    print(f"Saved figure to {outp}")


def cmd_solve(args: argparse.Namespace) -> None:
    script_dir = Path(__file__).resolve().parent
    pic_dir = script_dir / "pic"
    data = np.load(args.load, allow_pickle=True)
    t = np.array(data["tensor"], dtype=np.float64, copy=True)
    w = int(data.get("w", t.shape[2]))
    h = int(data.get("h", t.shape[1]))

    cfg = SolveConfig(
        bond_threshold=float(args.bond_threshold),
        connect_all=bool(args.connect_all),
        bond_floor=float(args.bond_floor),
        uniform_bonds=bool(getattr(args, "uniform_bonds", False)),
        uniform_bond_value=float(getattr(args, "uniform_bond_value", 1.0)),
        delta=float(args.delta),
        E=float(args.E),
        A=float(args.A),
        I=float(args.I),
        y_ext=float(args.y_ext),
        e_scale=float(args.e_scale),
    )
    ms = solve_tensor_mechanics(t, w, h, cfg)
    r = ms.result

    if ms.connect_bridge is not None:
        ag, al = ms.connect_bridge
        print(f"connect-all: +{ag} half-edges (merge components), +{al} (left reach)")
    if getattr(args, "uniform_bonds", False):
        ub = float(np.clip(args.uniform_bond_value, 0.0, 1.0))
        print(f"uniform bonds: all active half-edges → {ub:.4g}")

    if not r.ok:
        raise SystemExit(r.error or "Mechanics solve failed.")

    print(
        f"Reduced model: {r.n_nodes_reduced} nodes, {r.n_edges_reduced} edges "
        f"(reachable from left)"
    )
    print(
        f"struts: ΣL={r.strut_length_sum:.6g}  Σ(L·w) mass proxy={r.strut_mass_metric:.6g}"
    )
    E_use = float(args.E) * float(args.e_scale)
    print(f"max stress est: {r.sigma_max:.6g} (same units as E)")
    print(f"E_eff (homogenized): {r.E_eff:.6g}")
    metrics = {
        "mean_ux_left": r.mean_ux_left,
        "mean_ux_right": r.mean_ux_right,
        "delta_applied": float(args.delta),
        "strain_like_mean": r.strain_like_mean,
    }
    effective = {
        "E_eff": r.E_eff,
        "sigma_macro_end": r.sigma_macro_end,
        "eps_macro": r.eps_macro,
        "K_sec": r.K_sec,
        "sum_Rx_left": r.sum_rx_left,
    }
    print(f"metrics: {metrics}")
    print(f"effective: {effective}")

    stress_label = "scaled" if args.e_scale != 1.0 else "model units"
    if not args.no_plot:
        outp = _resolve_pic_path(script_dir, pic_dir, args.plot, "tensor_lattice_solve.png")
        note_p = f"{args.note} | " if getattr(args, "note", "") else ""
        eff_dict = {
            "E_eff": r.E_eff,
            "sigma_macro_end": r.sigma_macro_end,
            "eps_macro": r.eps_macro,
            "K_sec": r.K_sec,
        }
        visualize_mechanics(
            ms.frame,
            ms.sigma,
            outp,
            disp_scale=args.plot_scale,
            title=(
                f"{note_p}E_base={args.E}, e_scale={args.e_scale}, E_use={E_use:.4g}, δ={args.delta}"
            ),
            eff=eff_dict,
            stress_units=stress_label,
        )
        print(f"Saved mechanics figure to {outp}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="4-channel grid tensor lattice + frame solve")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("random", help="Generate random tensor and optional plot")
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument(
        "--full-grid",
        action="store_true",
        help="Full bond grid (all beams on), dx/dy scaled by --perturb from ideal square",
    )
    pr.add_argument(
        "--perturb",
        type=float,
        default=1.0,
        help="With --full-grid: amplitude in [0,1] for random node offsets (0 = ideal lattice)",
    )
    pr.add_argument("--soft-bonds", action="store_true", help="Continuous bond weights in (0,1]")
    pr.add_argument("--bond-threshold", type=float, default=0.5)
    pr.add_argument("--out", type=str, default="", help="Output .npz path")
    pr.add_argument("--plot", type=str, default="")
    pr.add_argument("--no-plot", action="store_true")
    pr.add_argument(
        "--connect-all",
        action="store_true",
        help="Add/raise bonds so all nodes are in one component and reachable from the left",
    )
    pr.add_argument(
        "--bond-floor",
        type=float,
        default=1.0,
        help="Bond weight applied when connect-all adds a half-edge",
    )
    pr.add_argument("--note", type=str, default="", help="Extra line prefix on figure title")
    pr.set_defaults(func=cmd_random)

    pv = sub.add_parser("visualize", help="Plot channels + geometry from .npz")
    pv.add_argument("--load", type=str, required=True)
    pv.add_argument("--bond-threshold", type=float, default=0.5)
    pv.add_argument("--plot", type=str, default="")
    pv.add_argument("--note", type=str, default="", help="Extra line prefix on figure title")
    pv.set_defaults(func=cmd_visualize)

    ps = sub.add_parser("solve", help="Mechanical solve for a saved tensor")
    ps.add_argument("--load", type=str, required=True)
    ps.add_argument("--bond-threshold", type=float, default=0.5)
    ps.add_argument("--delta", type=float, default=0.01, help="Prescribed u_x on right column")
    ps.add_argument("--E", type=float, default=1.0)
    ps.add_argument("--A", type=float, default=1.0)
    ps.add_argument("--I", type=float, default=1e-2)
    ps.add_argument(
        "--y-ext",
        type=float,
        default=None,
        help="Outer-fiber distance for bending stress; default sqrt(I/A)",
    )
    ps.add_argument("--plot-scale", type=float, default=30.0)
    ps.add_argument("--plot", type=str, default="")
    ps.add_argument("--no-plot", action="store_true")
    ps.add_argument(
        "--connect-all",
        action="store_true",
        help="Add/raise bonds before solve (same as random)",
    )
    ps.add_argument("--bond-floor", type=float, default=1.0)
    ps.add_argument(
        "--e-scale",
        type=float,
        default=10000.0,
        help="Multiply base Young's modulus E so stresses are O(1)–O(10) (default 1e4)",
    )
    ps.add_argument(
        "--uniform-bonds",
        action="store_true",
        help="After mask/connect-all, set every active beam weight to --uniform-bond-value",
    )
    ps.add_argument(
        "--uniform-bond-value",
        type=float,
        default=1.0,
        help="Target weight for --uniform-bonds (clipped to [0,1])",
    )
    ps.add_argument("--note", type=str, default="", help="Extra prefix on mechanics figure title")
    ps.set_defaults(func=cmd_solve)

    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    if args.cmd == "solve" and args.y_ext is None:
        # attach default sqrt(I/A) for stress helper
        args.y_ext = float(np.sqrt(args.I / max(args.A, 1e-30)))
    args.func(args)


if __name__ == "__main__":
    main()
