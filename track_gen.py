# track_gen.py
# Generator + visualizer for a VALID closed, non-self-intersecting track
# Track rows: [turn_radius, is_pit_stop(0/1), idx]
from __future__ import annotations

import math
from typing import Tuple, Dict, Optional

# Force a headless backend BEFORE importing pyplot
import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt


# =========================
# Utilities: geometry
# =========================

def _resample_closed_polyline(xy: np.ndarray, N: int) -> np.ndarray:
    """Resample a CLOSED polyline to N equidistant points along its perimeter."""
    P = np.vstack([xy, xy[0]])
    seg = np.diff(P, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    clen = np.concatenate([[0.0], np.cumsum(seglen)])
    L = clen[-1]

    targets = np.linspace(0, L, N + 1)[:-1]
    res = np.empty((N, 2), dtype=float)
    j = 0
    for i, t in enumerate(targets):
        while not (clen[j] <= t <= clen[j + 1]):
            j += 1
        u = (t - clen[j]) / (clen[j + 1] - clen[j] + 1e-12)
        res[i] = P[j] + u * seg[j]
    return res


def _angle_wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _segments_intersect(p1, p2, q1, q2) -> bool:
    def orient(a, b, c):
        return np.cross(b - a, c - a)

    if (max(p1[0], p2[0]) < min(q1[0], q2[0]) or
        max(q1[0], q2[0]) < min(p1[0], p2[0]) or
        max(p1[1], p2[1]) < min(q1[1], q2[1]) or
        max(q1[1], q2[1]) < min(p1[1], p2[1])):
        return False

    o1 = orient(p1, p2, q1)
    o2 = orient(p1, p2, q2)
    o3 = orient(q1, q2, p1)
    o4 = orient(q1, q2, p2)
    return (o1 * o2 < 0) and (o3 * o4 < 0)


def _count_self_intersections(xy: np.ndarray) -> int:
    n = len(xy)
    count = 0
    for i in range(n):
        p1 = xy[i]
        p2 = xy[(i + 1) % n]
        for j in range(i + 2, n):
            if j == n - 1 and i == 0:
                continue
            q1 = xy[j]
            q2 = xy[(j + 1) % n]
            if _segments_intersect(p1, p2, q1, q2):
                count += 1
    return count


# =========================
# Smooth closed loop (polar)
# =========================

def _smooth_polar_loop(M: int, R0: float, eps: float,
                       rng: np.random.Generator,
                       n_terms: int = 3,
                       freq_low: int = 1,
                       freq_high: int = 4) -> np.ndarray:
    phi = np.linspace(0, 2 * np.pi, M, endpoint=False)
    r = np.ones_like(phi) * R0

    amps = rng.uniform(0.3, 1.0, size=n_terms)
    amps = amps / amps.sum()
    ks = rng.integers(freq_low, freq_high + 1, size=n_terms)

    for a, k in zip(amps, ks):
        phase = rng.uniform(0, 2 * np.pi)
        sign = rng.choice([-1.0, 1.0])
        r *= (1 + eps * sign * a * np.sin(k * phi + phase))

    r = np.clip(r, 0.5 * R0, 1.5 * R0)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    xy = np.stack([x, y], axis=1)
    xy -= xy.mean(axis=0, keepdims=True)
    return xy


# =========================
# Radii <-> XY conversions
# =========================

def _radii_from_xy_equally_spaced(xy: np.ndarray) -> Tuple[np.ndarray, float]:
    N = len(xy)
    v = xy[(np.arange(N) + 1) % N] - xy
    seglen = np.linalg.norm(v, axis=1)
    s = float(seglen.mean())

    theta = np.arctan2(v[:, 1], v[:, 0])
    dtheta = np.array([_angle_wrap(theta[(i + 1) % N] - theta[i]) for i in range(N)], dtype=float)

    R = np.empty(N, dtype=float)
    for i in range(N):
        if abs(dtheta[i]) < 1e-9:
            R[i] = math.inf
        else:
            R[i] = (s / abs(dtheta[i])) * (-1 if dtheta[i] > 0 else 1)
    return R, s


def track_to_xy(track: np.ndarray, step_length: float) -> np.ndarray:
    R = track[:, 0]
    N = len(R)
    xy = np.zeros((N + 1, 2), dtype=float)
    theta = 0.0
    x, y = 0.0, 0.0
    for i in range(N):
        Ri = R[i]
        dtheta = 0.0 if math.isinf(Ri) else - step_length / Ri
        theta += dtheta
        x += step_length * math.cos(theta)
        y += step_length * math.sin(theta)
        xy[i + 1] = (x, y)
    return xy


# =========================
# Public API
# =========================

def generate_track(
    N: int = 1000,
    seed: Optional[int] = None,
    *,
    max_tries: int = 25,
    R0: float = 200.0,
    eps_range: Tuple[float, float] = (0.08, 0.2),
    n_terms_range: Tuple[int, int] = (2, 4),
    freq_range: Tuple[int, int] = (1, 4),
    pit_len_frac: float = 0.02
) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    pit_len = max(1, int(round(pit_len_frac * N)))

    for _ in range(max_tries):
        M = max(4 * N, 512)
        eps = float(rng.uniform(*eps_range))
        n_terms = int(rng.integers(n_terms_range[0], n_terms_range[1] + 1))
        base_xy = _smooth_polar_loop(M, R0, eps, rng, n_terms, freq_range[0], freq_range[1])

        xy = _resample_closed_polyline(base_xy, N)
        if _count_self_intersections(xy) > 0:
            continue

        R, s = _radii_from_xy_equally_spaced(xy)

        xy2 = track_to_xy(np.column_stack([R, np.zeros(N), np.arange(N)]), s)
        closure_err = float(np.linalg.norm(xy2[-1] - xy2[0]))
        if closure_err > 2.0 * s:
            continue

        pit_start = int(rng.integers(0, N))
        pit_idx = [(pit_start + k) % N for k in range(pit_len)]
        is_pit = np.zeros(N, dtype=int)
        is_pit[pit_idx] = 1

        track = np.column_stack([R, is_pit, np.arange(N, dtype=int)])
        return track, s

    raise RuntimeError("Failed to generate a valid track. Try a different seed or parameters.")


def validate_track(track: np.ndarray, step_length: float) -> Dict[str, float]:
    xy = track_to_xy(track, step_length)
    closure_err = float(np.linalg.norm(xy[-1] - xy[0]))
    intersections = int(_count_self_intersections(xy[:-1]))
    return {
        "closure_error": closure_err,
        "self_intersections": intersections,
        "valid": int((closure_err <= 2.0 * step_length) and (intersections == 0))
    }


def visualize_track(
    track: np.ndarray,
    step_length: float,
    show_indices: bool = False,
    save_path: Optional[str] = None
) -> str:
    """
    Always saves a PNG (headless-safe). Returns the output path.
    """
    xy_full = track_to_xy(track, step_length)  # N+1
    xy = xy_full[:-1]
    N = len(track)
    pit_mask = track[:, 1].astype(bool)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(xy[:, 0], xy[:, 1], linewidth=1.5, label="Track")
    if pit_mask.any():
        ax.plot(xy[pit_mask, 0], xy[pit_mask, 1], linewidth=2.5, label="Pit segment")
    ax.scatter([xy[0, 0]], [xy[0, 1]], s=80, marker='o', label="Start")
    ax.set_aspect('equal', 'box')
    ax.set_title("Generated Track (centerline)")
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if show_indices:
        step = max(1, N // 20)
        for i in range(0, N, step):
            ax.text(xy[i, 0], xy[i, 1], str(i), fontsize=8)

    plt.tight_layout()

    if save_path is None:
        save_path = "track.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved track image to {save_path}")
    return save_path
