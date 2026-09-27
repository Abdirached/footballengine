"""Geometric primitives: passing-lane openness, line-break counting, hulls.

Coordinate system: metres, origin at pitch centre, x along length, y across width.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError

# Physical constants (tunable)
V_PASS = 18.0        # m/s ground-pass ball speed
V_DEF = 5.0          # m/s defender sprint speed
T_REACT = 0.35       # s defender reaction delay
REACH = 1.1          # m interception radius around defender


def seg_projection(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Projection of point p on segment a->b -> (s∈[0,1] clamped, perpendicular dist m)."""
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-9:
        return 0.0, float(np.linalg.norm(p - a))
    s = float(((p - a) @ ab) / denom)
    sc = min(1.0, max(0.0, s))
    closest = a + ab * sc
    return sc, float(np.linalg.norm(p - closest))


def lane_risk(a: np.ndarray, b: np.ndarray, opponents: np.ndarray,
              v_pass: float = V_PASS, v_def: float = V_DEF,
              t_react: float = T_REACT, reach: float = REACH) -> tuple[float, dict]:
    """Interception risk of pass segment a->b (0=safe, 1=almost certain pick-off)."""
    return float(lane_risk_batch(a, b[None, :], opponents, v_pass, v_def,
                                 t_react, reach)[0]), {}


def lane_risk_batch(origin: np.ndarray, targets: np.ndarray, opponents: np.ndarray,
                    v_pass: float = V_PASS, v_def: float = V_DEF,
                    t_react: float = T_REACT, reach: float = REACH) -> np.ndarray:
    """Vectorised interception risk for N segments sharing an origin.

    Model per opponent: they may close (t_ball − t_react)·v_def metres toward
    their nearest point on the lane; risk_i = clip(1 − d_perp/closing, 0, 1),
    aggregated by max across opponents. Opponents behind the passer (s ≤ 5%)
    are ignored.
    """
    targets = np.atleast_2d(targets)
    n = targets.shape[0]
    if opponents.size == 0 or n == 0:
        return np.zeros(n)

    AB = targets - origin                                   # (N,2)
    L2 = np.einsum("ij,ij->i", AB, AB)
    L2 = np.maximum(L2, 1e-9)
    t_ball = np.sqrt(L2) / v_pass                           # (N,)

    rel = opponents[None, :, :] - origin                    # (1,M,2)
    s = np.einsum("nmk,nk->nm", rel, AB) / L2[:, None]      # (N,M)
    proj = origin + s[..., None] * AB[:, None, :]           # (N,M,2)
    d_perp = np.linalg.norm(opponents[None, :, :] - proj, axis=2)

    closing = np.maximum(0.0, t_ball - t_react) * v_def + reach
    r = np.clip(1.0 - d_perp / closing[:, None], 0.0, 1.0)
    r = np.where(s > 0.05, r, 0.0)
    return r.max(axis=1)


def lane_openness_batch(origin: np.ndarray, targets: np.ndarray,
                        opponents: np.ndarray) -> np.ndarray:
    return 1.0 - lane_risk_batch(origin, targets, opponents)


def lane_openness(a: np.ndarray, b: np.ndarray, opponents: np.ndarray, **kw) -> float:
    """1 - risk."""
    return 1.0 - lane_risk(a, b, opponents, **kw)[0]


def count_lines_broken(a: np.ndarray, b: np.ndarray, opponents: np.ndarray,
                       gap_m: float = 4.0) -> int:
    """Number of distinct defensive lines (x-clustered) strictly between a and b."""
    x_lo, x_hi = sorted((a[0], b[0]))
    xs = np.array(sorted(o[0] for o in opponents if x_lo < o[0] < x_hi))
    if xs.size == 0:
        return 0
    lines = 1
    for i in range(1, xs.size):
        if xs[i] - xs[i - 1] > gap_m:
            lines += 1
    return lines


def hull_area(points: np.ndarray) -> float:
    """Convex hull area in m²; 0 when degenerate (<3 pts or collinear)."""
    if points.shape[0] < 3:
        return 0.0
    try:
        return float(ConvexHull(points).volume)  # 2-D: volume == area
    except QhullError:
        return 0.0


def corridor_index(y: float, pitch_width: float = 68.0) -> int:
    """Map y ∈ [−W/2, W/2] to one of 5 vertical corridors (0=left wing … 4=right)."""
    c = int((y + pitch_width / 2) / (pitch_width / 5))
    return int(min(4, max(0, c)))


def corridor_of(x: float, pitch_length: float = 105.0) -> int:
    """Horizontal band index 0..2 (defensive/middle/final third relative to +x)."""
    third = pitch_length / 3
    return int(min(2, max(0, floor_div(x + pitch_length / 2, third))))


def floor_div(v: float, step: float) -> int:
    import math
    return math.floor(v / step)


def angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    cu = float(np.linalg.norm(u)); cv = float(np.linalg.norm(v))
    if cu < 1e-9 or cv < 1e-9:
        return 180.0
    cosang = float(np.clip((u @ v) / (cu * cv), -1, 1))
    return float(np.degrees(np.arccos(cosang)))
