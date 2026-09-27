"""Pressing analytics: cover shadows, passing-lane occlusion, press triggers."""
from __future__ import annotations

import numpy as np

from .geometry import seg_projection


# --------------------------------------------------------- cover shadows -----
def cover_shadows(defenders_xy: np.ndarray, carrier_xy: np.ndarray
                  ) -> list[dict]:
    """Cone each defender casts while marking the ball carrier.

    Apex at defender, axis toward the carrier, half-angle narrows with distance
    (closer markers cast wider shadows). Length extends ~10 m past the carrier.
    """
    out = []
    for d in defenders_xy:
        v = carrier_xy - d
        dist = float(np.linalg.norm(v))
        if dist < 0.5:
            continue
        u = v / dist
        half_deg = float(np.clip(65.0 - 1.2 * dist, 28.0, 65.0))
        out.append({
            "x": float(d[0]), "y": float(d[1]),
            "dir_x": round(float(u[0]), 3), "dir_y": round(float(u[1]), 3),
            "half_angle_deg": round(half_deg, 1),
        })
    return out


def point_in_shadow(p: np.ndarray, shadow: dict) -> bool:
    d = np.array([shadow["x"], shadow["y"]])
    u = np.array([shadow["dir_x"], shadow["dir_y"]])
    w = p - d
    proj = float(w @ u)
    if proj <= 0:
        return False
    perp = float(np.linalg.norm(w - u * proj))
    half_tan = np.tan(np.radians(shadow["half_angle_deg"]))
    return perp <= proj * half_tan


def lane_occlusion_pct(a: np.ndarray, b: np.ndarray, shadows: list[dict],
                       n_samples: int = 24) -> float:
    """Percentage of the pass corridor a->b hidden inside cover shadows."""
    if not shadows:
        return 0.0
    hits = 0
    for s in np.linspace(0.05, 1.0, n_samples):
        p = a + (b - a) * s
        hits += any(point_in_shadow(p, sh) for sh in shadows)
    return round(100.0 * hits / n_samples, 1)


# ------------------------------------------------------------- triggers ------
def detect_triggers(prev_ball_v: np.ndarray, ball_xy: np.ndarray, ball_v: np.ndarray,
                    receiver_xy: np.ndarray | None, receiver_v: np.ndarray | None,
                    defenders_xy: np.ndarray, attack_dir: int) -> list[dict]:
    """Detect pressing triggers on the receiving player / ball state."""
    trig: list[dict] = []
    # heavy touch: sudden ball velocity change between frames
    dv = float(np.linalg.norm(ball_v - prev_ball_v))
    if dv > 6.0:
        trig.append({"type": "heavy_touch", "player": None})
    # pass into touchline corridor moving outward
    if abs(ball_xy[1]) > 26.0 and np.sign(ball_v[1]) == np.sign(ball_xy[1]) \
            and abs(ball_v[1]) > 2.0:
        trig.append({"type": "touchline_trap", "player": None})
    if receiver_xy is not None and receiver_v is not None:
        speed = float(np.linalg.norm(receiver_v))
        to_goal = np.array([-attack_dir, 0.0])
        facing = float((receiver_v / max(speed, 1e-6)) @ to_goal) if speed > 0.5 else 0.0
        nearest_def = float(np.min(np.linalg.norm(defenders_xy - receiver_xy, axis=1))) \
            if defenders_xy.size else 99.0
        if speed < 2.0 and facing < -0.3 and nearest_def < 4.0:
            trig.append({"type": "back_to_goal", "player": "receiver"})
        elif speed < 2.0 and nearest_def < 2.5:
            trig.append({"type": "pressure_spike", "player": "receiver"})
    return trig


# ------------------------------------------------------------ intensity ------
def pressing_intensity(defenders_xy: np.ndarray, defenders_vel: np.ndarray,
                       ball_xy: np.ndarray) -> dict:
    """Nearest-presser distance, number of pressers within 5 m, closing speed."""
    if defenders_xy.shape[0] == 0:
        return {"nearest_m": 99.0, "pressers_5m": 0, "closing_speed": 0.0}
    d_vec = defenders_xy - ball_xy
    dists = np.linalg.norm(d_vec, axis=1)
    order = np.argsort(dists)[:3]
    closing = 0.0
    for i in order:
        if dists[i] < 1e-3:
            continue
        approach_unit = -d_vec[i] / dists[i]           # toward ball
        closing += float(max(0.0, defenders_vel[i] @ approach_unit))
    closing /= len(order)
    return {
        "nearest_m": round(float(dists.min()), 2),
        "pressers_5m": int((dists < 5.0).sum()),
        "closing_speed": round(closing, 2),
    }
