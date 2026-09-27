"""Structural Evaluation Score (chess 'pawn-structure' analogue).

In-possession score (0…100) for the team on the ball:
  rewards — corridor coverage (5 vertical lanes), depth+width spread,
            local numerical overloads near the ball;
  penalties — vertical/horizontal stacking, isolated players,
              over-congested clusters ('pawn islands') away from the ball.

Out-of-possession block metrics: compactness (hull area ratio), max internal
line gap, defensive-line height.
"""
from __future__ import annotations

import numpy as np

from .geometry import corridor_index, hull_area


# --------------------------------------------------------------------- in ----
def structure_score(team_xy: np.ndarray, opponents_xy: np.ndarray, ball_xy: np.ndarray,
                    pitch_length: float = 105.0, pitch_width: float = 68.0
                    ) -> dict:
    """Return {score, width, depth, corridors, penalties{}, bonuses{}}."""
    n = team_xy.shape[0]
    pens: dict[str, float] = {}
    bons: dict[str, float] = {}

    width = float(team_xy[:, 1].max() - team_xy[:, 1].min()) if n else 0.0
    depth = float(team_xy[:, 0].max() - team_xy[:, 0].min()) if n else 0.0

    # --- width / depth rewards (scaled 0…15 each) -------------------------
    bons["width"] = round(15.0 * min(1.0, width / 50.0), 2)
    bons["depth"] = round(15.0 * min(1.0, depth / 40.0), 2)

    # --- corridor coverage (5 vertical lanes) -----------------------------
    corridors = [0] * 5
    for p in team_xy:
        corridors[corridor_index(p[1], pitch_width)] += 1
    covered = sum(1 for c in corridors if c > 0)
    bons["corridors"] = round(20.0 * covered / 5.0, 2)

    # --- stacking penalties ------------------------------------------------
    stack = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            same_corridor = corridor_index(team_xy[i][1], pitch_width) == \
                            corridor_index(team_xy[j][1], pitch_width)
            near_same_line = abs(team_xy[i][0] - team_xy[j][0]) < 6.0
            if same_corridor and near_same_line:
                stack += 1.0
    pens["stacking"] = round(min(20.0, stack * 4.0), 2)

    # --- isolation penalty (min teammate distance > 25 m) ------------------
    iso = 0.0
    if n >= 2:
        d = np.linalg.norm(team_xy[:, None, :] - team_xy[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        min_d = d.min(axis=1)
        iso = float((min_d > 25.0).sum())
    pens["isolation"] = round(min(12.0, iso * 6.0), 2)

    # --- congestion ('pawn islands' >3 players clustered away from ball) ---
    congest = 0.0
    if n >= 4:
        d = np.linalg.norm(team_xy[:, None, :] - team_xy[None, :, :], axis=2)
        adj = d < 10.0
        visited = np.zeros(n, dtype=bool)
        for s in range(n):
            if visited[s]:
                continue
            stack_, q = [s], [s]
            visited[s] = True
            while q:
                u = q.pop()
                for v_ in np.where(adj[u] & ~visited)[0]:
                    visited[v_] = True
                    stack_.append(int(v_))
                    q.append(int(v_))
            cluster = team_xy[stack_]
            far_from_ball = float(np.linalg.norm(cluster - ball_xy, axis=1).mean()) > 22.0
            if len(stack_) > 3 and far_from_ball:
                congest += 1.0
    pens["congestion"] = round(min(10.0, congest * 5.0), 2)

    # --- local overload bonus near ball ------------------------------------
    r_zone = 18.0
    atk_n = int((np.linalg.norm(team_xy - ball_xy, axis=1) < r_zone).sum())
    def_n = int((np.linalg.norm(opponents_xy - ball_xy, axis=1) < r_zone).sum())
    surplus = atk_n - def_n
    zone = ("centre" if abs(ball_xy[1]) < 11.33 else
            "left-half-space" if ball_xy[1] < 0 else "right-half-space")
    bons["overload_" + zone.replace("-", "_")] = round(min(18.0, max(0.0, surplus) * 6.0), 2)
    if surplus < 0:
        pens["underload"] = round(min(8.0, -surplus * 4.0), 2)

    base = 30.0 + sum(bons.values()) - sum(pens.values())
    return {
        "score": round(float(np.clip(base, 0.0, 100.0)), 1),
        "width": round(width, 1),
        "depth": round(depth, 1),
        "corridors": corridors,
        "penalties": pens,
        "bonuses": bons,
    }


# -------------------------------------------------------------------- out ----
def defensive_block_metrics(def_xy: np.ndarray, own_goal_x: float,
                            pitch_length: float = 105.0, pitch_width: float = 68.0
                            ) -> dict:
    """Compactness / line-gap / block-height metrics for the defending unit."""
    n = def_xy.shape[0]
    area = hull_area(def_xy)
    compactness = area / (pitch_length * pitch_width)

    line_gap_max, def_height = 0.0, 0.0
    if n:
        xs = np.sort(def_xy[:, 0])
        gaps = np.diff(xs)
        line_gap_max = float(gaps.max()) if gaps.size else 0.0
        # deepest line toward own goal
        deepest = xs.min() if own_goal_x < 0 else xs.max()
        def_height = abs(deepest - own_goal_x)

    return {
        "compactness": round(compactness, 3),
        "line_gap_max": round(line_gap_max, 1),
        "def_line_height": round(float(def_height), 1),
    }


def block_score(block: dict) -> float:
    """Defensive-block quality 0…100 (higher = better organised).

    Good: tight compactness, small internal gaps, sane height (not suicidal,
    not passive)."""
    comp_pen = min(50.0, block["compactness"] * 250.0)          # sprawled block
    gap_pen = min(25.0, max(0.0, block["line_gap_max"] - 8.0) * 2.5)
    h = block["def_line_height"]
    h_pen = 0.0 if 8.0 <= h <= 35.0 else min(15.0, (abs(h - 21.5) - 13.5) * 1.2)
    return round(float(np.clip(100.0 - comp_pen - gap_pen - h_pen, 0.0, 100.0)), 1)
