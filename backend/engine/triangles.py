"""Dynamic Triangle Engine — Delaunay-based active passing-triangle detection.

An *active triangle* is a Delaunay facet of the in-possession outfield players that:
  • has all sides within [min_side, max_side] metres,
  • includes the ball carrier as a vertex (direct participation), and
  • every edge is an open pass lane (openness ≥ threshold).

Triangles are scored by mean lane openness × angle quality (~60° equilateral ideal)
and returned best-first.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay, QhullError

from .geometry import angle_deg, lane_openness


def active_triangles(carrier_idx: int, team_xy: np.ndarray, opponents_xy: np.ndarray,
                     min_side: float = 4.0, max_side: float = 30.0,
                     open_thresh: float = 0.55, top_n: int = 3
                     ) -> list[dict]:
    """Return up to `top_n` active triangle units for the possession team.

    Returns dicts: {players: [i,j,k], pts: [[x,y]×3], score}.
    Indices refer to rows of `team_xy` (carrier_idx included).
    """
    n = team_xy.shape[0]
    if n < 3:
        return []
    try:
        tri = Delaunay(team_xy)
    except QhullError:
        return []

    results = []
    for simplex in tri.simplices:
        i, j, k = (int(v) for v in simplex)
        if carrier_idx not in (i, j, k):
            continue
        pts = team_xy[[i, j, k]]
        edges = [(pts[0], pts[1]), (pts[1], pts[2]), (pts[2], pts[0])]
        lens = [float(np.linalg.norm(b - a)) for a, b in edges]
        if min(lens) < min_side or max(lens) > max_side:
            continue

        opens = [lane_openness(a, b, opponents_xy) for a, b in edges]
        if min(opens) < open_thresh:
            continue

        angles = [
            angle_deg(pts[1] - pts[0], pts[2] - pts[0]),
            angle_deg(pts[0] - pts[1], pts[2] - pts[1]),
            angle_deg(pts[0] - pts[2], pts[1] - pts[2]),
        ]
        # angle quality: 1 at 60° each, decaying toward degenerate shapes
        ang_q = float(np.mean([max(0.0, 1.0 - abs(a - 60.0) / 60.0) for a in angles]))
        score = float(np.mean(opens) * 0.6 + ang_q * 0.4)

        results.append({
            "players": sorted([i, j, k]),
            "pts": pts.tolist(),
            "score": round(score, 3),
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_n]
