"""Stockfish-style move evaluator: candidate passes → tiers + global eval bar.

Tiers
  brilliant  "!!"  unobvious third-man / big switch that disorganises the block
  best       "!"   top-scoring option; breaks ≥1 line at low risk
  good             safe retention / structure-preserving shift
  inaccuracy "?!"  low completion or redundant (negative marginal value)
  blunder    "?"   high risk into dense zone, likely turnover

Global eval bar (−10…+10, home-positive) blends:
  control_area  — pitch-control share of possession team
  structure_in  — attacking structure score (0…100)
  structure_out — opponent block quality (0…100)
  threat        — xT-like threat of ball position
"""
from __future__ import annotations

import numpy as np

from .geometry import (count_lines_broken, lane_openness, lane_openness_batch,
                       lane_risk)
from .grid_notation import GridSpec
from .pitch_control import PitchControlModel


# ------------------------------------------------------------ candidates -----
def evaluate_options(carrier_xy: np.ndarray, teammates_xy: np.ndarray,
                     teammate_ids: list[str], opponents_xy: np.ndarray,
                     pcm: PitchControlModel, control_grid: np.ndarray,
                     value_grid: np.ndarray, attack_dir: int,
                     prev_receiver_idx: int | None = None,
                     max_pass_m: float = 45.0, min_pass_m: float = 3.0,
                     top_n: int = 5, grid: GridSpec | None = None) -> list[dict]:
    """Score every viable pass from the carrier and tier the results."""
    opts: list[dict] = []
    for i, mate in enumerate(teammates_xy):
        vec = mate - carrier_xy
        dist = float(np.linalg.norm(vec))
        if dist < min_pass_m or dist > max_pass_m:
            continue

        openn = lane_openness(carrier_xy, mate, opponents_xy)
        ctrl = pcm.control_at(control_grid, float(mate[0]), float(mate[1]))
        completion = float(np.clip(0.65 * openn + 0.35 * ctrl, 0.0, 1.0))

        v_now = pcm.threat_at(value_grid, float(carrier_xy[0]), float(carrier_xy[1]))
        v_next = pcm.threat_at(value_grid, float(mate[0]), float(mate[1]))
        progress = v_next - v_now

        lines = count_lines_broken(carrier_xy, mate, opponents_xy)
        lateral_switch = abs(float(vec[1])) > 25.0 and abs(float(vec[0])) < 20.0

        # third-man signal: previous receiver is nearby & drawing pressure,
        # making this (different) target less obvious to defenders
        third_man = (prev_receiver_idx is not None
                     and 0 <= prev_receiver_idx < len(teammates_xy)
                     and prev_receiver_idx != i
                     and float(np.linalg.norm(
                         teammates_xy[prev_receiver_idx] - carrier_xy)) < 14.0)

        value = (
            progress * 30.0                       # positional gain (xT units)
            + lines * 1.4                         # line-breaking premium
            + (2.2 if lateral_switch else 0.0)    # switches stretch the block
            + (2.6 if third_man else 0.0)         # unobvious pattern bonus
            + completion * 3.0                    # reliability component
        )

        # density around receiving point (blunder detector)
        if opponents_xy.size:
            near_opp = int((np.linalg.norm(opponents_xy - mate, axis=1) < 6.0).sum())
        else:
            near_opp = 0

        reason_bits = []
        if lines:
            reason_bits.append(f"breaks {lines} line(s)")
        if lateral_switch:
            reason_bits.append("switches play")
        if third_man:
            reason_bits.append("third-man geometry")
        if progress > 0.05:
            reason_bits.append("progressive")

        opts.append({
            "to_player": teammate_ids[i],
            "to": [float(mate[0]), float(mate[1])],
            "to_cell": grid.cell_name(float(mate[0]), float(mate[1])) if grid else None,
            "completion": round(completion, 3),
            "value": round(value, 2),
            "lines_broken": lines,
            "near_defenders": near_opp,
            "third_man": bool(third_man),
            "switch": bool(lateral_switch),
            "reason": ", ".join(reason_bits) or "retention",
        })

    opts.sort(key=lambda o: o["value"], reverse=True)

    if not opts:
        return []

    best_val = opts[0]["value"]
    for rank, o in enumerate(opts):
        if o["completion"] < 0.35 or (o["near_defenders"] >= 3 and o["lines_broken"] == 0):
            o["tier"] = "blunder"
            o["symbol"] = "?"
        elif o["completion"] < 0.5:
            o["tier"] = "inaccuracy"
            o["symbol"] = "?!"
        elif o["third_man"] or (o["switch"] and o["lines_broken"] >= 1):
            o["tier"] = "brilliant"
            o["symbol"] = "!!"
        elif rank == 0 and o["lines_broken"] >= 1 and o["completion"] >= 0.7:
            o["tier"] = "best"
            o["symbol"] = "!"
        elif o["completion"] >= 0.75 and o["value"] >= best_val - 2.0:
            o["tier"] = "good"
            o["symbol"] = ""
        else:
            o["tier"] = "good"
            o["symbol"] = ""

    return opts[:top_n]


def best_of(options: list[dict]) -> dict | None:
    """Pick the headline move: brilliant > best > good, avoiding blunders."""
    order = {"brilliant": 0, "best": 1, "good": 2, "inaccuracy": 3, "blunder": 4}
    playable = [o for o in options if order[o["tier"]] <= 2]
    if not playable:
        return options[0] if options else None
    playable.sort(key=lambda o: (order[o["tier"]], -o["value"]))
    return playable[0]


# ------------------------------------------------- triangle-forming moves ----
def suggest_triangle_move(carrier_xy: np.ndarray, mates_xy: np.ndarray,
                          mate_ids: list[str], opponents_xy: np.ndarray,
                          grid: GridSpec, min_side: float = 5.0,
                          max_side: float = 30.0, open_thresh: float = 0.6
                          ) -> dict | None:
    """When no active triangle includes the carrier, propose the highest-value
    off-ball movement that completes one.

    Geometry is mover-independent, so all lane checks are computed once as
    batched NumPy ops over (third-vertex × candidate-cell) pairs; the mover is
    then chosen as the nearest valid outfielder to the winning cell.
    """
    n_m = mates_xy.shape[0]
    if n_m < 2:
        return None

    names = grid.all_names()
    centers = np.array([grid.cell_center(nm) for nm in names])
    d_car = np.linalg.norm(centers - carrier_xy, axis=1)
    keep = (d_car >= min_side) & (d_car <= max_side)
    if not keep.any():
        return None
    cell_pts = centers[keep]
    cell_names = [nm for nm, k in zip(names, keep) if k]

    # openness carrier -> each candidate destination (one batched call)
    opA = lane_openness_batch(carrier_xy, cell_pts, opponents_xy)
    ok_cells = opA >= open_thresh
    if not ok_cells.any():
        return None

    thirds = mates_xy
    d_third_car = np.linalg.norm(thirds - carrier_xy, axis=1)
    opB = lane_openness_batch(carrier_xy, thirds, opponents_xy)      # (J,)
    # pairwise third->cell segments in a single batched call
    J, C = thirds.shape[0], cell_pts.shape[0]
    seg_from = np.repeat(thirds, C, axis=0)              # (J*C,2)
    seg_to = np.tile(cell_pts, (J, 1))                   # (J*C,2)
    opC = lane_openness_batch(seg_from[0], seg_to, opponents_xy).reshape(J, C)
    d_third_cell = np.linalg.norm(cell_pts[None, :, :] - thirds[:, None, :],
                                  axis=2)                # (J,C)

    sides_a = d_car[keep][None, :]                       # carrier->cell (1,C)
    sides_b = d_third_cell                               # third->cell   (J,C)
    sides_c = d_third_car[:, None] * np.ones((1, C))     # carrier->third(1,C)->(J,C)

    geo_ok = (
        (sides_b >= min_side) & (sides_b <= max_side) &
        (sides_c >= min_side) & (sides_c <= max_side) &
        ok_cells[None, :] & (opB[:, None] >= open_thresh) & (opC >= open_thresh))
    if not geo_ok.any():
        return None

    lo = np.minimum(np.minimum(sides_a, sides_b), sides_c)
    hi = np.maximum(np.maximum(sides_a, sides_b), sides_c)
    equil = 1.0 - (hi - lo) / max_side
    score = ((opA[None, :] + opB[:, None] + opC) / 3.0) * 0.7 + equil * 0.3
    score = np.where(geo_ok, score, -1.0)
    j, c = np.unravel_index(int(score.argmax()), score.shape)

    # mover: closest valid teammate to the target cell (excluding the vertex j)
    dist_to_cell = np.linalg.norm(mates_xy - cell_pts[c], axis=1)
    dist_to_cell[j] = np.inf
    i = int(dist_to_cell.argmin())

    return {
        "action": "move",
        "tier": "best",
        "symbol": "!",
        "player": mate_ids[i],
        "to_player": None,
        "to": [float(cell_pts[c][0]), float(cell_pts[c][1])],
        "to_cell": cell_names[c],
        "completion": 1.0,
        "value": round(4.0 + float(score[j, c]) * 4.0, 2),
        "lines_broken": 0,
        "reason": f"forms passing triangle into {cell_names[c]}",
    }


# ------------------------------------------------------------- eval bar ------
def eval_bar(control_area_home: float, structure_in_home: float,
             structure_out_away: float, threat_home: float,
             home_has_ball: bool) -> float:
    """Compose instantaneous advantage on a −10…+10 scale (home-positive).

    When away is in possession the polarity flips so the bar always reads from
    home's perspective.
    """
    s = (
        5.0 * (control_area_home - 0.5) * 2.0     # ±5 space dominance
        + 0.03 * (structure_in_home - structure_out_away)   # ±3 structure edge
        + 2.0 * threat_home                        # +0…2 immediate threat
    )
    s = float(np.clip(s, -10.0, 10.0))
    return round(s if home_has_ball else -s, 2)
