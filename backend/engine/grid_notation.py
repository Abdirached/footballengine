"""Chess-style pitch notation: 12 files (a–l) × 8 ranks (1–8) = 96 named cells.

Conventions (fixed for the whole match — turnovers do NOT flip names):
  • Files `a`→`l` run from home's own goal line toward the opponent goal
    (home attacks along +x by convention).
  • Ranks `1`→`8` run from home's LEFT touchline to home's RIGHT touchline,
    as seen by a home player facing the opponent goal. Facing +x means the
    left hand points to +y, so rank 1 hugs y=+W/2 and rank 8 hugs y=−W/2.
  • A cell is ~8.75 × 8.5 m — approximately one outfield player's actionable
    control radius ("can receive here and do something with the ball").
  • Piece notation: `H8g8` = Home #8 moves to g8; carrier annotated `●e4`.

Receivability per cell aggregates the engine's fine 2 m physics grid:
    P(receive & act) ≈ w1·pitch_control(possession)
                     + w2·lane_openness(carrier → cell centre)
                     + w3·action_space(distance of nearest opponent)
"""
from __future__ import annotations

import numpy as np

FILES = "abcdefghijkl"
N_FILES = len(FILES)          # 12 columns along x
N_RANKS = 8                   # rows across y


class GridSpec:
    """Named-cell geometry over a standard pitch."""

    def __init__(self, length: float = 105.0, width: float = 68.0):
        self.length = length
        self.width = width
        self.cw = length / N_FILES      # 8.75 m
        self.ch = width / N_RANKS       # 8.5 m

    # ------------------------------------------------------------- naming --
    def file_of(self, x: float) -> int:
        fi = int((x + self.length / 2) // self.cw)
        return int(min(N_FILES - 1, max(0, fi)))

    def rank_of(self, y: float) -> int:
        ri = int(((self.width / 2 - y)) // self.ch)   # +y edge -> rank 1
        return int(min(N_RANKS - 1, max(0, ri)))

    def cell_name(self, x: float, y: float) -> str:
        return f"{FILES[self.file_of(x)]}{self.rank_of(y) + 1}"

    def cell_index(self, name: str) -> int:
        """Flat index row-major by rank then file: idx = rank*12 + file."""
        f, r = self.parse(name)
        return r * N_FILES + f

    @staticmethod
    def parse(name: str) -> tuple[int, int]:
        """'g5' -> (file_idx 6, rank_idx 4)."""
        name = name.strip().lower()
        if len(name) != 2 or name[0] not in FILES or not name[1].isdigit():
            raise ValueError(f"bad cell name {name!r}")
        return FILES.index(name[0]), int(name[1]) - 1

    def cell_center(self, name: str) -> np.ndarray:
        f, r = self.parse(name)
        x = -self.length / 2 + (f + 0.5) * self.cw
        y = self.width / 2 - (r + 0.5) * self.ch          # rank 1 at +y edge
        return np.array([x, y])

    def all_names(self) -> list[str]:
        out = []
        for r in range(N_RANKS):
            for f in range(N_FILES):
                out.append(f"{FILES[f]}{r + 1}")
        return out


# ------------------------------------------------------- receivability -------
def cell_receivability(grid: GridSpec,
                       pcm,                      # PitchControlModel
                       control_grid: np.ndarray, # home-positive control field
                       possession: str,
                       ball_xy: np.ndarray,
                       opponents_xy: np.ndarray,
                       w_ctrl: float = 0.45,
                       w_lane: float = 0.35,
                       w_space: float = 0.20) -> list[float]:
    """Aggregate P(receive & act) into the 96 named cells (order: see all_names).

    ctrl  — mean possession-team control across the cell's fine sub-cells
    lane  — lane openness from the current ball position to the cell centre
    space — nearest-opponent clearance at the cell centre (saturating ~7 m)
    """
    from .geometry import lane_openness_batch

    centers = np.array([grid.cell_center(n) for n in grid.all_names()])   # (96,2)
    # vectorised control sampling (nearest cell of the fine grid)
    gx = np.clip(((centers[:, 0] + pcm.length / 2) / pcm.length * pcm.nx).astype(int),
                 0, pcm.nx - 1)
    gy = np.clip(((centers[:, 1] + pcm.width / 2) / pcm.width * pcm.ny).astype(int),
                 0, pcm.ny - 1)
    ctrl = control_grid[gy, gx]
    if possession == "away":
        ctrl = 1.0 - ctrl
    lanes = lane_openness_batch(ball_xy.astype(float), centers,
                                opponents_xy.astype(float))
    if opponents_xy.size:
        d_opp = np.linalg.norm(centers[:, None, :] - opponents_xy[None, :, :],
                               axis=2).min(axis=1)
    else:
        d_opp = np.full(len(centers), 30.0)
    space = 1.0 - np.exp(-np.maximum(0.0, d_opp - 2.0) / 3.0)
    v = w_ctrl * ctrl + w_lane * lanes + w_space * space
    return [round(float(x), 3) for x in np.clip(v, 0.0, 1.0)]


def top_cells(values: list[float], k: int = 3) -> list[tuple[str, float]]:
    """Best-k cells by receivability -> [(name, value)]."""
    names = GridSpec().all_names()
    pairs = sorted(zip(names, values), key=lambda p: p[1], reverse=True)
    return [(n, v) for n, v in pairs[:k]]
