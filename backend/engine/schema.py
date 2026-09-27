"""TEF v1 (input tracking) and EEF v1 (computed engine output) schemas."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class TeamInfo(BaseModel):
    name: str = "Team"
    color: str = "#888888"
    attack_dir: Literal[1, -1] = 1


class PitchSpec(BaseModel):
    length: float = 105.0
    width: float = 68.0


class TrackingMeta(BaseModel):
    match_id: str
    fps: int = 20
    pitch: PitchSpec = PitchSpec()
    teams: dict[str, TeamInfo] = Field(
        default_factory=lambda: {
            "home": TeamInfo(name="Home", color="#3b82f6", attack_dir=1),
            "away": TeamInfo(name="Away", color="#ef4444", attack_dir=-1),
        }
    )


class BallState(BaseModel):
    x: float
    y: float
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0


class PlayerState(BaseModel):
    id: str
    team: Literal["home", "away"]
    num: int = 0                     # squad number (chess-piece label)
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0


class TrackingFrame(BaseModel):
    t: float
    period: int = 1
    ball: BallState
    players: List[PlayerState]


class MatchTracking(BaseModel):
    """TEF v1 document."""
    meta: TrackingMeta
    frames: List[TrackingFrame]


# ---------------------------------------------------------------- EEF v1 -----

class EvalComponents(BaseModel):
    control_area: float
    structure_in: float
    structure_out: float
    threat: float


class Evaluation(BaseModel):
    score: float                      # −10…+10, positive favours home
    components: EvalComponents


class MoveOption(BaseModel):
    tier: Literal["best", "good", "brilliant", "inaccuracy", "blunder"]
    symbol: str                       # "!" best, "!!" brilliant, "?!" inaccuracy, "?" blunder
    action: Literal["pass", "move"] = "pass"
    player: Optional[str] = None      # acting player (carrier for passes, mover for moves)
    to_cell: Optional[str] = None     # chess-style destination cell, e.g. "g8"
    from_x: float
    from_y: float
    to_x: float
    to_y: float
    to_player: Optional[str] = None   # pass receiver (None for movements)
    completion: float = 1.0           # P(pass completes) 0…1 (movement ⇒ 1)
    value: float                      # composite tactical gain
    lines_broken: int = 0
    reason: str = ""


class TriangleUnit(BaseModel):
    players: List[str]
    pts: List[List[float]]            # [[x,y]×3]
    score: float


class ControlGrid(BaseModel):
    nx: int
    ny: int
    values: List[float]


class StructureIn(BaseModel):
    score: float
    width: float
    depth: float
    corridors: List[int]              # occupancy of 5 vertical corridors
    penalties: dict[str, float]
    bonuses: dict[str, float]


class DefensiveBlock(BaseModel):
    compactness: float                # hull area / pitch area
    line_gap_max: float
    def_line_height: float            # distance own goal -> deepest defender line


class StructureOut(BaseModel):
    score: float
    block: DefensiveBlock


class CoverShadow(BaseModel):
    x: float
    y: float
    dir_x: float
    dir_y: float
    half_angle_deg: float


class PressTrigger(BaseModel):
    type: Literal["back_to_goal", "heavy_touch", "touchline_trap", "pressure_spike"]
    player: Optional[str]


class PressingInfo(BaseModel):
    shadows: List[CoverShadow]
    lane_occlusion_pct: float
    intensity: dict[str, float]
    triggers: List[PressTrigger]


class EngineFrame(BaseModel):
    """EEF v1 document — one per 50 ms tracking frame."""
    t: float
    possession: Literal["home", "away", "loose"]
    carrier_id: Optional[str]
    carrier_cell: Optional[str] = None       # e.g. "e4" — ball's chess cell
    eval_: Evaluation = Field(alias="eval")
    best_move: Optional[MoveOption]
    options: List[MoveOption]
    triangles: List[TriangleUnit]
    control_grid: ControlGrid
    cells: Optional[ControlGrid] = None       # 12×8 receivability map
    structure: dict[str, StructureIn | StructureOut]
    pressing: PressingInfo

    model_config = ConfigDict(populate_by_name=True)


class EngineResult(BaseModel):
    meta: TrackingMeta
    frames: List[EngineFrame]
