/** EEF v1 + TEF v1 wire types (mirror of backend/engine/schema.py). */

export type Team = "home" | "away";
export type MoveTier = "best" | "good" | "brilliant" | "inaccuracy" | "blunder";

export interface TeamInfo {
  name: string;
  color: string;
  attack_dir: 1 | -1;
}

export interface TrackingMeta {
  match_id: string;
  fps: number;
  pitch: { length: number; width: number };
  teams: { home: TeamInfo; away: TeamInfo };
}

export interface PlayerFrameState {
  id: string;
  team: Team;
  num: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
}

export interface BallFrameState {
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
}

export interface TrackingFrame {
  t: number;
  period: number;
  ball: BallFrameState;
  players: PlayerFrameState[];
}

export interface MatchTrackingDoc {
  meta: TrackingMeta;
  frames: TrackingFrame[];
}

export interface EvalComponents {
  control_area: number;
  structure_in: number;
  structure_out: number;
  threat: number;
}

export interface Evaluation {
  score: number; // −10…+10
  components: EvalComponents;
}

export interface MoveOption {
  tier: MoveTier;
  symbol: string;
  action: "pass" | "move";
  player: string | null;
  to_cell: string | null;
  from_x: number;
  from_y: number;
  to_x: number;
  to_y: number;
  to_player: string | null;
  completion: number;
  value: number;
  lines_broken: number;
  reason: string;
}

export interface TriangleUnit {
  players: string[];
  pts: [number, number][];
  score: number;
}

export interface ControlGrid {
  nx: number;
  ny: number;
  values: number[];
}

export interface StructureIn {
  score: number;
  width: number;
  depth: number;
  corridors: number[];
  penalties: Record<string, number>;
  bonuses: Record<string, number>;
}

export interface DefensiveBlock {
  compactness: number;
  line_gap_max: number;
  def_line_height: number;
}

export interface StructureOut {
  score: number;
  block: DefensiveBlock;
}

export interface CoverShadow {
  x: number;
  y: number;
  dir_x: number;
  dir_y: number;
  half_angle_deg: number;
}

export interface PressTrigger {
  type: "back_to_goal" | "heavy_touch" | "touchline_trap" | "pressure_spike";
  player: string | null;
}

export interface PressingInfo {
  shadows: CoverShadow[];
  lane_occlusion_pct: number;
  intensity: { nearest_m: number; pressers_5m: number; closing_speed: number };
  triggers: PressTrigger[];
}

export interface EngineFrame {
  t: number;
  possession: Team | "loose";
  carrier_id: string | null;
  carrier_cell: string | null;
  eval: Evaluation;
  best_move: MoveOption | null;
  options: MoveOption[];
  triangles: TriangleUnit[];
  control_grid: ControlGrid;
  cells: ControlGrid | null;
  structure: Record<string, StructureIn | StructureOut>;
  pressing: PressingInfo;
}

export interface OverlayToggles {
  triangles: boolean;
  spaceControl: boolean;
  pressingLanes: boolean;
  engineBestMove: boolean;
  cellGrid: boolean;
}
