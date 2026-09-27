"""Per-frame analysis pipeline: TEF frame → EEF engine frame."""
from __future__ import annotations

import numpy as np

from . import pressing as pr
from . import structure as st
from .evaluator import best_of, eval_bar, evaluate_options, suggest_triangle_move
from .grid_notation import GridSpec, cell_receivability
from .pitch_control import PitchControlModel
from .schema import (BallState, CoverShadow, ControlGrid, DefensiveBlock,
                     EngineFrame, Evaluation, EvalComponents, MoveOption,
                     PressingInfo, PressTrigger, StructureIn, StructureOut,
                     TrackingFrame, TrackingMeta, TriangleUnit)
from .triangles import active_triangles

GK_DEPTH_FROM_GOAL = 12.0   # players within this of own goal treated as GK-ish


class FrameContext:
    """Convenience NumPy views of one tracking frame."""

    def __init__(self, frame: TrackingFrame, meta: TrackingMeta):
        self.frame = frame
        self.meta = meta
        self.home = np.array([[p.x, p.y] for p in frame.players if p.team == "home"])
        self.away = np.array([[p.x, p.y] for p in frame.players if p.team == "away"])
        self.home_ids = [p.id for p in frame.players if p.team == "home"]
        self.away_ids = [p.id for p in frame.players if p.team == "away"]
        self.home_v = np.array([[p.vx, p.vy] for p in frame.players if p.team == "home"])
        self.away_v = np.array([[p.vx, p.vy] for p in frame.players if p.team == "away"])
        self.ball = np.array([frame.ball.x, frame.ball.y])

    def team(self, name: str):
        return (self.home, self.away) if name == "home" else (self.away, self.home)


def _possession(ctx: FrameContext) -> tuple[str, float]:
    """Nearest player to the ball owns it (within 2.5 m), else loose."""
    d_home = float(np.min(np.linalg.norm(ctx.home - ctx.ball, axis=1))) \
        if ctx.home.size else 99
    d_away = float(np.min(np.linalg.norm(ctx.away - ctx.ball, axis=1))) \
        if ctx.away.size else 99
    if min(d_home, d_away) > 2.5:
        return ("loose", min(d_home, d_away))
    return ("home", d_home) if d_home <= d_away else ("away", d_away)


def _carrier_index(ctx: FrameContext, team_name: str) -> int:
    xy, ids = (ctx.home, ctx.home_ids) if team_name == "home" else \
              (ctx.away, ctx.away_ids)
    return int(np.argmin(np.linalg.norm(xy - ctx.ball, axis=1))), ids


def analyse_frame(frame: TrackingFrame, prev: TrackingFrame | None,
                  meta: TrackingMeta, pcm: PitchControlModel,
                  value_grid: np.ndarray, grid: GridSpec | None = None,
                  prev_receiver: dict[str, str] | None = None
                  ) -> EngineFrame:
    """Compute the full EEF payload for a single 50 ms frame.

    `prev_receiver` maps possession-team -> id of last pass receiver, enabling
    third-man detection; callers may thread it across frames.
    """
    ctx = FrameContext(frame, meta)
    poss_team, _d = _possession(ctx)
    home_has_ball = poss_team == "home"

    # ---- pitch control ---------------------------------------------------
    control_grid = pcm.control(ctx.home, ctx.away)
    control_area_home = float(control_grid.mean())
    threat = pcm.threat_at(pcm.value_grid(1),
                           frame.ball.x, frame.ball.y)

    # ---- attacking unit / defending unit ---------------------------------
    if poss_team == "home":
        atk_xy, atk_ids = ctx.home, ctx.home_ids
        def_xy = ctx.away
        attack_dir = meta.teams["home"].attack_dir
    else:
        atk_xy, atk_ids = ctx.away, ctx.away_ids
        def_xy = ctx.home
        attack_dir = meta.teams["away"].attack_dir
    if poss_team == "loose":  # fall back to home as reference unit
        atk_xy, atk_ids = ctx.home, ctx.home_ids

    # exclude keeper from outfield analyses: prefer squad number 1 (sim data);
    # fall back to deepest player heuristic for vendor feeds without numbers.
    own_goal_x = -attack_dir * meta.pitch.length / 2
    nums = [p.num for p in frame.players if p.team ==
            ("home" if poss_team != "away" else "away")]
    outfield_mask = np.ones(atk_xy.shape[0], dtype=bool)
    gk_row = None
    if nums.count(1) == 1:
        gk_row = nums.index(1)
    elif atk_xy.shape[0] >= 8:
        depth = np.abs(atk_xy[:, 0] - own_goal_x)
        cand = int(np.argmin(depth))
        if depth[cand] < GK_DEPTH_FROM_GOAL:
            gk_row = cand
    if gk_row is not None and atk_xy.shape[0] >= 8:
        outfield_mask[gk_row] = False

    out_xy, out_ids = atk_xy[outfield_mask], [i for i, m in zip(atk_ids, outfield_mask) if m]

    carrier_idx_rel = _carrier_index(ctx, poss_team)[0] if poss_team != "loose" \
        else int(np.argmin(np.linalg.norm(atk_xy - ctx.ball, axis=1)))

    struct_in = st.structure_score(out_xy, def_xy, ctx.ball,
                                   meta.pitch.length, meta.pitch.width)

    # ---- triangles --------------------------------------------------------
    carrier_pid = atk_ids[carrier_idx_rel] if poss_team != "loose" else None
    gk_pid = atk_ids[gk_row] if (gk_row is not None) else None
    is_gk_carrier = carrier_pid is not None and carrier_pid == gk_pid
    carrier_out_row = out_ids.index(carrier_pid) \
        if carrier_pid in out_ids else -1
    tri = active_triangles(carrier_out_row, out_xy, def_xy) \
        if poss_team != "loose" and carrier_out_row >= 0 else []

    # ---- pressing ---------------------------------------------------------
    shadows = pr.cover_shadows(def_xy, ctx.ball) if poss_team != "loose" else []
    occ = 0.0
    options: list[dict] = []
    best: dict | None = None

    if poss_team != "loose" and atk_xy.shape[0] >= 2:
        mates_idx = [i for i in range(atk_xy.shape[0])
                     if i != carrier_idx_rel and i != gk_row]
        prev_recv_idx = None
        if prev_receiver and poss_team in prev_receiver:
            rid = prev_receiver[poss_team]
            mate_ids_full = [atk_ids[i] for i in mates_idx]
            if rid in mate_ids_full:
                prev_recv_idx = mate_ids_full.index(rid)

        vg_dir = pcm.value_grid(attack_dir if poss_team == "home" else -attack_dir)
        raw = evaluate_options(
            atk_xy[carrier_idx_rel], atk_xy[mates_idx],
            [atk_ids[i] for i in mates_idx], def_xy,
            pcm, control_grid, vg_dir,
            attack_dir, prev_recv_idx, grid=grid,
        )
        options = [
            MoveOption(tier=o["tier"], symbol=o["symbol"],
                       action="pass", player=carrier_pid,
                       to_cell=o.get("to_cell"),
                       from_x=float(ctx.ball[0]), from_y=float(ctx.ball[1]),
                       to_x=o["to"][0], to_y=o["to"][1],
                       to_player=o["to_player"], completion=o["completion"],
                       value=o["value"], lines_broken=o["lines_broken"],
                       reason=o["reason"])
            for o in raw
        ]
        best = best_of(raw)

        # no active triangle through the carrier → headline move is a run
        if (not tri) and grid is not None and not is_gk_carrier \
                and len(mates_idx) >= 2:
            mv = suggest_triangle_move(
                atk_xy[carrier_idx_rel], atk_xy[mates_idx],
                [atk_ids[i] for i in mates_idx], def_xy, grid)
            if mv is not None:
                best = mv
                options.insert(0, MoveOption(
                    tier="best", symbol="!", action="move",
                    player=mv["player"], to_cell=mv["to_cell"],
                    from_x=float(atk_xy[carrier_idx_rel][0]),
                    from_y=float(atk_xy[carrier_idx_rel][1]),
                    to_x=mv["to"][0], to_y=mv["to"][1],
                    to_player=None, completion=1.0,
                    value=mv["value"], lines_broken=0,
                    reason=mv["reason"]))

        # occlusion on the headline option's lane
        if best is not None:
            b_pt = np.array(best["to"])
            occ = pr.lane_occlusion_pct(ctx.ball, b_pt, shadows)

    intensity = pr.pressing_intensity(def_xy, ctx.away_v if poss_team == "home"
                                       else ctx.home_v, ctx.ball)

    triggers: list[PressTrigger] = []
    if prev is not None:
        recv_xy = recv_v = None
        if poss_team != "loose":
            rrow = carrier_idx_rel
            recv_xy, recv_v = atk_xy[rrow], (ctx.home_v[rrow] if poss_team == "home"
                                             else ctx.away_v[rrow])
        prev_bv = np.array([prev.ball.vx, prev.ball.vy])
        bv = np.array([frame.ball.vx, frame.ball.vy])
        for t in pr.detect_triggers(prev_bv, ctx.ball, bv, recv_xy, recv_v,
                                    def_xy, attack_dir):
            triggers.append(PressTrigger(**t))

    # ---- defensive block (opponent of possession) --------------------------
    def_own_goal_x = attack_dir * meta.pitch.length / 2
    block = st.defensive_block_metrics(def_xy, def_own_goal_x,
                                       meta.pitch.length, meta.pitch.width)
    struct_out_score = st.block_score(block)
    struct_in_score = struct_in["score"]

    score = eval_bar(control_area_home, struct_in_score if home_has_ball
                     else struct_out_score,
                     struct_out_score if home_has_ball else struct_in_score,
                     threat, home_has_ball)

    best_move = None
    if best is not None:
        is_move = best.get("action") == "move"
        origin = atk_xy[carrier_idx_rel] if poss_team != "loose" else ctx.ball
        best_move = MoveOption(
            tier=best["tier"], symbol=best["symbol"],
            action="move" if is_move else "pass",
            player=best.get("player") if is_move else atk_ids[carrier_idx_rel]
            if poss_team != "loose" else None,
            to_cell=best.get("to_cell"),
            from_x=float(origin[0]), from_y=float(origin[1]),
            to_x=best["to"][0], to_y=best["to"][1],
            to_player=None if is_move else best["to_player"],
            completion=best["completion"], value=best["value"],
            lines_broken=best["lines_broken"], reason=best["reason"])

    cells_payload = None
    if grid is not None and poss_team != "loose":
        cells_vals = cell_receivability(grid, pcm, control_grid, poss_team,
                                        ctx.ball, def_xy)
        cells_payload = ControlGrid(nx=12, ny=8, values=cells_vals)

    return EngineFrame(
        t=frame.t,
        possession=poss_team,                      # type: ignore[arg-type]
        carrier_id=atk_ids[carrier_idx_rel] if poss_team != "loose" else None,
        carrier_cell=(grid.cell_name(frame.ball.x, frame.ball.y)
                      if grid is not None else None),
        eval_=Evaluation(score=score, components=EvalComponents(
            control_area=round(control_area_home, 3),
            structure_in=struct_in_score,
            structure_out=struct_out_score,
            threat=round(threat, 3))),
        best_move=best_move,
        options=options,
        triangles=[TriangleUnit(players=[out_ids[i] for i in t["players"]],
                                pts=t["pts"], score=t["score"]) for t in tri],
        control_grid=ControlGrid(nx=pcm.nx, ny=pcm.ny,
                                 values=np.round(control_grid.ravel(), 3).tolist()),
        cells=cells_payload,
        structure={
            poss_team if poss_team != "loose" else "home": StructureIn(
                score=struct_in["score"], width=struct_in["width"],
                depth=struct_in["depth"], corridors=struct_in["corridors"],
                penalties=struct_in["penalties"], bonuses=struct_in["bonuses"]),
            ("away" if poss_team == "home" else "home") if poss_team != "loose"
            else "away": StructureOut(score=struct_out_score,
                                      block=DefensiveBlock(**block)),
        },
        pressing=PressingInfo(
            shadows=[CoverShadow(**s) for s in shadows],
            lane_occlusion_pct=occ,
            intensity=intensity,
            triggers=triggers),
    )


# ------------------------------------------------------------------ batch ----
def analyse_match(match) -> list[EngineFrame]:
    """Analyse a full MatchTracking document into EEF frames."""
    meta = match.meta
    pcm = PitchControlModel(meta.pitch.length, meta.pitch.width)
    vg = pcm.value_grid(1)
    grid = GridSpec(meta.pitch.length, meta.pitch.width)
    out: list[EngineFrame] = []
    prev: TrackingFrame | None = None
    prev_receiver: dict[str, str] | None = None
    for f in match.frames:
        ef = analyse_frame(f, prev, meta, pcm, vg, grid, prev_receiver)
        if ef.best_move is not None:
            prev_receiver = {ef.possession: ef.best_move.to_player} \
                if ef.best_move.to_player else None
        out.append(ef)
        prev = f
    return out
