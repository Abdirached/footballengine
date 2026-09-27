"""Event-driven 20 Hz match simulator — the 5-minute two-way demo generator.

Theory-first design:
  • Attacking brain is TRIANGLE-FIRST (in cell coordinates): if the carrier has
    no active passing triangle, the best move becomes an off-ball *movement*
    into a triangle-forming cell, and the sim actually executes it.
  • Defensive brain presses the ball's cell plus the top-K dangerous
    progression cells (receivability-ranked).
  • Turnovers emerge from real pass-lane risk rolls + pressure dispossession,
    so possession flows both ways.

Outputs standard TEF v1 (MatchTracking) + a human-readable event feed used by
the UI move ticker.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import lane_openness
from .grid_notation import FILES, GridSpec
from .schema import BallState, MatchTracking, PlayerState, TeamInfo, TrackingFrame, TrackingMeta
from .triangles import active_triangles

V_PASS_SHORT = 14.0     # m/s inside ~15 m
V_PASS_LONG = 20.0      # m/s beyond
V_SHOT = 26.0
A_MAX = 9.0             # m/s^2 accel limit
V_OUT = 5.2             # outfield cruise speed
V_GK = 4.0

HOME_433 = [  # (role, num, fx from own goal 0..1, fy -1..1)
    ("GK", 1, 0.045, 0.0),
    ("RB", 2, 0.215, -0.66), ("RCB", 4, 0.175, -0.24),
    ("LCB", 5, 0.175, 0.24), ("LB", 3, 0.215, 0.66),
    ("DM", 6, 0.335, 0.0), ("CM", 8, 0.445, -0.36), ("CM", 10, 0.445, 0.36),
    ("RW", 7, 0.655, -0.72), ("ST", 9, 0.700, 0.0), ("LW", 11, 0.655, 0.72),
]
AWAY_4231 = [
    ("GK", 1, 0.045, 0.0),
    ("RB", 2, 0.205, -0.62), ("RCB", 4, 0.170, -0.22),
    ("LCB", 5, 0.170, 0.22), ("LB", 3, 0.205, 0.62),
    ("DM", 6, 0.320, -0.16), ("DM", 8, 0.320, 0.16),
    ("AM", 10, 0.480, 0.0), ("RW", 7, 0.605, -0.68), ("LW", 11, 0.605, 0.68),
    ("ST", 9, 0.665, 0.0),
]


@dataclass
class SimPlayer:
    pid: str
    team: str                 # home | away
    num: int
    role: str
    dir: int                  # attacking direction ±1
    base: np.ndarray          # formation anchor in metres
    pos: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    order: np.ndarray | None = None   # explicit movement target (metres)
    is_gk: bool = False


class MatchSim:
    def __init__(self, seconds: float = 300.0, fps: int = 20, seed: int = 7):
        self.fps = fps
        self.dt = 1.0 / fps
        self.seconds = seconds
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.grid = GridSpec()
        self.players: list[SimPlayer] = []
        self.ball_pos = np.zeros(2)
        self.ball_vel = np.zeros(2)
        self.ball_z = 0.0
        self.state = "lineup"            # lineup | held | pass | loose | shot
        self.carrier: SimPlayer | None = None
        self.possession = "home"
        self.pass_t = 0.0
        self.pass_T = 1e-9
        self.pass_from = np.zeros(2)
        self.pass_to = np.zeros(2)
        self.pass_receiver: SimPlayer | None = None
        self.intercept_at_s: float | None = None
        self.brain_cd = 0.0
        self.hold_until = 3.2            # kickoff lineup hold
        self.recover_until = 0.0         # rest-defence retreat after turnover
        self.score = {"home": 0, "away": 0}
        self.events: list[dict] = []
        self._build_teams()

    # ------------------------------------------------------------- setup ----
    def _build_teams(self):
        L, W = 105.0, 68.0
        for team, tpl in (("home", HOME_433), ("away", AWAY_4231)):
            d = 1 if team == "home" else -1
            for role, num, fx, fy in tpl:
                base = np.array([d * (-L / 2 + fx * L), fy * (W / 2 - 3.0)])
                pid = f"{'H' if team == 'home' else 'A'}{num:02d}"
                self.players.append(SimPlayer(
                    pid=pid, team=team, num=num, role=role, dir=d,
                    base=base.copy(), pos=base.copy(),
                    is_gk=(role == "GK")))

    def _team(self, name: str) -> list[SimPlayer]:
        return [p for p in self.players if p.team == name]

    def _opponents(self, name: str) -> list[SimPlayer]:
        return self._team("away" if name == "home" else "home")

    # ------------------------------------------------------------- kickoff --
    def reset_kickoff(self, kicking_team: str, t: float):
        for p in self.players:
            p.pos = p.base.copy()
            p.vel = np.zeros(2)
            p.order = None
        kicker = next(p for p in self._team(kicking_team) if p.num == 9)
        helper = next(p for p in self._team(kicking_team) if p.num == 6)
        off = 0.8 if kicking_team == "home" else -0.8
        kicker.pos = np.array([off, -0.6])
        helper.pos = np.array([off + 1.6 * np.sign(off or 1) * -1, 0.6])
        self.possession = kicking_team
        self.carrier = kicker
        self.state = "held"
        self.hold_until = t + 2.8
        self.ball_pos = np.array([off, -0.6])
        self.ball_vel = np.zeros(2)
        self.ball_z = 0.0

    # ------------------------------------------------------------- helpers --
    def _nearest(self, point: np.ndarray, pool: list[SimPlayer]) -> SimPlayer:
        return min(pool, key=lambda p: float(np.linalg.norm(p.pos - point)))

    # Per-role mobility bands: how far (m) a player may shift forward/back
    # from his formation base and laterally toward the ball. Football is semi-
    # structured: centre backs barely move, strikers roam.
    SHIFT = {   # role: (x_band_m, y_band_m)
        "GK":  (0.0, 0.0),
        "RB":  (8.0, 6.0), "LB": (8.0, 6.0),
        "RCB": (3.5, 4.0), "LCB": (3.5, 4.0),
        "DM":  (6.0, 5.0),
        "CM":  (9.0, 7.0), "AM": (9.0, 7.0),
        "RW":  (10.0, 4.0), "LW": (10.0, 4.0),
        "ST":  (12.0, 8.0),
    }

    def _anchor(self, p: SimPlayer) -> np.ndarray:
        """Role-banded formation anchor — hold position, shift nearby space."""
        sx, sy = self.SHIFT[p.role]
        bx = float(np.clip(self.ball_pos[0], -52, 52))
        by = float(np.clip(self.ball_pos[1], -34, 34))
        a = p.base.copy()
        if not p.is_gk:
            if self.possession == p.team:
                adv = max(0.0, bx * p.dir)                 # ball past halfway?
                push = min(2.0 + adv * 0.22, sx)
                if p.role in ("RCB", "LCB"):               # CBs only step to
                    push = min(push, 3.5) if adv < 10 else min(2.0 + adv * 0.16, sx)
                a[0] += p.dir * push
            else:
                drop = min(1.5 + max(0.0, -bx * p.dir) * 0.10, sx * 0.6)
                if self.t_now < self.recover_until:        # rest-defence retreat
                    drop += 4.0
                a[0] -= p.dir * drop
            a[1] += float(np.clip(by - a[1], -sy, sy)) * \
                (0.35 if self.possession == p.team else 0.45)
        return np.clip(a, [-51, -32.5], [51, 32.5])

    def _move(self, p: SimPlayer, target: np.ndarray, v_max: float):
        d = target - p.pos
        dist = float(np.linalg.norm(d))
        want = d / dist * min(v_max, dist / self.dt * 0.5) if dist > 1e-4 \
            else np.zeros(2)
        dv = want - p.vel
        n = float(np.linalg.norm(dv))
        if n > A_MAX * self.dt:
            dv = dv / n * A_MAX * self.dt
        p.vel = p.vel + dv

    def _dangerous_cells(self) -> list[np.ndarray]:
        """(Retired) danger-cell marking removed in favour of role-band shape."""
        return []

    # --------------------------------------------------------- attack brain -
    def attack_brain(self):
        c = self.carrier
        if c is None:
            return
        mates = [p for p in self._team(c.team) if p is not c]
        opps = self._opponents(c.team)
        mate_xy = np.array([m.pos for m in mates])
        opp_xy = np.array([o.pos for o in opps])
        team_all = np.array([c.pos] + [m.pos for m in mates])

        tri = active_triangles(0, team_all, opp_xy)

        tri_exists = any(0 in t["players"] for t in tri)

        # ---- final-third central cells → shoot sometimes --------------------
        cell_name = self.grid.cell_name(c.pos[0], c.pos[1])
        cx_file = FILES.index(cell_name[0])
        rank = int(cell_name[1])
        if cx_file >= 9 and 3 <= rank <= 6 and self.rng.random() < 0.35:
            self._shoot(c)
            return

        # ---- no triangle through the carrier → usually best move = form one -
        if not tri_exists:
            mover, target = self._triangle_fix(c, mates, opp_xy)
            if mover is not None and self.rng.random() < 0.7:
                mover.order = target
                self.events.append({
                    "t": round(self.t_now, 2), "type": "move",
                    "team": c.team, "player": mover.pid,
                    "cell": self.grid.cell_name(*target),
                    "note": f"{mover.pid} moves to "
                            f"{self.grid.cell_name(*target)} to form triangle"})
                return
            # otherwise: play a safe pass anyway (retention under pressure)
        # ---- triangle alive: pick a pass ------------------------------------
        opts = self._pass_options(c, mates, opp_xy)
        if not opts:
            return                                   # keep carrying
        weights = np.array([0.62, 0.26, 0.12, 0.06][:len(opts)])
        weights = weights / weights.sum()
        pick = self.rng.choice(len(opts), p=weights)
        opt = opts[int(pick)]
        receiver = opt["receiver"]
        dist = opt["dist"]
        T = dist / (V_PASS_SHORT if dist < 15 else V_PASS_LONG)
        self.state = "pass"
        self.pass_from = c.pos.copy()
        lead = receiver.vel * T * 0.55
        self.pass_to = np.clip(receiver.pos + lead, [-51.5, -33.5], [51.5, 33.5])
        self.pass_to[1] = float(self.pass_to[1])
        self.pass_receiver = receiver
        self.pass_t = 0.0
        self.pass_T = max(T, 1e-3)
        self.carrier = None
        # interception roll from true lane risk
        risk = 1.0 - lane_openness(self.pass_from, self.pass_to, opp_xy)
        self.intercept_at_s = None
        if self.rng.random() < risk * 0.85:
            self.intercept_at_s = float(self.rng.uniform(0.35, 0.85))
        self.events.append({
            "t": round(self.t_now, 2), "type": "pass", "team": c.team,
            "player": c.pid, "to": receiver.pid,
            "cell": self.grid.cell_name(*self.pass_to),
            "completed": self.intercept_at_s is None})

    def _shoot(self, c: SimPlayer):
        goal_x = c.dir * 52.5
        aim = np.array([goal_x, float(self.rng.uniform(-3.4, 3.4))])
        dist = float(np.linalg.norm(aim - c.pos))
        self.state = "shot"
        self.pass_from = c.pos.copy()
        self.pass_to = aim
        self.pass_t = 0.0
        self.pass_T = dist / V_SHOT
        self.intercept_at_s = None
        self.shot_outcome = self.rng.choice(
            ["goal", "save", "wide"], p=[0.42, 0.38, 0.20])
        self.pass_receiver = None
        self.events.append({"t": round(self.t_now, 2), "type": "shot",
                            "team": c.team, "player": c.pid})

    def _triangle_fix(self, c: SimPlayer, mates: list[SimPlayer],
                      opp_xy: np.ndarray):
        """Nearest teammate whose run into a nearby open cell completes a triangle."""
        best = (None, None, -1.0)
        for m in mates:
            if m.is_gk or m.order is not None:
                continue
            for r in range(8):
                for fi in range(12):
                    cell_c = self.grid.cell_center(f"{FILES[fi]}{r + 1}")
                    # semi-structured: runners only leave their zone briefly
                    if float(np.linalg.norm(cell_c - m.pos)) > 15.0:
                        continue
                    d_car = float(np.linalg.norm(cell_c - c.pos))
                    if not (5.0 <= d_car <= 30.0):
                        continue
                    # some third teammate within triangle range of both?
                    for o in mates:
                        if o is m or o.is_gk:
                            continue
                        d1 = float(np.linalg.norm(cell_c - o.pos))
                        d2 = float(np.linalg.norm(o.pos - c.pos))
                        if not (5.0 <= d1 <= 30.0 and 5.0 <= d2 <= 30.0):
                            continue
                        op = lane_openness(c.pos, cell_c, opp_xy)
                        if op < 0.6:
                            continue
                        score = op + 0.05 * (30.0 - abs(d_car - 14.0))
                        if score > best[2]:
                            best = (m, cell_c, score)
        return best[0], best[1]

    def _pass_options(self, c: SimPlayer, mates: list[SimPlayer],
                      opp_xy: np.ndarray):
        opts = []
        for m in mates:
            if m.is_gk:
                continue
            vec = m.pos - c.pos
            dist = float(np.linalg.norm(vec))
            if dist < 5.0 or dist > 40.0:
                continue
            comp = lane_openness(c.pos, m.pos, opp_xy)
            progress = (m.pos[0] - c.pos[0]) * c.dir
            lateral = abs(vec[1])
            score = comp * 2.2 + progress * 0.06 + lateral * 0.03 \
                + self.rng.normal(0, 0.25)
            if comp < 0.45:
                continue
            opts.append({"receiver": m, "dist": dist, "completion": comp,
                         "score": score})
        opts.sort(key=lambda o: o["score"], reverse=True)
        return opts[:4]

    # -------------------------------------------------------- defense brain -
    def defense_brain(self):
        """Semi-structured pressing: the 1-2 nearest defenders chase the ball
        (handled in update_players); every other defender holds his role band,
        sliding compactly toward the ball side via _anchor — no cross-field
        chasing, no marker orders."""
        return

    # ------------------------------------------------------------ physics ---
    def update_players(self):
        def_team = "away" if self.possession == "home" else "home"
        recovering = self.t_now < self.recover_until
        pressers = [] if recovering else \
            sorted([p for p in self._team(def_team) if not p.is_gk],
                   key=lambda p: float(np.linalg.norm(p.pos - self.ball_pos)))[:2]
        for p in self.players:
            v_max = V_GK if p.is_gk else V_OUT + (0.6 if p in pressers else 0)
            if p.is_gk:
                gx = -p.dir * 50.0
                gy = float(np.clip(self.ball_pos[1] * 0.28, -8.5, 8.5))
                rush = 3.5 if abs(self.ball_pos[0] + p.dir * 46) < 14 else 0.0
                self._move(p, np.array([gx + p.dir * rush, gy]), v_max)
            elif p.order is not None:
                self._move(p, p.order, v_max + 1.2)
                if np.linalg.norm(p.order - p.pos) < 1.0:
                    p.order = None
            elif p in pressers:
                self._move(p, self.ball_pos, v_max)
            else:
                self._move(p, self._anchor(p), v_max)
            p.pos += p.vel * self.dt
            p.pos = np.clip(p.pos, [-51.5, -33.5], [51.5, 33.5])

    def update_ball(self):
        if self.state == "held" and self.carrier is not None:
            spd = float(np.linalg.norm(self.carrier.vel))
            u = self.carrier.vel / spd if spd > 0.5 else \
                np.array([self.carrier.dir, 0.0])
            self.ball_pos = self.carrier.pos + u * 0.9
            self.ball_vel = self.carrier.vel.copy()
            self.ball_z = 0.0
        elif self.state in ("pass", "shot"):
            self.pass_t += self.dt
            s = min(1.0, self.pass_t / self.pass_T)
            self.ball_pos = self.pass_from + (self.pass_to - self.pass_from) * s
            arc = 0.0
            if self.state == "pass" and float(np.linalg.norm(
                    self.pass_to - self.pass_from)) > 24.0:
                arc = 4.5 * np.sin(np.pi * s)
            self.ball_z = arc if self.state == "pass" else max(
                0.0, 2.0 * np.sin(np.pi * min(1.0, s)) *
                (1 if self.shot_outcome != "wide" else 0.4))
            self.ball_vel = (self.pass_to - self.pass_from) / self.pass_T

            if self.state == "shot":
                if s >= 1.0:
                    self._resolve_shot()
            elif self.intercept_at_s is not None and s >= self.intercept_at_s:
                self._intercept_pass()
            elif s >= 1.0:
                assert self.pass_receiver is not None
                self.carrier = self.pass_receiver
                self.possession = self.carrier.team
                self.state = "held"

    def _intercept_pass(self):
        thief_pool = self._opponents(self.pass_receiver.team) \
            if self.pass_receiver else self.players
        pt = self.ball_pos.copy()
        thief = self._nearest(pt, [p for p in thief_pool if not p.is_gk])
        self.carrier = thief
        self.possession = thief.team
        self.state = "held"
        self.recover_until = self.t_now + 2.0
        self.events.append({"t": round(self.t_now, 2), "type": "turnover",
                            "team": thief.team, "player": thief.pid})

    def _resolve_shot(self):
        outcome = getattr(self, "shot_outcome", "wide")
        scoring_team = self.possession
        if outcome == "goal":
            self.score[scoring_team] += 1
            self.events.append({"t": round(self.t_now, 2), "type": "goal",
                                "team": scoring_team,
                                "score": f"{self.score['home']}-{self.score['away']}"})
            conceding = "away" if scoring_team == "home" else "home"
            self.reset_kickoff(conceding, self.t_now)
            return
        gk_unused = None  # (kept minimal — GK handled via defending team lookup)
        defending = "away" if scoring_team == "home" else "home"
        dgk = next(p for p in self._team(defending) if p.is_gk)
        if outcome == "save":
            self.ball_pos = np.array([-dgk.dir * 49.5, dgk.pos[1]])
            self.carrier = dgk
            self.possession = defending
            self.state = "held"
        else:  # wide -> goal kick
            self.ball_pos = np.array([-dgk.dir * 47.0, 9.0])
            self.carrier = dgk
            self.possession = defending
            self.state = "held"
        self.events.append({"t": round(self.t_now, 2), "type": outcome,
                            "team": defending})

    # ---------------------------------------------------------------- main --
    def run(self) -> tuple[MatchTracking, list[dict]]:
        n = int(self.seconds * self.fps)
        frames: list[TrackingFrame] = []
        self.t_now = 0.0
        self.reset_kickoff("home", 0.0)
        prev_pos: dict[str, np.ndarray] = {}

        for k in range(n):
            self.t_now = k * self.dt

            if self.state == "lineup" or self.t_now < self.hold_until:
                pass                                        # stand in formation
            else:
                self.brain_cd -= self.dt
                if self.brain_cd <= 0.0 and self.state in ("held",):
                    # pressure dispossession chance on the carrier
                    # (GK carriers skip the steal roll but still play the ball)
                    c = self.carrier
                    stole = False
                    if c is not None and not c.is_gk:
                        opps = self._opponents(c.team)
                        nd = min(float(np.linalg.norm(o.pos - c.pos)) for o in opps)
                        if nd < 1.4 and self.rng.random() < 0.30:
                            thief = self._nearest(c.pos, [o for o in opps
                                                          if not o.is_gk])
                            self.carrier = thief
                            self.possession = thief.team
                            self.recover_until = self.t_now + 2.0
                            self.events.append({
                                "t": round(self.t_now, 2), "type": "turnover",
                                "team": thief.team, "player": thief.pid})
                            stole = True
                    if c is not None and not stole:
                        self.attack_brain()
                        if self.state == "held":
                            self.defense_brain()
                    self.brain_cd = 0.4

                self.update_players()
                self.update_ball()

            players = []
            for p in self.players:
                pv = (p.pos - prev_pos[p.pid]) / self.dt \
                    if p.pid in prev_pos else np.zeros(2)
                prev_pos[p.pid] = p.pos.copy()
                players.append(PlayerState(
                    id=p.pid, team=p.team, num=p.num,
                    x=round(float(p.pos[0]), 3), y=round(float(p.pos[1]), 3),
                    vx=round(float(pv[0]), 3), vy=round(float(pv[1]), 3)))

            bv = (self.ball_vel).tolist()
            frames.append(TrackingFrame(
                t=round(self.t_now, 3), period=1,
                ball=BallState(x=round(float(self.ball_pos[0]), 3),
                               y=round(float(self.ball_pos[1]), 3),
                               z=round(float(self.ball_z), 3),
                               vx=round(float(bv[0]), 3),
                               vy=round(float(bv[1]), 3)),
                players=players))

        meta = TrackingMeta(
            match_id=f"demo-{int(self.seconds)}s-seed{self.seed}",
            fps=self.fps,
            teams={"home": TeamInfo(name="Alpha FC", color="#3b82f6", attack_dir=1),
                   "away": TeamInfo(name="Beta United", color="#ef4444", attack_dir=-1)})
        return MatchTracking(meta=meta, frames=frames), self.events


def simulate(seconds: float = 300.0, fps: int = 20, seed: int = 7
             ) -> tuple[MatchTracking, list[dict]]:
    return MatchSim(seconds=seconds, fps=fps, seed=seed).run()
