"""Loaders normalising vendor formats into TEF v1.

Supported:
  • TEF v1 JSON (native)
  • Metrica Sports sample data (CSV: home/away player + ball columns, pitch 105×68,
    origin bottom-left → converted to centre-origin metres)
  • SkillCorner open match data (JSON: frames[].player_data with x/y in metres,
    centre origin — passes through nearly directly)
"""
from __future__ import annotations

import csv
import io
import json

import numpy as np

from .schema import BallState, MatchTracking, PlayerState, TeamInfo, TrackingFrame, TrackingMeta


def load_tef_json(raw: bytes) -> MatchTracking:
    return MatchTracking.model_validate_json(raw)


# ------------------------------------------------------------- Metrica ------
def _metrica_team_frames(path_or_bytes, team_prefix: str):
    """Parse one Metrica home/away CSV into {frame_id: {pid: (x,y)}}."""
    if isinstance(path_or_bytes, bytes):
        rows = csv.DictReader(io.StringIO(path_or_bytes.decode("utf-8")))
    else:
        rows = csv.DictReader(open(path_or_bytes, encoding="utf-8"))
    out: dict[int, dict[str, tuple[float, float]]] = {}
    for r in rows:
        fid = int(r["Frame"])
        players = {}
        for col, val in r.items():
            if col.startswith(team_prefix) and col.endswith(":x") and val:
                pid = col.split(":")[0]
                ycol = f"{pid}:y"
                # Metrica normalised coords, origin top-left of their pitch
                x = float(val) * 105.0 - 52.5
                y = float(r[ycol]) * 68.0 - 34.0
                players[pid.replace("Player", "P")] = (x, y)
        out[fid] = players
    return out


def load_metrica_csv(home_csv: bytes, away_csv: bytes, match_id: str = "metrica") -> MatchTracking:
    H = _metrica_team_frames(home_csv, "Home")
    A = _metrica_team_frames(away_csv, "Away")
    meta = TrackingMeta(
        match_id=match_id, fps=25,
        teams={"home": TeamInfo(name="Home", color="#3b82f6"),
               "away": TeamInfo(name="Away", color="#ef4444")})
    frames: list[TrackingFrame] = []
    for i, fid in enumerate(sorted(H)):
        t = fid / 25.0
        players = [PlayerState(id=pid, team="home", x=x, y=y)
                   for pid, (x, y) in sorted(H[fid].items())]
        if fid in A:
            players += [PlayerState(id=pid, team="away", x=x, y=y)
                        for pid, (x, y) in sorted(A[fid].items())]
        # Metrica sample has no ball file here; synthesise at pitch centre if absent
        frames.append(TrackingFrame(t=t, period=1,
                                    ball=BallState(x=0, y=0), players=players))
    return MatchTracking(meta=meta, frames=frames)


# ---------------------------------------------------------- SkillCorner -----
def load_skillcorner_json(raw: bytes, match_id: str | None = None) -> MatchTracking:
    doc = json.loads(raw)
    mid = match_id or str(doc.get("match_id", "skillcorner"))
    fps = int(doc.get("frequency", 10)) or 10
    pl_meta = {p["trackable_object"]: p for p in doc.get("players", [])}
    frames: list[TrackingFrame] = []
    for fr in doc.get("frames", []):
        t = float(fr.get("time", 0.0))
        players: list[PlayerState] = []
        ball = BallState(x=0.0, y=0.0)
        for pd in fr.get("player_data", []):
            to = pd.get("trackable_object")
            x, y = float(pd["x"]), float(pd["y"])
            info = pl_meta.get(to, {})
            team = "home" if info.get("team_name", "home").lower().startswith(
                doc.get("home_team", {}).get("name", "h")[:4].lower() or "h") else "away"
            players.append(PlayerState(id=f"{team[0].upper()}{to}", team=team,  # type: ignore[index]
                                       x=x, y=y))
        for bd in fr.get("ball_data", []):
            if bd.get("time") is not None or True:
                ball = BallState(x=float(bd["x"]), y=float(bd["y"]),
                                 z=float(bd.get("z", 0) or 0))
                break
        frames.append(TrackingFrame(t=t, period=int(fr.get("period", 1)),
                                    ball=ball, players=players))
    meta = TrackingMeta(match_id=mid, fps=fps)
    return MatchTracking(meta=meta, frames=frames)


# ------------------------------------------------------------ dispatch ------
def auto_load(filename: str, raw: bytes) -> MatchTracking:
    name = filename.lower()
    if name.endswith(".json"):
        head = raw[:512].decode("utf-8", errors="ignore")
        if '"frames"' in head or '"player_data"' in head or '"match_id"' in head:
            return load_skillcorner_json(raw)
        return load_tef_json(raw)
    raise ValueError("Provide TEF/SkillCorner .json; for Metrica upload both "
                     "home/away CSVs via /api/upload/metrica")
