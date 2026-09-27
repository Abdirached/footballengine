#!/usr/bin/env python3
"""Generate a synthetic 20 Hz TEF v1 tracking document.

Simulates a possession phase of ~22 s: home team builds in a 4-3-3 with a
third-man pattern on the left half-space; away defends in a compact 4-4-2
mid-block and presses triggers. Deterministic seed → reproducible demos.
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

L, W = 105.0, 68.0


def formation_433(dir_x: float, shift: float = 0.0):
    """Return [(id,x,y)] relative positions, attacking along dir_x."""
    base = [
        ("GK", -48 * dir_x, 0),
        ("RB", -28 * dir_x, -22), ("RCB", -32 * dir_x, -8),
        ("LCB", -32 * dir_x, 8), ("LB", -28 * dir_x, 22),
        ("DM", -18 * dir_x, 0), ("CM", -6 * dir_x, -12), ("CM2", -6 * dir_x, 12),
        ("RW", 14 * dir_x, -24), ("ST", 18 * dir_x, 0), ("LW", 14 * dir_x, 24),
    ]
    return [(n, x + shift * dir_x, y) for n, x, y in base]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--out", default="data/sample_match.json")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    fps, dur = args.fps, args.seconds
    n = int(dur * fps)

    home = formation_433(+1, shift=-5)
    away = formation_442 = [
        ("GK", 48, 0), ("RB", 26, -20), ("RCB", 30, -7), ("LCB", 30, 7),
        ("LB", 26, 20), ("RM", 8, -20), ("RCM", 12, -7), ("LCM", 12, 7),
        ("LM", 8, 20), ("ST", -6, -6), ("ST2", -6, 6)]

    ids_home = ["H%02d" % i for i in range(1, 12)]
    ids_away = ["A%02d" % i for i in range(1, 12)]

    # scripted ball sequence: build-up -> third-man switch into left channel
    waypoints = [
        (-23, 0, 0.0), (-16, -11, 3.0), (-6, -13, 6.5), (2, 9, 10.0),
        (-2, 24, 13.0), (14, 21, 17.0), (26, 12, 21.0)]
    wp_t = [p[2] for p in waypoints]
    total_end = wp_t[-1] + 3.0

    frames = []
    for k in range(n):
        t = k / fps
        tt = min(t / total_end, 1.0)
        # ball position: piecewise linear through waypoints
        bx = np.interp(t, wp_t, [p[0] for p in waypoints])
        by = np.interp(t, wp_t, [p[1] for p in waypoints])
        if t > wp_t[-1]:
            bx += (t - wp_t[-1]) * 4.0
        bvx = (np.interp(t + 0.05, wp_t, [p[0] for p in waypoints]) -
               np.interp(t - 0.05, wp_t, [p[0] for p in waypoints])) * fps / 2
        bvy = (np.interp(t + 0.05, wp_t, [p[1] for p in waypoints]) -
               np.interp(t - 0.05, wp_t, [p[1] for p in waypoints])) * fps / 2

        players = []
        for i, (name, hx, hy) in enumerate(home):
            # gentle possession shape oscillation + noise
            wobble = 2.5 * math.sin(t * 0.8 + i)
            px = np.clip(hx + wobble * 0.6 + rng.normal(0, 0.15), -51, 51)
            py = np.clip(hy + wobble + rng.normal(0, 0.15), -33, 33)
            if i == 5:  # deep midfielder dribbles/carries at the ball
                px, py = float(np.clip(bx - 1.0, -51, 51)), float(np.clip(by - 0.8, -33, 33))
            vx = vy = 0.0
            if frames:
                prev = frames[-1]["players"][i]
                vx, vy = (px - prev["x"]) * fps, (py - prev["y"]) * fps
            players.append({"id": ids_home[i], "team": "home",
                            "x": round(float(px), 3), "y": round(float(py), 3),
                            "vx": round(float(vx), 3), "vy": round(float(vy), 3)})
        for i, (name, ax, ay) in enumerate(formation_442):
            # block slides toward ball with inertia; presses when ball nears
            pull = np.clip((bx - ax) * 0.12, -6, 6)
            press = 1.0 if abs(bx - ax) < 14 and abs(by - ay) < 14 else 0.35
            px = np.clip(ax + pull * press + rng.normal(0, 0.12), -51, 51)
            py = np.clip(ay + (by - ay) * 0.08 * press + rng.normal(0, 0.12), -33, 33)
            vx = vy = 0.0
            if frames:
                prev = frames[-1]["players"][11 + i]
                vx, vy = (px - prev["x"]) * fps, (py - prev["y"]) * fps
            players.append({"id": ids_away[i], "team": "away",
                            "x": round(float(px), 3), "y": round(float(py), 3),
                            "vx": round(float(vx), 3), "vy": round(float(vy), 3)})

        frames.append({"t": round(t, 3), "period": 1,
                       "ball": {"x": round(float(bx), 3), "y": round(float(by), 3),
                                "z": 0.0, "vx": round(float(bvx), 3),
                                "vy": round(float(bvy), 3)},
                       "players": players})

    doc = {
        "meta": {
            "match_id": "demo-001",
            "fps": fps,
            "pitch": {"length": L, "width": W},
            "teams": {
                "home": {"name": "Alpha FC", "color": "#3b82f6", "attack_dir": 1},
                "away": {"name": "Beta United", "color": "#ef4444", "attack_dir": -1},
            },
        },
        "frames": frames,
    }
    with open(args.out, "w") as fh:
        json.dump(doc, fh)
    print(f"wrote {args.out}: {len(frames)} frames @ {fps} Hz ({dur}s)")


if __name__ == "__main__":
    main()
