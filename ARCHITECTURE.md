# FootballEngine — Architecture & System Design

Tactical analysis engine for 20 Hz optical tracking data (50 ms frames): positional-play
structure scoring, dynamic triangle detection, pitch control, pressing/cover-shadow
analytics and a Stockfish-style move evaluator (Best / Good / Brilliant / Inaccuracy /
Blunder) with a live −10…+10 evaluation bar.

---

## 1. Tech stack & justification

| Layer | Choice | Why |
|---|---|---|
| Backend engine | **Python 3.12 + FastAPI + NumPy/SciPy/Pandas** | Delaunay/Voronoi/ConvexHull are one-liners in `scipy.spatial` (C-speed, Qhull). Vectorised NumPy evaluates pitch control over a ~7 k-cell grid × 22 players per frame in <5 ms. Rust/WASM would win on cold-start client compute but loses on iteration speed, scientific-library maturity and server-side batch re-analysis of full matches. Heavy per-frame work is precomputed once and streamed — the client never recomputes. |
| API | **FastAPI + Uvicorn** | Native async WebSockets for frame streaming, Pydantic = schema-as-code (TEF/EEF v1), OpenAPI docs free. |
| Frontend | **Next.js 14 (App Router) + TypeScript + Tailwind** | RSC shell + client-only canvas island; TS types mirror EEF v1 exactly. |
| Visualization | **PixiJS (WebGL2), dedicated 2D “dot board”** | See §2. |
| Data format | **TEF v1 JSON** (normalised) with adapters for Metrica Sports CSV, SkillCorner open JSON, Second Spectrum JSON | Normalise vendor quirks at ingest; the engine sees one schema. |

## 2. Visualisation decision: dot board vs. video overlay

Chosen: **dedicated 2D WebGL mini-map board** (FIFA-HUD style).

* **Latency/precision.** Video overlay demands camera calibration (homography from
  broadcast feed, lens distortion, pan/tilt compensation). Errors of 0.5–1 m are typical
  and *move with the camera*, corrupting Voronoi/pitch-control maths that need absolute
  metres. A dot board consumes raw tracking coordinates directly → exact geometry.
* **Rendering cost.** 22 player nodes + ball + vector overlays at 60 fps ≈ <200 sprites;
  PixiJS batches this trivially. Video overlay would additionally decode/composite an H.264
  stream per frame (WebCodecs) — high GPU cost, sync drift between media clock and data clock.
* **Product scope.** Overlay is a *presentation* feature; the engine is presentation-agnostic.
  The board ships first; an `<video>`+homography layer can reuse the same EEF overlays later.

Interpolation: backend stores native 20 Hz frames; frontend runs a fixed-timestep
accumulator and renders at display rate (60 fps) by lerping positions between adjacent
frames — smooth motion without inflating payload size.

## 3. Data pipeline

```
Vendor files (Metrica/SkillCorner/Second Spectrum)
        │  loaders.py  (normalise to TEF v1)
        ▼
MatchTracking (20 Hz frames: 22 players + ball, velocities)
        │  pipeline.py  (one pass, NumPy-vectorised per frame)
        ├── geometry.py      passing lanes, line-breaks, hulls
        ├── pitch_control.py Spearman-style control grid + xT value grid
        ├── triangles.py     Delaunay active-triangle units
        ├── structure.py     structural score (in/out of possession)
        ├── pressing.py      cover shadows, occlusion %, press triggers
        └── evaluator.py     candidate-pass scoring → move tiers + eval bar
        ▼
EEF v1 EngineFrames[]  ──REST /ws──▶  Next.js client
                                        ├─ PixiJS PitchBoard (60 fps lerp)
                                        ├─ EvalBar (−10…+10)
                                        └─ ControlPanel (scrub ±50 ms, toggles)
```

Performance model: ~4–8 ms/frame ⇒ a 60 s clip (1200 frames) analyses in ≲10 s,
then plays back instantly from cache (`/api/match/{id}/engine`).

## 4. Schemas

### 4.1 TEF v1 — input tracking frame (normalised)

Coordinates: metres, origin at pitch centre, `x ∈ [−52.5, 52.5]`, `y ∈ [−34, 34]`,
home attacks `+x`.

```jsonc
{
  "meta": {
    "match_id": "demo-001",
    "fps": 20,
    "pitch": { "length": 105, "width": 68 },
    "teams": {
      "home": { "name": "Alpha FC", "color": "#3b82f6", "attack_dir": 1 },
      "away": { "name": "Beta United", "color": "#ef4444", "attack_dir": -1 }
    }
  },
  "frames": [{
    "t": 12.35,                    // seconds since period start (50 ms steps)
    "period": 1,
    "ball":   { "x": 8.2, "y": -6.1, "z": 0.05, "vx": 3.1, "vy": -0.4 },
    "players": [
      { "id": "H07", "team": "home", "x": 9.9, "y": -7.0, "vx": 1.2, "vy": 0.8 }
      // … 22 entries
    ]
  }]
}
```

### 4.2 EEF v1 — computed engine frame (server output)

```jsonc
{
  "t": 12.35,
  "possession": "home",
  "carrier_id": "H07",
  "eval": {                       // evaluation bar source
    "score": 3.42,                // −10…+10, home-positive
    "components": {
      "control_area": 0.58,       // fraction of pitch controlled by possession team
      "structure_in": 74.1,       // 0…100 structural quality of team-in-possession
      "structure_out": 41.0,      // 0…100 defensive-block quality of opponents
      "threat": 0.31              // xT-style threat of current ball position
    }
  },
  "best_move": {
    "tier": "brilliant",          // best | good | brilliant | inaccuracy | blunder
    "symbol": "!!",
    "from": {"x": 9.9, "y": -7.0},
    "to":   {"x": 24.0, "y": 12.0},
    "to_player": "H11",
    "completion": 0.87,
    "value": 6.9,                 // composite tactical gain
    "lines_broken": 2,
    "reason": "Third-man pass behind midfield line; switch disorganises block"
  },
  "options": [ /* top N candidates, same shape as best_move */ ],
  "triangles": [
    { "players": ["H07","H04","H11"], "pts": [[9.9,-7.0],[2.0,10.5],[24.0,12.0]], "score": 0.81 }
  ],
  "control_grid": {               // coarse pitch-control field for heat overlay
    "nx": 53, "ny": 34,
    "values": [ /* ny*nx floats, 0=away … 1=home */ ]
  },
  "structure": {
    "home": { "score": 74.1, "width": 52.0, "depth": 38.5, "corridors": [1,1,1,1,1],
              "penalties": {"stacking": 4.0}, "overload_zone": "left-half-space" },
    "away": { "block": { "compactness": 0.18, "line_gap_max": 11.5,
                          "def_line_height": 28.0 } }
  },
  "pressing": {
    "shadows": [ {"x": -3.0, "y": 2.0, "dir_x": 0.94, "dir_y": -0.33, "half_angle_deg": 51} ],
    "lane_occlusion_pct": 44.0,
    "intensity": { "nearest_m": 1.8, "pressers_5m": 2, "closing_speed": 4.1 },
    "triggers": [ {"type": "back_to_goal", "player": "H07"} ]
  }
}
```

### 4.3 REST / WS API

| Route | Purpose |
|---|---|
| `POST /api/upload` | Upload TEF/vendor file → `{match_id}` (analysed lazily/on demand) |
| `GET  /api/match/{id}` | Meta + frame count |
| `GET  /api/match/{id}/engine?from=&to=` | Batch EEF frames (initial load) |
| `WS   /api/match/{id}/stream?from=t` | Streams EEF frames live while analysing |
| `POST /api/demo/generate` | Generate synthetic demo match |

## 5. Repository layout

```
backend/
  main.py                 FastAPI app (upload, batch, ws, demo)
  engine/
    schema.py             Pydantic TEF/EEF models
    loaders.py            Metrica / SkillCorner / TEF adapters
    geometry.py           lanes, line-break counting, hull helpers
    pitch_control.py      control grid + xT value grid
    triangles.py          Delaunay triangle engine
    structure.py          structure & defensive-block scores
    pressing.py           cover shadows, triggers, occlusion
    evaluator.py          move tiers + eval bar composition
    pipeline.py           per-frame orchestration
scripts/generate_sample_match.py
frontend/                 Next.js 14 + PixiJS board
```
