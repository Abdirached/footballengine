# FootballEngine

Tactical analysis engine for 20 Hz optical tracking data — positional-play structure
scoring, dynamic triangle detection, pitch control, pressing/cover-shadow analytics,
and a Stockfish-style move evaluator with a live −10…+10 evaluation bar.

## Quick start

```bash
# 1) Backend  (Python 3.12)
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# 2) Frontend (Node 20+)
cd ../frontend
npm install
npm run dev          # http://localhost:3000
```

Open http://localhost:3000 → **Load Demo Match** generates a synthetic 25 s possession
phase, analyses every 50 ms frame server-side, and streams the result to a PixiJS board.

## What you get

* **Dot-board renderer** (WebGL/PixiJS, 60 fps interpolated from native 20 Hz)
* **Overlays:** active passing triangles · pitch-control heatmap · cover-shadow cones +
  press triggers · Best/Brilliant move arrows with option ticker
* **Eval bar** blending space dominance, structure scores and xT-style threat
* **Controls:** play/pause, ±50 ms stepping, 0.25–4× speed, scrubbing, overlay toggles
* **REST + WebSocket API** for your own data (`POST /api/upload` accepts TEF v1 or
  SkillCorner JSON; Metrica CSV via `/api/upload/metrica`)

## Tests

```bash
cd backend && python -m pytest tests -q     # geometry / control / pipeline smoke tests
```

## Layout

See `ARCHITECTURE.md` for stack justification, data-pipeline design and the full
TEF v1 (input) / EEF v1 (engine output) schemas.
