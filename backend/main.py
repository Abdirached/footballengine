"""FootballEngine API — upload, analyse, stream engine frames."""
from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from engine.loaders import auto_load, load_tef_json
from engine.pipeline import analyse_match
from engine.schema import EngineFrame, MatchTracking

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

app = FastAPI(title="FootballEngine", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# match_id -> (MatchTracking, [EngineFrame])
_STORE: dict[str, tuple[MatchTracking, list[EngineFrame]]] = {}
# match_id -> simulator event feed (passes / moves / goals / turnovers)
_EVENTS: dict[str, list[dict]] = {}


def _get(match_id: str) -> tuple[MatchTracking, list[EngineFrame]]:
    if match_id not in _STORE:
        raise HTTPException(404, f"match {match_id!r} not loaded")
    return _STORE[match_id]


def _register(match: MatchTracking) -> str:
    mid = match.meta.match_id or uuid.uuid4().hex[:8]
    cache = DATA_DIR / f"{mid}.eef.json.gz"
    if cache.exists():
        import gzip
        eng = [EngineFrame.model_validate(d) for d in
               json.loads(gzip.decompress(cache.read_bytes()))]
    else:
        eng = analyse_match(match)
        try:
            import gzip
            cache.write_bytes(gzip.compress(
                json.dumps([json.loads(f.model_dump_json(by_alias=True))
                            for f in eng]).encode()))
        except OSError:
            pass
    _STORE[mid] = (match, eng)
    return mid


# ------------------------------------------------------------------ routes ---
@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        match = auto_load(file.filename or "upload.json", raw)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, f"parse error: {e}") from e
    return {"match_id": _register(match), "frames": len(match.frames)}


@app.post("/api/upload/tef")
async def upload_tef(file: UploadFile = File(...)):
    raw = await file.read()
    match = load_tef_json(raw)
    return {"match_id": _register(match), "frames": len(match.frames)}


@app.post("/api/demo/generate")
def demo(seconds: float = 300.0, seed: int = 7):
    """Generate a synthetic two-way match (triangle-first theory) and analyse it."""
    from engine.sim import simulate
    match, events = simulate(seconds=min(600.0, max(5.0, seconds)), seed=seed)
    mid = _register(match)          # heavy CPU — runs in threadpool (sync route)
    _EVENTS[mid] = events
    return {"match_id": mid, "frames": len(match.frames)}


@app.get("/api/match/{match_id}")
async def meta(match_id: str):
    match, eng = _get(match_id)
    return {"meta": match.meta.model_dump(), "frames": len(eng),
            "duration_s": match.frames[-1].t if match.frames else 0}


@app.get("/api/match/{match_id}/tracking")
async def tracking(match_id: str):
    match, _ = _get(match_id)
    return json.loads(match.model_dump_json())


@app.get("/api/match/{match_id}/events")
async def events(match_id: str):
    if match_id not in _EVENTS:
        raise HTTPException(404, "no event feed for match")
    return {"events": _EVENTS[match_id]}


@app.get("/api/match/{match_id}/engine")
async def engine_frames(match_id: str, from_t: float = 0.0,
                        to_t: float | None = None, grid_every: int = 1):
    """Batch EEF frames. grid_every>1 keeps control grids only every Nth frame
    (≈90% payload cut; heat overlay refreshes at 5 Hz — visually identical)."""
    _, eng = _get(match_id)
    to_t = to_t if to_t is not None else 10**9
    sel = [(k, f) for k, f in enumerate(eng) if from_t <= f.t <= to_t]
    out = []
    for k, f in sel:
        d = json.loads(f.model_dump_json(by_alias=True))
        if grid_every > 1 and k % grid_every != 0:
            d["control_grid"]["values"] = []
            if d.get("cells"):
                d["cells"]["values"] = []
        out.append(d)
    return {"count": len(out), "frames": out}


@app.websocket("/api/match/{match_id}/stream")
async def stream(ws: WebSocket, match_id: str, from_t: float = 0.0, hz: int = 20):
    await ws.accept()
    _, eng = _get(match_id)
    idx = next((i for i, f in enumerate(eng) if f.t >= from_t), 0)
    delay = 1.0 / max(1, min(hz, 60))
    try:
        while idx < len(eng):
            payload = eng[idx].model_dump_json(by_alias=True)
            await ws.send_text(payload)
            idx += 1
            await asyncio.sleep(delay)
        await ws.send_text(json.dumps({"eof": True}))
    except WebSocketDisconnect:
        pass
