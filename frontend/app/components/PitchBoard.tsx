"use client";

import { useEffect, useRef } from "react";
import { Application, Container, Graphics, Sprite, Text, Texture } from "pixi.js";
import type { EngineFrame, OverlayToggles, TrackingMeta } from "../lib/types";
import type { FrameSlice } from "../hooks/usePlayback";

interface Props {
  meta: TrackingMeta | null;
  slice: FrameSlice | null;
  engine: EngineFrame | null;
  overlays: OverlayToggles;
}

const FILES = "abcdefghijkl";

/** Board geometry shared by every layer rebuild / frame render. */
interface Geo {
  w: number; h: number; s: number; ox: number; oy: number;
  X: (x: number) => number; Y: (y: number) => number;
}

function geoFor(w: number, h: number, meta: TrackingMeta): Geo {
  const pad = 26;
  const L = meta.pitch.length, W = meta.pitch.width;
  const s = Math.min((w - 2 * pad) / L, (h - 2 * pad) / W);
  const ox = (w - L * s) / 2, oy = (h - W * s) / 2;
  return {
    w, h, s, ox, oy,
    X: (x) => ox + (x + L / 2) * s,
    Y: (y) => oy + (y + W / 2) * s,
  };
}

export default function PitchBoard({ meta, slice, engine, overlays }: Props) {
  const hostRef = useRef<HTMLDivElement>(null);
  const appRef = useRef<Application | null>(null);
  const propsRef = useRef<Props>({ meta, slice, engine, overlays });
  propsRef.current = { meta, slice, engine, overlays };
  const lastEngineT = useRef(-1);
  const lastScale = useRef(-1);
  const numTexts = useRef<Map<string, Text>>(new Map());

  useEffect(() => {
    const app = new Application({
      background: 0x071007,
      resizeTo: hostRef.current ?? undefined,
      antialias: true,
    });
    hostRef.current?.appendChild(app.view as HTMLCanvasElement);

    const heat = new Sprite(Texture.WHITE);
    heat.alpha = 0.40;
    const cells = new Container();
    const markings = new Graphics();
    const shadows = new Graphics();
    const triangles = new Graphics();
    const moves = new Graphics();
    const moveLabel = new Text("", {
      fontSize: 15, fontWeight: "800", fill: 0xffffff,
      stroke: 0x000000, strokeThickness: 4,
    });
    moveLabel.anchor.set(0.5);
    moveLabel.visible = false;
    const nodes = new Graphics();
    app.stage.addChild(cells, heat, markings, shadows, triangles, moves,
      moveLabel, nodes);

    let raf = 0;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      render(app, { cells, heat, markings, shadows, triangles, moves, moveLabel, nodes },
        propsRef.current, { lastEngineT, lastScale, numTexts });
    };
    raf = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(raf);
      numTexts.current.clear();
      app.destroy(true, { children: true });
    };
  }, []);

  return (
    <div className="relative h-full w-full">
      <div ref={hostRef} className="absolute inset-0" />
    </div>
  );
}

interface Layers {
  cells: Container;
  heat: Sprite;
  markings: Graphics;
  shadows: Graphics;
  triangles: Graphics;
  moves: Graphics;
  moveLabel: Text;
  nodes: Graphics;
}
interface Persist {
  lastEngineT: { current: number };
  lastScale: { current: number };
  numTexts: { current: Map<string, Text> };
}

// ===========================================================================
function render(app: Application, L: Layers, props: Props, P: Persist) {
  const { meta, slice, engine, overlays } = props;
  if (!meta || !slice) return;

  const w = app.renderer.width / app.renderer.resolution;
  const h = app.renderer.height / app.renderer.resolution;
  const G = geoFor(w, h, meta);
  const { X, Y, s } = G;
  const pL = meta.pitch.length, pW = meta.pitch.width;

  // ---- interpolated world state -------------------------------------------
  const pos = new Map<string, { x: number; y: number; team: string; num: number }>();
  for (const p of slice.prev.players) {
    const q = slice.next.players.find((r) => r.id === p.id);
    pos.set(p.id, {
      x: lerp(p.x, q ? q.x : p.x, slice.alpha),
      y: lerp(p.y, q ? q.y : p.y, slice.alpha),
      team: p.team, num: p.num,
    });
  }
  const ball = {
    x: lerp(slice.prev.ball.x, slice.next.ball.x, slice.alpha),
    y: lerp(slice.prev.ball.y, slice.next.ball.y, slice.alpha),
  };

  // ---- static layers (cells chessboard + FIFA markings) --------------------
  if (Math.abs(s - P.lastScale.current) > 0.01) {
    P.lastScale.current = s;
    buildCells(L.cells, G, meta, overlays.cellGrid);
    buildMarkings(L.markings, G, meta);
  }

  // cells visibility follows toggle without rebuilding
  if (L.cells.visible !== overlays.cellGrid && meta) {
    // rebuild cheap enough only on change of scale; here just toggle alpha
    L.cells.children.forEach((c) => (c.renderable = overlays.cellGrid));
  }

  // ---- space-control heat ---------------------------------------------------
  if (engine && overlays.spaceControl &&
      engine.t !== P.lastEngineT.current && engine.control_grid.values.length) {
    P.lastEngineT.current = engine.t;
    updateHeat(L.heat, engine.control_grid);
    L.heat.visible = true;
  } else if (!overlays.spaceControl) {
    L.heat.visible = false;
  }
  L.heat.x = X(-pL / 2); L.heat.y = Y(-pW / 2);
  L.heat.width = pL * s; L.heat.height = pW * s;

  // ---- cover shadows --------------------------------------------------------
  const sh = L.shadows;
  sh.clear();
  if (overlays.pressingLanes && engine?.possession !== "loose") {
    for (const shd of engine?.pressing.shadows ?? []) {
      const ax = X(shd.x), ay = Y(shd.y);
      const len = 16 * s;
      const half = (shd.half_angle_deg * Math.PI) / 180;
      const a0 = Math.atan2(shd.dir_y, shd.dir_x) - half;
      const a1 = Math.atan2(shd.dir_y, shd.dir_x) + half;
      sh.beginFill(0xff4444, 0.12);
      sh.lineStyle(1, 0xff6666, 0.22);
      sh.moveTo(ax, ay);
      sh.arc(ax, ay, len, a0, a1);
      sh.lineTo(ax, ay);
      sh.endFill();
    }
    // highlight the ball's cell
    if (engine?.carrier_cell) {
      const [rx, ry, rw, rh] = cellRect(engine.carrier_cell, G, meta);
      sh.lineStyle(2, 0xffc83d, 0.85);
      sh.drawRect(rx, ry, rw, rh);
    }
  }

  // ---- active triangles ------------------------------------------------------
  const tr = L.triangles;
  tr.clear();
  if (overlays.triangles) {
    for (const t of engine?.triangles ?? []) {
      const pts = t.pts.map(([x, y]) => ({ x: X(x), y: Y(y) }));
      tr.lineStyle(2, 0x22ff88, 0.35 + t.score * 0.55);
      tr.beginFill(0x22ff88, 0.07);
      tr.drawPolygon(pts.flatMap((p) => [p.x, p.y]));
      tr.endFill();
    }
  }

  // ---- engine moves (arrows + algebraic label) --------------------------------
  const mv = L.moves;
  mv.clear();
  L.moveLabel.visible = false;
  const bm = engine?.best_move;
  if (overlays.engineBestMove && bm) {
    const color =
      bm.tier === "brilliant" ? 0xffd700 :
      bm.tier === "best" ? 0x00ff88 :
      bm.tier === "good" ? 0x66ccff :
      bm.tier === "inaccuracy" ? 0xffaa00 : 0xff3344;
    // single arrow only — the engine's headline move
    arrow(mv, X(bm.from_x), Y(bm.from_y), X(bm.to_x), Y(bm.to_y), color,
          bm.tier === "brilliant" ? 4.5 : 3.5);
    if (bm.to_cell) {
      L.moveLabel.text = `${bm.to_cell} ${bm.symbol}`;
      L.moveLabel.style.fill = color;
      L.moveLabel.position.set(X(bm.to_x), Y(bm.to_y) - 18);
      L.moveLabel.visible = true;
    }
  }

  // ---- numbered player dots + ball --------------------------------------------
  const nd = L.nodes;
  nd.clear();
  const carrier = engine?.carrier_id;
  const r = Math.max(9, Math.min(14, s * 0.85));
  pos.forEach((p, id) => {
    const cxp = X(p.x), cyp = Y(p.y);
    nd.beginFill(p.team === "home" ? 0x3b82f6 : 0xef4444);
    nd.drawCircle(cxp, cyp, r);
    nd.endFill();
    nd.lineStyle(id === carrier ? 3 : 1.4, 0xffffff, id === carrier ? 1 : 0.55);
    nd.drawCircle(cxp, cyp, r);
    let tx = P.numTexts.current.get(id);
    if (!tx) {
      tx = new Text(String(p.num), {
        fontSize: Math.max(11, Math.round(r * 1.35)), fontWeight: "800",
        fill: 0xffffff, fontFamily: "system-ui, sans-serif",
      });
      tx.anchor.set(0.5);
      app.stage.addChild(tx);
      P.numTexts.current.set(id, tx);
    }
    tx.position.set(cxp, cyp);
    tx.visible = true;
  });
  // hide stale texts
  P.numTexts.current.forEach((tx, id) => {
    if (!pos.has(id)) tx.visible = false;
  });

  const br = Math.max(4.5, Math.min(7.5, s * 0.42));
  nd.beginFill(0xffffff);
  nd.drawCircle(X(ball.x), Y(ball.y), br);
  nd.endFill();
  nd.lineStyle(1.2, 0x000000, 0.6);
  nd.drawCircle(X(ball.x), Y(ball.y), br);
}

// ------------------------------------------------------------------ static ---
function buildCells(layer: Container, G: Geo, meta: TrackingMeta, visible: boolean) {
  layer.removeChildren().forEach((c) => c.destroy());
  const { X, Y, s } = G;
  const L = meta.pitch.length, W = meta.pitch.width;
  const cw = L / 12, ch = W / 8;
  const gw = Math.max(1.5, s * 0.05);          // bold, scale-aware grid lines
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 12; f++) {
      const g = new Graphics();
      const tint = (f + r) % 2 === 0 ? 0x3a4258 : 0x20263a;   // slate / graphite
      g.beginFill(tint, 0.96);
      g.drawRect(X(-L / 2 + f * cw), Y(-W / 2 + r * ch), cw * s, ch * s);
      g.endFill();
      g.lineStyle(gw, 0xffffff, 0.85);
      g.drawRect(X(-L / 2 + f * cw), Y(-W / 2 + r * ch), cw * s, ch * s);
      layer.addChild(g);
    }
  }
  // file letters (bottom) + rank numbers (left)
  for (let f = 0; f < 12; f++) {
    const t = new Text(FILES[f], { fontSize: 11, fontWeight: "700", fill: 0xe2e8f0 });
    t.anchor.set(0.5);
    t.position.set(X(-L / 2 + (f + 0.5) * cw), Y(W / 2) + 12);
    layer.addChild(t);
  }
  for (let r = 0; r < 8; r++) {
    const t = new Text(String(r + 1), { fontSize: 11, fontWeight: "700", fill: 0xe2e8f0 });
    t.anchor.set(0.5);
    t.position.set(X(-L / 2) - 12, Y(-W / 2 + (r + 0.5) * ch));
    layer.addChild(t);
  }
  layer.renderable = visible;
}

function buildMarkings(g: Graphics, G: Geo, meta: TrackingMeta) {
  const { X, Y, s } = G;
  const HL = meta.pitch.length / 2, HW = meta.pitch.width / 2;
  g.clear();
  g.beginFill(0x0e5c2f);
  g.drawRect(G.X(-HL) - 4, G.Y(-HW) - 4, meta.pitch.length * G.s + 8,
             meta.pitch.width * G.s + 8);
  g.endFill();

  const line = (fn: () => void) => { g.lineStyle(1.6, 0xf8fafc, 0.95); fn(); };
  const rect = (x0: number, y0: number, ww: number, hh: number) =>
    g.drawRect(X(x0), Y(y0), ww * G.s, hh * G.s);

  line(() => rect(-HL, -HW, meta.pitch.length, meta.pitch.width));   // boundary
  line(() => { g.moveTo(X(0), Y(-HW)); g.lineTo(X(0), Y(HW)); });     // halfway
  line(() => g.drawCircle(X(0), Y(0), 9.15 * s));                     // centre circle
  line(() => { g.beginFill(0xf8fafc, 0.95); g.drawCircle(X(0), Y(0), 2.2); g.endFill(); });

  for (const side of [-1, 1]) {
    const gl = side * HL;                       // goal-line x
    const paD = 16.5 * side, gaD = 5.5 * side;  // depths (signed)
    // penalty area 16.5 × 40.32, goal area 5.5 × 18.32
    line(() => rect(Math.min(gl, gl - paD), -20.16, Math.abs(paD), 40.32));
    line(() => rect(Math.min(gl, gl - gaD), -9.16, Math.abs(gaD), 18.32));
    // penalty spot @11 m
    const spot = gl - side * 11;
    line(() => { g.beginFill(0xf8fafc, 0.95); g.drawCircle(X(spot), Y(0), 2.2); g.endFill(); });
    // penalty arc: r 9.15 about the spot, clipped outside the box
    const dxr = (16.5 - 11) / 9.15;             // cos of clip angle
    const a = Math.acos(dxr);
    line(() => {
      if (side === -1) {
        g.moveTo(X(spot) + 9.15 * s * Math.cos(-a), Y(0) + 9.15 * s * Math.sin(-a));
        g.arc(X(spot), Y(0), 9.15 * s, -a, a);
      } else {
        g.moveTo(X(spot) + 9.15 * s * Math.cos(Math.PI - a),
                 Y(0) + 9.15 * s * Math.sin(Math.PI - a));
        g.arc(X(spot), Y(0), 9.15 * s, Math.PI - a, Math.PI + a);
      }
    });
    // corner arcs r=1 (moveTo first — PixiJS arcs otherwise connect paths)
    line(() => {
      for (const [cy2, a0, a1] of [
        [-HW, side === -1 ? 0 : Math.PI / 2, side === -1 ? Math.PI / 2 : Math.PI],
        [HW, side === -1 ? -Math.PI / 2 : Math.PI,
         side === -1 ? 0 : Math.PI * 1.5],
      ] as [number, number, number][]) {
        g.moveTo(X(gl) + s * Math.cos(a0), Y(cy2) + s * Math.sin(a0));
        g.arc(X(gl), Y(cy2), 1 * s, a0, a1);
      }
    });
    // goal 7.32 m wide, 2 m deep behind the line
    line(() => rect(side === -1 ? gl - 2 : gl, -3.66, 2, 7.32));
  }
}

function cellRect(name: string, G: Geo, meta: TrackingMeta): [number, number, number, number] {
  const f = FILES.indexOf(name[0]);
  const r = parseInt(name[1], 10) - 1;
  const L = meta.pitch.length, W = meta.pitch.width;
  const cw = L / 12, ch = W / 8;
  return [G.X(-L / 2 + f * cw), G.Y(-W / 2 + r * ch), cw * G.s, ch * G.s];
}

function updateHeat(sprite: Sprite, grid: { nx: number; ny: number; values: number[] }) {
  const cv = document.createElement("canvas");
  cv.width = grid.nx; cv.height = grid.ny;
  const ctx = cv.getContext("2d");
  if (!ctx) return;
  const img = ctx.createImageData(grid.nx, grid.ny);
  for (let i = 0; i < grid.values.length; i++) {
    const c = grid.values[i];
    const homeSide = c >= 0.5;
    const inten = Math.abs(c - 0.5) * 2;
    img.data[i * 4 + 0] = homeSide ? 59 * inten : 239 * inten;
    img.data[i * 4 + 1] = homeSide ? 130 * inten : 68 * inten;
    img.data[i * 4 + 2] = homeSide ? 246 * inten : 68 * inten;
    img.data[i * 4 + 3] = Math.round(inten * 200);
  }
  ctx.putImageData(img, 0, 0);
  sprite.texture = Texture.from(cv);
}

function arrow(g: Graphics, x0: number, y0: number, x1: number, y1: number,
               color: number, width: number) {
  const ang = Math.atan2(y1 - y0, x1 - x0);
  const headLen = 8 + width * 2;
  const bx = x1 - headLen * Math.cos(ang);
  const by = y1 - headLen * Math.sin(ang);
  g.lineStyle(width, color, 0.95);
  g.moveTo(x0, y0);
  g.lineTo(bx, by);
  g.beginFill(color, 0.95);
  g.drawPolygon([
    x1, y1,
    bx - headLen * 0.5 * Math.cos(ang - Math.PI / 2.6),
    by - headLen * 0.5 * Math.sin(ang - Math.PI / 2.6),
    bx - headLen * 0.5 * Math.cos(ang + Math.PI / 2.6),
    by - headLen * 0.5 * Math.sin(ang + Math.PI / 2.6),
  ]);
  g.endFill();
}

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
