"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../lib/api";
import type {
  EngineFrame,
  OverlayToggles,
  TrackingFrame,
  TrackingMeta,
} from "../lib/types";

export interface FrameSlice {
  prev: TrackingFrame;
  next: TrackingFrame;
  alpha: number;
}

const WINDOW_S = 45;      // seconds of EEF frames per fetch
const PREFETCH_AHEAD = 12; // start next window this many seconds before running out

export function usePlayback() {
  const [status, setStatus] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const [meta, setMeta] = useState<TrackingMeta | null>(null);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [cursor, setCursor] = useState(0);

  const framesRef = useRef<TrackingFrame[]>([]);
  const engineRef = useRef<EngineFrame[]>([]);
  const eventsRef = useRef<{ t: number; type: string; team?: string }[]>([]);
  const loadedUntil = useRef(0);
  const matchIdRef = useRef<string>("");
  const fetching = useRef(false);
  const cursorRef = useRef(0);
  const playingRef = useRef(false);
  const speedRef = useRef(1);
  const lastTsRef = useRef<number | null>(null);
  const rafRef = useRef<number>(0);
  const [, forceTick] = useState(0);

  const duration = useMemo(() => {
    const f = framesRef.current;
    return f.length ? f[f.length - 1].t : 0;
  }, [status]);

  // ------------------------------------------------------------- loading --
  const loadDemo = useCallback(async (seconds: number) => {
    setStatus("loading");
    setError(null);
    try {
      const id = await api.loadDemo(seconds);
      matchIdRef.current = id;
      const [m, trackDoc, eng, ev] = await Promise.all([
        api.meta(id),
        api.tracking(id),
        api.engineWindow(id, -1, WINDOW_S),
        api.events(id),
      ]);
      setMeta(m.meta);
      framesRef.current = trackDoc.frames;
      engineRef.current = eng;
      eventsRef.current = ev.events;
      loadedUntil.current = eng.length ? eng[eng.length - 1].t : WINDOW_S;
      cursorRef.current = trackDoc.frames[0]?.t ?? 0;
      setCursor(cursorRef.current);
      setStatus("ready");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus("error");
    }
  }, []);

  // ------------------------------------------- sliding-window EEF loader --
  const ensureWindows = useCallback(async () => {
    if (fetching.current || !matchIdRef.current) return;
    if (cursorRef.current < loadedUntil.current - PREFETCH_AHEAD) return;
    fetching.current = true;
    const fromT = loadedUntil.current + 0.05;
    try {
      const chunk = await api.engineWindow(
        matchIdRef.current, fromT, fromT + WINDOW_S
      );
      if (chunk.length) {
        engineRef.current = engineRef.current.concat(chunk);
        loadedUntil.current = chunk[chunk.length - 1].t;
        forceTick((n) => n + 1);
      }
    } catch {
      /* transient — retried on next tick */
    } finally {
      fetching.current = false;
    }
  }, []);

  // -------------------------------------------------------------- raf loop --
  useEffect(() => {
    const tick = (ts: number) => {
      const f = framesRef.current;
      if (playingRef.current && f.length > 1) {
        if (lastTsRef.current != null) {
          const dt = Math.min(0.1, (ts - lastTsRef.current) / 1000);
          let c = cursorRef.current + dt * speedRef.current;
          const end = f[f.length - 1].t;
          if (c >= end) c = f[0].t;                       // loop demo
          cursorRef.current = c;
          setCursor(c);
        }
        lastTsRef.current = ts;
      } else {
        lastTsRef.current = null;
      }
      void ensureWindows();
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [ensureWindows]);

  // ------------------------------------------------------------ controls ---
  const play = useCallback(() => setPlaying(true), []);
  const pause = useCallback(() => setPlaying(false), []);
  const toggle = useCallback(() => setPlaying((p) => !p), []);

  const stepFrames = useCallback((n: number) => {
    setPlaying(false);
    const f = framesRef.current;
    if (!f.length) return;
    const fps = f.length > 1 ? Math.round(1 / (f[1].t - f[0].t)) : 20;
    const idx = nearestIdx(f, cursorRef.current);
    const ni = Math.min(f.length - 1, Math.max(0, idx + n * Math.sign(fps)));
    cursorRef.current = f[ni].t;
    setCursor(f[ni].t);
  }, []);

  const seek = useCallback((t: number) => {
    const f = framesRef.current;
    if (!f.length) return;
    cursorRef.current = Math.min(f[f.length - 1].t, Math.max(f[0].t, t));
    setCursor(cursorRef.current);
  }, []);

  useEffect(() => { playingRef.current = playing; }, [playing]);
  useEffect(() => { speedRef.current = speed; }, [speed]);

  // --------------------------------------------------------------- views ---
  const frameSlice = useCallback((t: number): FrameSlice | null => {
    const f = framesRef.current;
    if (f.length < 2) return null;
    let i = nearestIdx(f, t);
    if (f[i].t > t && i > 0) i -= 1;
    const a = f[i];
    const b = f[Math.min(i + 1, f.length - 1)];
    const span = b.t - a.t || 1e-3;
    const alpha = Math.min(1, Math.max(0, (t - a.t) / span));
    return { prev: a, next: b, alpha };
  }, []);

  const engineFrame = useMemo<EngineFrame | null>(() => {
    const e = engineRef.current;
    if (!e.length) return null;
    return e[nearestIdx(e, cursor)];
  }, [cursor, status]);

  const score = useMemo(() => {
    let home = 0, away = 0;
    for (const e of eventsRef.current) {
      if (e.type !== "goal" || e.t > cursor) continue;
      if (e.team === "home") home += 1; else away += 1;
    }
    return { home, away };
  }, [cursor]);

  return {
    status, error, meta, loadDemo,
    playing, toggle, play, pause,
    speed, setSpeed,
    cursor, duration,
    frameSlice, engineFrame,
    stepFrames, seek,
    score,
  };
}

function nearestIdx<T extends { t: number }>(arr: T[], t: number): number {
  if (arr.length < 2) return 0;
  const est = Math.round(
    ((t - arr[0].t) / (arr[arr.length - 1].t - arr[0].t)) * (arr.length - 1)
  );
  const i = Math.min(arr.length - 1, Math.max(0, est));
  return arr[i].t <= t
    ? (i + 1 < arr.length && arr[i + 1].t <= t ? i + 1 : i)
    : Math.max(0, i - 1);
}
