"use client";

import { useCallback, useState } from "react";
import PitchBoard from "./components/PitchBoard";
import EvalBar from "./components/EvalBar";
import ControlPanel from "./components/ControlPanel";
import MoveFeed from "./components/MoveFeed";
import { usePlayback } from "./hooks/usePlayback";
import type { OverlayToggles } from "./lib/types";

export default function Home() {
  const pb = usePlayback();
  const [overlays, setOverlays] = useState<OverlayToggles>({
    triangles: true,
    spaceControl: false,
    pressingLanes: true,
    engineBestMove: true,
    cellGrid: true,
  });
  const setOverlay = useCallback(
    (k: keyof OverlayToggles, v: boolean) =>
      setOverlays((o) => ({ ...o, [k]: v })),
    []
  );

  return (
    <main className="flex h-screen flex-col overflow-hidden">
      <header className="flex items-center gap-3 px-4 py-2">
        <h1 className="text-lg font-bold tracking-tight">
          Football<span className="text-emerald-400">Engine</span>
        </h1>
        <span className="hidden text-[11px] font-mono text-neutral-500 sm:inline">
          Juego de Posición · 20 Hz · a1–l8 board
        </span>
        {pb.meta && (
          <span className="font-mono text-xs text-neutral-400">
            {pb.meta.teams.home.name}{" "}
            <b className="text-neutral-100">{pb.score.home}</b>
            {" – "}
            <b className="text-neutral-100">{pb.score.away}</b>{" "}
            {pb.meta.teams.away.name}
          </span>
        )}
        <button
          className="btn ml-auto"
          onClick={() => void pb.loadDemo(300)}
          disabled={pb.status === "loading"}
        >
          {pb.status === "loading" ? "Simulating…" : "▶ 5-min Demo"}
        </button>
      </header>

      {pb.status === "error" && (
        <div className="mx-4 mb-2 rounded-lg border border-red-800 bg-red-950/50 p-2 text-sm text-red-200">
          Backend unreachable on :8000 — {pb.error}
        </div>
      )}

      {pb.status !== "ready" && pb.status !== "error" && (
        <div className="flex flex-1 items-center justify-center text-neutral-500">
          Load the demo match to begin.
        </div>
      )}

      {pb.status === "ready" && pb.meta && (
        <div className="flex min-h-0 flex-1 gap-3 px-4 pb-3">
          {/* board column */}
          <section className="flex min-h-0 min-w-0 flex-1 flex-col gap-2">
            <div
              className="panel mx-auto min-h-0 w-full flex-1 overflow-hidden !p-0"
              style={{ maxWidth: "calc(74vh * 1.5441)" }}   /* 105:68, slightly smaller */
            >
              <PitchBoard
                meta={pb.meta}
                slice={pb.frameSlice(pb.cursor)}
                engine={pb.engineFrame}
                overlays={overlays}
              />
            </div>
            <ControlPanel
              pb={pb}
              engine={pb.engineFrame}
              overlays={overlays}
              setOverlay={setOverlay}
            />
          </section>

          {/* notation sidebar */}
          <aside className="flex w-[340px] shrink-0 flex-col gap-3 overflow-y-auto pr-1">
            <EvalBar evaluation={pb.engineFrame?.eval ?? null} />
            <MoveFeed engine={pb.engineFrame} />
          </aside>
        </div>
      )}
    </main>
  );
}
