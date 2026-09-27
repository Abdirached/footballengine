"use client";

import type { EngineFrame, OverlayToggles, Team } from "../lib/types";
import type { usePlayback } from "../hooks/usePlayback";
import { notation } from "./MoveFeed";

interface Props {
  pb: ReturnType<typeof usePlayback>;
  engine: EngineFrame | null;
  overlays: OverlayToggles;
  setOverlay: (k: keyof OverlayToggles, v: boolean) => void;
}

/** Slim horizontal control bar under the board + overlay toggles. */
export default function ControlPanel({ pb, engine, overlays, setOverlay }: Props) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <button className="btn" onClick={pb.toggle}>
        {pb.playing ? "⏸ Pause" : "▶ Play"}
      </button>
      <button className="btn" onClick={() => pb.stepFrames(-1)}>⏮ −50ms</button>
      <button className="btn" onClick={() => pb.stepFrames(1)}>+50ms ⏭</button>
      <select
        className="btn"
        value={pb.speed}
        onChange={(e) => pb.setSpeed(Number(e.target.value))}
      >
        {[0.25, 0.5, 1, 2, 4].map((sp) => (
          <option key={sp} value={sp}>{sp}×</option>
        ))}
      </select>
      <input
        type="range" min={0} max={Math.max(0.001, pb.duration)} step={0.05}
        value={pb.cursor}
        onChange={(e) => pb.seek(Number(e.target.value))}
        className="mx-2 min-w-[160px] flex-1 accent-blue-500"
      />
      <span className="font-mono text-xs text-neutral-400">
        {fmt(pb.cursor)} / {fmt(pb.duration)}
      </span>

      <div className="ml-auto flex flex-wrap gap-x-3 gap-y-1 text-xs">
        {(
          [
            ["cellGrid", "Cells"],
            ["triangles", "Triangles"],
            ["spaceControl", "Space"],
            ["pressingLanes", "Pressing"],
            ["engineBestMove", "Moves"],
          ] as [keyof OverlayToggles, string][]
        ).map(([k, label]) => (
          <label key={k} className="inline-flex cursor-pointer items-center gap-1">
            <input
              type="checkbox" checked={overlays[k]}
              onChange={(e) => setOverlay(k, e.target.checked)}
              className="accent-emerald-500"
            />
            {label}
          </label>
        ))}
      </div>

      {engine && (
        <div className="w-full font-mono text-[11px] text-neutral-400">
          ● {cap(engine.possession)} ball · carrier{" "}
          <b className="text-neutral-200">{engine.carrier_id ?? "—"}</b>
          {" @ "}
          <b className="text-yellow-300">{engine.carrier_cell ?? "—"}</b>
          {" · press "}
          {engine.pressing.intensity.pressers_5m} in 5 m · occlusion{" "}
          {engine.pressing.lane_occlusion_pct}%
          {engine.pressing.triggers.length > 0 && (
            <span className="text-orange-300"> · ⚡ {engine.pressing.triggers[0].type}</span>
          )}
        </div>
      )}
    </div>
  );
}

function fmt(t: number) {
  const m = Math.floor(t / 60), sec = (t % 60).toFixed(1);
  return `${m}:${sec.padStart(4, "0")}`;
}
function cap(s: string) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}
