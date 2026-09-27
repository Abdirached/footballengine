"use client";

import type { Evaluation } from "../lib/types";

/** Horizontal chess-style eval strip for the sidebar header: −10…+10 home-positive. */
export default function EvalBar({ evaluation }: { evaluation: Evaluation | null }) {
  const s = evaluation?.score ?? 0;
  const pct = ((s + 10) / 20) * 100; // fill from left

  return (
    <div className="panel py-3">
      <div className="mb-1 flex items-center justify-between text-xs font-mono text-neutral-400">
        <span>AWAY</span>
        <span className={`text-base ${s >= 0 ? "text-blue-300" : "text-red-300"}`}>
          {s > 0 ? "+" : ""}{s.toFixed(2)}
        </span>
        <span>HOME</span>
      </div>
      <div className="relative h-3 w-full overflow-hidden rounded-full border border-neutral-700 bg-gradient-to-r from-red-900 to-red-700">
        <div
          className="absolute inset-y-0 right-0 bg-gradient-to-l from-blue-500 to-blue-300 transition-all duration-150"
          style={{ width: `${pct}%` }}
        />
        <div className="absolute inset-y-0 left-1/2 w-px bg-white/70" />
      </div>
    </div>
  );
}
