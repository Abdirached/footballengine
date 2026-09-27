"use client";

import { useMemo, useRef } from "react";
import type { EngineFrame, MoveOption } from "../lib/types";

/** Compact algebraic string: "H9g8 !!" (pass) / "H10 f6 !" (movement). */
export function notation(o: MoveOption): string {
  const who = o.player ?? "?";
  const num = who.replace(/\D/g, "");
    const side = who.startsWith("A") ? "a" : "";
  const cell = o.to_cell ?? "";
  return o.action === "move"
    ? `${side}${num} → ${cell} ${o.symbol}`.trim()
    : `${side}${num}${cell} ${o.symbol}`.trim();
}

const TIER_CLS: Record<string, string> = {
  brilliant: "text-yellow-300 border-yellow-400/60 bg-yellow-400/10",
  best: "text-emerald-300 border-emerald-400/60 bg-emerald-400/10",
  good: "text-sky-300 border-sky-400/50 bg-sky-400/5",
  inaccuracy: "text-orange-300 border-orange-400/60 bg-orange-400/10",
  blunder: "text-red-300 border-red-400/60 bg-red-400/10",
};

interface FeedEntry {
  t: number;
  text: string;
  tier: string;
  detail: string;
}

export default function MoveFeed({ engine }: { engine: EngineFrame | null }) {
  const seenT = useRef<number>(-1);
  const feed = useRef<FeedEntry[]>([]);

  // accumulate new engine frames as playback advances forward
  if (engine && engine.t > seenT.current) {
    seenT.current = engine.t;
    for (const o of engine.options.slice(0, 3)) {
      const last = feed.current[0];
      if (last && last.text === notation(o)) continue;
      feed.current.unshift({
        t: engine.t,
        text: notation(o),
        tier: o.tier,
        detail: `p=${o.completion.toFixed(2)} · v=${o.value.toFixed(1)} · L${o.lines_broken}`,
      });
    }
    feed.current = feed.current.slice(0, 30);
  }

  const bm = engine?.best_move ?? null;
  const cells = useMemo(() => {
    if (!engine?.cells?.values.length) return null;
    let bi = 0;
    engine.cells.values.forEach((v, i) => { if (v > engine.cells!.values[bi]) bi = i; });
    const f = bi % 12, r = Math.floor(bi / 12);
    return `${FILES[f]}${r + 1}`;
  }, [engine]);

  return (
    <div className="flex flex-col gap-2">
      {/* hero best move */}
      <div className={`rounded-lg border p-3 ${
        bm ? TIER_CLS[bm.tier] : "border-neutral-800 bg-neutral-900"
      }`}>
        <div className="text-[10px] uppercase tracking-widest text-neutral-400">
          Engine move
        </div>
        <div className="font-mono text-2xl font-extrabold leading-tight">
          {bm ? (
            <>
              {notation(bm).split(" ")[0]}
              <span className="ml-2">{bm.symbol}</span>
            </>
          ) : (
            <span className="text-neutral-500">—</span>
          )}
        </div>
        <div className="mt-1 text-xs text-neutral-300">{bm?.reason || "waiting…"}</div>
        {cells && (
          <div className="mt-1 font-mono text-[11px] text-neutral-400">
            hottest cell: {cells}
          </div>
        )}
      </div>

      {/* rolling feed */}
      <div className="rounded-lg border border-neutral-800 bg-neutral-900/70 p-2">
        <div className="mb-1 px-1 text-[10px] uppercase tracking-widest text-neutral-500">
          Notation
        </div>
        <div className="max-h-64 overflow-y-auto font-mono text-sm">
          {feed.current.length === 0 && (
            <div className="px-1 py-2 text-xs text-neutral-500">no moves yet…</div>
          )}
          {feed.current.map((e, i) => (
            <div key={`${e.t}-${i}`}
                 className="flex items-center justify-between rounded px-1.5 py-1 hover:bg-neutral-800/60">
              <span className={`border-b ${TIER_CLS[e.tier]?.split(" ").slice(0, 2).join(" ")}`}>
                {e.text}
              </span>
              <span className="text-[11px] text-neutral-500">{e.detail}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

const FILES = "abcdefghijkl";
