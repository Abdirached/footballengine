import type { EngineFrame, MatchTrackingDoc, TrackingMeta } from "./types";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function j<T>(r: Response | Promise<Response>): Promise<T> {
  const res = await r;
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json() as Promise<T>;
}

export const api = {
  async loadDemo(seconds = 25): Promise<string> {
    const r = await fetch(`${API}/api/demo/generate?seconds=${seconds}`, {
      method: "POST",
    });
    return (await j<{ match_id: string }>(r)).match_id;
  },

  meta(matchId: string): Promise<{ meta: TrackingMeta; frames: number }> {
    return j(fetch(`${API}/api/match/${matchId}`));
  },

  events(matchId: string): Promise<{ events: { t: number; type: string; team?: string }[] }> {
    return j(fetch(`${API}/api/match/${matchId}/events`));
  },

  tracking(matchId: string): Promise<MatchTrackingDoc> {
    return j<MatchTrackingDoc>(fetch(`${API}/api/match/${matchId}/tracking`));
  },

  /** Batched engine frames; grid_every>1 thins control grids (payload cut). */
  async engineWindow(matchId: string, fromT: number, toT: number,
                     gridEvery = 4): Promise<EngineFrame[]> {
    const doc = await fetch(
      `${API}/api/match/${matchId}/engine?from_t=${fromT}&to_t=${toT}&grid_every=${gridEvery}`
    ).then(j<{ frames: EngineFrame[] }>);
    return doc.frames;
  },
};
