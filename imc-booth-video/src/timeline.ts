// Timeline engine: every scene/shot duration is derived from the real narration audio
// durations (src/data/narration.json) and the real clip durations (src/data/manifest_rt.json), so the video
// can never drift out of sync with the voice — change a sentence, re-run make_narration.py,
// and everything re-flows.
//
// All demo footage is REAL TIME (1x): clips are cut straight from the original recordings by
// scripts/cut_realtime.py. Result screens are held with a freeze-frame, never sped up.
import narration from './data/narration.json';
import manifestRt from './data/manifest_rt.json';
import { FPS } from './theme';

const fr = (s: number) => Math.round(s * FPS);

export const LEAD = 0.6; // seconds of visual before a beat's first sentence
export const GAP = 0.5; // seconds between sentences inside a beat (slower, calmer pacing)
export const TAIL = 0.9; // seconds of breathing room after a beat's last sentence
const MIN_GAP = 0.35; // minimum silence between two sentences even when `at` asks for less

// ---------- clips ----------
export type Region = { name?: string; x: number; y: number; w: number; h: number; tStartSec: number; tEndSec: number };
export type ClipInfo = { id: string; file: string; durationSec: number; speed: number; zoomRegions: Region[] };

export const CLIPS: Record<string, ClipInfo> = {};
((manifestRt as any).clips as any[]).forEach((c) => {
  CLIPS[c.id] = { id: c.id, file: String(c.file).replace(/^.*[\\/]/, ''), durationSec: c.durationSec, speed: c.speed ?? 1, zoomRegions: c.zoomRegions ?? [] };
});

/** A clip placed inside a shot. `from`/`to` are seconds in the clip's own time; `hold` freezes its last frame for N more seconds. */
export type ClipUse = { id: string; from?: number; to?: number; rate?: number; step?: number; zoom?: Region[]; hold?: number };

export type ShotSpec = {
  beat?: string;
  clips?: ClipUse[];
  /** start time (s, relative to the shot) of each sentence; falls back to back-to-back placement */
  at?: number[];
  minSec?: number;
  lead?: number;
  tail?: number;
};
export type SceneSpec = { id: string; shots: ShotSpec[] };

// ---------- placed timeline ----------
export type Sentence = { show: string; file: string; start: number; dur: number }; // frames, relative to SCENE start
export type PlacedClip = { id: string; file: string; start: number; dur: number; hold: number; from: number; rate: number; step?: number; srcFrames: number; zoom?: Region[]; eff: number };
export type PlacedRegion = { start: number; end: number; x: number; y: number; w: number; h: number };
export type Shot = { beat?: string; start: number; dur: number; sentences: Sentence[]; clips: PlacedClip[]; regions: PlacedRegion[] };
export type Scene = { id: string; start: number; dur: number; shots: Shot[]; sentences: Sentence[] };

const beatMap: Record<string, { show: string; file: string; durationSec: number }[]> = {};
(narration as any).beats.forEach((b: any) => (beatMap[b.id] = b.sentences));

// ---------- magnifier windows (source px 720x1560, CLIP time) ----------
// The app's text is too small to read when the whole phone is shown at ~55%, so a "live zoom" panel follows the action.
const R = (x: number, y: number, w: number, h: number, tStartSec: number, tEndSec: number): Region => ({ x, y, w, h, tStartSec, tEndSec });
const CAPTION = { x: 30, y: 1250, w: 660, h: 190 };
const RIDGES = { x: 60, y: 440, w: 600, h: 330 };
const FINGERS = { x: 20, y: 330, w: 680, h: 760 };
const Z: Record<string, Region[]> = {
  rt_single_home: [R(40, 200, 640, 790, 0, 2.0), R(40, 680, 640, 650, 2.0, 99)],
  rt_single_form: [R(30, 300, 660, 400, 0, 99)],
  // capture: alternate the guidance caption with the fingertip ridges (ridges only once the camera has focused)
  rt_single_capture: [
    { ...CAPTION, tStartSec: 0, tEndSec: 3.0 },
    { ...RIDGES, tStartSec: 3.0, tEndSec: 5.9 },
    { ...CAPTION, tStartSec: 5.9, tEndSec: 7.5 },
    { ...RIDGES, tStartSec: 7.5, tEndSec: 99 },
  ],
  rt_single_enrolled: [R(40, 1040, 640, 430, 0, 2.4), R(40, 670, 640, 555, 2.4, 5.7), R(40, 1085, 640, 275, 5.7, 6.6), R(40, 930, 640, 295, 6.6, 9.0), R(40, 975, 640, 130, 9.0, 99)],
  rt_single_auth_open: [R(40, 200, 640, 540, 0, 99)],
  rt_single_auth_cap: [
    { ...CAPTION, tStartSec: 0, tEndSec: 2.4 },
    { ...RIDGES, tStartSec: 2.4, tEndSec: 3.9 },
    { ...CAPTION, tStartSec: 3.9, tEndSec: 6.0 },
    { ...RIDGES, tStartSec: 6.0, tEndSec: 99 },
  ],
  rt_single_auth_result: [R(40, 1000, 640, 470, 0, 4.5), R(40, 810, 640, 660, 4.5, 99)], // later windows include the 'MATCHING / API TIME' row
  // Pipeline Visualizer: we play to the Preprocessed stage and jump-cut past the 'Minutiae: Not available' panel (see spec)
  rt_single_pipeline: [
    R(30, 150, 660, 700, 0, 3.0),
    { ...CAPTION, tStartSec: 3.0, tEndSec: 11.0 },
    R(40, 1050, 640, 420, 11.0, 15.5),
    R(40, 925, 640, 490, 15.5, 17.0),
    R(30, 200, 660, 700, 17.0, 19.9),
    R(40, 450, 640, 850, 20.0, 99),
  ],
  rt_slap_home: [R(30, 990, 660, 220, 0, 99)], // the 4x / 1:N / 1:1 tiles - keeps the hero card's badge and tech chips out of the magnifier
  rt_slap_menu: [R(30, 600, 660, 330, 0, 3.0), R(30, 950, 660, 450, 3.0, 99)], // Enroll/Authenticate, then Verify / Pipeline / QC while they are named
  rt_slap_form: [R(30, 215, 660, 660, 0, 99)],
  rt_slap_capture: [{ ...FINGERS, tStartSec: 0, tEndSec: 99 }],
  rt_slap_enrolled: [R(40, 1019, 640, 451, 0, 5.8), R(30, 1040, 660, 380, 5.8, 99)],
  rt_slap_auth: [
    { ...FINGERS, tStartSec: 0, tEndSec: 6.6 },
    R(40, 1110, 640, 360, 6.6, 10.4),
    R(40, 746, 640, 480, 10.4, 13.0),
    R(30, 900, 660, 520, 13.0, 14.6),
    R(30, 520, 660, 840, 14.6, 99), // AUTHENTICATED card incl. the MATCHING / API TIME row, panning down to the per-finger results
  ],
  rt_slap_pipeline: [R(30, 150, 660, 700, 0, 2.8), { ...FINGERS, tStartSec: 2.8, tEndSec: 9.0 }, R(30, 200, 660, 650, 9.0, 20.4), R(30, 430, 660, 760, 20.4, 99)],
};

export const SCENE_SPECS: SceneSpec[] = [
  { id: 'hook', shots: [{ beat: 'B01', tail: 1.4 }] },
  { id: 'problem', shots: [{ beat: 'B02', clips: [{ id: 'rt_single_home' }], tail: 0.9 }] },
  {
    id: 'enrol',
    shots: [
      { beat: 'B03', clips: [{ id: 'rt_single_form', step: 0, hold: 1.0 }], at: [0.5, 3.4, 11.0] },
      { beat: 'B04', clips: [{ id: 'rt_single_capture', step: 1 }], at: [0.4, 5.2] },
      {
        beat: 'B05',
        clips: [
          { id: 'rt_single_enrolled', from: 0, to: 5.7, step: 2 },
          { id: 'rt_single_enrolled', from: 5.7, step: 3, hold: 3.5 },
        ],
        at: [0.4, 6.4, 11.2],
      },
    ],
  },
  {
    id: 'auth',
    shots: [
      { beat: 'B06', clips: [{ id: 'rt_single_auth_open', step: 0 }, { id: 'rt_single_auth_cap', step: 1 }], at: [0.2, 4.9, 9.0] },
      {
        beat: 'B07',
        clips: [
          { id: 'rt_single_auth_result', from: 0, to: 5.3, step: 2 },
          { id: 'rt_single_auth_result', from: 5.3, step: 3, hold: 3.0 },
        ],
        at: [0.3, 4.4],
      },
    ],
  },
  {
    id: 'pipeline',
    shots: [
      {
        beat: 'B08',
        clips: [
          { id: 'rt_single_pipeline', from: 3.0, to: 18.5 },
          { id: 'rt_single_pipeline', from: 20.1, to: 20.3 },
        ],
        at: [0.6, 8.0, 10.4, 15.4, 17.9, 23.4],
      },
    ],
  },
  {
    id: 'slap',
    shots: [
      { beat: 'B09', clips: [{ id: 'rt_slap_home', step: 0, hold: 2.0 }, { id: 'rt_slap_menu', step: 0, hold: 7.0 }], at: [0.5, 4.9] },
      { beat: 'B10', clips: [{ id: 'rt_slap_form', step: 0, hold: 0.8 }], at: [1.4, 14.0] },
      { beat: 'B11', clips: [{ id: 'rt_slap_capture', step: 0 }], at: [0.4, 7.0] },
      { beat: 'B12', clips: [{ id: 'rt_slap_enrolled', step: 0 }], at: [0.8, 7.4], minSec: 10.5 },
      { beat: 'B13', clips: [{ id: 'rt_slap_auth', step: 1 }], at: [0.6, 7.8, 11.6, 15.3] },
      { beat: 'B14', clips: [{ id: 'rt_slap_pipeline', from: 2.4, step: 2, hold: 3.5 }], at: [0.6, 8.5, 18.3] },
    ],
  },
  { id: 'accuracy', shots: [{ beat: 'B15', minSec: 9 }] },
  { id: 'pillars', shots: [{ beat: 'B16', tail: 2.4 }] },
  { id: 'closing', shots: [{ beat: 'B17', tail: 0.5 }, { beat: 'B18', tail: 1.8 }] },
];

function build(): { scenes: Scene[]; total: number } {
  const scenes: Scene[] = [];
  let cursor = 0;
  for (const spec of SCENE_SPECS) {
    let sc = 0; // scene-relative cursor (frames)
    const shots: Shot[] = [];
    const allSent: Sentence[] = [];
    for (const s of spec.shots) {
      const lead = s.lead ?? LEAD;
      const tail = s.tail ?? TAIL;
      const sents = s.beat ? beatMap[s.beat] : [];
      if (s.beat && !sents) throw new Error('unknown beat ' + s.beat);
      // narration placement (relative to shot start)
      const placedSent: Sentence[] = [];
      let t = lead;
      let prevEnd = -Infinity;
      sents.forEach((se, i) => {
        const wanted = s.at && s.at[i] !== undefined ? s.at[i] : i === 0 ? lead : prevEnd + GAP;
        const start = Math.max(wanted, prevEnd + MIN_GAP);
        placedSent.push({ show: se.show, file: se.file, start: fr(start), dur: fr(se.durationSec) });
        prevEnd = start + se.durationSec;
        t = prevEnd;
      });
      const narrDur = sents.length ? t + tail : 0;
      // clips back-to-back from shot start (each followed by its freeze-hold)
      const placedClips: PlacedClip[] = [];
      let ct = 0;
      for (const cu of s.clips ?? []) {
        const info = CLIPS[cu.id];
        if (!info) throw new Error('missing clip in manifest: ' + cu.id);
        const from = cu.from ?? 0;
        const to = Math.min(cu.to ?? info.durationSec, info.durationSec);
        const rate = cu.rate ?? 1;
        const outSec = (to - from) / rate;
        const hold = cu.hold ?? 0;
        placedClips.push({
          id: cu.id, file: info.file, start: fr(ct), dur: fr(outSec), hold: fr(hold), from: fr(from), rate, step: cu.step,
          srcFrames: fr(to - from), zoom: cu.zoom ?? Z[cu.id] ?? info.zoomRegions, eff: (info.speed ?? 1) * rate,
        });
        ct += outSec + hold;
      }
      const dur = fr(Math.max(narrDur, ct, s.minSec ?? 0));
      // zoom regions mapped into shot time
      const regions: PlacedRegion[] = [];
      for (const pc of placedClips) {
        for (const r of pc.zoom ?? []) {
          const t0 = Math.max(r.tStartSec, pc.from / FPS);
          const t1 = Math.min(r.tEndSec, (pc.from + pc.srcFrames) / FPS + pc.hold / FPS + 0.001);
          if (t1 <= t0) continue;
          regions.push({
            start: pc.start + fr((t0 - pc.from / FPS) / pc.rate),
            end: pc.start + fr((t1 - pc.from / FPS) / pc.rate),
            x: r.x, y: r.y, w: r.w, h: r.h,
          });
        }
      }
      regions.sort((a, b) => a.start - b.start);
      shots.push({ beat: s.beat, start: sc, dur, sentences: placedSent, clips: placedClips, regions });
      placedSent.forEach((p) => allSent.push({ ...p, start: p.start + sc }));
      sc += dur;
    }
    scenes.push({ id: spec.id, start: cursor, dur: sc, shots, sentences: allSent });
    cursor += sc;
  }
  return { scenes, total: cursor };
}

const built = build();
export const SCENES = built.scenes;
export const TOTAL_FRAMES = built.total;
export const SCENE: Record<string, Scene> = Object.fromEntries(SCENES.map((s) => [s.id, s]));
