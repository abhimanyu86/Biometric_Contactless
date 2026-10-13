// Demo-footage building blocks: a phone mockup, the shot's video stack, and a live "magnifier"
// panel that zooms the same footage into the UI area being talked about (the app's own text is
// too small to read at TV distance when the whole phone is shown).
import React, { FC, ReactNode } from 'react';
import { AbsoluteFill, Freeze, OffthreadVideo, Sequence, staticFile, useCurrentFrame } from 'remotion';
import { C } from '../theme';
import { inter, oswald } from '../fonts';
import { clamp01, ease, Logo, Tag, Heading } from './ui';
import { Subtitles } from './Subtitles';
import type { PlacedRegion, Scene, Shot } from '../timeline';

export const SRC_W = 720;
export const SRC_H = 1560;

// ---------- video stack for one shot, in 720x1560 source space ----------
export const ShotVideo: FC<{ shot: Shot }> = ({ shot }) => {
  const last = shot.clips[shot.clips.length - 1];
  const lastEnd = last ? last.start + last.dur + last.hold : 0;
  const vid = (c: (typeof shot.clips)[number]) => (
    <OffthreadVideo src={staticFile(`media/clips/${c.file}`)} startFrom={c.from} playbackRate={c.rate} muted style={{ width: SRC_W, height: SRC_H }} />
  );
  return (
    <div style={{ position: 'absolute', left: 0, top: 0, width: SRC_W, height: SRC_H, background: '#0b0b0b', overflow: 'hidden' }}>
      {shot.clips.map((c, i) => (
        <React.Fragment key={i}>
          <Sequence from={c.start} durationInFrames={Math.max(1, c.dur)}>
            {vid(c)}
          </Sequence>
          {c.hold > 0 && (
            <Sequence from={c.start + c.dur} durationInFrames={c.hold}>
              <Freeze frame={Math.max(0, c.dur - 1)}>{vid(c)}</Freeze>
            </Sequence>
          )}
        </React.Fragment>
      ))}
      {last && shot.dur > lastEnd && (
        <Sequence from={lastEnd} durationInFrames={shot.dur - lastEnd}>
          <Freeze frame={Math.max(0, last.dur - 1)}>{vid(last)}</Freeze>
        </Sequence>
      )}
    </div>
  );
};

// ---------- region logic ----------
// The magnifier shows a WINDOW of the 720x1560 source with the same aspect as the zoom box. A region that is
// too tall to fit at a readable scale is zoomed to a readable scale and panned top -> bottom over its duration.
export const ZOOM_W = 920;
export const ZOOM_H = 516;
const MIN_READABLE = 1.15;
const MAX_SCALE = 1.5; // the source is only 720 px wide — beyond this it goes soft
type Rect = { x: number; y: number; w: number; h: number };
const DEFAULT_REGION = { x: 0, y: 150, w: 720, h: 520, start: 0, end: 1 } as PlacedRegion;
const TRANS = 14;
const clampN = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

function windowFor(r: PlacedRegion, t: number): Rect {
  let sc = Math.min(ZOOM_W / r.w, ZOOM_H / r.h);
  if (sc < MIN_READABLE) sc = Math.min(ZOOM_W / r.w, 1.3);
  sc = Math.min(sc, MAX_SCALE);
  sc = Math.max(sc, ZOOM_W / SRC_W); // never show beyond the 720 px source edge
  sc = sc * (1 + 0.04 * t); // slow 4% push-in over each window, so held result screens keep moving
  const wv = ZOOM_W / sc;
  const hv = ZOOM_H / sc;
  const x = clampN(r.x + r.w / 2 - wv / 2, 0, Math.max(0, SRC_W - wv));
  const y = hv >= r.h ? r.y + r.h / 2 - hv / 2 : r.y + (r.h - hv) * t;
  return { x, y: clampN(y, 0, SRC_H - hv), w: wv, h: hv };
}
const lerpRect = (a: Rect, b: Rect, t: number): Rect => ({
  x: a.x + (b.x - a.x) * t,
  y: a.y + (b.y - a.y) * t,
  w: a.w + (b.w - a.w) * t,
  h: a.h + (b.h - a.h) * t,
});
export function activeRect(regions: PlacedRegion[], frame: number): Rect {
  if (!regions.length) return windowFor(DEFAULT_REGION, 0);
  let i = -1;
  for (let k = 0; k < regions.length; k++) if (regions[k].start <= frame) i = k;
  if (i === -1) return windowFor(regions[0], 0);
  const cur = regions[i];
  const prev = i > 0 ? regions[i - 1] : cur;
  const pan = ease(clamp01((frame - cur.start) / Math.max(1, cur.end - cur.start)));
  const move = ease(clamp01((frame - cur.start) / TRANS));
  return lerpRect(windowFor(prev, 1), windowFor(cur, pan), move);
}

// ---------- phone ----------
export const PhoneFrame: FC<{ x: number; y: number; screenH: number; children: ReactNode }> = ({ x, y, screenH, children }) => {
  const s = screenH / SRC_H;
  const screenW = SRC_W * s;
  const pad = 14;
  return (
    <div
      style={{
        position: 'absolute',
        left: x,
        top: y,
        width: screenW + pad * 2,
        height: screenH + pad * 2,
        borderRadius: 56,
        background: 'linear-gradient(145deg,#2b2b2e,#0d0d0f 55%,#1d1d20)',
        boxShadow: '0 40px 80px rgba(60,35,10,0.32), 0 8px 18px rgba(60,35,10,0.25), inset 0 0 0 2px rgba(255,255,255,0.10)',
        padding: pad,
      }}
    >
      <div style={{ position: 'relative', width: screenW, height: screenH, borderRadius: 42, overflow: 'hidden', background: '#000' }}>
        <div style={{ position: 'absolute', left: 0, top: 0, width: SRC_W, height: SRC_H, transform: `scale(${s})`, transformOrigin: '0 0' }}>{children}</div>
        {/* punch-hole camera */}
        <div style={{ position: 'absolute', top: 16, left: '50%', width: 14, height: 14, marginLeft: -7, borderRadius: 7, background: '#050505', boxShadow: '0 0 0 3px rgba(255,255,255,0.05)' }} />
      </div>
    </div>
  );
};

/** Orange bracket on the phone screen showing what the magnifier is currently zoomed into. */
export const Highlight: FC<{ shot: Shot }> = ({ shot }) => {
  const frame = useCurrentFrame();
  if (!shot.regions.length) return null;
  const r = activeRect(shot.regions, frame);
  const arm = 48;
  const t = 8;
  const corner = (pos: React.CSSProperties) => <div style={{ position: 'absolute', width: arm, height: arm, border: `${t}px solid ${C.orange}`, ...pos }} />;
  return (
    <div style={{ position: 'absolute', left: r.x - 6, top: r.y - 6, width: r.w + 12, height: r.h + 12, pointerEvents: 'none' }}>
      <div style={{ position: 'absolute', inset: 0, background: 'rgba(194,110,16,0.10)', borderRadius: 10 }} />
      {corner({ left: 0, top: 0, borderRight: 'none', borderBottom: 'none', borderTopLeftRadius: 12 })}
      {corner({ right: 0, top: 0, borderLeft: 'none', borderBottom: 'none', borderTopRightRadius: 12 })}
      {corner({ left: 0, bottom: 0, borderRight: 'none', borderTop: 'none', borderBottomLeftRadius: 12 })}
      {corner({ right: 0, bottom: 0, borderLeft: 'none', borderTop: 'none', borderBottomRightRadius: 12 })}
    </div>
  );
};

const ZoomShot: FC<{ shot: Shot; boxW: number; boxH: number }> = ({ shot, boxW, boxH }) => {
  const frame = useCurrentFrame();
  const r = activeRect(shot.regions, frame);
  const s = boxW / r.w;
  return (
    <div
      style={{
        position: 'absolute',
        left: 0,
        top: 0,
        width: boxW,
        height: boxH,
        overflow: 'hidden',
        borderRadius: 14,
        boxShadow: '0 0 0 2px rgba(21,18,14,0.08)',
        background: '#0b0b0b',
      }}
    >
      <div style={{ position: 'absolute', left: 0, top: 0, width: SRC_W, height: SRC_H, transform: `scale(${s}) translate(${-r.x}px, ${-r.y}px)`, transformOrigin: '0 0' }}>
        <ShotVideo shot={shot} />
      </div>
    </div>
  );
};

// ---------- the full demo-scene layout ----------
function activeStep(scene: Scene, frame: number): number {
  for (const sh of scene.shots) {
    if (frame >= sh.start && frame < sh.start + sh.dur) {
      const rel = frame - sh.start;
      let step = sh.clips[0]?.step ?? 0;
      for (const c of sh.clips) if (rel >= c.start) step = c.step ?? step;
      return step;
    }
  }
  return 0;
}

export const StepChips: FC<{ steps: string[]; active: number }> = ({ steps, active }) => (
  <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
    {steps.map((s, i) => {
      const on = i === active;
      const done = i < active;
      return (
        <div
          key={s}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            padding: '10px 20px 10px 12px',
            borderRadius: 999,
            fontFamily: inter,
            fontWeight: 700,
            fontSize: 24,
            background: on ? C.orange : done ? C.orangeSoft : 'rgba(255,255,255,0.7)',
            color: on ? '#fff' : done ? C.orangeDeep : C.mute,
            boxShadow: on ? '0 8px 20px rgba(194,110,16,0.35)' : '0 0 0 1.5px rgba(21,18,14,0.08) inset',
            transform: `scale(${on ? 1.04 : 1})`,
          }}
        >
          <span
            style={{
              width: 30,
              height: 30,
              borderRadius: 15,
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 18,
              background: on ? 'rgba(255,255,255,0.25)' : done ? C.orange : 'rgba(21,18,14,0.08)',
              color: on || done ? '#fff' : C.mute,
            }}
          >
            {done ? '✓' : i + 1}
          </span>
          {s}
        </div>
      );
    })}
  </div>
);

function activeEff(scene: Scene, frame: number): number {
  for (const sh of scene.shots) {
    if (frame >= sh.start && frame < sh.start + sh.dur) {
      const rel = frame - sh.start;
      for (const c of sh.clips) if (rel >= c.start && rel < c.start + c.dur) return c.eff;
      return 1; // frozen tail
    }
  }
  return 1;
}

export const DemoScene: FC<{
  scene: Scene;
  tag: string;
  heading: ReactNode;
  steps?: string[];
  zoomLabel?: string;
  rightExtra?: ReactNode; // replaces the zoom card (e.g. pipeline cards)
  headingSize?: number;
}> = ({ scene, tag, heading, steps, zoomLabel = 'Live zoom', rightExtra, headingSize = 80 }) => {
  const frame = useCurrentFrame();
  const step = activeStep(scene, frame);
  const sped = activeEff(scene, frame) >= 1.5;
  const enter = ease(clamp01(frame / 22));
  const screenH = 900;
  const phoneW = (SRC_W * screenH) / SRC_H + 28;
  const phoneX = 450 - phoneW / 2;
  const phoneY = (1080 - (screenH + 28)) / 2 - 8;
  const ZB = { x: 900, y: 328, w: 960, h: 556 };
  return (
    <AbsoluteFill>
      {/* glow behind phone */}
      <div style={{ position: 'absolute', left: 450 - 330, top: 540 - 330, width: 660, height: 660, borderRadius: 330, background: 'radial-gradient(closest-side, rgba(194,110,16,0.20), rgba(194,110,16,0))' }} />
      <div style={{ transform: `translateY(${(1 - enter) * 60}px)`, opacity: enter }}>
        <PhoneFrame x={phoneX} y={phoneY} screenH={screenH}>
          {scene.shots.map((sh, i) => (
            <Sequence key={i} from={sh.start} durationInFrames={sh.dur}>
              <ShotVideo shot={sh} />
              {!rightExtra && <Highlight shot={sh} />}
            </Sequence>
          ))}
        </PhoneFrame>
      </div>

      {/* right column */}
      <div style={{ position: 'absolute', left: 900, top: 70, width: 960, display: 'flex', justifyContent: 'space-between', alignItems: 'center', opacity: enter }}>
        <Tag>{tag}</Tag>
        <Logo height={50} />
      </div>
      <div style={{ position: 'absolute', left: 900, top: 140, width: 960, opacity: enter, transform: `translateY(${(1 - enter) * 20}px)` }}>
        <Heading size={headingSize}>{heading}</Heading>
      </div>
      {steps && (
        <div style={{ position: 'absolute', left: 900, top: 248, opacity: enter }}>
          <StepChips steps={steps} active={step} />
        </div>
      )}
      {rightExtra ?? (
        <div
          style={{
            position: 'absolute',
            left: ZB.x,
            top: ZB.y,
            width: ZB.w,
            height: ZB.h,
            borderRadius: 30,
            background: C.paper,
            boxShadow: '0 30px 60px rgba(60,35,10,0.14), 0 0 0 1.5px rgba(21,18,14,0.06)',
            opacity: enter,
            transform: `translateY(${(1 - enter) * 30}px)`,
          }}
        >
          <div style={{ position: 'absolute', left: 24, top: -16 }}>
            <div style={{ background: C.ink, color: '#fff', fontFamily: inter, fontWeight: 700, fontSize: 17, letterSpacing: '0.16em', textTransform: 'uppercase', padding: '7px 16px', borderRadius: 999, display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ width: 9, height: 9, borderRadius: 5, background: C.orange, display: 'inline-block' }} />
              {zoomLabel}
            </div>
          </div>
          <div style={{ position: 'absolute', left: 20, top: 20, width: ZB.w - 40, height: ZB.h - 40 }}>
            {scene.shots.map((sh, i) => (
              <Sequence key={i} from={sh.start} durationInFrames={sh.dur}>
                <ZoomShot shot={sh} boxW={ZB.w - 40} boxH={ZB.h - 40} />
              </Sequence>
            ))}
          </div>
        </div>
      )}
      {/* the footage is sped up in places - say so rather than let timers look inconsistent */}
      <div style={{ position: 'absolute', left: 450 - 190, top: 1020, width: 380, display: 'flex', justifyContent: 'center', opacity: sped ? 1 : 0 }}>
        <div style={{ fontFamily: inter, fontWeight: 700, fontSize: 17, letterSpacing: '0.14em', color: C.orangeDeep, background: C.orangeSoft, padding: '6px 16px', borderRadius: 999, textTransform: 'uppercase', whiteSpace: 'nowrap' }}>⏩ Sped-up footage</div>
      </div>
      <Subtitles sentences={scene.sentences} pos="right" bottom={34} />
    </AbsoluteFill>
  );
};

export { oswald };
