import React, { FC } from 'react';
import { useCurrentFrame } from 'remotion';
import { C } from '../theme';
import { inter, oswald } from '../fonts';
import { Check, Orange, prog, ease, clamp01, Fingerprint, ShieldCheck, Layers, Sparkles, Cloud } from '../components/ui';
import { DemoScene } from '../components/Demo';
import { SCENE } from '../timeline';

const STAGES = [
  { n: 'DETECT', model: 'YOLO', desc: 'Finds and crops the fingertip', icon: (c: string) => <Fingerprint size={40} color={c} /> },
  { n: 'LIVENESS', model: 'Liveness check', desc: 'Classifies each capture as live or not', icon: (c: string) => <ShieldCheck size={40} color={c} /> },
  { n: 'SEGMENT', model: 'U²-Net', desc: 'Isolates the fingerprint region', icon: (c: string) => <Layers size={40} color={c} /> },
  { n: 'ENHANCE', model: 'Zero-DCE', desc: 'Brings out ridge detail (low-light correction when needed)', icon: (c: string) => <Sparkles size={40} color={c} /> },
  { n: 'TEMPLATE & MATCH', model: 'Cloud', desc: 'Builds the template and matches it in the cloud', icon: (c: string) => <Cloud size={40} color={c} /> },
];

const Cards: FC = () => {
  const frame = useCurrentFrame();
  const sents = SCENE.pipeline.shots[0].sentences; // [0]=intro, [1..5]=one per stage
  const CH = 110;
  const GAP = 10;
  return (
    <div style={{ position: 'absolute', left: 900, top: 252, width: 960 }}>
      {STAGES.map((s, i) => {
        const start = sents[i + 1].start;
        const end = i + 2 < sents.length ? sents[i + 2].start : sents[i + 1].start + sents[i + 1].dur + 60;
        const enter = prog(frame, 10 + i * 6, 22);
        const active = frame >= start && frame < end;
        const done = frame >= end;
        const sweep = clamp01((frame - start) / Math.max(10, end - start - 6));
        const col = active ? C.orange : done ? C.green : C.mute;
        return (
          <div
            key={s.n}
            style={{
              position: 'absolute',
              top: i * (CH + GAP),
              left: 0,
              width: 960,
              height: CH,
              borderRadius: 26,
              background: active || done ? C.paper : 'rgba(255,255,255,0.55)',
              boxShadow: active ? `0 18px 40px rgba(194,110,16,0.28), 0 0 0 3px ${C.orange}` : '0 8px 22px rgba(60,35,10,0.08), 0 0 0 1.5px rgba(21,18,14,0.06)',
              opacity: enter * (active || done ? 1 : 0.62),
              transform: `translateX(${(1 - enter) * 70}px) scale(${active ? 1.015 : 1})`,
              overflow: 'hidden',
              display: 'flex',
              alignItems: 'center',
              padding: '0 28px',
              gap: 22,
            }}
          >
            <div style={{ width: 62, height: 62, borderRadius: 31, background: active ? C.orangeSoft : done ? C.greenSoft : 'rgba(21,18,14,0.05)', display: 'flex', alignItems: 'center', justifyContent: 'center', flex: '0 0 auto' }}>
              {done ? <Check size={38} color={C.green} stroke={2.4} /> : s.icon(col)}
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                <div style={{ fontFamily: oswald, fontWeight: 700, fontSize: 40, letterSpacing: '0.03em', color: C.ink }}>
                  <span style={{ color: C.orange, marginRight: 14 }}>{i + 1}</span>
                  {s.n}
                </div>
                <div style={{ fontFamily: inter, fontWeight: 700, fontSize: 21, padding: '5px 14px', borderRadius: 999, background: active ? C.orange : C.orangeSoft, color: active ? '#fff' : C.orangeDeep, letterSpacing: '0.04em' }}>{s.model}</div>
              </div>
              <div style={{ fontFamily: inter, fontWeight: 500, fontSize: 25, color: C.mute, marginTop: 2 }}>{s.desc}</div>
            </div>
            {active && <div style={{ position: 'absolute', left: 0, bottom: 0, height: 6, width: `${sweep * 100}%`, background: C.orange }} />}
          </div>
        );
      })}
    </div>
  );
};

export const Pipeline: FC = () => (
  <DemoScene scene={SCENE.pipeline} tag="5-stage pipeline" heading={<>Under the <Orange>hood</Orange></>} rightExtra={<Cards />} />
);
