import React, { FC } from 'react';
import { AbsoluteFill, useCurrentFrame } from 'remotion';
import { C, SHOW_UNDEMONSTRATED_POSTER_CLAIMS } from '../theme';
import { inter, oswald } from '../fonts';
import { Bolt, Heading, Layers, Logo, Orange, ShieldCheck, Sparkles, Tag, prog, usePop, Hand } from '../components/ui';
import { Subtitles } from '../components/Subtitles';
import { SCENE } from '../timeline';

// Wording is taken from the YellowSense poster.
const PILLARS = [
  { title: 'Secure Authentication', body: 'Built for privacy and compliance', icon: (c: string) => <ShieldCheck size={72} color={c} />, at: 1.55 },
  { title: 'Real-Time Verification', body: 'Match step in under a second', icon: (c: string) => <Bolt size={72} color={c} />, at: 2.5 },
  { title: 'AI-Powered Intelligence', body: 'Advanced models for smarter decisions', icon: (c: string) => <Sparkles size={72} color={c} />, at: 3.55 },
  { title: 'Enterprise Ready', body: 'Scalable, reliable and future-ready', icon: (c: string) => <Layers size={72} color={c} />, at: 4.85 },
];
// Poster claims the demo footage does not show (see theme.ts flag).
const EXTRAS = [
  { title: 'Deepfake Detection', body: 'AI-powered protection against synthetic threats', icon: (c: string) => <ShieldCheck size={72} color={c} />, at: 3.3 },
  { title: 'Fake App Detection', body: 'Detect and block malicious applications in real-time', icon: (c: string) => <Hand size={72} color={c} />, at: 3.6 },
];

const Card: FC<{ title: string; body: string; icon: (c: string) => React.ReactNode; start: number; w: number; h: number; x: number; y: number }> = ({ title, body, icon, start, w, h, x, y }) => {
  const p = usePop(start, 15, 140);
  return (
    <div
      style={{
        position: 'absolute',
        left: x,
        top: y,
        width: w,
        height: h,
        borderRadius: 32,
        background: C.paper,
        boxShadow: '0 22px 48px rgba(60,35,10,0.13), 0 0 0 1.5px rgba(21,18,14,0.06)',
        display: 'flex',
        alignItems: 'center',
        gap: 28,
        padding: '0 34px',
        transform: `translateY(${(1 - p) * 60}px) scale(${0.9 + 0.1 * p})`,
        opacity: Math.min(1, p * 1.5),
      }}
    >
      <div style={{ width: 112, height: 112, borderRadius: 56, background: C.orangeSoft, display: 'flex', alignItems: 'center', justifyContent: 'center', flex: '0 0 auto' }}>{icon(C.orange)}</div>
      <div>
        <div style={{ fontFamily: oswald, fontWeight: 700, fontSize: 50, lineHeight: 1.05, color: C.orangeDeep, textTransform: 'uppercase' }}>{title}</div>
        <div style={{ fontFamily: inter, fontWeight: 500, fontSize: 28, color: C.ink, marginTop: 8, lineHeight: 1.3 }}>{body}</div>
      </div>
    </div>
  );
};

export const Pillars: FC = () => {
  const frame = useCurrentFrame();
  const sc = SCENE.pillars;
  const s = sc.shots[0].sentences[0];
  const enter = prog(frame, 0, 22);
  const items = SHOW_UNDEMONSTRATED_POSTER_CLAIMS ? [...PILLARS, ...EXTRAS] : PILLARS;
  const cols = items.length > 4 ? 3 : 2;
  const rows = Math.ceil(items.length / cols);
  const gx = 40;
  const gy = 40;
  const gridW = 1740;
  const w = (gridW - gx * (cols - 1)) / cols;
  const h = 250;
  const x0 = 90;
  const y0 = 350;
  return (
    <AbsoluteFill>
      <div style={{ position: 'absolute', left: 90, top: 54, opacity: enter }}>
        <Logo height={60} />
      </div>
      <div style={{ position: 'absolute', left: 90, top: 150, opacity: enter, transform: `translateY(${(1 - enter) * 20}px)` }}>
        <Tag>Design goals</Tag>
        <div style={{ height: 22 }} />
        <Heading size={96}>
          Designed for <Orange>trust</Orange>
        </Heading>
      </div>
      {items.map((it, i) => (
        <Card key={it.title} title={it.title} body={it.body} icon={it.icon} start={s.start + Math.round(it.at * 30)} w={w} h={h} x={x0 + (i % cols) * (w + gx)} y={y0 + Math.floor(i / cols) * (h + gy)} />
      ))}
      <Subtitles sentences={sc.sentences} pos="center" bottom={44} />
    </AbsoluteFill>
  );
};
