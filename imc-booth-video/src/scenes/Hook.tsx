import React, { FC } from 'react';
import { AbsoluteFill, Img, staticFile, useCurrentFrame } from 'remotion';
import { C, BOOTH } from '../theme';
import { inter, oswald } from '../fonts';
import { Hand, Logo, Phone, Scanner, prog, usePop } from '../components/ui';
import { Subtitles } from '../components/Subtitles';
import { SCENE } from '../timeline';

const Ridges: FC<{ cx: number; cy: number }> = ({ cx, cy }) => {
  const frame = useCurrentFrame();
  const items = Array.from({ length: 15 }, (_, i) => i);
  return (
    <svg width={1920} height={1080} style={{ position: 'absolute', inset: 0 }}>
      <g transform={`translate(${cx} ${cy}) scale(${1 + frame * 0.0007})`}>
        {items.map((i) => (
          <ellipse
            key={i}
            rx={70 + i * 34}
            ry={96 + i * 44}
            fill="none"
            stroke={C.orange}
            strokeOpacity={0.09 + (i % 3) * 0.015}
            strokeWidth={6}
            strokeLinecap="round"
            strokeDasharray={`${180 + i * 22} ${26 + (i % 4) * 10}`}
            transform={`rotate(${(i % 2 ? 1 : -1) * frame * 0.06 + i * 11})`}
          />
        ))}
      </g>
    </svg>
  );
};

const NoChip: FC<{ label: string; icon: React.ReactNode; slash?: boolean; hot?: boolean; start: number }> = ({ label, icon, slash, hot, start }) => {
  const p = usePop(start, 13, 150);
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        padding: '10px 22px 10px 14px',
        borderRadius: 999,
        background: hot ? C.orange : C.paper,
        color: hot ? '#fff' : C.ink,
        fontFamily: oswald,
        fontWeight: 600,
        fontSize: 33,
        letterSpacing: '0.04em',
        boxShadow: hot ? '0 14px 30px rgba(194,110,16,0.35)' : '0 10px 24px rgba(60,35,10,0.12), 0 0 0 1.5px rgba(21,18,14,0.07)',
        transform: `scale(${0.6 + 0.4 * p}) translateY(${(1 - p) * 20}px)`,
        opacity: Math.min(1, p * 1.6),
      }}
    >
      <div style={{ position: 'relative', width: 44, height: 44 }}>
        {icon}
        {slash && (
          <div style={{ position: 'absolute', left: -3, top: 20, width: 52, height: 4, background: C.orange, borderRadius: 3, transform: 'rotate(-42deg)', boxShadow: '0 0 0 2px #fff' }} />
        )}
      </div>
      {label}
    </div>
  );
};

export const Hook: FC = () => {
  const frame = useCurrentFrame();
  const sc = SCENE.hook;
  const s2 = sc.shots[0].sentences[1];
  const t1 = prog(frame, 6, 26);
  const t2 = prog(frame, 16, 26);
  const t3 = prog(frame, 34, 24);
  const heroP = prog(frame, 4, 34);
  return (
    <AbsoluteFill>
      <Ridges cx={1450} cy={520} />
      <div style={{ position: 'absolute', left: 90, top: 54, opacity: prog(frame, 0, 20) }}>
        <Logo height={78} />
      </div>

      {/* headline */}
      <div style={{ position: 'absolute', left: 84, top: 168, fontFamily: oswald, fontWeight: 700, fontSize: 222, lineHeight: 0.94, letterSpacing: '-0.005em' }}>
        <div style={{ overflow: 'hidden', height: 214 }}>
          <div style={{ color: C.ink, transform: `translateY(${(1 - t1) * 120}%)` }}>TRUST</div>
        </div>
        <div style={{ overflow: 'hidden', height: 214 }}>
          <div style={{ color: C.orange, transform: `translateY(${(1 - t2) * 120}%)` }}>INSTANTLY.</div>
        </div>
      </div>
      <div style={{ position: 'absolute', left: 90, top: 620, opacity: t3, transform: `translateY(${(1 - t3) * 16}px)` }}>
        <div style={{ width: 150 * t3, height: 6, borderRadius: 3, background: C.orange, marginBottom: 26 }} />
        <div style={{ fontFamily: inter, fontWeight: 600, fontSize: 48, lineHeight: 1.25, color: C.ink, width: 900 }}>
          Contactless Identity
          <br />
          Verification for the <span style={{ color: C.orange }}>AI Era</span>
        </div>
      </div>

      {/* "no scanner / no touch / just a camera" */}
      <div style={{ position: 'absolute', left: 90, top: 830, display: 'flex', gap: 16 }}>
        <NoChip label="NO SCANNER" icon={<Scanner size={44} color={C.ink} />} slash start={s2.start + 2} />
        <NoChip label="NO TOUCH" icon={<Hand size={44} color={C.ink} />} slash start={s2.start + 36} />
        <NoChip label="JUST A CAMERA" icon={<Phone size={44} color="#fff" />} hot start={s2.start + 66} />
      </div>

      {/* hero */}
      <div
        style={{
          position: 'absolute',
          left: 1090,
          top: 80,
          width: 730,
          height: 835,
          borderRadius: 48,
          overflow: 'hidden',
          boxShadow: '0 40px 80px rgba(60,35,10,0.30), 0 0 0 1.5px rgba(21,18,14,0.06)',
          transform: `translateY(${(1 - heroP) * 50}px) scale(${0.96 + 0.04 * heroP})`,
          opacity: heroP,
          background: C.orangeSoft,
        }}
      >
        <Img src={staticFile('media/assets/hero_man_phone.png')} style={{ width: '100%', height: '100%', objectFit: 'cover', objectPosition: 'center 20%' }} />
        <div style={{ position: 'absolute', inset: 0, background: 'linear-gradient(180deg, rgba(0,0,0,0) 62%, rgba(21,18,14,0.35))' }} />
        <div
          style={{
            position: 'absolute',
            left: 24,
            bottom: 22,
            fontFamily: inter,
            fontWeight: 700,
            fontSize: 20,
            letterSpacing: '0.16em',
            color: '#fff',
            textTransform: 'uppercase',
          }}
        >
          {BOOTH.eventShort} · {BOOTH.hall} · Booth {BOOTH.booth}
        </div>
        <div style={{ position: 'absolute', right: 20, top: 18, fontFamily: inter, fontWeight: 600, fontSize: 17, letterSpacing: '0.12em', color: '#fff', background: 'rgba(21,18,14,0.55)', padding: '5px 12px', borderRadius: 999, textTransform: 'uppercase' }}>
          Illustrative
        </div>
      </div>
      <Subtitles sentences={sc.sentences} pos="center" bottom={44} />
    </AbsoluteFill>
  );
};
