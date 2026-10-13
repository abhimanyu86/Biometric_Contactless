import React, { FC } from 'react';
import { AbsoluteFill, interpolate, useCurrentFrame } from 'remotion';
import { ACCURACY_IS_MEASURED, C } from '../theme';
import { inter, oswald } from '../fonts';
import { Heading, Logo, Orange, Tag, ease, prog, usePop, Bolt, Fingerprint, Layers } from '../components/ui';
import { Subtitles } from '../components/Subtitles';
import { SCENE } from '../timeline';

const Tile: FC<{ big: string; small: string; icon: React.ReactNode; start: number }> = ({ big, small, icon, start }) => {
  const p = usePop(start, 15, 130);
  return (
    <div
      style={{
        width: 330,
        height: 230,
        borderRadius: 30,
        background: C.paper,
        padding: '26px 28px',
        boxShadow: '0 20px 44px rgba(60,35,10,0.12), 0 0 0 1.5px rgba(21,18,14,0.06)',
        transform: `translateY(${(1 - p) * 50}px) scale(${0.9 + 0.1 * p})`,
        opacity: Math.min(1, p * 1.5),
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
      }}
    >
      <div>{icon}</div>
      <div>
        <div style={{ fontFamily: oswald, fontWeight: 700, fontSize: 70, lineHeight: 1, color: C.ink }}>{big}</div>
        <div style={{ fontFamily: inter, fontWeight: 600, fontSize: 24, color: C.mute, marginTop: 6 }}>{small}</div>
      </div>
    </div>
  );
};

export const Accuracy: FC = () => {
  const frame = useCurrentFrame();
  const sc = SCENE.accuracy;
  const s = sc.shots[0].sentences[0];
  const count = interpolate(frame, [s.start + 10, s.start + 70], [0, 85], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: ease });
  const R = 250;
  const CIRC = 2 * Math.PI * R;
  const enter = prog(frame, 0, 24);
  return (
    <AbsoluteFill>
      <div style={{ position: 'absolute', left: 90, top: 54, opacity: enter }}>
        <Logo height={60} />
      </div>
      {/* gauge */}
      <div style={{ position: 'absolute', left: 120, top: 200, width: 640, height: 640, opacity: enter, transform: `scale(${0.92 + 0.08 * enter})` }}>
        <svg width={640} height={640} viewBox="0 0 640 640">
          <circle cx={320} cy={320} r={R} fill="none" stroke={C.orangeSoft} strokeWidth={44} />
          <circle
            cx={320}
            cy={320}
            r={R}
            fill="none"
            stroke={C.orange}
            strokeWidth={44}
            strokeLinecap="round"
            strokeDasharray={`${(count / 100) * CIRC} ${CIRC}`}
            transform="rotate(-90 320 320)"
          />
        </svg>
        <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
          {!ACCURACY_IS_MEASURED && (
            <div style={{ fontFamily: inter, fontWeight: 700, fontSize: 32, letterSpacing: '0.2em', color: C.orangeDeep, background: C.orangeSoft, padding: '8px 24px', borderRadius: 999, marginBottom: 10 }}>PoC TARGET</div>
          )}
          <div style={{ fontFamily: oswald, fontWeight: 700, fontSize: 190, lineHeight: 0.95, color: C.ink }}>
            {ACCURACY_IS_MEASURED ? '' : '>'}
            {Math.round(count)}
            <span style={{ fontSize: 92, color: C.orange }}>%</span>
          </div>
          <div style={{ fontFamily: inter, fontWeight: 700, fontSize: 25, letterSpacing: '0.1em', color: C.mute, marginTop: 10 }}>MATCHING ACCURACY</div>
        </div>
      </div>

      {!ACCURACY_IS_MEASURED && (
        <div style={{ position: 'absolute', left: 70, top: 856, width: 740, textAlign: 'center', whiteSpace: 'nowrap', fontFamily: inter, fontWeight: 600, fontSize: 22, color: C.mute, opacity: enter }}>
          PoC target · user-wise · contactless-to-contactless
        </div>
      )}
      <div style={{ position: 'absolute', left: 900, top: 215, width: 960 }}>
        <div style={{ opacity: enter }}>
          <Tag>{ACCURACY_IS_MEASURED ? 'Results' : 'PoC target · demo build'}</Tag>
          <div style={{ height: 24 }} />
          <Heading size={104}>
            Contactless.
            <br />
            <Orange>Match step &lt;1 sec.</Orange>
          </Heading>
        </div>
        <div style={{ display: 'flex', gap: 24, marginTop: 64 }}>
          <Tile big="<1 sec" small="Match step (demo build)" icon={<Bolt size={44} color={C.orange} />} start={s.start + 40} />
          <Tile big="1:N" small="Matched within the selected region" icon={<Fingerprint size={44} color={C.orange} />} start={s.start + 58} />
          <Tile big="4" small="Fingers in one slap capture" icon={<Layers size={44} color={C.orange} />} start={s.start + 76} />
        </div>
      </div>
      <Subtitles sentences={sc.sentences} pos="center" bottom={44} />
    </AbsoluteFill>
  );
};
