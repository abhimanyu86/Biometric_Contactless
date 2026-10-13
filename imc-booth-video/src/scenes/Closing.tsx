import React, { FC } from 'react';
import { AbsoluteFill, Img, staticFile, useCurrentFrame } from 'remotion';
import { BOOTH, C, SHOW_QR } from '../theme';
import { inter, oswald } from '../fonts';
import { Logo, Orange, Tag, prog, usePop } from '../components/ui';
import { Subtitles } from '../components/Subtitles';
import { SCENE } from '../timeline';

const IconGlobe: FC = () => (
  <svg width="38" height="38" viewBox="0 0 24 24" fill="none" stroke={C.orange} strokeWidth="1.8" strokeLinecap="round">
    <circle cx="12" cy="12" r="9" />
    <path d="M3 12h18M12 3c3 3.2 3 14.8 0 18M12 3c-3 3.2-3 14.8 0 18" />
  </svg>
);
const IconMail: FC = () => (
  <svg width="38" height="38" viewBox="0 0 24 24" fill="none" stroke={C.orange} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="5.5" width="18" height="13" rx="2.2" />
    <path d="M3.5 7 L12 13 L20.5 7" />
  </svg>
);
const IconPhone: FC = () => (
  <svg width="38" height="38" viewBox="0 0 24 24" fill="none" stroke={C.orange} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M5 4.5 h3.5 l1.6 4 -2 1.4 a11 11 0 0 0 5 5 l1.4 -2 4 1.6 V18.5 a2 2 0 0 1 -2 2 C10.4 20.5 3.5 13.6 3.5 6.5 a2 2 0 0 1 1.5 -2 Z" />
  </svg>
);

export const Closing: FC = () => {
  const frame = useCurrentFrame();
  const sc = SCENE.closing;
  const [b14, b15] = sc.shots;
  const sVision = b14.sentences[0];
  const sBrand = b14.sentences[1];
  const sBooth = { start: b15.start + b15.sentences[0].start };

  const vis = prog(frame, 4, 24) * (1 - prog(frame, sBrand.start - 8, 14));
  const brandP = prog(frame, sBrand.start - 6, 26);
  const boothP = usePop(sBooth.start - 8, 15, 120);
  const contactP = usePop(sBooth.start + 26, 15, 120);
  const qrP = usePop(sBooth.start + 44, 14, 130);

  return (
    <AbsoluteFill>
      {/* phase 1: vision statement */}
      <div style={{ position: 'absolute', left: 160, top: 250, width: 1600, textAlign: 'center', opacity: vis, transform: `translateY(${(1 - vis) * 30}px)` }}>
        <Tag>Our vision</Tag>
        <div style={{ height: 34 }} />
        <div style={{ fontFamily: oswald, fontWeight: 700, fontSize: 112, lineHeight: 1.04, color: C.ink, textTransform: 'uppercase' }}>
          A world where every <Orange>identity</Orange>
          <br />
          is verified securely and seamlessly.
        </div>
      </div>

      {/* phase 2: brand */}
      <div style={{ position: 'absolute', left: 90, top: 84, opacity: brandP, transform: `translateY(${(1 - brandP) * 24}px)` }}>
        <Logo height={116} />
      </div>
      <div style={{ position: 'absolute', left: 84, top: 262, opacity: brandP, transform: `translateY(${(1 - brandP) * 40}px)`, fontFamily: oswald, fontWeight: 700, fontSize: 200, lineHeight: 0.94 }}>
        <div style={{ color: C.ink }}>TRUST</div>
        <div style={{ color: C.orange }}>INSTANTLY.</div>
      </div>
      <div style={{ position: 'absolute', left: 90, top: 690, opacity: brandP, fontFamily: inter, fontWeight: 600, fontSize: 38, lineHeight: 1.3, color: C.ink, width: 900 }}>
        Contactless Identity Verification for the <Orange>AI Era</Orange>
      </div>

      {/* phase 3: booth + contact */}
      <div
        style={{
          position: 'absolute',
          left: 1060,
          top: 84,
          width: 770,
          height: 462,
          borderRadius: 44,
          background: `linear-gradient(135deg, ${C.orange}, ${C.orangeDeep})`,
          color: '#fff',
          padding: '34px 46px 40px',
          boxShadow: '0 36px 70px rgba(168,90,10,0.38)',
          opacity: Math.min(1, boothP * 1.6),
          transform: `translateY(${(1 - boothP) * 60}px) scale(${0.94 + 0.06 * boothP})`,
        }}
      >
        <div style={{ fontFamily: inter, fontWeight: 700, fontSize: 26, letterSpacing: '0.2em', opacity: 0.9 }}>VISIT US AT</div>
        <div style={{ fontFamily: oswald, fontWeight: 600, fontSize: 54, letterSpacing: '0.04em', marginTop: 8, textTransform: 'uppercase', whiteSpace: 'nowrap' }}>
          {BOOTH.event}
        </div>
        <div style={{ height: 3, background: 'rgba(255,255,255,0.45)', margin: '22px 0 14px' }} />
        <div style={{ fontFamily: oswald, fontWeight: 700, fontSize: 124, lineHeight: 1.0, textTransform: 'uppercase', whiteSpace: 'nowrap' }}>{BOOTH.hall}</div>
        <div style={{ fontFamily: oswald, fontWeight: 700, fontSize: 124, lineHeight: 1.0, textTransform: 'uppercase', whiteSpace: 'nowrap' }}>Booth {BOOTH.booth}</div>
      </div>

      <div
        style={{
          position: 'absolute',
          left: 1060,
          top: 580,
          width: 770,
          height: 296,
          borderRadius: 36,
          background: C.paper,
          boxShadow: '0 24px 50px rgba(60,35,10,0.14), 0 0 0 1.5px rgba(21,18,14,0.06)',
          padding: '0 36px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          opacity: Math.min(1, contactP * 1.6),
          transform: `translateY(${(1 - contactP) * 50}px)`,
        }}
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: 26 }}>
          {[
            [<IconGlobe />, BOOTH.web],
            [<IconMail />, BOOTH.email],
            [<IconPhone />, BOOTH.phone],
          ].map(([icon, text], i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 18, fontFamily: inter, fontWeight: 600, fontSize: 31, color: C.ink }}>
              <div style={{ width: 56, height: 56, borderRadius: 28, background: C.orangeSoft, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>{icon}</div>
              {text}
            </div>
          ))}
        </div>
        {SHOW_QR && (
        <div style={{ textAlign: 'center', opacity: Math.min(1, qrP * 1.6), transform: `scale(${0.8 + 0.2 * qrP})` }}>
          <div style={{ width: 236, height: 236, borderRadius: 22, background: '#fff', boxShadow: '0 0 0 2px rgba(21,18,14,0.12)', padding: 6 }}>
            <Img src={staticFile('media/assets/qr_clean.png')} style={{ width: '100%', height: '100%' }} />
          </div>
          <div style={{ fontFamily: inter, fontWeight: 700, fontSize: 22, letterSpacing: '0.18em', color: C.orangeDeep, marginTop: 10 }}>EXPLORE MORE</div>
        </div>
        )}
      </div>
      <Subtitles sentences={sc.sentences} pos="center" bottom={44} />
    </AbsoluteFill>
  );
};
