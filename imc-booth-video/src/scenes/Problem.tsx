import React, { FC } from 'react';
import { AbsoluteFill, useCurrentFrame } from 'remotion';
import { C } from '../theme';
import { inter, oswald } from '../fonts';
import { Heading, Logo, Orange, Scanner, Tag, prog, usePop, Phone, Hand, Fingerprint, Layers } from '../components/ui';
import { Subtitles } from '../components/Subtitles';
import { Highlight, PhoneFrame, ShotVideo } from '../components/Demo';
import { SCENE } from '../timeline';

const Chip: FC<{ text: string; start: number; icon?: React.ReactNode; hot?: boolean }> = ({ text, start, icon, hot }) => {
  const p = usePop(start, 14, 140);
  return (
    <div
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 12,
        padding: '12px 26px 12px 18px',
        borderRadius: 999,
        background: hot ? C.orange : C.paper,
        color: hot ? '#fff' : C.ink,
        fontFamily: inter,
        fontWeight: 700,
        fontSize: 30,
        boxShadow: hot ? '0 12px 26px rgba(194,110,16,0.30)' : '0 8px 20px rgba(60,35,10,0.10), 0 0 0 1.5px rgba(21,18,14,0.07)',
        transform: `scale(${0.7 + 0.3 * p}) translateX(${(1 - p) * -24}px)`,
        opacity: Math.min(1, p * 1.6),
      }}
    >
      {icon}
      {text}
    </div>
  );
};

export const Problem: FC = () => {
  const frame = useCurrentFrame();
  const sc = SCENE.problem;
  const shot = sc.shots[0];
  const s1 = shot.sentences[0];
  const s2 = shot.sentences[1];
  const s3 = shot.sentences[2];
  const flip = prog(frame, s2.start - 10, 20); // 0 -> 1 when the idea starts
  const enter = prog(frame, 0, 22);

  // delay the home clip until the phone is on screen
  const phoneShot = { ...shot, clips: shot.clips.map((c) => ({ ...c, start: Math.max(0, s2.start - 6) })), regions: [] };

  const phoneScreenH = 900;
  const phoneW = (720 * phoneScreenH) / 1560 + 28;
  const phoneX = 450 - phoneW / 2;
  const phoneY = (1080 - (phoneScreenH + 28)) / 2 - 8;

  return (
    <AbsoluteFill>
      <div style={{ position: 'absolute', left: 900, top: 56, width: 960, display: 'flex', justifyContent: 'flex-end', opacity: enter }}>
        <Logo height={56} />
      </div>

      {/* left visual: scanner card -> phone */}
      <div style={{ position: 'absolute', left: 450 - 330, top: 540 - 330, width: 660, height: 660, borderRadius: 330, background: 'radial-gradient(closest-side, rgba(194,110,16,0.18), rgba(194,110,16,0))', opacity: flip }} />
      <div
        style={{
          position: 'absolute',
          left: 450 - 270,
          top: 250,
          width: 540,
          height: 540,
          borderRadius: 44,
          background: C.paper,
          boxShadow: '0 30px 60px rgba(60,35,10,0.14), 0 0 0 1.5px rgba(21,18,14,0.06)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 18,
          opacity: enter * (1 - flip),
          transform: `translateY(${(1 - enter) * 40}px) translateX(${-flip * 120}px) scale(${1 - flip * 0.15})`,
        }}
      >
        <Scanner size={300} color="#8B8378" stroke={1.2} />
        <div style={{ fontFamily: oswald, fontWeight: 600, fontSize: 36, letterSpacing: '0.08em', color: C.mute, textAlign: 'center', padding: '0 30px', lineHeight: 1.1 }}>
          DEDICATED
          <br />
          CONTACT SCANNER
        </div>
      </div>
      <div style={{ opacity: flip, transform: `translateY(${(1 - flip) * 70}px)` }}>
        <PhoneFrame x={phoneX} y={phoneY} screenH={phoneScreenH}>
          <ShotVideo shot={phoneShot as any} />
        </PhoneFrame>
      </div>

      {/* right: copy */}
      <div style={{ position: 'absolute', left: 900, top: 190, width: 960 }}>
        <div style={{ position: 'absolute', opacity: 1 - flip, transform: `translateY(${-flip * 30}px)` }}>
          <div style={{ opacity: enter }}>
            <Tag>The challenge</Tag>
            <div style={{ height: 26 }} />
            <Heading size={104}>
              Fingerprint auth
              <br />
              needs <Orange>dedicated</Orange>
              <br />
              scanners.
            </Heading>
          </div>
          <div style={{ display: 'flex', gap: 18, marginTop: 56 }}>
            <Chip text="Buy" start={s1.start + 122} icon={<Layers size={34} color={C.mute} />} />
            <Chip text="Deploy" start={s1.start + 142} icon={<Hand size={34} color={C.mute} />} />
            <Chip text="Maintain" start={s1.start + 164} icon={<Fingerprint size={34} color={C.mute} />} />
          </div>
        </div>
        <div style={{ position: 'absolute', opacity: flip, transform: `translateY(${(1 - flip) * 40}px)` }}>
          <Tag>The idea</Tag>
          <div style={{ height: 26 }} />
          <Heading size={104}>
            The phone camera
            <br />
            becomes the <Orange>reader.</Orange>
          </Heading>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 18, marginTop: 56, alignItems: 'flex-start' }}>
            <Chip text="Just a smartphone camera" start={s2.start + 55} icon={<Phone size={34} color={C.orange} />} />
            <Chip text="Contactless capture" start={s2.start + 112} icon={<Hand size={34} color="#fff" />} hot />
            <Chip text="Enroll + Authenticate in one app" start={s3.start + 8} icon={<Fingerprint size={34} color={C.orange} />} />
          </div>
        </div>
      </div>
      <Subtitles sentences={sc.sentences} pos="right" bottom={40} />
    </AbsoluteFill>
  );
};
