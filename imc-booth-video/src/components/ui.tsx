import React, { FC, ReactNode } from 'react';
import { AbsoluteFill, Easing, Img, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig } from 'remotion';
import { C } from '../theme';
import { inter, oswald } from '../fonts';

export const ease = Easing.bezier(0.22, 1, 0.36, 1);
export const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

/** 0 -> 1 ease-out progress starting at `start` frames, lasting `dur` frames. */
export const prog = (frame: number, start: number, dur = 18) =>
  interpolate(frame, [start, start + dur], [0, 1], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: ease });

/** Bouncy pop-in 0 -> 1. */
export const usePop = (start: number, damping = 14, stiffness = 130) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return spring({ frame: frame - start, fps, config: { damping, stiffness, mass: 0.8 } });
};

export const Logo: FC<{ height?: number; onDark?: boolean; style?: React.CSSProperties }> = ({ height = 64, onDark, style }) => (
  <Img
    src={staticFile(onDark ? 'media/assets/logo_yellowsense_onDark.png' : 'media/assets/logo_yellowsense.png')}
    style={{ height, width: 'auto', display: 'block', ...style }}
  />
);

export const Mark: FC<{ size?: number; style?: React.CSSProperties }> = ({ size = 80, style }) => (
  <Img src={staticFile('media/assets/logo_mark.png')} style={{ height: size, width: 'auto', display: 'block', ...style }} />
);

export const Tag: FC<{ children: ReactNode; style?: React.CSSProperties }> = ({ children, style }) => (
  <div
    style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: 10,
      background: C.orangeSoft,
      color: C.orangeDeep,
      fontFamily: inter,
      fontWeight: 700,
      fontSize: 22,
      letterSpacing: '0.14em',
      textTransform: 'uppercase',
      padding: '9px 20px',
      borderRadius: 999,
      ...style,
    }}
  >
    {children}
  </div>
);

export const Heading: FC<{ children: ReactNode; size?: number; style?: React.CSSProperties }> = ({ children, size = 76, style }) => (
  <div style={{ fontFamily: oswald, fontWeight: 700, fontSize: size, lineHeight: 1.0, letterSpacing: '0.005em', color: C.ink, textTransform: 'uppercase', ...style }}>
    {children}
  </div>
);

export const Orange: FC<{ children: ReactNode }> = ({ children }) => <span style={{ color: C.orange }}>{children}</span>;

/** Soft cream backdrop with a faint dot grid and warm glow — constant across scenes. */
export const Backdrop: FC = () => (
  <AbsoluteFill style={{ background: C.cream }}>
    <AbsoluteFill
      style={{
        background:
          'radial-gradient(900px 700px at 88% 8%, rgba(194,110,16,0.13), rgba(194,110,16,0) 70%), radial-gradient(800px 700px at 4% 100%, rgba(201,138,28,0.10), rgba(201,138,28,0) 70%)',
      }}
    />
    <AbsoluteFill
      style={{
        backgroundImage: 'radial-gradient(rgba(21,18,14,0.07) 1.2px, transparent 1.3px)',
        backgroundSize: '34px 34px',
        opacity: 0.55,
      }}
    />
  </AbsoluteFill>
);

/** Fades a scene's content in/out against the constant backdrop (dip-to-cream transition). */
export const SceneWrap: FC<{ dur: number; children: ReactNode; fadeIn?: number; fadeOut?: number }> = ({ dur, children, fadeIn = 10, fadeOut = 10 }) => {
  const f = useCurrentFrame();
  const o = interpolate(f, [0, fadeIn, dur - fadeOut, dur], [0, 1, 1, 0], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' });
  return <AbsoluteFill style={{ opacity: o }}>{children}</AbsoluteFill>;
};

// ---------- icons (stroke style, 24x24 viewBox) ----------
type IconProps = { size?: number; color?: string; stroke?: number };
const Svg: FC<IconProps & { children: ReactNode }> = ({ size = 56, color = C.orange, stroke = 1.7, children }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={stroke} strokeLinecap="round" strokeLinejoin="round">
    {children}
  </svg>
);
export const ShieldCheck: FC<IconProps> = (p) => (
  <Svg {...p}>
    <path d="M12 2.5 L19.5 5.4 V11.2 C19.5 16 16.3 19.7 12 21.5 C7.7 19.7 4.5 16 4.5 11.2 V5.4 Z" />
    <path d="M8.6 12.1 L11 14.5 L15.6 9.6" />
  </Svg>
);
export const Bolt: FC<IconProps> = (p) => (
  <Svg {...p}>
    <path d="M13.2 2.5 L5 13.4 H11 L10.2 21.5 L19 10.2 H13 Z" />
  </Svg>
);
export const Sparkles: FC<IconProps> = (p) => (
  <Svg {...p}>
    <path d="M11 3 L13 9 L19 11 L13 13 L11 19 L9 13 L3 11 L9 9 Z" />
    <path d="M19 3.5 L19.8 5.7 L22 6.5 L19.8 7.3 L19 9.5 L18.2 7.3 L16 6.5 L18.2 5.7 Z" />
  </Svg>
);
export const Layers: FC<IconProps> = (p) => (
  <Svg {...p}>
    <path d="M12 3 L21.5 8 L12 13 L2.5 8 Z" />
    <path d="M2.5 12 L12 17 L21.5 12" />
    <path d="M2.5 16 L12 21 L21.5 16" />
  </Svg>
);
export const Cloud: FC<IconProps> = (p) => (
  <Svg {...p}>
    <path d="M7 18 a4.2 4.2 0 0 1 -.6 -8.3 A5.6 5.6 0 0 1 17.2 8.6 A4.7 4.7 0 0 1 17.5 18 Z" />
  </Svg>
);
export const Check: FC<IconProps> = (p) => (
  <Svg {...p}>
    <path d="M5 12.5 L10 17.5 L19.5 7" />
  </Svg>
);
export const Hand: FC<IconProps> = (p) => (
  <Svg {...p}>
    <path d="M8 13 V5.5 a1.5 1.5 0 0 1 3 0 V11 M11 10.5 V4 a1.5 1.5 0 0 1 3 0 V10.5 M14 10.5 V5.5 a1.5 1.5 0 0 1 3 0 V13 M17 9.5 a1.5 1.5 0 0 1 3 0 V15 c0 4 -2.5 6.5 -6.5 6.5 h-1 c-2.5 0 -4 -1 -5.5 -3 L4 14 a1.5 1.5 0 0 1 2.4 -1.8 L8 14" />
  </Svg>
);
export const Phone: FC<IconProps> = (p) => (
  <Svg {...p}>
    <rect x="7" y="2.5" width="10" height="19" rx="2.2" />
    <circle cx="12" cy="18.6" r="0.6" fill={p.color ?? C.orange} />
  </Svg>
);
export const Fingerprint: FC<IconProps> = (p) => (
  <Svg {...p}>
    <path d="M12 3.2 C7.6 3.2 4.7 6.3 4.7 10.2 V13" />
    <path d="M19.3 12.5 V10.2 C19.3 6.3 16.4 3.2 12 3.2" />
    <path d="M7.7 10.5 C7.7 7.9 9.5 6.2 12 6.2 s4.3 1.7 4.3 4.3 V13.5 c0 2.2 .5 4 1.3 5.4" />
    <path d="M10 10.8 V14.5 c0 2.2 -.5 4.2 -1.4 5.7" />
    <path d="M12 10.2 V14.7 c0 2.7 .5 4.4 1.2 5.9" />
    <path d="M4.7 16 c0 2.1 -.4 3.3 -1 4" />
  </Svg>
);
export const Scanner: FC<IconProps> = (p) => (
  <Svg {...p}>
    <rect x="3" y="12" width="18" height="8.5" rx="1.8" />
    <rect x="6.5" y="4" width="11" height="8" rx="1.6" />
    <path d="M9 8 h6" />
    <path d="M7 16.2 h2.5" />
  </Svg>
);
