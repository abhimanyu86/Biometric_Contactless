import React from 'react';
import { Composition } from 'remotion';
import { Booth } from './Booth';
import { TOTAL_FRAMES } from './timeline';
import { FPS, H, W } from './theme';

export const Root: React.FC = () => (
  <>
    {/* Voiced version (narration + burned-in subtitles) */}
    <Composition id="Booth" component={Booth} durationInFrames={TOTAL_FRAMES} fps={FPS} width={W} height={H} defaultProps={{ withAudio: true }} />
    {/* Silent loop (burned-in subtitles only) — same visuals, no audio track */}
    <Composition id="BoothSilent" component={Booth} durationInFrames={TOTAL_FRAMES} fps={FPS} width={W} height={H} defaultProps={{ withAudio: false }} />
  </>
);
