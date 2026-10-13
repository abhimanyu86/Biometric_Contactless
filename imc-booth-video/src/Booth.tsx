import React, { FC } from 'react';
import { AbsoluteFill, Audio, Sequence, staticFile } from 'remotion';
import { Backdrop, Orange, SceneWrap } from './components/ui';
import { DemoScene } from './components/Demo';
import { SCENES, SCENE } from './timeline';
import { Hook } from './scenes/Hook';
import { Problem } from './scenes/Problem';
import { Pipeline } from './scenes/Pipeline';
import { Accuracy } from './scenes/Accuracy';
import { Pillars } from './scenes/Pillars';
import { Closing } from './scenes/Closing';

const renderScene = (id: string) => {
  switch (id) {
    case 'hook':
      return <Hook />;
    case 'problem':
      return <Problem />;
    case 'enrol':
      return (
        <DemoScene
          scene={SCENE.enrol}
          tag="Single finger · Enrollment"
          heading={<>Enroll in a <Orange>few steps</Orange></>}
          steps={['Details', 'Capture', 'Checks', 'Enrolled']}
        />
      );
    case 'auth':
      return (
        <DemoScene
          scene={SCENE.auth}
          tag="Single finger · Authentication"
          heading={<>Authenticate with <Orange>one scan</Orange></>}
          headingSize={68}
          steps={['Open', 'Capture', 'Match', 'Matched']}
        />
      );
    case 'pipeline':
      return <Pipeline />;
    case 'slap':
      return (
        <DemoScene
          scene={SCENE.slap}
          tag="Slap mode · 4 fingers"
          heading={<>Four fingers. <Orange>One shot.</Orange></>}
          steps={['Enroll', 'Authenticate', 'Per-finger view']}
        />
      );
    case 'accuracy':
      return <Accuracy />;
    case 'pillars':
      return <Pillars />;
    case 'closing':
      return <Closing />;
    default:
      return null;
  }
};

export const Booth: FC<{ withAudio?: boolean }> = ({ withAudio = true }) => (
  <AbsoluteFill style={{ background: '#FBF7F0' }}>
    <Backdrop />
    {SCENES.map((sc) => (
      <Sequence key={sc.id} from={sc.start} durationInFrames={sc.dur} name={sc.id}>
        <SceneWrap dur={sc.dur}>{renderScene(sc.id)}</SceneWrap>
      </Sequence>
    ))}
    {withAudio &&
      SCENES.flatMap((sc) =>
        sc.shots.flatMap((sh) =>
          sh.sentences.map((se, i) => (
            <Sequence key={`${sc.id}-${sh.start}-${i}`} from={sc.start + sh.start + se.start} durationInFrames={se.dur + 6}>
              <Audio src={staticFile(se.file)} />
            </Sequence>
          )),
        ),
      )}
  </AbsoluteFill>
);
