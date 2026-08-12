import {Composition} from 'remotion';

import {ArmBenchDemo, ArmBenchPoster, DEMO_FRAMES, FPS} from './video';

export const Root = () => (
  <>
    <Composition
      id="ArmBenchDemo"
      component={ArmBenchDemo}
      durationInFrames={DEMO_FRAMES}
      fps={FPS}
      width={1920}
      height={1080}
    />
    <Composition
      id="ArmBenchPoster"
      component={ArmBenchPoster}
      durationInFrames={1}
      fps={FPS}
      width={1800}
      height={1200}
    />
  </>
);
