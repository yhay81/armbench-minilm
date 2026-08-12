# ArmBench MiniLM demo video

This directory contains the reproducible source for the 74-second English demo video. Its claims
are imported directly from the retained benchmark JSON files rather than copied into the animation
by hand.

## Storyboard

1. Optimization claims versus evidence
2. One-command reproduction
3. FP32-to-INT8 transformation
4. Native Arm64 latency by batch
5. Predeclared BF16 downstream quality gate
6. Evidence and provenance chain
7. Closing proposition

The video is readable without sound. The accompanying low-volume ambient bed is synthesized by
FFmpeg from mathematical waveforms and pink noise, so no third-party music or recorded media is
distributed.

## Render

Requirements: Node.js 24, pnpm, and FFmpeg/FFprobe.

```powershell
pnpm install --frozen-lockfile
pnpm run check
pnpm run render
```

When only the final playback encoding or synthesized audio has changed, an existing silent render
can be reused explicitly:

```powershell
pwsh -File .\scripts\render.ps1 -ReuseSilent
```

Outputs:

- `assets/armbench-demo.mp4` — 1920x1080, 30 fps, H.264/AAC, 74 seconds
- `assets/armbench-demo-poster.png` — 1800x1200 poster image

Run `pnpm run verify` to check duration, streams, dimensions, pixel format, size, and SHA-256.

## Tools and licensing

- Video composition: [Remotion 4.0.508](https://www.remotion.dev/), used under its free
  individual-use license for this individual competition entry.
- Encoding and original synthesized ambience: [FFmpeg 8.0.1](https://ffmpeg.org/).
- Demo source: MIT, matching the parent project.

No Arm logo, third-party music, stock footage, generated voice, or private data is included.
