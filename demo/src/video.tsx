import type {CSSProperties, ReactNode} from 'react';
import {
  AbsoluteFill,
  Easing,
  interpolate,
  Sequence,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';

import benchmark from '../../benchmarks/github-arm64-run-31405378460/benchmark.json';
import aggregate from '../../benchmarks/r2-bf16-fastmath-a7ed33f/aggregate.json';
import quality from '../../benchmarks/r2-bf16-task-quality-af45559/run-31495451767/quality.json';

export const FPS = 30;

const scenes = [180, 180, 300, 360, 330, 330, 300, 240] as const;
export const DEMO_FRAMES = scenes.reduce((sum, value) => sum + value, 0);

const C = {
  ink: '#f6fbff',
  muted: '#98adc1',
  navy: '#020712',
  panel: 'rgba(8, 21, 38, 0.76)',
  cyan: '#20d7f2',
  cyan2: '#5bf2ff',
  blue: '#2488ff',
  green: '#58e68b',
  amber: '#ffcc66',
  red: '#ff7185',
};

const font = '"Segoe UI Variable Display", "Segoe UI", Arial, sans-serif';
const mono = '"Cascadia Code", "SFMono-Regular", Consolas, monospace';

const clamp = (value: number) => Math.max(0, Math.min(1, value));
const ease = (value: number) => Easing.bezier(0.22, 1, 0.36, 1)(clamp(value));
const format = (value: number, digits: number) => value.toFixed(digits);

const speedup = benchmark.summary.geometric_mean_latency_speedup;
const sizeReduction = benchmark.summary.model_size_reduction_percent;
const embeddingCosine = benchmark.quality.mean_embedding_cosine;
const fp32Size = benchmark.models.baseline.size_bytes / 1024 / 1024;
const int8Size = benchmark.models.optimized.size_bytes / 1024 / 1024;
const batch32 = benchmark.batches.find((item) => item.batch_size === 32)!;
const bf16 = aggregate.comparisons.fp32_bf16_vs_control.run_geometric_mean_speedup;
const taskGate = quality.comparisons.fp32_bf16_vs_control;

const GlobalBackground = () => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill
      style={{
        overflow: 'hidden',
        background:
          'radial-gradient(circle at 77% 22%, rgba(32,215,242,0.11), transparent 29%), radial-gradient(circle at 20% 80%, rgba(36,136,255,0.13), transparent 35%), linear-gradient(140deg, #020712 0%, #06101c 52%, #020712 100%)',
      }}
    >
      <div
        style={{
          position: 'absolute',
          inset: -200,
          opacity: 0.16,
          transform: `translate(${Math.sin(frame / 180) * 24}px, ${Math.cos(frame / 220) * 18}px)`,
          backgroundImage:
            'linear-gradient(rgba(91,242,255,0.14) 1px, transparent 1px), linear-gradient(90deg, rgba(91,242,255,0.14) 1px, transparent 1px)',
          backgroundSize: '68px 68px',
          maskImage: 'radial-gradient(circle at center, black, transparent 72%)',
        }}
      />
      {Array.from({length: 24}, (_, i) => {
        const x = (i * 83 + 41) % 1920;
        const y = (i * 137 + 97) % 1080;
        const drift = ((frame * (0.18 + (i % 5) * 0.03)) % 1300) - 160;
        return (
          <div
            key={i}
            style={{
              position: 'absolute',
              left: x,
              top: (y + drift) % 1180,
              width: 2 + (i % 3),
              height: 2 + (i % 3),
              borderRadius: 99,
              background: i % 4 === 0 ? C.green : C.cyan,
              opacity: 0.2 + (i % 4) * 0.08,
              boxShadow: `0 0 16px ${i % 4 === 0 ? C.green : C.cyan}`,
            }}
          />
        );
      })}
      <div
        style={{
          position: 'absolute',
          inset: 34,
          border: '1px solid rgba(91,242,255,0.12)',
          borderRadius: 28,
          pointerEvents: 'none',
        }}
      />
    </AbsoluteFill>
  );
};

const Chrome = ({scene, children}: {scene: string; children: ReactNode}) => (
  <AbsoluteFill style={{fontFamily: font, color: C.ink}}>
    <div
      style={{
        position: 'absolute',
        top: 58,
        left: 74,
        right: 74,
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        fontSize: 18,
        letterSpacing: 4.2,
        textTransform: 'uppercase',
        color: C.muted,
      }}
    >
      <span><b style={{color: C.cyan}}>ARMBENCH</b> MINILM</span>
      <span>{scene}</span>
    </div>
    {children}
    <div
      style={{
        position: 'absolute',
        bottom: 52,
        left: 76,
        right: 76,
        display: 'flex',
        justifyContent: 'space-between',
        fontSize: 16,
        letterSpacing: 2.2,
        color: '#6f8498',
      }}
    >
      <span>NATIVE ARM64 · ONNX RUNTIME · OPEN SOURCE</span>
      <span>github.com/yhay81/armbench-minilm</span>
    </div>
  </AbsoluteFill>
);

const Scene = ({children, duration, label}: {children: ReactNode; duration: number; label: string}) => {
  const frame = useCurrentFrame();
  const opacity = Math.min(
    interpolate(frame, [0, 18], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}),
    interpolate(frame, [duration - 18, duration], [1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}),
  );
  return (
    <AbsoluteFill style={{opacity}}>
      <Chrome scene={label}>{children}</Chrome>
    </AbsoluteFill>
  );
};

const Reveal = ({children, at = 0, y = 30, style}: {children: ReactNode; at?: number; y?: number; style?: CSSProperties}) => {
  const frame = useCurrentFrame();
  const p = ease((frame - at) / 24);
  return <div style={{opacity: p, transform: `translateY(${(1 - p) * y}px)`, ...style}}>{children}</div>;
};

const Pill = ({children, tone = C.cyan}: {children: ReactNode; tone?: string}) => (
  <div
    style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: 12,
      border: `1px solid ${tone}55`,
      background: `${tone}12`,
      color: tone,
      padding: '10px 18px',
      borderRadius: 999,
      fontSize: 18,
      fontWeight: 700,
      letterSpacing: 2,
      textTransform: 'uppercase',
    }}
  >
    <span style={{width: 8, height: 8, borderRadius: 99, background: tone, boxShadow: `0 0 14px ${tone}`}} />
    {children}
  </div>
);

const Metric = ({value, label, color = C.cyan}: {value: string; label: string; color?: string}) => (
  <div
    style={{
      background: C.panel,
      border: `1px solid ${color}3d`,
      borderRadius: 24,
      padding: '28px 30px',
      boxShadow: `0 20px 70px rgba(0,0,0,0.34), inset 0 1px ${color}26`,
    }}
  >
    <div style={{fontSize: 66, lineHeight: 1, fontWeight: 760, color, letterSpacing: -2}}>{value}</div>
    <div style={{marginTop: 13, color: C.muted, fontSize: 20, letterSpacing: 1.2}}>{label}</div>
  </div>
);

const TitleScene = () => {
  const frame = useCurrentFrame();
  const pulse = 0.6 + Math.sin(frame / 18) * 0.12;
  return (
    <Scene duration={scenes[0]} label="Cloud AI optimization">
      <div style={{position: 'absolute', inset: '180px 120px 150px', display: 'grid', alignContent: 'center'}}>
        <Reveal at={0}><Pill>Evidence-first optimization</Pill></Reveal>
        <Reveal at={18}>
          <div style={{fontSize: 136, fontWeight: 800, letterSpacing: -7, lineHeight: 0.94, marginTop: 35}}>
            ArmBench <span style={{color: C.cyan}}>MiniLM</span>
          </div>
        </Reveal>
        <Reveal at={36}>
          <div style={{fontSize: 42, color: C.muted, marginTop: 32, maxWidth: 1200, lineHeight: 1.32}}>
            One command turns a pinned FP32 model into a smaller, faster, auditable Arm64 result.
          </div>
        </Reveal>
        <div
          style={{
            position: 'absolute',
            right: 85,
            top: 60,
            width: 280,
            height: 280,
            borderRadius: '38% 62% 56% 44%',
            border: `2px solid ${C.cyan}77`,
            boxShadow: `0 0 ${80 * pulse}px ${C.cyan}33, inset 0 0 70px ${C.blue}22`,
            transform: `rotate(${frame / 6}deg)`,
          }}
        />
      </div>
    </Scene>
  );
};

const ProblemScene = () => (
  <Scene duration={scenes[1]} label="The problem">
    <div style={{position: 'absolute', left: 130, right: 130, top: 250}}>
      <Reveal at={0}>
        <div style={{fontSize: 86, fontWeight: 760, lineHeight: 1.05, maxWidth: 1500}}>
          Optimization claims are easy.
          <br />
          <span style={{color: C.cyan}}>Evidence is hard.</span>
        </div>
      </Reveal>
      <div style={{display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 24, marginTop: 70}}>
        {[
          ['01', 'Was the model actually faster?'],
          ['02', 'Did numerical quality survive?'],
          ['03', 'Can anyone reproduce it?'],
        ].map(([index, text], i) => (
          <Reveal key={index} at={34 + i * 10}>
            <div style={{padding: '28px 30px', minHeight: 126, borderLeft: `2px solid ${C.cyan}`, background: 'rgba(9,25,43,0.54)'}}>
              <div style={{fontFamily: mono, color: C.cyan, fontSize: 18}}>{index}</div>
              <div style={{fontSize: 25, marginTop: 13, lineHeight: 1.25}}>{text}</div>
            </div>
          </Reveal>
        ))}
      </div>
    </div>
  </Scene>
);

const CommandScene = () => {
  const frame = useCurrentFrame();
  const command = 'uv run armbench-minilm all --work-dir .armbench --output-dir results';
  const typed = command.slice(0, Math.max(0, Math.floor((frame - 34) * 1.15)));
  const rows = [
    ['PIN', 'source model revision', 94],
    ['INT8', 'dynamic per-channel quantization', 125],
    ['RUN', 'native Arm64 benchmark', 158],
    ['PROVE', 'JSON · Markdown · HTML evidence', 191],
  ] as const;
  return (
    <Scene duration={scenes[2]} label="One command">
      <div style={{position: 'absolute', left: 150, right: 150, top: 185}}>
        <Reveal><div style={{fontSize: 62, fontWeight: 730}}>From clean clone to evidence.</div></Reveal>
        <div style={{marginTop: 45, border: '1px solid rgba(91,242,255,0.25)', background: 'rgba(2,8,15,0.94)', borderRadius: 24, overflow: 'hidden', boxShadow: '0 38px 100px rgba(0,0,0,0.45)'}}>
          <div style={{height: 54, background: 'rgba(255,255,255,0.035)', display: 'flex', alignItems: 'center', gap: 10, padding: '0 22px'}}>
            {[C.red, C.amber, C.green].map((color) => <span key={color} style={{width: 12, height: 12, borderRadius: 99, background: color}} />)}
            <span style={{fontFamily: mono, color: C.muted, marginLeft: 14}}>armbench / reproducible run</span>
          </div>
          <div style={{fontFamily: mono, padding: '30px 34px 32px', fontSize: 24, minHeight: 420}}>
            <div><span style={{color: C.green}}>$</span> {typed}<span style={{opacity: frame % 24 < 12 ? 1 : 0, color: C.cyan}}>▌</span></div>
            <div style={{marginTop: 32, display: 'grid', gap: 18}}>
              {rows.map(([tag, text, at]) => {
                const p = ease((frame - at) / 18);
                return (
                  <div key={tag} style={{opacity: p, transform: `translateX(${(1 - p) * 22}px)`, display: 'grid', gridTemplateColumns: '115px 1fr 60px', color: C.muted}}>
                    <span style={{color: C.cyan}}>{tag}</span><span>{text}</span><span style={{color: C.green}}>✓</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </Scene>
  );
};

const BlockCloud = ({count, color, progress, compact}: {count: number; color: string; progress: number; compact?: boolean}) => {
  const cols = compact ? 6 : 8;
  return (
    <div style={{display: 'grid', gridTemplateColumns: `repeat(${cols}, ${compact ? 30 : 34}px)`, gap: 8, alignContent: 'center'}}>
      {Array.from({length: count}, (_, i) => {
        const p = ease((progress - i / count * 0.45) / 0.42);
        return <div key={i} style={{width: compact ? 30 : 34, height: compact ? 30 : 34, borderRadius: 7, background: `linear-gradient(145deg, ${color}, ${color}55)`, opacity: p, transform: `scale(${0.55 + p * 0.45})`, boxShadow: `0 0 15px ${color}25`, border: `1px solid ${color}aa`}} />;
      })}
    </div>
  );
};

const TransformScene = () => {
  const frame = useCurrentFrame();
  const p = ease((frame - 25) / 95);
  return (
    <Scene duration={scenes[3]} label="Measured transformation">
      <div style={{position: 'absolute', left: 115, right: 115, top: 180}}>
        <Reveal><div style={{fontSize: 62, fontWeight: 750}}>FP32 → INT8, measured on native Arm64.</div></Reveal>
        <div style={{display: 'grid', gridTemplateColumns: '1fr 220px 1fr', alignItems: 'center', marginTop: 62, gap: 30}}>
          <div style={{height: 310, borderRadius: 28, background: 'rgba(36,136,255,0.08)', border: '1px solid rgba(36,136,255,0.3)', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 40}}>
            <BlockCloud count={48} color={C.blue} progress={1} />
            <div><div style={{fontSize: 30, color: C.muted}}>FP32</div><div style={{fontSize: 62, fontWeight: 750}}>{format(fp32Size, 2)}<small style={{fontSize: 24}}> MiB</small></div></div>
          </div>
          <div style={{textAlign: 'center', color: C.cyan}}>
            <div style={{fontSize: 92, transform: `translateX(${(1 - p) * -25}px)`, opacity: p}}>→</div>
            <div style={{fontFamily: mono, fontSize: 17, color: C.muted, lineHeight: 1.5}}>MatMul<br />Gemm</div>
          </div>
          <div style={{height: 310, borderRadius: 28, background: 'rgba(88,230,139,0.08)', border: '1px solid rgba(88,230,139,0.3)', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 40}}>
            <BlockCloud count={30} color={C.green} progress={p} compact />
            <div style={{opacity: p}}><div style={{fontSize: 30, color: C.muted}}>INT8</div><div style={{fontSize: 62, fontWeight: 750, color: C.green}}>{format(int8Size, 2)}<small style={{fontSize: 24}}> MiB</small></div></div>
          </div>
        </div>
        <div style={{display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 24, marginTop: 28}}>
          <Metric value={`${format(speedup, 2)}×`} label="geometric-mean latency speedup" />
          <Metric value={`${format(sizeReduction, 1)}%`} label="smaller model file" color={C.green} />
          <Metric value={`${format(embeddingCosine, 8)}`} label="mean embedding cosine" color={C.cyan2} />
        </div>
      </div>
    </Scene>
  );
};

const SpeedBars = () => {
  const frame = useCurrentFrame();
  const items = benchmark.batches.map((item) => ({batch: item.batch_size, speedup: item.median_latency_speedup}));
  return (
    <div style={{display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 42, alignItems: 'end', height: 370}}>
      {items.map((item, i) => {
        const p = spring({frame: frame - 34 - i * 10, fps: FPS, config: {damping: 16, stiffness: 95}});
        const h = 90 + item.speedup / 3.2 * 220;
        return (
          <div key={item.batch} style={{display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'flex-end', height: '100%'}}>
            <div style={{fontSize: 48, fontWeight: 760, color: C.cyan, opacity: p}}>{format(item.speedup, 2)}×</div>
            <div style={{width: 180, height: h * p, marginTop: 16, borderRadius: '16px 16px 4px 4px', background: `linear-gradient(180deg, ${C.cyan}, ${C.blue}44)`, boxShadow: `0 0 45px ${C.cyan}22`, border: `1px solid ${C.cyan}88`}} />
            <div style={{marginTop: 14, color: C.muted, fontSize: 22}}>batch {item.batch}</div>
          </div>
        );
      })}
    </div>
  );
};

const EvidenceScene = () => (
  <Scene duration={scenes[4]} label="Native performance evidence">
    <div style={{position: 'absolute', left: 140, right: 140, top: 175, display: 'grid', gridTemplateColumns: '1.25fr 0.75fr', gap: 90}}>
      <div>
        <Reveal><Pill>100 measured iterations per model and batch</Pill></Reveal>
        <Reveal at={20}><div style={{fontSize: 66, lineHeight: 1.06, fontWeight: 760, marginTop: 34}}>The gain grows with useful cloud batch sizes.</div></Reveal>
        <SpeedBars />
      </div>
      <div style={{paddingTop: 95}}>
        <Reveal at={35}><Metric value={`${format(batch32.baseline.median_ms, 3)} ms`} label="FP32 median · batch 32" color={C.blue} /></Reveal>
        <div style={{height: 20}} />
        <Reveal at={50}><Metric value={`${format(batch32.optimized.median_ms, 3)} ms`} label="INT8 median · batch 32" color={C.green} /></Reveal>
        <Reveal at={70} style={{marginTop: 26, color: C.muted, fontSize: 21, lineHeight: 1.55}}>
          Same four-core Arm64 runner.<br />Same runtime settings. Same timing boundary.
        </Reveal>
      </div>
    </div>
  </Scene>
);

const GateRow = ({label, value, limit, at}: {label: string; value: string; limit: string; at: number}) => (
  <Reveal at={at}>
    <div style={{display: 'grid', gridTemplateColumns: '1.25fr 0.8fr 0.8fr 90px', alignItems: 'center', minHeight: 82, padding: '0 24px', borderBottom: '1px solid rgba(255,255,255,0.07)', fontSize: 21}}>
      <span>{label}</span><span style={{fontFamily: mono, color: C.cyan}}>{value}</span><span style={{color: C.muted}}>{limit}</span><span style={{color: C.green, fontWeight: 800}}>PASS</span>
    </div>
  </Reveal>
);

const QualityScene = () => (
  <Scene duration={scenes[5]} label="Predeclared quality gate">
    <div style={{position: 'absolute', left: 135, right: 135, top: 170}}>
      <Reveal><div style={{fontSize: 64, fontWeight: 760}}>Fast math only ships after task quality passes.</div></Reveal>
      <Reveal at={18}><div style={{fontSize: 25, color: C.muted, marginTop: 18}}>FP32 + BF16 vs the unchanged FP32 artifact · exact, hash-checked task revisions</div></Reveal>
      <div style={{marginTop: 46, background: C.panel, border: '1px solid rgba(91,242,255,0.22)', borderRadius: 24, overflow: 'hidden'}}>
        <div style={{display: 'grid', gridTemplateColumns: '1.25fr 0.8fr 0.8fr 90px', padding: '18px 24px', color: C.muted, fontSize: 17, letterSpacing: 1.6, textTransform: 'uppercase'}}>
          <span>Gate</span><span>Measured</span><span>Limit</span><span>Status</span>
        </div>
        <GateRow label="STS score loss" value={`${format(taskGate.sts_absolute_loss_points, 4)} pt`} limit="≤ 0.5 pt" at={38} />
        <GateRow label="ArguAna relative loss" value={`${format(taskGate.retrieval_relative_loss * 100, 4)}%`} limit="≤ 1.0%" at={52} />
        <GateRow label="STS embedding cosine" value={format(taskGate.sts_mean_corresponding_embedding_cosine, 8)} limit="≥ 0.99" at={66} />
        <GateRow label="Retrieval embedding cosine" value={format(taskGate.retrieval_mean_corresponding_embedding_cosine, 8)} limit="≥ 0.99" at={80} />
      </div>
      <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, marginTop: 26}}>
        <Reveal at={96}><Metric value={`${format(bf16.median, 4)}×`} label={`FP32+BF16 median · ${bf16.count} native runs`} color={C.green} /></Reveal>
        <Reveal at={110}><Metric value={`${format(bf16.cv_percent, 2)}%`} label="run-to-run coefficient of variation" color={C.cyan} /></Reveal>
      </div>
    </div>
  </Scene>
);

const TrustScene = () => {
  const items = [
    ['PINNED', 'Source model and evaluation revisions'],
    ['CHECKED', 'SHA-256, schemas, and cardinalities'],
    ['RETAINED', 'Raw samples and machine-readable reports'],
    ['PUBLIC', 'Native Arm64 GitHub Actions workflow'],
  ];
  return (
    <Scene duration={scenes[6]} label="Trust by construction">
      <div style={{position: 'absolute', left: 145, right: 145, top: 175}}>
        <Reveal><div style={{fontSize: 70, fontWeight: 770}}>Every claim has a path back to evidence.</div></Reveal>
        <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 22, marginTop: 52}}>
          {items.map(([tag, text], i) => (
            <Reveal key={tag} at={28 + i * 12}>
              <div style={{minHeight: 150, padding: '30px 32px', border: '1px solid rgba(91,242,255,0.18)', borderRadius: 22, background: C.panel}}>
                <div style={{fontFamily: mono, color: C.cyan, fontSize: 20}}>{tag} / 0{i + 1}</div>
                <div style={{fontSize: 27, marginTop: 17, color: C.ink}}>{text}</div>
              </div>
            </Reveal>
          ))}
        </div>
        <Reveal at={95} style={{marginTop: 34, fontSize: 26, color: C.muted}}>
          Limitations stay visible. Failed gates stay visible. The original headline stays immutable.
        </Reveal>
      </div>
    </Scene>
  );
};

const CloseScene = () => {
  const frame = useCurrentFrame();
  const ring = spring({frame: frame - 18, fps: FPS, config: {damping: 14, stiffness: 80}});
  return (
    <Scene duration={scenes[7]} label="ArmBench MiniLM">
      <div style={{position: 'absolute', inset: '185px 150px 130px', display: 'grid', alignContent: 'center', textAlign: 'center'}}>
        <div style={{position: 'absolute', left: '50%', top: '43%', width: 540 * ring, height: 540 * ring, transform: 'translate(-50%, -50%)', borderRadius: 999, border: `1px solid ${C.cyan}44`, boxShadow: `0 0 100px ${C.cyan}1f`}} />
        <Reveal at={0}><Pill>Cloud AI · Native Arm64</Pill></Reveal>
        <Reveal at={22}>
          <div style={{fontSize: 91, fontWeight: 800, lineHeight: 1.08, marginTop: 32}}>
            Faster is a claim.<br /><span style={{color: C.cyan}}>Reproducible faster is a result.</span>
          </div>
        </Reveal>
        <Reveal at={52}>
          <div style={{fontFamily: mono, fontSize: 25, marginTop: 46, color: C.muted}}>github.com/yhay81/armbench-minilm</div>
        </Reveal>
      </div>
    </Scene>
  );
};

export const ArmBenchDemo = () => {
  let start = 0;
  const entries: Array<[number, number, ReactNode]> = [
    [start, scenes[0], <TitleScene key="title" />],
    [(start += scenes[0]), scenes[1], <ProblemScene key="problem" />],
    [(start += scenes[1]), scenes[2], <CommandScene key="command" />],
    [(start += scenes[2]), scenes[3], <TransformScene key="transform" />],
    [(start += scenes[3]), scenes[4], <EvidenceScene key="evidence" />],
    [(start += scenes[4]), scenes[5], <QualityScene key="quality" />],
    [(start += scenes[5]), scenes[6], <TrustScene key="trust" />],
    [(start += scenes[6]), scenes[7], <CloseScene key="close" />],
  ];
  return (
    <AbsoluteFill style={{background: C.navy}}>
      <GlobalBackground />
      {entries.map(([from, duration, element]) => <Sequence key={from} from={from} durationInFrames={duration}>{element}</Sequence>)}
      <Progress />
    </AbsoluteFill>
  );
};

const Progress = () => {
  const frame = useCurrentFrame();
  const {durationInFrames} = useVideoConfig();
  return <div style={{position: 'absolute', left: 34, right: 34, bottom: 24, height: 2, background: 'rgba(255,255,255,0.06)'}}><div style={{height: '100%', width: `${frame / durationInFrames * 100}%`, background: `linear-gradient(90deg, ${C.blue}, ${C.cyan}, ${C.green})`, boxShadow: `0 0 12px ${C.cyan}`}} /></div>;
};

export const ArmBenchPoster = () => (
  <AbsoluteFill style={{fontFamily: font, color: C.ink, background: 'radial-gradient(circle at 75% 20%, #0b3d4a 0%, #051424 33%, #020712 72%)', padding: 86}}>
    <div style={{fontSize: 22, letterSpacing: 5, color: C.cyan, fontWeight: 700}}>NATIVE ARM64 · EVIDENCE-FIRST AI OPTIMIZATION</div>
    <div style={{fontSize: 115, lineHeight: 0.95, fontWeight: 820, marginTop: 38, letterSpacing: -6}}>ArmBench<br /><span style={{color: C.cyan}}>MiniLM</span></div>
    <div style={{fontSize: 35, color: C.muted, marginTop: 38, maxWidth: 1300}}>One command. Native performance. Retained quality. Reproducible evidence.</div>
    <div style={{display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 24, marginTop: 70}}>
      <Metric value={`${format(speedup, 2)}×`} label="median-latency speedup" />
      <Metric value={`${format(sizeReduction, 1)}%`} label="smaller model" color={C.green} />
      <Metric value={format(embeddingCosine, 6)} label="mean embedding cosine" color={C.cyan2} />
    </div>
    <div style={{position: 'absolute', left: 86, bottom: 74, right: 86, display: 'flex', justifyContent: 'space-between', fontFamily: mono, fontSize: 20, color: C.muted}}><span>FP32 → DYNAMIC QINT8</span><span>github.com/yhay81/armbench-minilm</span></div>
  </AbsoluteFill>
);
