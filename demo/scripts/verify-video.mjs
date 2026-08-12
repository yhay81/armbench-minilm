import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {readFileSync, statSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import path from 'node:path';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(scriptDir, '..', '..');
const videoPath = path.join(projectRoot, 'assets', 'armbench-demo.mp4');
const posterPath = path.join(projectRoot, 'assets', 'armbench-demo-poster.png');

const metadata = JSON.parse(execFileSync('ffprobe', [
  '-v', 'error',
  '-show_entries', 'format=duration:stream=index,codec_type,codec_name,width,height,r_frame_rate,pix_fmt',
  '-of', 'json',
  videoPath,
], {encoding: 'utf8'}));

const video = metadata.streams.find((stream) => stream.codec_type === 'video');
const audio = metadata.streams.find((stream) => stream.codec_type === 'audio');
const duration = Number(metadata.format.duration);

if (!video || video.codec_name !== 'h264' || video.width !== 1920 || video.height !== 1080) {
  throw new Error(`Unexpected video stream: ${JSON.stringify(video)}`);
}
if (video.pix_fmt !== 'yuv420p' || video.r_frame_rate !== '30/1') {
  throw new Error(`Unexpected playback format: ${JSON.stringify(video)}`);
}
if (!audio || audio.codec_name !== 'aac') {
  throw new Error(`Unexpected audio stream: ${JSON.stringify(audio)}`);
}
if (duration < 73.8 || duration > 74.2) {
  throw new Error(`Unexpected duration: ${duration}`);
}
if (statSync(videoPath).size > 100 * 1024 * 1024) {
  throw new Error('Video exceeds the 100 MiB repository delivery limit.');
}
if (statSync(posterPath).size < 100_000) {
  throw new Error('Poster render is unexpectedly small.');
}

const sha256 = createHash('sha256').update(readFileSync(videoPath)).digest('hex');
console.log(JSON.stringify({
  video: path.relative(projectRoot, videoPath).replaceAll('\\', '/'),
  poster: path.relative(projectRoot, posterPath).replaceAll('\\', '/'),
  duration_seconds: duration,
  size_bytes: statSync(videoPath).size,
  sha256,
  streams: metadata.streams,
}, null, 2));
