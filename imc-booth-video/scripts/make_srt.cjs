// Writes out/imc_booth_loop.srt from the SAME timeline the video renders from, so the sidecar
// subtitle file can never disagree with the burned-in captions.
const esbuild = require('esbuild');
const fs = require('fs');
esbuild.buildSync({ entryPoints: ['src/timeline.ts'], bundle: true, platform: 'node', format: 'cjs', outfile: 'build/timeline.cjs', loader: { '.json': 'json' }, logLevel: 'error' });
const t = require('../build/timeline.cjs');
const FPS = 30;
const ts = (frame) => {
  const ms = Math.round((frame / FPS) * 1000);
  const p = (n, w = 2) => String(n).padStart(w, '0');
  return `${p(Math.floor(ms / 3600000))}:${p(Math.floor((ms % 3600000) / 60000))}:${p(Math.floor((ms % 60000) / 1000))},${p(ms % 1000, 3)}`;
};
const wrap = (t, n = 46) => {
  // balance across two lines when the cue is longer than one line
  if (t.length <= n) return t;
  const w = t.split(' ');
  let best = null;
  for (let i = 1; i < w.length; i++) {
    const a = w.slice(0, i).join(' '), b = w.slice(i).join(' ');
    const score = Math.max(a.length, b.length);
    if (a.length <= n + 6 && b.length <= n + 6 && (!best || score < best.score)) best = { a, b, score };
  }
  return best ? best.a + '\n' + best.b : t;
};
const cues = [];
for (const sc of t.SCENES) for (const sh of sc.shots) for (const se of sh.sentences) {
  const start = sc.start + sh.start + se.start;
  cues.push({ start, end: start + se.dur, text: wrap(se.show.replace(/U²-Net/g, 'U2-Net')) });
}
cues.sort((a, b) => a.start - b.start);
const srt = cues.map((c, i) => `${i + 1}\n${ts(c.start)} --> ${ts(c.end)}\n${c.text}\n`).join('\n');
fs.mkdirSync('out', { recursive: true });
fs.writeFileSync('out/imc_booth_loop.srt', srt, 'utf8');
console.log(`wrote out/imc_booth_loop.srt  (${cues.length} cues, ${(t.TOTAL_FRAMES / FPS).toFixed(1)} s)`);
