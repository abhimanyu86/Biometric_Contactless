// Bundles src/timeline.ts with esbuild (same code the video uses) and prints scene/shot/sentence frames.
const esbuild = require('esbuild');
esbuild.buildSync({ entryPoints: ['src/timeline.ts'], bundle: true, platform: 'node', format: 'cjs', outfile: 'build/timeline.cjs', loader: { '.json': 'json' }, logLevel: 'error' });
const t = require('../build/timeline.cjs');
const fps = 30;
const mmss = (f) => `${Math.floor(f / fps / 60)}:${String(((f / fps) % 60).toFixed(1)).padStart(4, '0')}`;
console.log(`TOTAL ${t.TOTAL_FRAMES} frames = ${(t.TOTAL_FRAMES / fps).toFixed(1)} s`);
for (const s of t.SCENES) {
  console.log(`\n[${s.id}] start=${s.start} (${mmss(s.start)})  dur=${s.dur} (${(s.dur / fps).toFixed(1)}s)  end=${s.start + s.dur}`);
  for (const sh of s.shots) {
    console.log(`   shot ${sh.beat ?? '-'} @+${sh.start} dur=${sh.dur} (${(sh.dur / fps).toFixed(1)}s) clips=${sh.clips.map((c) => `${c.id}[${(c.dur / fps).toFixed(1)}s]`).join(' ')}`);
    for (const se of sh.sentences) console.log(`      "${se.show.slice(0, 60)}"  abs ${s.start + sh.start + se.start}..${s.start + sh.start + se.start + se.dur}`);
  }
}
if (process.argv[2] === 'json') require('fs').writeFileSync('build/timeline.json', JSON.stringify({ total: t.TOTAL_FRAMES, scenes: t.SCENES }, null, 1));
