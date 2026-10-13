import React, { FC, useMemo } from 'react';
import { interpolate, useCurrentFrame } from 'remotion';
import { inter } from '../fonts';
import type { Sentence } from '../timeline';

type Page = { text: string; start: number; end: number };

const STOP = new Set(['a', 'an', 'the', 'to', 'of', 'in', 'on', 'at', 'and', 'or', 'but', 'for', 'with', 'into', 'your', 'our', 'that', 'is', 'it', 'its', 'as', 'by', 'from', 'then']);
const MAX_CHARS = 74; // ~2 lines at 42px in a 960px column

function splitWords(show: string): string[] {
  const words = show.split(' ');
  const pages: string[][] = [];
  let rest = words.slice();
  while (rest.join(' ').length > MAX_CHARS) {
    let acc = 0;
    let cut = 0;
    let punct = -1;
    for (let i = 0; i < rest.length; i++) {
      acc += (i ? 1 : 0) + rest[i].length;
      if (acc > MAX_CHARS) break;
      cut = i;
      if (/[,;:.?!]$/.test(rest[i]) && acc > MAX_CHARS * 0.45) punct = i;
    }
    let k = punct >= 0 ? punct : cut;
    // never end a page on a dangling article / preposition / conjunction
    while (k > 2 && STOP.has(rest[k].toLowerCase().replace(/[^a-z]/g, ''))) k--;
    pages.push(rest.slice(0, k + 1));
    rest = rest.slice(k + 1);
  }
  pages.push(rest);
  // never leave a tiny orphan page: borrow words from the previous page
  while (pages.length > 1 && pages[pages.length - 1].join(' ').length < 20 && pages[pages.length - 2].length > 3) {
    pages[pages.length - 1].unshift(pages[pages.length - 2].pop() as string);
  }
  // the borrow step can leave a page ending on a stop-word: push those words to the next page
  for (let i = 0; i < pages.length - 1; i++) {
    while (pages[i].length > 3 && STOP.has(pages[i][pages[i].length - 1].toLowerCase().replace(/[^a-z]/g, ''))) pages[i + 1].unshift(pages[i].pop() as string);
  }
  return pages.map((p) => p.join(' '));
}

function paginate(se: Sentence): Page[] {
  const texts = splitWords(se.show);
  if (texts.length === 1) return [{ text: se.show, start: se.start, end: se.start + se.dur }];
  const total = texts.reduce((a, t) => a + t.length, 0);
  let cursor = se.start;
  return texts.map((t, i) => {
    const d = i === texts.length - 1 ? se.start + se.dur - cursor : Math.round((se.dur * t.length) / total);
    const pg = { text: t, start: cursor, end: cursor + d };
    cursor += d;
    return pg;
  });
}

export const Subtitles: FC<{ sentences: Sentence[]; pos?: 'center' | 'right'; bottom?: number }> = ({ sentences, pos = 'center', bottom = 44 }) => {
  const frame = useCurrentFrame();
  const pages = useMemo(() => sentences.flatMap(paginate), [sentences]);
  // latest page whose window covers this frame (so page 2 takes over from page 1 exactly at its own start)
  let idx = -1;
  pages.forEach((p, i) => {
    if (frame >= p.start - 2 && frame < p.end + 8) idx = i;
  });
  if (idx < 0) return null;
  const active = pages[idx];
  const prev = pages[idx - 1];
  const next = pages[idx + 1];
  const joinedPrev = !!prev && active.start - prev.end <= 3; // hand-over inside one sentence: no fade between pages
  const joinedNext = !!next && next.start - active.end <= 3;
  const fin = joinedPrev ? 1 : interpolate(frame, [active.start - 2, active.start + 3], [0, 1], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' });
  const fout = joinedNext ? 1 : interpolate(frame, [active.end + 3, active.end + 8], [1, 0], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' });
  const o = Math.min(fin, fout);
  const left = pos === 'right' ? 900 : 160;
  const width = pos === 'right' ? 960 : 1600;
  return (
    <div style={{ position: 'absolute', left, width, bottom, display: 'flex', justifyContent: 'center', pointerEvents: 'none' }}>
      <div
        style={{
          background: 'rgba(21,18,14,0.92)',
          color: '#fff',
          fontFamily: inter,
          fontWeight: 600,
          fontSize: 42,
          lineHeight: 1.28,
          padding: '14px 30px 16px',
          borderRadius: 20,
          textAlign: 'center',
          maxWidth: width - 20,
          opacity: o,
          textWrap: 'balance' as any,
          transform: `translateY(${(1 - o) * 10}px)`,
          boxShadow: '0 10px 30px rgba(21,18,14,0.25)',
        }}
      >
        {active.text.split(' ').map((w, i, arr) => (
          <React.Fragment key={i}>
            {w.includes('-') ? <span style={{ whiteSpace: 'nowrap' }}>{w}</span> : w}
            {i < arr.length - 1 ? ' ' : ''}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
};
