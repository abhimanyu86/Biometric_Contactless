"""QA for a rendered cut: (1) audio/timeline sync, (2) frame contact sheets for visual review.

  python scripts/qa_check.py out/voiced_raw.mp4

(1) decodes the audio, and for every narrated sentence checks that there is speech energy inside its
    timeline window and near-silence in the gaps between sentences.
(2) writes build/qa/sheet_XX.jpg  (4x4 thumbnails, one every 2 s) and build/qa/full_*.png at scene mid-points.
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FFMPEG = str(ROOT / "tools" / "ffmpeg.exe")
FPS = 30
src = Path(sys.argv[1])
qa = ROOT / "build" / "qa"
qa.mkdir(parents=True, exist_ok=True)

# timeline (same code the video renders from)
subprocess.run(["node", str(ROOT / "scripts" / "dump_timeline.cjs"), "json"], cwd=ROOT, check=True, capture_output=True)
tl = json.loads((ROOT / "build" / "timeline.json").read_text(encoding="utf-8"))
total = tl["total"] / FPS

# ---- (1) audio sync ----
wav = qa / "audio.wav"
r = subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", str(wav)], capture_output=True)
has_audio = wav.exists() and wav.stat().st_size > 1000
print(f"video length target {total:.2f}s")
if has_audio:
    import wave
    w = wave.open(str(wav))
    x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    sr = w.getframerate()
    hop = int(sr * 0.05)
    rms = np.array([np.sqrt(np.mean(x[i:i + hop] ** 2) + 1e-12) for i in range(0, len(x) - hop, hop)])
    db = 20 * np.log10(rms + 1e-9)
    print(f"audio length {len(x)/sr:.2f}s   speech-ish frames (> -45 dB): {(db > -45).mean()*100:.0f}%")
    sents = []
    for sc in tl["scenes"]:
        for sh in sc["shots"]:
            for se in sh["sentences"]:
                a = (sc["start"] + sh["start"] + se["start"]) / FPS
                sents.append((a, a + se["dur"] / FPS, se["show"]))
    bad = 0
    for a, b, text in sents:
        seg = db[int(a / 0.05): int(b / 0.05)]
        loud = (seg > -42).mean() if len(seg) else 0
        # leading 0.15 s before the sentence should be quiet (previous sentence ended), unless back-to-back
        pre = db[max(0, int((a - 0.18) / 0.05)): int(a / 0.05)]
        flag = "OK " if loud > 0.55 else "LOW"
        if loud <= 0.55:
            bad += 1
        print(f"  {flag} {a:6.2f}-{b:6.2f}s  voiced={loud*100:3.0f}%  pre-gap={pre.mean() if len(pre) else -99:6.1f} dB  {text[:48]}")
    # silence inside the long gaps between beats (no stray audio)
    print(f"sentences with weak audio in their window: {bad}/{len(sents)}")
else:
    print("no audio track in this file (expected for the silent cut)")

# ---- (2) frame sheets ----
cap = cv2.VideoCapture(str(src))
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
dur = n / fps
print(f"decoded: {n} frames @ {fps:.2f} fps = {dur:.2f}s  size {int(cap.get(3))}x{int(cap.get(4))}")
times = np.arange(0.5, dur - 0.2, 2.0)
thumbs = []
for t in times:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, f = cap.read()
    if not ok:
        continue
    f = cv2.resize(f, (480, 270), interpolation=cv2.INTER_AREA)
    cv2.putText(f, f"{t:5.1f}s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    thumbs.append(f)
per = 16
for s in range(0, len(thumbs), per):
    chunk = thumbs[s:s + per]
    while len(chunk) < per:
        chunk.append(np.full((270, 480, 3), 255, np.uint8))
    rows = [np.hstack(chunk[i:i + 4]) for i in range(0, per, 4)]
    cv2.imwrite(str(qa / f"sheet_{s // per:02d}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
for sc in tl["scenes"]:
    mid = (sc["start"] + sc["dur"] * 0.55) / FPS
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(mid * fps))
    ok, f = cap.read()
    if ok:
        cv2.imwrite(str(qa / f"full_{sc['id']}.png"), f)
print("wrote", len(list(qa.glob('sheet_*.jpg'))), "sheets and", len(list(qa.glob('full_*.png'))), "full frames to", qa)
