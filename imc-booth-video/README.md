# IMC booth loop — YellowSense contactless fingerprint auth

A ~4:25, 1920×1080 / 30 fps looping video for the IMC booth (Hall 1, Booth TS 14), built with
[Remotion](https://www.remotion.dev). Two cuts come out of the same project:

| Output | What it is |
|---|---|
| `out/imc_booth_loop_voiced.mp4` | Indian-English narration + burned-in subtitles |
| `out/imc_booth_loop_silent.mp4` | Same visuals, no audio track, burned-in subtitles (for a muted booth TV) |
| `out/imc_booth_loop.srt` | Sidecar subtitles for the voiced cut (timing comes from the same timeline) |

Booth halls are loud — if the TV is muted, play the **silent** cut; the story still reads from the captions.

## How it fits together

```
scripts/make_narration.py   script text -> one MP3 per sentence (edge-tts, en-IN-NeerjaNeural) + src/data/narration.json
build/rt/*.mp4              demo recordings cut at 1x REAL TIME to 30 fps, no audio (scripts/cut_realtime.py -> src/data/manifest_rt.json)
assets/                     logo, hero photo, QR, brand palette (extracted from the poster)
scripts/sync_media.py       copies the above into public/media/ and src/data/
src/timeline.ts             derives every scene length from the real audio + clip durations (nothing is hard-timed)
src/scenes/*.tsx            hook, problem, pipeline, accuracy, pillars, closing   (src/components/Demo.tsx = phone + live-zoom layout)
```

Change a sentence in `make_narration.py`, re-run it, and the subtitles, animations and scene lengths re-flow on their own.

## Commands

```bash
cd imc-booth-video
python scripts/sync_media.py          # after (re)cutting clips or assets
python scripts/make_narration.py      # after editing the script / switching voice (needs internet)
node scripts/dump_timeline.cjs        # print scene / sentence frame numbers and total length
npx remotion studio                   # live preview
npx remotion render src/index.ts Booth       out/voiced_raw.mp4
npx remotion render src/index.ts BoothSilent out/imc_booth_loop_silent.mp4
node scripts/make_srt.cjs             # writes out/imc_booth_loop.srt
# loudness-normalise the voiced cut to -16 LUFS:
./tools/ffmpeg.exe -y -i out/voiced_raw.mp4 -c:v copy -af loudnorm=I=-16:TP=-1.5:LRA=11 -c:a aac -b:a 192k -movflags +faststart out/imc_booth_loop_voiced.mp4
```

`tools/ffmpeg.exe` is the binary bundled with `pip install imageio-ffmpeg` (copied here; gitignored).
Different voice: `python scripts/make_narration.py --voice en-IN-PrabhatNeural` (male) — nothing else changes.

## Switches (`src/theme.ts`)

- `ACCURACY_IS_MEASURED` (default `false`) — the **85%** is the PoC acceptance gate (">85% user-wise, contactless-to-contactless"), and
  the repo (`context.md`) lists it as *not measured*; the benchmarks in the repo are far lower. So the video shows it as a
  **PoC target** ("Our target is over 85% matching accuracy"). Once there is a measured, documented result, set this to `true`,
  edit beat B15 in `scripts/make_narration.py` (and the conditions on the slide), re-run narration, re-render.
- `SHOW_UNDEMONSTRATED_POSTER_CLAIMS` — the poster's *Deepfake detection* and *Fake app detection* are not shown in the
  demo footage, so the "Why YellowSense" slide leaves them out by default. Set `true` to add them.
- `SHOW_QR` — the poster's "Explore more" QR decodes to a Google Drive **folder** link. Confirm that folder is public and
  intended for visitors, or set this to `false`.
- `BOOTH` — event name, hall, booth, contact details.

## What the video does and does not claim

On screen / in the narration: contactless capture with a phone camera, enrol + authenticate in one app, slap mode (4 fingers,
"at least two must match"), the app's own *Matching / API time* readout (0.45 s single, 0.69 s slap) as **"under a second"** for the
match step, and a five-stage pipeline whose last stage is described only as cloud-side "template & match" (the demo build shows
"Embedding: 512 dimensions", not minutiae).

Deliberately **not** claimed: on-device processing; end-to-end speed (the app's timers read ~8–11 s; all demo footage plays at real time, nothing is sped up); that liveness catches photo / print attacks; any UIDAI certification or endorsement; and the
app's confidence scores (97.3% / 89.3%) are per-attempt similarity scores, **not** the accuracy figure.

Known, left as-is: the app UI in the footage shows "UIDAI · SITAA Cohort 1" and "for Aadhaar — no scanner needed" on its home
card, and test data (name "shiv", UID 121212121212).
