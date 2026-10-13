export const meta = {
  name: 'imc-video-review-v2',
  description: 'Independent review of the 4-minute IMC booth video: spelling/copy, pacing, visual defects, claims, audio/subtitle/file integrity',
  phases: [{ title: 'Review', detail: 'five independent reviewers, each told to find real defects' }],
}

const PROJ = 'D:\\UIDAI\\imc-booth-video'
const FFMPEG = PROJ + '\\tools\\ffmpeg.exe'
const VIDEO = PROJ + '\\out\\imc_booth_loop_voiced.mp4'
const SILENT = PROJ + '\\out\\imc_booth_loop_silent.mp4'
const SRT = PROJ + '\\out\\imc_booth_loop.srt'
const SCRATCH = 'C:\\Users\\ABHIMA~1\\AppData\\Local\\Temp\\claude\\D--UIDAI\\b6756823-3c94-4f4a-85cf-9cb5298632ae\\scratchpad\\review2'

const FINDINGS = {
  type: 'object',
  properties: {
    verdict: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          severity: { type: 'string', enum: ['blocker', 'major', 'minor', 'nit'] },
          where: { type: 'string' },
          issue: { type: 'string' },
          evidence: { type: 'string' },
          fix: { type: 'string' },
        },
        required: ['severity', 'where', 'issue', 'evidence', 'fix'],
      },
    },
  },
  required: ['verdict', 'findings'],
}

const CONTEXT = `
CONTEXT: ${PROJ} is a Remotion project that rendered a ~4:18, 1920x1080, 30fps looping booth video for YellowSense Technologies' contactless fingerprint authentication product, to be shown on a landscape TV at the India Mobile Congress (Hall 1, Booth TS 14). Final files: voiced cut ${VIDEO}, silent cut ${SILENT}, sidecar subtitles ${SRT}. ffmpeg is ${FFMPEG} (no ffprobe; Python 3.9 with cv2/numpy/PIL is available). Put scratch files in ${SCRATCH} (create it). Source: scenes in ${PROJ}\\src\\scenes\\*.tsx, ${PROJ}\\src\\components\\*.tsx, ${PROJ}\\src\\timeline.ts (shot plan + zoom windows), narration text in ${PROJ}\\src\\data\\narration.json, booth details/flags in ${PROJ}\\src\\theme.ts. All demo footage is the REAL phone screen recording played at 1x real time (clips in ${PROJ}\\build\\rt, cut by scripts\\cut_realtime.py from two recordings); result screens are held with a freeze frame. Print the scene/shot time table with: node ${PROJ}\\scripts\\dump_timeline.cjs .
DECISIONS ALREADY MADE BY THE OWNER (do not re-raise these as defects; you may mention them only if you find a NEW problem with how they are executed): American spelling throughout (Enroll/Enrollment, program, etc.); the 85% figure is deliberately shown as a PoC TARGET, not a measured result; the poster's Google-Drive QR code is intentionally kept on the closing card; the app footage itself contains the app's own UI text (including 'UIDAI · SITAA Cohort 1', test data 'shiv' / UID 121212121212, and the app's own spellings such as 'ZeroDCE') which cannot be edited; stage 5 of the pipeline slide is deliberately worded as cloud 'template & match'.
You are an INDEPENDENT reviewer. Your job is to find REAL defects the author missed — not to praise. Verify every suspicion with evidence (a frame you actually looked at, a number you measured, a file you read) before reporting it, and drop anything you cannot substantiate. Do NOT modify project files. Severity: blocker = would embarrass the company on the booth TV / factually wrong; major = clearly visible flaw; minor = polish; nit = trivial. If you find nothing in an area, say so in the verdict rather than inventing issues.
`

const COPY = `${CONTEXT}
YOUR LENS: SPELLING AND COPY-EDITING (the owner's original complaint was "there are some spelling errors in this video"). Build the complete text inventory and proof-read it as a careful American-English copy editor: (1) every spoken/subtitled line in ${PROJ}\\src\\data\\narration.json ('show' and 'say'); (2) every user-visible string in the TSX files under ${PROJ}\\src\\scenes and ${PROJ}\\src\\components and ${PROJ}\\src\\Booth.tsx, ${PROJ}\\src\\theme.ts (headings, tags, chips, cards, tiles, stage names, badges, booth/contact details); (3) the burned-in text you can actually SEE in decoded frames of the voiced video — sample a full-resolution frame every ~2.5 s across the entire video and read the text off the frames, since the build could differ from the source; (4) the SRT. Look for: misspellings, British spellings that slipped through (enrol, enrolment, programme, colour, centre, licence...), inconsistent spellings of the same word, wrong capitalization or hyphenation (e.g. 'AI-powered', 'Zero-DCE', 'U²-Net', 'YellowSense', 'IMC', 'Hall 1', 'TS 14'), punctuation errors, doubled words, wrong or inconsistent contact details (compare against ${PROJ}\\assets\\poster_text.json: website, email, phone), and inconsistent product naming across scenes. Separately list text that belongs to the APP footage (cannot be fixed by us) so the owner can decide whether to mask it. Report each issue with exact text, location (file/line or timestamp) and the corrected text.`

const PACING = `${CONTEXT}
YOUR LENS: PACING. The owner said the previous 2:10 cut was "too fast" and asked for a slower, longer cut (~4 min) with real-time footage and the full ~10 s processing waits shown. Judge whether this cut actually achieves that without becoming boring. Measure: total duration; speaking rate (words per minute from narration.json and the sentence durations); the gap between consecutive sentences; the longest stretch with NO narration (from the timeline) and what is on screen during it; the longest freeze-frame hold and whether the screen still has some motion/animation then; whether any footage still runs faster than real time (compare each clip's duration against build\\rt manifest and the spec in src/timeline.ts; there must be no rate > 1); whether any text on screen (subtitles, chips, headings) stays up long enough to read at normal speed (subtitle reading speed in characters per second; chip/card dwell times); whether scene transitions are abrupt; whether the narration is ahead of / behind the footage anywhere (e.g. a sentence describing a screen that is not yet visible, or a result card visible for several seconds before it is mentioned — check ~12 moments by decoding frames at sentence start times). Report concrete problems with timestamps and a specific fix (e.g. 'shift sentence X from 7.0 s to 9.5 s in shot B13').`

const VISUAL = `${CONTEXT}
YOUR LENS: VISUAL QA as a viewer standing 2-3 m from a 55-inch TV. Decode the voiced video with cv2 and look at (Read) real frames: at least one full-resolution frame every ~3 s across the whole video, PLUS frames at every scene boundary (scene fades), PLUS the first and last 10 frames (the video LOOPS, check the end flows into the start), PLUS frames inside every freeze-hold. Look for: text clipped/overflowing/overlapping other elements, subtitles covering important content or wrapping badly, elements colliding with the phone mock-up or each other, illegible text that is meant to be read (the 'Live zoom' panel should make app text readable — judge whether it does, and whether the zoom window shows the thing being narrated), awkward crops in the zoom panel (cut-off lines, black bars from beyond the video edge), animations caught mid-way at moments that should be settled, dead/blank screens longer than ~1.5 s, wrong step chip highlighted (e.g. 'Checks' vs 'Enrolled'), cards in the pipeline slide highlighting out of sync with the spoken stage, inconsistent branding, anything broken.`

const CLAIMS = `${CONTEXT}
YOUR LENS: CLAIMS AUDIT. Read ${PROJ}\\src\\data\\narration.json and every user-visible string in the scene/component TSX files. For each factual statement about the product, find evidence: (a) what the demo footage on screen actually shows (clip stills in ${PROJ}\\build\\rt and frames of the video), (b) the codebase at D:\\UIDAI (read D:\\UIDAI\\CLAUDE.md first; then backend files as needed), (c) ${PROJ}\\assets\\poster_text.json for poster wording. Scrutinise anything that could mislead a technically literate visitor or an evaluator: implying on-device processing; end-to-end speed claims that the app's own ~8-11 s timers contradict (the narration now says 'about ten seconds' for enrollment and 'under a second' only for the match step - verify both against the footage); 'enrolled users in that region' vs what the app says; 'liveness check' wording vs what is demonstrated; statements about the slap flow ('at least two matching fingers', 'all four passed'); the 'five stages' slide vs the narration; '1:N'; claims on the accuracy slide beyond the PoC-target framing; any wording that suggests UIDAI certification or endorsement. Also verify that nothing the narrator says is contradicted by the screen at that moment. Give the exact text, where it appears, the evidence and a safer replacement wording.`

const INTEGRITY = `${CONTEXT}
YOUR LENS: AUDIO / SUBTITLE / FILE INTEGRITY. Measure, don't assume. (1) Container facts for both mp4s via 'ffmpeg -i': resolution, fps, pix_fmt/range (must be limited-range yuv420p, bt709), duration, audio present in the voiced cut and ABSENT in the silent cut, faststart. (2) Audio: integrated loudness / true peak (ebur128), clipping, clicks at sentence boundaries, leading/trailing silence (the video loops), silence gaps > 4 s with no on-screen reason. (3) Subtitles: parse the .srt - monotonic, non-overlapping, within the duration, text identical to narration.json 'show'; confirm the BOM/encoding is acceptable; sample ~10 frames at cue midpoints and read the burned-in caption off the frame. (4) Sync: for at least 12 sentences, confirm via audio energy that speech is present inside each cue window and the caption appears within +-0.3 s. (5) Pronunciation risk: read the 'say' strings and flag words an Indian-English TTS voice could mispronounce (acronyms, 'U squared Net', 'Zero D C E', 'Yellow Sense', 'T S fourteen', 'one-to-one', 'liveness'); state clearly what you could and could not verify. (6) Confirm the SRT and the video have the same duration/timing basis (the .srt must match THIS render, not an older one).`

phase('Review')
const lenses = [
  ['copy', COPY],
  ['pacing', PACING],
  ['visual', VISUAL],
  ['claims', CLAIMS],
  ['integrity', INTEGRITY],
]
const results = await parallel(lenses.map(([n, p]) => () => agent(p, { label: 'review:' + n, phase: 'Review', schema: FINDINGS })))
const rank = { blocker: 0, major: 1, minor: 2, nit: 3 }
const all = results.flatMap((r, i) => (r ? r.findings.map((f) => ({ lens: lenses[i][0], ...f })) : []))
all.sort((a, b) => rank[a.severity] - rank[b.severity])
return { verdicts: Object.fromEntries(lenses.map(([n], i) => [n, results[i] ? results[i].verdict : null])), findings: all }
