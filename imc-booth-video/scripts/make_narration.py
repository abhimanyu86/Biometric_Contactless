"""Synthesise the booth narration, one MP3 per sentence, and write src/data/narration.json.

Each sentence is its own TTS utterance so subtitle timing is exact by construction and the
video can trigger an animation on a specific sentence. `say` is what the voice reads
(phonetic spellings for acronyms), `show` is what the subtitle displays.

Usage:  python scripts/make_narration.py [--voice en-IN-NeerjaNeural] [--rate +0%]
Needs internet (edge-tts talks to Microsoft's speech service) and tools/ffmpeg.exe.
"""
import argparse
import asyncio
import json
import re
import subprocess
from pathlib import Path

import edge_tts

ROOT = Path(__file__).resolve().parent.parent
FFMPEG = ROOT / "tools" / "ffmpeg.exe"
OUT_DIR = ROOT / "public" / "media" / "narration"
DATA = ROOT / "src" / "data" / "narration.json"

# (beat id, scene, [(show, say)])  -- say == None means same as show
BEATS = [
    ("B01", "hook", [
        ("Imagine verifying identity with just the phone in your hand.", None),
        ("No scanner. No touch. Just a camera.", "No scanner, no touch, just a camera."),
    ]),
    ("B02", "problem", [
        ("Today, fingerprint authentication needs dedicated scanners to buy, deploy and maintain.", None),
        ("YellowSense turns the camera of a smartphone into a contactless fingerprint reader.", "Yellow Sense turns the camera of a smartphone into a contactless fingerprint reader."),
        ("One app handles both enrollment and authentication.", None),
    ]),
    ("B03", "enroll_form", [
        ("Let's start with enrollment.", None),
        ("The operator enters the person's name and ID number.", None),
        ("Then selects the region.", None),
    ]),
    ("B04", "enroll_capture", [
        ("The person holds one index finger in front of the rear camera.", None),
        ("Auto-capture takes the shot once the finger is steady and in focus.", None),
    ]),
    ("B05", "enroll_process", [
        ("The app checks image quality, finds the finger, and runs a liveness check.", None),
        ("Including the capture, enrollment takes about ten seconds.", None),
        ("The result card reports the capture as live.", None),
    ]),
    ("B06", "auth_open", [
        ("To authenticate, the operator opens the Authenticate screen.", None),
        ("The match runs against the users enrolled in the chosen region.", None),
        ("The person scans the same finger again.", None),
    ]),
    ("B07", "auth_result", [
        ("The same processing steps run first.", None),
        ("The match itself takes under a second, and the person is authenticated.", None),
    ]),
    ("B08", "pipeline", [
        ("Under the hood, five stages do the heavy lifting.", None),
        ("YOLO finds the finger.", None),
        ("A liveness check classifies each capture as live or not.", None),
        ("U²-Net segments it.", "U squared Net segments it."),
        ("Enhancement brings out the ridges, with Zero-DCE for low light.", "Enhancement brings out the ridges, with Zero D C E for low light."),
        ("And in the cloud, the print becomes a template and is matched.", None),
    ]),
    ("B09", "slap_intro", [
        ("Slap mode captures four fingers in a single shot.", None),
        ("The home screen also offers one-to-one verification, a pipeline view, and a quality check tool.", None),
    ]),
    ("B10", "slap_form", [
        ("The operator fills in the same details, and chooses which hand to scan.", None),
        ("Here, it is the left hand.", None),
    ]),
    ("B11", "slap_capture", [
        ("The person holds four fingers sideways in front of the rear camera.", None),
        ("Detection boxes lock onto each fingertip, and the app checks liveness.", None),
    ]),
    ("B12", "slap_enrolled", [
        ("Every finger is detected, checked for liveness, and processed on its own.", None),
        ("All four fingers are enrolled.", None),
    ]),
    ("B13", "slap_auth", [
        ("To authenticate, the person scans the same hand again.", None),
        ("The same checks run on every finger.", None),
        ("Authentication needs at least two matching fingers.", None),
        ("Here, all four passed, and the match took under a second.", None),
    ]),
    ("B14", "slap_pipeline", [
        ("The slap pipeline view lets you watch a capture step by step.", None),
        ("While it processes, each finger is cropped and checked on its own.", None),
        ("Then every fingertip appears with its crop, its liveness result, and a processed ridge image.", None),
    ]),
    ("B15", "accuracy", [
        ("Our target is over 85% matching accuracy.", "Our target is over eighty-five percent matching accuracy."),
        ("That is the acceptance target for this proof of concept.", None),
    ]),
    ("B16", "pillars", [
        ("Our design goals: secure, real-time, AI-powered, enterprise ready.", "Our design goals: secure, real-time, A I-powered, enterprise ready."),
    ]),
    ("B17", "closing", [
        ("A world where every identity is verified securely and seamlessly.", None),
        ("YellowSense Technologies. Trust, instantly.", "Yellow Sense Technologies, trust, instantly."),
    ]),
    ("B18", "closing", [
        ("Visit us at IMC, Hall 1, Booth TS 14, and try it yourself.", "Visit us at the India Mobile Congress, Hall 1, Booth T S fourteen, and try it yourself."),
    ]),
]


def duration_sec(path: Path) -> float:
    r = subprocess.run([str(FFMPEG), "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", r.stderr)
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


TRIM = (
    "silenceremove=start_periods=1:start_duration=0:start_threshold=-45dB:start_silence=0.04:detection=peak,"
    "areverse,"
    "silenceremove=start_periods=1:start_duration=0:start_threshold=-45dB:start_silence=0.10:detection=peak,"
    "areverse"
)


def trim_silence(path: Path) -> None:
    """edge-tts pads each utterance (~0.25 s head, up to ~0.9 s tail) — strip it so
    pacing and subtitle timing are controlled by the video, not by the TTS service."""
    tmp = path.with_suffix(".trim.mp3")
    subprocess.run([str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
                    "-af", TRIM, "-c:a", "libmp3lame", "-q:a", "3", str(tmp)], check=True)
    tmp.replace(path)


async def synth(text: str, voice: str, rate: str, out: Path) -> None:
    for attempt in range(3):
        try:
            await edge_tts.Communicate(text, voice, rate=rate).save(str(out))
            if out.exists() and out.stat().st_size > 1000:
                trim_silence(out)
                return
        except Exception as exc:  # network hiccup -> retry
            print("  retry", attempt + 1, exc)
            await asyncio.sleep(1.5)
    raise RuntimeError(f"TTS failed for: {text}")


async def main(voice: str, rate: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob('*.mp3'):
        old.unlink()
    DATA.parent.mkdir(parents=True, exist_ok=True)
    beats = []
    for bid, scene, sentences in BEATS:
        items = []
        for i, (show, say) in enumerate(sentences, 1):
            fname = f"{bid}_{i}.mp3"
            await synth(say or show, voice, rate, OUT_DIR / fname)
            d = duration_sec(OUT_DIR / fname)
            items.append({"show": show, "say": say or show, "file": f"media/narration/{fname}", "durationSec": round(d, 3)})
            print(f"{fname}: {d:5.2f}s  {show}")
        beats.append({"id": bid, "scene": scene, "sentences": items})
    DATA.write_text(json.dumps({"voice": voice, "rate": rate, "beats": beats}, indent=2, ensure_ascii=False), encoding="utf-8")
    total = sum(s["durationSec"] for b in beats for s in b["sentences"])
    print(f"\nTotal speech: {total:.1f}s across {sum(len(b['sentences']) for b in beats)} sentences -> {DATA}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="en-IN-NeerjaNeural")
    ap.add_argument("--rate", default="-12%")
    a = ap.parse_args()
    asyncio.run(main(a.voice, a.rate))
