"""Re-cut the demo recordings at 1x REAL TIME (no speed ramps) and remap the zoom regions from the earlier
speed-ramped clips onto the new clip timing.

Cut points are the PTS-accurate source times the earlier prep pass verified (build/clips/manifest_*.json).
Outputs:  build/rt/<id>.mp4   (720x1560, constant 30 fps, no audio)   and   src/data/manifest_rt.json
"""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FFMPEG = str(ROOT / "tools" / "ffmpeg.exe")
OUT = ROOT / "build" / "rt"
OUT.mkdir(parents=True, exist_ok=True)

SRC = {
    "A": r"C:\Users\ABHIMANYU MALIK\Downloads\Record_2026-10-06-22-49-27_221aa898fa5b29237199bc159eb7fefc.mp4",
    "B": r"C:\Users\ABHIMANYU MALIK\Downloads\video2.mp4",
}

# id -> (source, start, end)   all true-PTS seconds in the original recordings
CLIPS = {
    # ---- single finger (video 1) ----
    "rt_single_home":        ("A", 0.00, 3.35),
    "rt_single_form":        ("A", 3.85, 17.05),
    "rt_single_capture":     ("A", 18.80, 28.75),
    "rt_single_enrolled":    ("A", 28.75, 37.00),   # camera closes -> processing checklist (timer to 10.76 s) -> Enrolled card
    "rt_single_auth_open":   ("A", 39.78, 46.12),
    "rt_single_auth_cap":    ("A", 46.55, 54.72),
    "rt_single_auth_result": ("A", 54.75, 63.70),   # processing (timer to 10.65 s) -> match -> AUTHENTICATED card
    "rt_single_pipeline":    ("A", 67.00, 87.40),   # played 67.0-85.5 then jump-cut to 87.1-87.3 (skips 'Minutiae: Not available')
    # ---- slap (video 2) ----
    "rt_slap_home":          ("B", 0.00, 0.87),     # only ~0.9 s of settled home exists; the scene holds the last frame
    "rt_slap_menu":          ("B", 68.76, 69.74),   # ACTIONS tiles: Enroll / Authenticate / Verify / Pipeline / QC Pipeline
    "rt_slap_form":          ("B", 3.906, 18.51),
    "rt_slap_capture":       ("B", 21.355, 30.976),
    "rt_slap_enrolled":      ("B", 30.993, 39.894),  # processing checklist -> '4 finger(s) enrolled successfully!'
    "rt_slap_auth":          ("B", 47.847, 64.983),  # capture -> processing (timer to 10.84 s) -> AUTHENTICATED (4 / 2 required)
    "rt_slap_pipeline":      ("B", 70.457, 95.296),  # pipeline visualizer: capture -> per-finger cards
}


def cut(cid, src, s, e):
    out = OUT / f"{cid}.mp4"
    vf = f"trim=start={s}:end={e},setpts=PTS-STARTPTS,fps=30"
    subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", SRC[src], "-vf", vf, "-an",
                    "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], check=True)
    return out


def frames(path):
    import cv2
    c = cv2.VideoCapture(str(path))
    n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = c.get(cv2.CAP_PROP_FPS)
    c.release()
    return n, fps


# ---------- old clips -> (src time <-> out time) so zoom regions can be remapped ----------
def old_segments(c, src_name):
    segs = []
    t_out = 0.0
    for s in c.get("speedSegments") or []:
        a = s.get("srcFromSec", s.get("srcStartSec"))
        b = s.get("srcToSec", s.get("srcEndSec"))
        if a is None or b is None or "speed" not in s:
            continue
        o = s.get("outFromSec", t_out)
        dur = (b - a) / s["speed"]
        segs.append((o, o + dur, a, b))
        t_out = o + dur
    return segs


def out_to_src(segs, t):
    for o0, o1, a, b in segs:
        if o0 - 1e-6 <= t <= o1 + 1e-6:
            return a + (t - o0) / (o1 - o0) * (b - a) if o1 > o0 else a
    return None


manifest = {"clips": []}
old = {}
for name, key in (("single", "A"), ("slap", "B")):
    d = json.load(open(ROOT / "build" / "clips" / f"manifest_{name}.json"))
    for c in d["clips"]:
        old.setdefault(key, []).append((c, old_segments(c, name)))

for cid, (src, s, e) in CLIPS.items():
    p = cut(cid, src, s, e)
    n, fps = frames(p)
    regions = []
    for c, segs in old.get(src, []):
        for r in c.get("zoomRegions", []):
            s0, s1 = out_to_src(segs, r["tStartSec"]), out_to_src(segs, r["tEndSec"])
            if s0 is None or s1 is None:
                continue
            a, b = max(s0, s), min(s1, e)
            if b - a < 0.2:
                continue
            regions.append({"x": r["x"], "y": r["y"], "w": r["w"], "h": r["h"], "tStartSec": round(a - s, 3), "tEndSec": round(b - s, 3), "name": r.get("why", "")})
    regions.sort(key=lambda r: (r["tStartSec"], r["tEndSec"]))
    manifest["clips"].append({"id": cid, "file": f"{cid}.mp4", "durationSec": round(n / fps, 3), "speed": 1, "srcStartSec": s, "srcEndSec": e, "zoomRegions": regions})
    print(f"{cid:24s} {n/fps:6.2f}s  regions={len(regions)}")

(ROOT / "src" / "data" / "manifest_rt.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
total = sum(c["durationSec"] for c in manifest["clips"])
print(f"\ntotal real-time footage: {total:.1f}s  -> src/data/manifest_rt.json")
