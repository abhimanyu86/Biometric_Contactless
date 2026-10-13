"""Copy prepared clips / brand assets / manifests into where Remotion reads them.

  build/clips/*.mp4          -> public/media/clips/
  assets/*.png               -> public/media/assets/
  build/clips/manifest_*.json -> src/data/
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

pairs = [
    (ROOT / "build" / "rt", "*.mp4", ROOT / "public" / "media" / "clips"),  # real-time (1x) clips
    (ROOT / "assets", "*.png", ROOT / "public" / "media" / "assets"),
]
for src, pat, dst in pairs:
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob(pat))
    for f in files:
        shutil.copy2(f, dst / f.name)
    print(f"{len(files):3d} x {pat:18s} -> {dst.relative_to(ROOT)}")
