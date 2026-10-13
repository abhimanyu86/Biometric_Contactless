"""Grid of per-point patches for review.
    python pilot_patches.py NAME [--mask MASK.png] [--ids 1,2,3] [--half 20] [--zoom 6]
Each patch is centred on one point; ticks mark the centre, the line is the current direction."""
import argparse, math, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_tiles as pt

ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("--mask"); ap.add_argument("--ids")
ap.add_argument("--half", type=int, default=20); ap.add_argument("--zoom", type=int, default=6)
ap.add_argument("--per", type=int, default=12)
a = ap.parse_args()
d = pt.load(a.name)
img = cv2.imread(d["image"], cv2.IMREAD_GRAYSCALE)
enh = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(img)
pts = d["points"]
if a.mask:
    m = cv2.imread(a.mask, 0); pts = [p for p in pts if m[int(p["y"]), int(p["x"])] > 0]
if a.ids:
    want = {int(i) for i in a.ids.split(",")}; pts = [p for p in pts if p["id"] in want]
h, z = a.half, a.zoom
padded = cv2.copyMakeBorder(enh, h, h, h, h, cv2.BORDER_REFLECT)
tiles = []
for p in pts:
    cx, cy = int(round(p["x"])), int(round(p["y"]))
    crop = padded[cy:cy + 2 * h + 1, cx:cx + 2 * h + 1]
    t = cv2.cvtColor(cv2.resize(crop, None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
    c = h * z + z // 2
    col = (0, 0, 255) if p["src"] == "fingernet" else (0, 170, 0)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):      # centre ticks, leaving the centre visible
        cv2.line(t, (c + dx * 2 * z, c + dy * 2 * z), (c + dx * 4 * z, c + dy * 4 * z), col, 2)
    L = 8 * z
    cv2.line(t, (c + int(4 * z * math.cos(p["direction"])), c + int(4 * z * math.sin(p["direction"]))),
             (c + int(L * math.cos(p["direction"])), c + int(L * math.sin(p["direction"]))), (255, 120, 0), 2)
    cv2.putText(t, "#%d (%d,%d) %s" % (p["id"], cx, cy, p["type"]), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
    tiles.append(t)
out = []
for k in range(0, len(tiles), a.per):
    group = tiles[k:k + a.per]
    while len(group) % 4: group.append(np.full_like(tiles[0], 255))
    g = np.vstack([np.hstack(group[i:i + 4]) for i in range(0, len(group), 4)])
    f = pt.path(a.name, "patches_%d.png" % (k // a.per + 1)); cv2.imwrite(f, g); out.append(f)
print(len(pts), "points ->", *out, sep="\n")
