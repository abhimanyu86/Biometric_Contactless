"""
Convert between this project's minutiae JSON and py-minutiae-viewer's .sim
files, so predictions can be corrected by hand in the viewer's editor tab.

    .sim line:  x y angle_deg END|BIF quality
    JSON item:  {x, y, direction (radians), type: "RIG"|"BIF"}

Both use image pixel coordinates with y pointing down. The viewer draws a
minutia's tail along (angle_deg - 90), so angle_deg = degrees(direction) + 90.

    python sim_convert.py to-sim  fingernet/output/scale1.5/*.json   # -> .sim next to each
    python sim_convert.py to-json labels/*.sim                       # -> .gt.json next to each

FingerNet predicts no type, so every imported point starts as END (ridge
ending). Fix bifurcations while correcting: in the editor, delete the point
(right-click) and re-place it with Ctrl+drag.
"""
import glob
import json
import math
import os
import sys


def to_sim(path):
    with open(path) as f:
        data = json.load(f)
    items = data["minutiae"] if isinstance(data, dict) else data
    lines = []
    for m in items:
        kind = "BIF" if m.get("type") == "BIF" else "END"
        angle = (math.degrees(m["direction"]) + 90) % 360
        lines.append("%d %d %.2f %s %.3f" % (round(m["x"]), round(m["y"]), angle, kind, m.get("score", 1.0)))
    out = os.path.splitext(path)[0] + ".sim"
    with open(out, "w") as f:
        f.write("\n".join(lines))  # the viewer's reader rejects a trailing blank line
    return out, len(lines)


def to_json(path):
    items = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 4:
                continue
            items.append({"x": int(p[0]), "y": int(p[1]),
                          "direction": math.radians((float(p[2]) - 90) % 360),
                          "type": "BIF" if p[3] == "BIF" else "RIG"})
    out = os.path.splitext(path)[0] + ".gt.json"
    with open(out, "w") as f:
        json.dump({"source": "manual (py-minutiae-viewer)", "minutiae": items}, f, indent=1)
    return out, len(items)


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("to-sim", "to-json"):
        sys.exit(__doc__)
    fn = to_sim if sys.argv[1] == "to-sim" else to_json
    for pattern in sys.argv[2:]:
        for p in glob.glob(pattern) or [pattern]:
            out, n = fn(p)
            print("%s  (%d minutiae)" % (out, n))
