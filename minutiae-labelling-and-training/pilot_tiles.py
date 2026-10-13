"""
Rendering + editing helper for the Claude labelling pilot.

A working set is ground_truth/pilot_claude/<name>/points.json:
    {"image": path, "points": [{"id", "x", "y", "direction", "type", "src"}]}
where src is "fingernet" (kept as predicted), "moved" or "added".

    python pilot_tiles.py init   NAME IMAGE PRELABEL_JSON   # seed from FingerNet
    python pilot_tiles.py render NAME [--tile 128 --zoom 4] # overview + grid tiles
    python pilot_tiles.py edit   NAME EDITS_JSON            # apply edits, re-render
    python pilot_tiles.py export NAME OUT.sim               # py-minutiae-viewer file

EDITS_JSON: {"delete": [ids], "move": {id: [x, y]}, "type": {id: "RIG"|"BIF"},
             "direction": {id: deg}, "add": [[x, y, deg, "RIG"|"BIF"], ...]}
Directions are degrees in image coordinates (y down), FingerNet/backend
convention: 0 = +x, 90 = +y (pointing down).
"""
import json
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "ground_truth", "pilot_claude")
ENHANCE = True


def path(name, *p):
    return os.path.join(ROOT, name, *p)


def load(name):
    with open(path(name, "points.json")) as f:
        return json.load(f)


def save(name, d):
    with open(path(name, "points.json"), "w") as f:
        json.dump(d, f, indent=1)


def init(name, image, prelabel):
    os.makedirs(path(name), exist_ok=True)
    with open(prelabel) as f:
        m = json.load(f)["minutiae"]
    pts = [{"id": i, "x": round(p["x"]), "y": round(p["y"]), "direction": p["direction"],
            "type": "RIG", "src": "fingernet"} for i, p in enumerate(m)]
    save(name, {"image": os.path.abspath(image), "points": pts, "next_id": len(pts)})
    render(name)


def draw_points(vis, pts, zoom, ox=0, oy=0, font=0.45):
    for p in pts:
        x, y = (p["x"] - ox) * zoom, (p["y"] - oy) * zoom
        col = (0, 0, 255) if p["type"] == "RIG" else (255, 0, 0)
        if p["src"] != "fingernet":
            col = (0, 170, 0) if p["type"] == "RIG" else (200, 0, 200)
        r = max(4, int(1.5 * zoom))
        c = (int(round(x)), int(round(y)))
        if p["type"] == "RIG":
            cv2.rectangle(vis, (c[0] - r, c[1] - r), (c[0] + r, c[1] + r), col, 1)
        else:
            cv2.circle(vis, c, r, col, 1)
        L = 6 * zoom
        cv2.line(vis, c, (int(x + L * math.cos(p["direction"])), int(y + L * math.sin(p["direction"]))), col, 1)
        cv2.putText(vis, str(p["id"]), (c[0] + r + 1, c[1] - r), cv2.FONT_HERSHEY_SIMPLEX, font, col, 1)


def render(name, tile=128, zoom=4):
    d = load(name)
    img = cv2.imread(d["image"], cv2.IMREAD_GRAYSCALE)
    out = path(name, "render")
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        os.remove(os.path.join(out, f))
    oz = 2
    ov = cv2.cvtColor(cv2.resize(img, None, fx=oz, fy=oz, interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    draw_points(ov, d["points"], oz, font=0.4)
    h, w = img.shape
    step = tile - 16  # 16 px overlap so nothing is lost on a tile edge
    k = 0
    for y0 in range(0, max(1, h - 16), step):
        for x0 in range(0, max(1, w - 16), step):
            k += 1
            cv2.rectangle(ov, (x0 * oz, y0 * oz), (min(w, x0 + tile) * oz, min(h, y0 + tile) * oz), (0, 200, 255), 1)
            cv2.putText(ov, "T%d" % k, (x0 * oz + 3, y0 * oz + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
            crop = img[y0:y0 + tile, x0:x0 + tile]
            if ENHANCE:  # display only: local contrast so faint ridges are judgeable
                crop = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(crop)
            vis = cv2.cvtColor(cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
            for gx in range((x0 // 10 + 1) * 10, x0 + crop.shape[1], 10):   # 10 px grid, labelled every 20
                X = (gx - x0) * zoom
                cv2.line(vis, (X, 0), (X, 6 if gx % 20 else 12), (0, 200, 255), 1)
                if gx % 20 == 0:
                    cv2.putText(vis, str(gx), (X + 2, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1)
            for gy in range((y0 // 10 + 1) * 10, y0 + crop.shape[0], 10):
                Y = (gy - y0) * zoom
                cv2.line(vis, (0, Y), (6 if gy % 20 else 12, Y), (0, 200, 255), 1)
                if gy % 20 == 0:
                    cv2.putText(vis, str(gy), (14, Y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1)
            sel = [p for p in d["points"] if x0 - 3 <= p["x"] < x0 + tile + 3 and y0 - 3 <= p["y"] < y0 + tile + 3]
            draw_points(vis, sel, zoom, x0, y0)
            cv2.imwrite(os.path.join(out, "T%02d_x%d_y%d.png" % (k, x0, y0)), vis)
    cv2.imwrite(os.path.join(out, "overview.png"), ov)
    print("%s: %d points, %d tiles -> %s" % (name, len(d["points"]), k, out))


def edit(name, edits_path):
    d = load(name)
    with open(edits_path) as f:
        e = json.load(f)
    by_id = {p["id"]: p for p in d["points"]}
    for i in e.get("delete", []):
        by_id.pop(int(i))
    for i, (x, y) in e.get("move", {}).items():
        by_id[int(i)].update(x=x, y=y, src="moved")
    for i, t in e.get("type", {}).items():
        by_id[int(i)]["type"] = t
    for i, deg in e.get("direction", {}).items():
        by_id[int(i)]["direction"] = math.radians(deg)
        if by_id[int(i)]["src"] == "fingernet":
            by_id[int(i)]["src"] = "moved"
    for x, y, deg, t in e.get("add", []):
        i = d["next_id"]
        d["next_id"] += 1
        by_id[i] = {"id": i, "x": x, "y": y, "direction": math.radians(deg), "type": t, "src": "added"}
    d["points"] = sorted(by_id.values(), key=lambda p: p["id"])
    d.setdefault("edit_log", []).append(e)
    save(name, d)
    render(name)


def export(name, out):
    d = load(name)
    lines = ["%d %d %.2f %s 1.000" % (p["x"], p["y"], (math.degrees(p["direction"]) + 90) % 360,
                                      "BIF" if p["type"] == "BIF" else "END") for p in d["points"]]
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print("wrote %s (%d minutiae)" % (out, len(lines)))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "init":
        init(*sys.argv[2:5])
    elif cmd == "render":
        render(sys.argv[2])
    elif cmd == "edit":
        edit(sys.argv[2], sys.argv[3])
    elif cmd == "export":
        export(sys.argv[2], sys.argv[3])
