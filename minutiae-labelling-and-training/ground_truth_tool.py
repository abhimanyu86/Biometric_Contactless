"""
Build and finalise a ground-truth minutiae set.

    python ground_truth_tool.py prepare SRC_DIR [--out ground_truth]
    python ground_truth_tool.py finalize        [--out ground_truth] [--annotator NAME]

prepare -- for every image in SRC_DIR:
    1. grayscale, resampled to 500 DPI equivalent. The ridge period is
       measured and the image scaled so it becomes fir.TARGET_RIDGE_PERIOD_PX
       (10 px), the same definition backend/fir.py uses. Phone crops have
       periods of 40-60 px, above fir's 30 px measuring band, so the period
       is measured on an 800 px working copy, on the central patch only
       (background and the finger outline dominate the spectrum otherwise).
    2. CLAHE, because raw skin photos have very low ridge contrast.
    3. FingerNet pre-labels on that exact image.
  Writes images_500dpi/, prelabels/ (.json + .sim), overlays/, manifest.csv.
  labels/ is left empty: that is where the human's corrected .sim files go.

finalize -- every labels/<name>.sim becomes final/<name>.json, carrying the
  manifest metadata so each label can be traced to its source photo and
  mapped back to it (x_src = x / scale_to_500dpi). Set a manifest row's
  status to "rejected" by hand for images too poor to label; finalize
  leaves those alone.

All coordinates are in the 500 DPI image's pixel frame, y down. Directions
are in radians, the backend/minutiae.py convention.
"""
import argparse
import csv
import datetime
import glob
import hashlib
import json
import math
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "fingernet"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "backend"))

from fir import TARGET_RIDGE_PERIOD_PX, estimate_ridge_period  # noqa: E402

WORK_MAX_DIM = 800          # working copy for period measurement
CENTRE_FRACTION = 0.5       # side of the central measuring patch, as a fraction of the short side
PLAUSIBLE_WORK_PERIOD = (5.0, 30.0)
CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

MANIFEST_FIELDS = ["image", "source", "source_sha256", "orig_w", "orig_h",
                   "scale_to_500dpi", "scale_method", "ridge_period_after", "w", "h",
                   "prelabel_count", "status", "annotator", "final_count", "notes"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def centre_patch(img, frac=CENTRE_FRACTION):
    h, w = img.shape
    n = int(min(h, w) * frac)
    y0, x0 = (h - n) // 2, (w - n) // 2
    return img[y0:y0 + n, x0:x0 + n]


def ridge_period(img, sigma):
    """fir.estimate_ridge_period after a high-pass. Without it, slow shading across the
    skin outweighs the ridges and the estimate lands near the 30 px top of fir's band."""
    img = img.astype(np.float32)
    return estimate_ridge_period(img - cv2.GaussianBlur(img, (0, 0), sigma))


def measure_scale(gray):
    """Factor that maps the source image to 500 DPI, or None if the period is not measurable."""
    s0 = min(1.0, WORK_MAX_DIM / max(gray.shape))
    work = cv2.resize(gray, None, fx=s0, fy=s0, interpolation=cv2.INTER_AREA)
    period = ridge_period(CLAHE.apply(centre_patch(work)), sigma=10)
    if period is None or not (PLAUSIBLE_WORK_PERIOD[0] <= period <= PLAUSIBLE_WORK_PERIOD[1]):
        return None
    return s0 * TARGET_RIDGE_PERIOD_PX / period


def natural_key(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return (0, int(stem), "") if stem.isdigit() else (1, 0, stem)


def to_sim_lines(mnt):
    # py-minutiae-viewer draws the tail along (angle_deg - 90); see sim_convert.py
    return ["%d %d %.2f END %.3f" % (round(x), round(y), (math.degrees(o) + 90) % 360, s)
            for x, y, o, s in mnt]


def write_manifest(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)


def read_manifest(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def prepare(src, out):
    from fingernet_tf2 import FingerNet, draw, run_image

    dirs = {k: os.path.join(out, k) for k in ("images_500dpi", "prelabels", "overlays", "labels", "final")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    paths = sorted((p for ext in ("jpg", "jpeg", "png", "bmp", "tif")
                    for p in glob.glob(os.path.join(src, "*." + ext))), key=natural_key)
    grays, scales = {}, {}
    for p in paths:
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is None:
            print("skip (unreadable):", p)
            continue
        grays[p] = g
        scales[p] = measure_scale(g)

    measured = [s for s in scales.values() if s]
    fallback = float(np.median(measured)) if measured else None
    print("measured %d/%d scales, median %.4f" % (len(measured), len(grays), fallback or float("nan")))

    net = FingerNet()
    rows = []
    for i, (p, g) in enumerate(grays.items(), 1):
        t0 = time.time()
        name = os.path.splitext(os.path.basename(p))[0]
        scale, method = scales[p], "ridge_period"
        if scale is None:
            scale, method = fallback, "median_fallback"
        img = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        img = CLAHE.apply(img)
        cv2.imwrite(os.path.join(dirs["images_500dpi"], name + ".png"), img)

        after = ridge_period(centre_patch(img), sigma=3)  # tighter: shading still wins at sigma>=4
        res = run_image(net, img)                      # already at 500 DPI: no further rescale
        mnt = res["mnt"]
        with open(os.path.join(dirs["prelabels"], name + ".json"), "w") as f:
            json.dump({"source": "FingerNet (prediction, not ground truth)",
                       "image": name + ".png",
                       "minutiae": [{"x": float(x), "y": float(y), "direction": float(o), "score": float(s)}
                                    for x, y, o, s in mnt]}, f, indent=1)
        with open(os.path.join(dirs["prelabels"], name + ".sim"), "w") as f:
            f.write("\n".join(to_sim_lines(mnt)))
        draw(img.astype(np.float32), mnt, os.path.join(dirs["overlays"], name + ".png"))

        rows.append({"image": name, "source": os.path.abspath(p), "source_sha256": sha256(p),
                     "orig_w": g.shape[1], "orig_h": g.shape[0],
                     "scale_to_500dpi": "%.5f" % scale, "scale_method": method,
                     "ridge_period_after": "%.2f" % after if after else "",
                     "w": img.shape[1], "h": img.shape[0], "prelabel_count": len(mnt),
                     "status": "prelabelled", "annotator": "", "final_count": "", "notes": ""})
        print("[%3d/%d] %-6s %4dx%-4d -> %3dx%-3d scale=%.4f (%s) period_after=%s prelabels=%3d  %.1fs" % (
            i, len(grays), name, g.shape[1], g.shape[0], img.shape[1], img.shape[0], scale, method,
            rows[-1]["ridge_period_after"] or "n/a", len(mnt), time.time() - t0))

    write_manifest(os.path.join(out, "manifest.csv"), rows)
    print("wrote", os.path.join(out, "manifest.csv"))


def finalize(out, annotator):
    man_path = os.path.join(out, "manifest.csv")
    rows = read_manifest(man_path)
    os.makedirs(os.path.join(out, "final"), exist_ok=True)
    today = datetime.date.today().isoformat()
    done = 0
    for r in rows:
        sim = os.path.join(out, "labels", r["image"] + ".sim")
        if r["status"] == "rejected" or not os.path.exists(sim):
            continue
        items = []
        with open(sim) as f:
            for line in f:
                p = line.split()
                if len(p) >= 4:
                    items.append({"x": int(p[0]), "y": int(p[1]),
                                  "direction": math.radians((float(p[2]) - 90) % 360),
                                  "type": "BIF" if p[3] == "BIF" else "RIG"})
        r["status"], r["final_count"] = "done", len(items)
        r["annotator"] = r["annotator"] or annotator
        with open(os.path.join(out, "final", r["image"] + ".json"), "w") as f:
            json.dump({"image": r["image"], "labelled_image": "images_500dpi/%s.png" % r["image"],
                       "source": r["source"], "source_sha256": r["source_sha256"],
                       "scale_to_500dpi": float(r["scale_to_500dpi"]), "scale_method": r["scale_method"],
                       "width": int(r["w"]), "height": int(r["h"]),
                       "status": "done", "annotator": r["annotator"], "labelled_on": today,
                       "minutiae": items}, f, indent=1)
        done += 1
    write_manifest(man_path, rows)
    counts = {s: sum(r["status"] == s for r in rows) for s in ("done", "prelabelled", "rejected")}
    print("finalised %d; manifest status: %s" % (done, counts))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prepare", "finalize"])
    ap.add_argument("src", nargs="?")
    ap.add_argument("--out", default=os.path.join(HERE, "ground_truth"))
    ap.add_argument("--annotator", default="")
    a = ap.parse_args()
    if a.cmd == "prepare":
        if not a.src:
            ap.error("prepare needs SRC_DIR")
        prepare(a.src, a.out)
    else:
        finalize(a.out, a.annotator)


if __name__ == "__main__":
    main()
