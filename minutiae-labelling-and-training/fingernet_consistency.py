"""
How repeatable are FingerNet's minutiae across captures of the same finger?

Every image in ground_truth/ is the same finger (Rahul), so the same physical
minutia should be found again in every capture that shows it. For each image
pair this script:

  1. registers A onto B with SIFT on the finger only (FingerNet's segmentation
     mask, so a shared background can't drive the alignment), and a RANSAC
     similarity transform. Images are already 500 DPI, so the scale should be
     ~1; pairs with scale outside 0.8-1.25 or too few inliers are "unregistered";
  2. maps A's minutiae into B, keeping only those that land inside B's mask,
     and does the same from B to A (the overlap);
  3. counts a minutia as repeated when the other image has one within 15 px
     and 30 degrees (the tolerance FingerNet itself is scored with), matched
     one-to-one.

repeatability = 2 * matched / (A in overlap + B in overlap), i.e. an F1 between
the two captures. A chance baseline repeats step 3 with B's points rotated
by random angles about the overlap centroid (same count and density, no
correspondence), so the number above chance is what FingerNet is really
reproducing.

    python fingernet_consistency.py            # writes ground_truth/consistency/
"""
import csv
import itertools
import json
import os
import sys

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
GT = os.path.join(HERE, "ground_truth")
OUT = os.path.join(GT, "consistency")
sys.path.insert(0, os.path.join(HERE, "fingernet"))

MAX_D, MAX_O = 15.0, np.pi / 6
MIN_INLIERS = 25
SCALE_OK = (0.8, 1.25)
RNG = np.random.default_rng(0)


def load(name, net, run_image):
    img = cv2.imread(os.path.join(GT, "images_500dpi", name + ".png"), cv2.IMREAD_GRAYSCALE)
    with open(os.path.join(GT, "prelabels", name + ".json")) as f:
        m = json.load(f)["minutiae"]
    mnt = np.array([[p["x"], p["y"], p["direction"]] for p in m]).reshape(-1, 3)
    seg = run_image(net, img)["seg"][:img.shape[0], :img.shape[1]].astype(np.uint8)
    return img, mnt, seg


def match_count(a, b):
    """One-to-one matches within MAX_D / MAX_O (Hungarian on distance)."""
    if len(a) == 0 or len(b) == 0:
        return 0
    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    o = np.abs((a[:, None, 2] - b[None, :, 2] + np.pi) % (2 * np.pi) - np.pi)
    ok = (d <= MAX_D) & (o <= MAX_O)
    cost = np.where(ok, d, 1e6)
    r, c = linear_sum_assignment(cost)
    return int(ok[r, c].sum())


def transform(mnt, M):
    xy = mnt[:, :2] @ M[:, :2].T + M[:, 2]
    rot = np.arctan2(M[1, 0], M[0, 0])
    return np.column_stack([xy, (mnt[:, 2] + rot) % (2 * np.pi)])


def inside(mnt, mask):
    if len(mnt) == 0:
        return np.zeros(0, bool)
    x, y = np.round(mnt[:, 0]).astype(int), np.round(mnt[:, 1]).astype(int)
    ok = (x >= 0) & (y >= 0) & (x < mask.shape[1]) & (y < mask.shape[0])
    ok[ok] = mask[y[ok], x[ok]] > 0
    return ok


def chance(a_in_b, b_in, trials=20):
    """Same points, correspondence destroyed: B rotated about its centroid by a random angle."""
    if len(a_in_b) == 0 or len(b_in) == 0:
        return 0.0
    c = b_in[:, :2].mean(0)
    vals = []
    for _ in range(trials):
        t = RNG.uniform(np.pi / 6, 2 * np.pi - np.pi / 6)
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        xy = (b_in[:, :2] - c) @ R.T + c
        vals.append(match_count(a_in_b, np.column_stack([xy, (b_in[:, 2] + t) % (2 * np.pi)])))
    return 2 * float(np.mean(vals)) / (len(a_in_b) + len(b_in))


def main():
    from fingernet_tf2 import FingerNet, run_image

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(GT, "manifest.csv"), newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] != "rejected"]
    names = [r["image"] for r in rows]

    net = FingerNet()
    sift = cv2.SIFT_create(nfeatures=3000)
    data = {}
    for n in names:
        img, mnt, seg = load(n, net, run_image)
        mask = cv2.erode(seg, np.ones((15, 15), np.uint8))  # keep keypoints off the finger outline
        kp, des = sift.detectAndCompute(img, mask * 255)
        data[n] = dict(mnt=mnt, seg=seg, kp=kp, des=des)
    print("features ready for", len(names), "images")

    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    results = []
    for a, b in itertools.combinations(names, 2):
        A, B = data[a], data[b]
        rec = dict(a=a, b=b, registered=False)
        if A["des"] is not None and B["des"] is not None and len(A["kp"]) > 10 and len(B["kp"]) > 10:
            good = [m for m, n2 in matcher.knnMatch(A["des"], B["des"], k=2) if m.distance < 0.75 * n2.distance]
            if len(good) >= MIN_INLIERS:
                src = np.float32([A["kp"][m.queryIdx].pt for m in good])
                dst = np.float32([B["kp"][m.trainIdx].pt for m in good])
                M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0)
                if M is not None:
                    scale = float(np.hypot(M[0, 0], M[1, 0]))
                    ninl = int(inl.sum())
                    rec.update(inliers=ninl, scale=round(scale, 3),
                               rotation_deg=round(float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))), 1))
                    if ninl >= MIN_INLIERS and SCALE_OK[0] <= scale <= SCALE_OK[1]:
                        Minv = cv2.invertAffineTransform(M)
                        a_in_b = transform(A["mnt"], M)
                        a_keep = a_in_b[inside(a_in_b, B["seg"])]
                        b_in_a = transform(B["mnt"], Minv)
                        b_keep = B["mnt"][inside(b_in_a, A["seg"])]
                        m = match_count(a_keep, b_keep)
                        denom = len(a_keep) + len(b_keep)
                        rec.update(registered=True, n_a=len(a_keep), n_b=len(b_keep), matched=m,
                                   repeatability=round(2 * m / denom, 4) if denom else 0.0,
                                   chance=round(chance(a_keep, b_keep), 4))
        results.append(rec)

    keys = ["a", "b", "registered", "inliers", "scale", "rotation_deg", "n_a", "n_b", "matched",
            "repeatability", "chance"]
    with open(os.path.join(OUT, "pairs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in keys})

    reg = [r for r in results if r["registered"]]
    rep = np.array([r["repeatability"] for r in reg])
    ch = np.array([r["chance"] for r in reg])
    per_img = {}
    for r in reg:
        for n in (r["a"], r["b"]):
            per_img.setdefault(n, []).append(r["repeatability"])
    with open(os.path.join(OUT, "per_image.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "registered_pairs", "mean_repeatability"])
        for n in names:
            v = per_img.get(n, [])
            w.writerow([n, len(v), "%.4f" % np.mean(v) if v else ""])

    summary = dict(images=len(names), pairs=len(results), registered=len(reg),
                   repeatability_mean=float(rep.mean()), repeatability_median=float(np.median(rep)),
                   repeatability_p10=float(np.percentile(rep, 10)), repeatability_p90=float(np.percentile(rep, 90)),
                   chance_mean=float(ch.mean()))
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
