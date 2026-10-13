"""
Independent checks on the Claude labelling pilot (38, 45, 69).

1. Cross-capture support. Every image is the same finger. For each FingerNet
   point on a pilot image, count how many *other* registered captures have a
   FingerNet minutia at the same place (15 px / 30 deg after registration).
   Real minutiae tend to be re-found; spurious ones don't. The question is
   whether the points Claude deleted have lower support than the ones kept.
   This uses only FingerNet's output and SIFT registration, never Claude's
   judgement, so it is an independent check (it cannot see a crease artefact
   that repeats in every capture, though).

2. Pair repeatability 38 <-> 45 (the only pilot pair that registers): the
   same measure as fingernet_consistency.py, for FingerNet's points vs
   Claude's corrected points.

    python pilot_checks.py      # writes ground_truth/pilot_claude/checks.json
"""
import json
import os

import cv2
import numpy as np

import fingernet_consistency as fc

GT = fc.GT
PILOT = os.path.join(GT, "pilot_claude")
PILOTS = ["38", "45", "69"]


def claude_points(name):
    with open(os.path.join(PILOT, "P" + name, "points.json")) as f:
        d = json.load(f)
    return d


def register(A, B, matcher):
    if A["des"] is None or B["des"] is None:
        return None
    good = [m for m, n in matcher.knnMatch(A["des"], B["des"], k=2) if m.distance < 0.75 * n.distance]
    if len(good) < fc.MIN_INLIERS:
        return None
    src = np.float32([A["kp"][m.queryIdx].pt for m in good])
    dst = np.float32([B["kp"][m.trainIdx].pt for m in good])
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    if M is None or inl.sum() < fc.MIN_INLIERS or not (fc.SCALE_OK[0] <= np.hypot(M[0, 0], M[1, 0]) <= fc.SCALE_OK[1]):
        return None
    return M


def near(p, pts, rot=0.0):
    if len(pts) == 0:
        return False
    d = np.hypot(pts[:, 0] - p[0], pts[:, 1] - p[1])
    o = np.abs((pts[:, 2] - p[2] + np.pi) % (2 * np.pi) - np.pi)
    return bool(((d <= fc.MAX_D) & (o <= fc.MAX_O)).any())


def main():
    import csv
    from fingernet_tf2 import FingerNet, run_image

    with open(os.path.join(GT, "manifest.csv"), newline="") as f:
        names = [r["image"] for r in csv.DictReader(f) if r["status"] != "rejected"]
    net = FingerNet()
    sift = cv2.SIFT_create(nfeatures=3000)
    data = {}
    for n in names:
        img, mnt, seg = fc.load(n, net, run_image)
        mask = cv2.erode(seg, np.ones((15, 15), np.uint8))
        kp, des = sift.detectAndCompute(img, mask * 255)
        data[n] = dict(mnt=mnt, seg=seg, kp=kp, des=des)
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))

    report = {}
    for n in PILOTS:
        d = claude_points(n)
        kept_ids = {p["id"] for p in d["points"] if p["src"] == "fingernet"}
        fn = data[n]["mnt"]                      # FingerNet ids are row indices
        support = np.zeros(len(fn), int)
        visible = np.zeros(len(fn), int)
        partners = 0
        for m in names:
            if m == n:
                continue
            M = register(data[n], data[m], matcher)
            if M is None:
                continue
            partners += 1
            mapped = fc.transform(fn, M)
            vis = fc.inside(mapped, data[m]["seg"])
            for i in range(len(fn)):
                if vis[i]:
                    visible[i] += 1
                    support[i] += near(mapped[i], data[m]["mnt"])
        rate = np.where(visible > 0, support / np.maximum(visible, 1), np.nan)
        kept = [i for i in range(len(fn)) if i in kept_ids and visible[i]]
        dele = [i for i in range(len(fn)) if i not in kept_ids and visible[i]]
        report[n] = dict(partners=partners, fingernet_points=len(fn), kept=len(kept_ids),
                         deleted=len(fn) - len(kept_ids),
                         support_rate_kept=float(np.nanmean(rate[kept])) if kept else None,
                         support_rate_deleted=float(np.nanmean(rate[dele])) if dele else None,
                         per_point=[dict(id=i, kept=i in kept_ids, visible_in=int(visible[i]),
                                         refound_in=int(support[i])) for i in range(len(fn))])
        print(n, {k: v for k, v in report[n].items() if k != "per_point"})

    # 38 <-> 45 pair repeatability, FingerNet vs Claude
    M = register(data["38"], data["45"], matcher)
    if M is not None:
        def rep(a, b):
            ab = fc.transform(a, M)
            ak = ab[fc.inside(ab, data["45"]["seg"])]
            bk = b[fc.inside(fc.transform(b, cv2.invertAffineTransform(M)), data["38"]["seg"])]
            m = fc.match_count(ak, bk)
            return dict(n_38=len(ak), n_45=len(bk), matched=m, repeatability=2 * m / max(1, len(ak) + len(bk)))
        c = lambda n: np.array([[p["x"], p["y"], p["direction"]] for p in claude_points(n)["points"]])
        report["pair_38_45"] = dict(fingernet=rep(data["38"]["mnt"], data["45"]["mnt"]), claude=rep(c("38"), c("45")))
        print("pair 38-45", report["pair_38_45"])
    with open(os.path.join(PILOT, "checks.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
