"""
Consensus minutiae across captures of the same finger (all ground_truth/ images are Rahul).

    python consensus_minutiae.py build     [--ref 36] [--min-support 0.5] [--min-visible 5]
    python consensus_minutiae.py propagate REVIEWED.sim [--ref 36]
    python consensus_minutiae.py assignment          # which reference each capture uses
    python consensus_minutiae.py propagate-all       # every ref_<r>/consensus_<r>_reviewed.sim -> all captures

build
  1. Registers every capture onto the reference (SIFT on the finger + RANSAC
     similarity, as in fingernet_consistency.py). Captures that don't
     register are left out.
  2. Maps each registered capture's FingerNet minutiae and finger mask into the
     reference frame, keeping points that land on the reference finger.
  3. Clusters the pooled points: average linkage, members on average within
     CLUSTER_D px, never joining points more than CLUSTER_O degrees apart.
     Complete linkage at 10 px fragmented real minutiae: registration offsets
     between captures are median 4 px, 90th percentile 10.5 px.
  4. support = distinct captures in the cluster / captures whose mask covers
     the cluster centre. Real minutiae are re-found; one-off detections are not.
  5. Keeps clusters with support >= --min-support seen in >= --min-visible
     captures. Split-half check: the same consensus from two random halves of
     the captures, and how well the halves agree.
  Writes ground_truth/consensus/ref_<ref>/: consensus_<ref>.json/.sim (for review in
  py-minutiae-viewer over images_500dpi/<ref>.png), clusters.csv,
  overlay_<ref>.png, support_hist.png, transforms.json, summary.json.

  Persistent physical features (creases, scars) are re-found in every capture
  too, so they pass consensus. That is what the human review is for. FingerNet
  gives no type, so every consensus point is exported as END; the reviewer sets
  bifurcations.

propagate
  Maps a reviewed consensus .sim (reference frame) back onto every registered
  capture with the inverse transform, keeping points inside that capture's
  finger mask, into ground_truth/consensus/ref_<ref>/propagated/<name>.sim. Those are
  labels for every capture from one review. Their accuracy depends on the
  registration (a similarity transform can't model the finger's curvature, so
  check the edges).
"""
import argparse
import csv
import json
import math
import os

import cv2
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

import fingernet_consistency as fc
from pilot_checks import register

GT = fc.GT
OUT = os.path.join(GT, "consensus")
CLUSTER_D = 10.0          # px: mean member distance (about one ridge period)
CLUSTER_O = math.pi / 6   # 30 deg, the tolerance FingerNet is scored with
RNG = np.random.default_rng(0)


def load_all():
    from fingernet_tf2 import FingerNet, run_image
    with open(os.path.join(GT, "manifest.csv"), newline="") as f:
        names = [r["image"] for r in csv.DictReader(f) if r["status"] != "rejected"]
    net = FingerNet()
    sift = cv2.SIFT_create(nfeatures=3000)
    data = {}
    for n in names:
        img, mnt, seg = fc.load(n, net, run_image)
        kp, des = sift.detectAndCompute(img, cv2.erode(seg, np.ones((15, 15), np.uint8)) * 255)
        data[n] = dict(img=img, mnt=mnt, seg=seg, kp=kp, des=des)
    return names, data


def register_all(names, data, ref):
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    T = {ref: np.array([[1.0, 0, 0], [0, 1.0, 0]])}
    for n in names:
        if n != ref:
            M = register(data[n], data[ref], matcher)       # maps n -> ref
            if M is not None:
                T[n] = M
    return T


def pool(captures, data, T, ref):
    h, w = data[ref]["seg"].shape
    ref_mask = data[ref]["seg"]
    pts, masks = [], {}
    for n in captures:
        masks[n] = cv2.warpAffine(data[n]["seg"], T[n], (w, h), flags=cv2.INTER_NEAREST) & ref_mask
        m = fc.transform(data[n]["mnt"], T[n])
        keep = fc.inside(m, ref_mask)
        for x, y, o in m[keep]:
            pts.append((x, y, o, n))
    return pts, masks


def cluster(pts):
    xy = np.array([[p[0], p[1]] for p in pts])
    th = np.array([p[2] for p in pts])
    d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    do = np.abs((th[:, None] - th[None, :] + np.pi) % (2 * np.pi) - np.pi)
    d = np.where(do <= CLUSTER_O, d, 1e3)                  # incompatible angles never join
    np.fill_diagonal(d, 0)
    lab = fcluster(linkage(squareform(d, checks=False), "average"), CLUSTER_D, "distance")
    return lab


def summarise(pts, lab, masks):
    out = []
    for c in np.unique(lab):
        idx = np.nonzero(lab == c)[0]
        caps = {pts[i][3] for i in idx}
        x = float(np.median([pts[i][0] for i in idx]))
        y = float(np.median([pts[i][1] for i in idx]))
        th = [pts[i][2] for i in idx]
        o = float(math.atan2(np.mean(np.sin(th)), np.mean(np.cos(th))) % (2 * math.pi))
        vis = sum(1 for m in masks.values() if m[int(round(y)), int(round(x))] > 0)
        out.append(dict(x=x, y=y, direction=o, captures=len(caps), visible=max(vis, len(caps)),
                        support=len(caps) / max(vis, len(caps)), members=len(idx)))
    return out


def select(clusters, min_support, min_visible):
    return [c for c in clusters if c["support"] >= min_support and c["visible"] >= min_visible]


def as_arr(cl):
    return np.array([[c["x"], c["y"], c["direction"]] for c in cl]).reshape(-1, 3)


def write_sim(path, cl):
    with open(path, "w") as f:
        f.write("\n".join("%d %d %.2f END %.3f" % (round(c["x"]), round(c["y"]),
                                                   (math.degrees(c["direction"]) + 90) % 360, c.get("support", 1.0))
                          for c in cl))


def ref_dir(ref):
    return os.path.join(OUT, "ref_" + ref)


def build(a):
    out = ref_dir(a.ref)
    os.makedirs(out, exist_ok=True)
    names, data = load_all()
    T = register_all(names, data, a.ref)
    caps = sorted(T, key=lambda s: (0, int(s)) if s.isdigit() else (1, s))
    print("reference %s: %d captures registered (incl. reference) of %d" % (a.ref, len(caps), len(names)))

    pts, masks = pool(caps, data, T, a.ref)
    clusters = summarise(pts, cluster(pts), masks)
    chosen = select(clusters, a.min_support, a.min_visible)
    chosen.sort(key=lambda c: (c["y"], c["x"]))
    flags = []   # close pairs: one minutia split by a direction flip, or two neighbours. Reviewer decides.
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            dd = math.hypot(chosen[i]["x"] - chosen[j]["x"], chosen[i]["y"] - chosen[j]["y"])
            if dd < 12:
                do = abs((chosen[i]["direction"] - chosen[j]["direction"] + math.pi) % (2 * math.pi) - math.pi)
                flags.append(dict(a=[round(chosen[i]["x"]), round(chosen[i]["y"])], b=[round(chosen[j]["x"]), round(chosen[j]["y"])],
                                  distance_px=round(dd, 1), angle_diff_deg=round(math.degrees(do))))
    print("pooled %d minutiae -> %d clusters -> %d consensus (support >= %.2f, visible >= %d)" % (
        len(pts), len(clusters), len(chosen), a.min_support, a.min_visible))

    # split-half reliability: two consensus sets from disjoint capture halves
    others = [c for c in caps if c != a.ref]
    agree = []
    for rep in range(5):
        perm = list(RNG.permutation(others))
        hs = [perm[::2], perm[1::2]]
        sets = []
        for h in hs:
            p2, m2 = pool(h, data, T, a.ref)
            sets.append(as_arr(select(summarise(p2, cluster(p2), m2), a.min_support, max(3, a.min_visible // 2))))
        m = fc.match_count(sets[0], sets[1])
        agree.append(2 * m / max(1, len(sets[0]) + len(sets[1])))
    print("split-half agreement (5 random splits): mean %.3f, min %.3f" % (np.mean(agree), np.min(agree)))

    # how the reference's own FingerNet output compares with the consensus
    ref_fn = data[a.ref]["mnt"]
    m = fc.match_count(ref_fn, as_arr(chosen))

    ref_img = data[a.ref]["img"]
    vis = cv2.cvtColor(cv2.resize(ref_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
    for c in clusters:
        if c["visible"] < a.min_visible:
            continue
        s = c["support"]
        col = (0, 200, 0) if s >= 0.75 else (0, 200, 255) if s >= a.min_support else (0, 0, 255)
        p = (int(c["x"] * 2), int(c["y"] * 2))
        if s >= a.min_support:
            cv2.rectangle(vis, (p[0] - 6, p[1] - 6), (p[0] + 6, p[1] + 6), col, 2)
            cv2.line(vis, p, (int(p[0] + 18 * math.cos(c["direction"])), int(p[1] + 18 * math.sin(c["direction"]))), col, 2)
        else:
            cv2.circle(vis, p, 3, col, -1)
    cv2.imwrite(os.path.join(out, "overlay_%s.png" % a.ref), vis)

    hist = np.histogram([c["support"] for c in clusters if c["visible"] >= a.min_visible], bins=10, range=(0, 1))[0]
    H = np.full((220, 420, 3), 255, np.uint8)
    for i, v in enumerate(hist):
        bh = int(180 * v / max(hist.max(), 1))
        col = (0, 0, 220) if (i + 1) / 10 <= a.min_support else (0, 170, 0)
        cv2.rectangle(H, (20 + i * 38, 195 - bh), (20 + i * 38 + 30, 195), col, -1)
        cv2.putText(H, str(v), (22 + i * 38, 190 - bh), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(H, "%.1f" % ((i + 1) / 10), (22 + i * 38, 212), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    cv2.imwrite(os.path.join(out, "support_hist.png"), H)

    with open(os.path.join(out, "clusters.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(clusters[0].keys()))
        w.writeheader()
        w.writerows(clusters)
    with open(os.path.join(out, "consensus_%s.json" % a.ref), "w") as f:
        json.dump({"reference": a.ref, "frame": "images_500dpi/%s.png" % a.ref,
                   "status": "consensus, NOT yet human-reviewed",
                   "captures": caps, "min_support": a.min_support, "min_visible": a.min_visible,
                   "review_flags": flags, "minutiae": chosen}, f, indent=1)
    write_sim(os.path.join(out, "consensus_%s.sim" % a.ref), chosen)
    with open(os.path.join(out, "transforms.json"), "w") as f:
        json.dump({"reference": a.ref, "to_reference": {n: M.tolist() for n, M in T.items()}}, f, indent=1)
    summary = dict(reference=a.ref, captures=len(caps), pooled=len(pts), clusters=len(clusters),
                   consensus=len(chosen), split_half_agreement=float(np.mean(agree)),
                   split_half_min=float(np.min(agree)),
                   ref_fingernet_points=len(ref_fn), ref_fingernet_in_consensus=m,
                   support_hist_0_to_1=hist.tolist())
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


def propagate(a):
    od = a.out or os.path.join(ref_dir(a.ref), "propagated")
    from fingernet_tf2 import FingerNet, run_image
    with open(os.path.join(ref_dir(a.ref), "transforms.json")) as f:
        T = {n: np.array(M) for n, M in json.load(f)["to_reference"].items()}
    rows = []
    with open(a.sim) as f:
        for line in f:
            p = line.split()
            if len(p) >= 4:
                rows.append((float(p[0]), float(p[1]), math.radians((float(p[2]) - 90) % 360), p[3]))
    ref_pts = np.array([[r[0], r[1], r[2]] for r in rows])
    kinds = [r[3] for r in rows]
    net = FingerNet()
    os.makedirs(od, exist_ok=True)
    for n, M in T.items():
        img = cv2.imread(os.path.join(GT, "images_500dpi", n + ".png"), cv2.IMREAD_GRAYSCALE)
        seg = run_image(net, img)["seg"][:img.shape[0], :img.shape[1]].astype(np.uint8)
        mapped = fc.transform(ref_pts, cv2.invertAffineTransform(M))
        keep = fc.inside(mapped, seg)
        with open(os.path.join(od, n + ".sim"), "w") as f:
            f.write("\n".join("%d %d %.2f %s 1.000" % (round(x), round(y), (math.degrees(o) + 90) % 360, k)
                              for (x, y, o), k, ok in zip(mapped, kinds, keep) if ok))
        print("%-5s %d minutiae" % (n, int(keep.sum())))


REF_PRIORITY = ["36", "16", "98", "2", "72"]   # bigger consensus groups first


def assignment():
    """One reference per capture: SIFT registration first (in REF_PRIORITY order), then an
    accepted minutiae-based alignment (align_by_minutiae.py)."""
    assign = {}
    for r in REF_PRIORITY:
        f = os.path.join(ref_dir(r), "transforms.json")
        if not os.path.exists(f):
            continue
        with open(f) as fh:
            for n, M in json.load(fh)["to_reference"].items():
                assign.setdefault(n, dict(reference=r, method="sift" if n != r else "reference", to_reference=M))
    f = os.path.join(OUT, "minutiae_alignment.json")
    if os.path.exists(f):
        with open(f) as fh:
            for n, v in json.load(fh)["captures"].items():
                if v["accepted"] and n not in assign:
                    assign[n] = dict(reference=v["reference"], method="minutiae", matches=v["matches"],
                                     to_reference=v["to_reference"])
    return assign


def propagate_all(a):
    """Reviewed consensus per reference (ref_<r>/consensus_<r>_reviewed.sim) -> one .sim per capture."""
    from fingernet_tf2 import FingerNet, run_image
    assign = assignment()
    od = a.out or os.path.join(OUT, "propagated_all")
    os.makedirs(od, exist_ok=True)
    reviewed = {}
    for r in REF_PRIORITY:
        f = os.path.join(ref_dir(r), a.reviewed_name.format(ref=r))
        if os.path.exists(f):
            rows = [l.split() for l in open(f) if len(l.split()) >= 4]
            reviewed[r] = (np.array([[float(x[0]), float(x[1]), math.radians((float(x[2]) - 90) % 360)] for x in rows]),
                           [x[3] for x in rows])
    print("reviewed references found:", sorted(reviewed), "| captures assigned:", len(assign))
    net = FingerNet()
    log = []
    for n, v in sorted(assign.items(), key=lambda t: int(t[0]) if t[0].isdigit() else 0):
        if v["reference"] not in reviewed:
            log.append(dict(image=n, reference=v["reference"], method=v["method"], written=False, count=0))
            continue
        pts, kinds = reviewed[v["reference"]]
        img = cv2.imread(os.path.join(GT, "images_500dpi", n + ".png"), cv2.IMREAD_GRAYSCALE)
        seg = run_image(net, img)["seg"][:img.shape[0], :img.shape[1]].astype(np.uint8)
        mapped = fc.transform(pts, cv2.invertAffineTransform(np.array(v["to_reference"])))
        keep = fc.inside(mapped, seg)
        with open(os.path.join(od, n + ".sim"), "w") as f:
            f.write("\n".join("%d %d %.2f %s 1.000" % (round(x), round(y), (math.degrees(o) + 90) % 360, k)
                              for (x, y, o), k, ok in zip(mapped, kinds, keep) if ok))
        log.append(dict(image=n, reference=v["reference"], method=v["method"], written=True, count=int(keep.sum())))
    with open(os.path.join(od, "propagation_log.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0].keys()))
        w.writeheader()
        w.writerows(log)
    print("wrote %d capture label files to %s" % (sum(l["written"] for l in log), od))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "propagate", "propagate-all", "assignment"])
    ap.add_argument("sim", nargs="?")
    ap.add_argument("--ref", default="36")
    ap.add_argument("--min-support", type=float, default=0.5)
    ap.add_argument("--min-visible", type=int, default=5)
    ap.add_argument("--out", help="propagate(-all): output folder")
    ap.add_argument("--reviewed-name", default="consensus_{ref}_reviewed.sim",
                    help="propagate-all: reviewed file name inside each ref_<r>/ folder")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a)
    elif a.cmd == "propagate":
        propagate(a)
    elif a.cmd == "propagate-all":
        propagate_all(a)
    else:
        v = assignment()
        by = {}
        for n, x in v.items():
            by.setdefault((x["reference"], x["method"]), []).append(n)
        for k in sorted(by):
            print(k, len(by[k]))
        print("assigned %d captures" % len(v))


if __name__ == "__main__":
    main()
