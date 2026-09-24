"""
FingerNet (Tang et al., IJCB 2017) inference, ported to TensorFlow 2.

Source: https://github.com/592692070/FingerNet -- Python 2.7 / TF 1.0.1 /
Keras 2.0.2, which no longer installs. This port rebuilds the deploy graph
of src/train_test_deploy.py:get_main_net as plain tf ops and reads the
released weights (Model.model, a Keras-2 HDF5 file) straight out of h5py by
layer name, so it does not depend on Keras 3 being able to load a Keras
2.0.2 file. Only inference is ported; training/test mode is not.

Output per image, same as the original deploy():
    <name>.mnt        FingerNet's own format: name / "N H W" / "x y theta"
    <name>.json       [{x, y, direction, score}] -- direction in radians,
                      image coordinates (y down), same convention as
                      backend/minutiae.py. FingerNet gives no RIG/BIF type.
    <name>_mnt.png    minutiae overlay
    <name>_seg.png    segmentation mask
    <name>_enh.png    Gabor-phase enhanced image

These are *predictions*, not ground truth -- see README.md.

Usage:
    python fingernet_tf2.py IMG_OR_DIR [...] [--out DIR] [--target-period 9 | --scale 1.5]
    python fingernet_tf2.py --validate     # check the port against the repo's labelled samples
"""
import argparse
import glob
import json
import os
import sys

import cv2
import h5py
import numpy as np
import tensorflow as tf
from scipy import signal, spatial

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(HERE, "Model.model")
BN_EPS = 1e-3  # Keras 2.0.2 BatchNormalization default
EPS = 1e-7     # K.epsilon()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "backend"))
try:
    from fir import estimate_ridge_period
except Exception:  # backend not importable -- rescaling just becomes unavailable
    estimate_ridge_period = None


# --------------------------------------------------------------------------- weights

def load_weights(path):
    w = {}
    with h5py.File(path, "r") as f:
        g = f["model_weights"] if "model_weights" in f else f
        for layer in g.attrs["layer_names"]:
            layer = layer.decode() if isinstance(layer, bytes) else layer
            for name in g[layer].attrs["weight_names"]:
                name = name.decode() if isinstance(name, bytes) else name
                key = name.split("/")[-1].split(":")[0]
                w.setdefault(layer, {})[key] = np.asarray(g[layer][name], dtype=np.float32)
    return w


# --------------------------------------------------------------------------- graph ops

def _atan2(y, x):
    # FingerNet's own atan2 (not tf.atan2): x==0 maps to 0, eps added to x.
    x = x + EPS
    a = tf.atan(y / x)
    angle = tf.where(x > 0, a, tf.zeros_like(x))
    angle = tf.where((x < 0) & (y >= 0), a + np.pi, angle)
    angle = tf.where((x < 0) & (y < 0), a - np.pi, angle)
    return angle


def _img_normalization(x, m0=0.0, var0=1.0):
    m = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
    var = tf.math.reduce_variance(x, axis=[1, 2, 3], keepdims=True)
    after = tf.sqrt(var0 * tf.square(x - m) / var)
    return tf.where(x > m, m0 + after, m0 - after)


def _gausslabel(length=180, stride=2):
    pdf = signal.windows.gaussian(length + 1, 3)
    label = np.reshape(np.arange(stride // 2, length, stride), [1, 1, -1, 1])
    y = np.reshape(np.arange(stride // 2, length, stride), [1, 1, 1, -1])
    delta = np.abs(label - y).astype(int)
    delta = np.minimum(delta, length - delta) + length // 2
    return pdf[delta].astype(np.float32)


GAUSS_LABEL = _gausslabel()


def ori_highest_peak(ori):
    return tf.nn.conv2d(ori, GAUSS_LABEL, strides=1, padding="SAME")


class FingerNet:
    def __init__(self, weights_path=DEFAULT_WEIGHTS):
        self.w = load_weights(weights_path)

    def _conv(self, x, name, dilation=1):
        p = self.w[name]
        return tf.nn.conv2d(x, p["kernel"], strides=1, padding="SAME", dilations=dilation) + p["bias"]

    def _cbp(self, x, name, dilation=1):
        conv_name = ("conv" if dilation == 1 else "atrousconv") + name
        x = self._conv(x, conv_name, dilation)
        bn = self.w["bn-" + name]
        x = (x - bn["moving_mean"]) / tf.sqrt(bn["moving_variance"] + BN_EPS) * bn["gamma"] + bn["beta"]
        alpha = self.w["prelu-" + name]["alpha"]
        return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

    @staticmethod
    def _pool(x):
        return tf.nn.max_pool2d(x, 2, 2, "VALID")

    @staticmethod
    def _up8(x):
        return tf.repeat(tf.repeat(x, 8, axis=1), 8, axis=2)

    @tf.function(reduce_retracing=True)
    def __call__(self, img):
        x = _img_normalization(img)
        x = self._pool(self._cbp(self._cbp(x, "1_1"), "1_2"))
        x = self._pool(self._cbp(self._cbp(x, "2_1"), "2_2"))
        x = self._pool(self._cbp(self._cbp(self._cbp(x, "3_1"), "3_2"), "3_3"))

        oris, segs = [], []
        for i, d in ((1, 1), (2, 4), (3, 8)):
            s = self._cbp(x, "4_%d" % i, dilation=d)
            oris.append(self._conv(self._cbp(s, "ori_%d_1" % i), "ori_%d_2" % i))
            segs.append(self._conv(self._cbp(s, "seg_%d_1" % i), "seg_%d_2" % i))
        ori_out = tf.sigmoid(tf.add_n(oris))
        seg_out = tf.sigmoid(tf.add_n(segs))

        # Gabor enhancement steered by the single strongest orientation per block.
        real = self._conv(img, "enh_img_real_1")
        imag = self._conv(img, "enh_img_imag_1")
        peak = ori_highest_peak(ori_out)
        peak = peak / (tf.reduce_max(peak, -1, keepdims=True) + EPS)
        peak = tf.where(peak > 0.999, peak, tf.zeros_like(peak))
        peak = peak / (tf.reduce_sum(peak, -1, keepdims=True) + EPS)
        up_ori = self._up8(peak)
        enh_real = tf.reduce_sum(real * up_ori, -1, keepdims=True)
        enh_imag = tf.reduce_sum(imag * up_ori, -1, keepdims=True)
        enh = _atan2(enh_imag, enh_real)
        up_seg = self._up8(tf.nn.softsign(seg_out))

        m = tf.concat([enh, up_seg], 3)
        m = self._pool(self._cbp(m, "mnt_1_1"))
        m = self._pool(self._cbp(m, "mnt_2_1"))
        m = self._pool(self._cbp(m, "mnt_3_1"))

        mnt_o = tf.sigmoid(self._conv(self._cbp(tf.concat([m, ori_out], 3), "mnt_o_1_1"), "mnt_o_1_2"))
        mnt_w = tf.sigmoid(self._conv(self._cbp(m, "mnt_w_1_1"), "mnt_w_1_2"))
        mnt_h = tf.sigmoid(self._conv(self._cbp(m, "mnt_h_1_1"), "mnt_h_1_2"))
        mnt_s = tf.sigmoid(self._conv(self._cbp(m, "mnt_s_1_1"), "mnt_s_1_2"))
        return enh_real, ori_out, seg_out, mnt_o, mnt_w, mnt_h, mnt_s


# --------------------------------------------------------------------------- post-processing

def label2mnt(mnt_s, mnt_w, mnt_h, mnt_o, thresh=0.5):
    mnt_s, mnt_w, mnt_h, mnt_o = (np.squeeze(a) for a in (mnt_s, mnt_w, mnt_h, mnt_o))
    rows, cols = np.nonzero(mnt_s > thresh)
    if len(rows) == 0:
        return np.zeros((0, 4))
    w = np.argmax(mnt_w, -1)[rows, cols]
    h = np.argmax(mnt_h, -1)[rows, cols]
    o = np.argmax(mnt_o, -1)[rows, cols]
    out = np.zeros((len(rows), 4))
    out[:, 0] = cols * 8 + w
    out[:, 1] = rows * 8 + h
    out[:, 2] = np.mod((o * 2 - 89.0) / 180 * np.pi, 2 * np.pi)
    out[:, 3] = mnt_s[rows, cols]
    return out


def _angle_delta(a, b, max_d=2 * np.pi):
    d = np.abs(a - b)
    return np.minimum(d, max_d - d)


def nms(mnt, max_d=16, max_o=np.pi / 6):
    if mnt.shape[0] == 0:
        return mnt
    mnt = mnt[np.argsort(-mnt[:, 3], kind="stable")]
    dist = spatial.distance.cdist(mnt[:, :2], mnt[:, :2])
    ang = _angle_delta(mnt[:, 2:3], mnt[:, 2:3].T)
    close = (dist <= max_d) & (ang <= max_o)
    keep = np.ones(len(mnt), bool)
    for i in range(len(mnt)):
        if keep[i]:
            keep[i + 1:] &= ~close[i, i + 1:]
    return mnt[keep]


def mnt_p_r_f(gt, pred, maxd=15, maxo=np.pi / 6):
    """Precision/recall exactly as the original utils.mnt_P_R_F computes it."""
    if len(pred) == 0 or len(gt) == 0:
        return 0.0, 0.0, 0.0
    dis = spatial.distance.cdist(pred[:, :2], gt[:, :2])
    mind, idx = dis.min(1), dis.argmin(1)
    ang = np.abs(np.mod(pred[:, 2], 2 * np.pi) - gt[idx, 2])
    ang = np.minimum(ang, 2 * np.pi - ang)
    hit = np.unique(idx[(mind <= maxd) & (ang <= maxo)])
    p, r = len(hit) / len(pred), len(hit) / len(gt)
    return p, r, 2 * p * r / (p + r + 1e-10)


def read_mnt(path):
    with open(path) as f:
        lines = f.read().split("\n")[2:]
    return np.array([[float(v) for v in ln.split()] for ln in lines if ln.strip()])


# --------------------------------------------------------------------------- driver

PLAUSIBLE_PERIOD = (4.0, 16.0)  # px; outside this the FFT peak is the finger outline, not ridges


def prepare(gray, target_period=None, fixed_scale=None):
    """Rescale to FingerNet's ~500 DPI ridge spacing (fixed factor or estimated), then pad to a multiple of 8."""
    scale = 1.0
    period = None
    if fixed_scale:
        scale = float(fixed_scale)
    elif target_period and estimate_ridge_period is not None:
        period = estimate_ridge_period(gray)
        if period and PLAUSIBLE_PERIOD[0] <= period <= PLAUSIBLE_PERIOD[1]:
            scale = float(target_period) / period
    if abs(scale - 1.0) > 0.05:
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    else:
        scale = 1.0
    h, w = gray.shape
    H, W = int(np.ceil(h / 8) * 8), int(np.ceil(w / 8) * 8)
    padded = np.full((H, W), gray.mean(), np.float32)
    padded[:h, :w] = gray
    return padded, scale, period


def run_image(net, gray, target_period=None, thresh=0.5, fixed_scale=None):
    img, scale, period = prepare(gray.astype(np.float32), target_period, fixed_scale)
    x = tf.constant(img[None, :, :, None] / 255.0, tf.float32)
    enh, ori, seg, mnt_o, mnt_w, mnt_h, mnt_s = [t.numpy() for t in net(x)]
    seg = np.round(np.squeeze(seg))
    seg = cv2.morphologyEx(seg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    mnt = nms(label2mnt(np.squeeze(mnt_s) * seg, mnt_w, mnt_h, mnt_o, thresh))
    seg_full = np.kron(seg, np.ones((8, 8)))
    return dict(img=img, mnt=mnt, seg=seg_full, enh=np.squeeze(enh) * seg_full,
                scale=scale, period=period)


def draw(img, mnt, path, r=12):
    vis = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    k = max(1, int(round(max(img.shape) / 400)))
    for x, y, o, s in mnt:
        p = (int(round(x)), int(round(y)))
        cv2.rectangle(vis, (p[0] - 4 * k, p[1] - 4 * k), (p[0] + 4 * k, p[1] + 4 * k), (0, 0, 255), k)
        cv2.line(vis, p, (int(round(x + r * k * np.cos(o))), int(round(y + r * k * np.sin(o)))), (0, 0, 255), k)
    cv2.imwrite(path, vis)


def save_outputs(res, name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    mnt, img = res["mnt"], res["img"]
    with open(os.path.join(out_dir, name + ".mnt"), "w") as f:
        f.write("%s\n%d %d %d\n" % (name, len(mnt), img.shape[0], img.shape[1]))
        for x, y, o, _ in mnt:
            f.write("%d %d %.6f\n" % (x, y, o))
    # JSON is in the ORIGINAL image's pixel frame (undo any rescale).
    s = res["scale"]
    with open(os.path.join(out_dir, name + ".json"), "w") as f:
        json.dump({"source": "FingerNet (predicted, not ground truth)",
                   "rescale_factor": s, "ridge_period_px": res["period"],
                   "minutiae": [{"x": float(x / s), "y": float(y / s), "direction": float(o),
                                 "score": float(sc)} for x, y, o, sc in mnt]}, f, indent=1)
    draw(img, mnt, os.path.join(out_dir, name + "_mnt.png"))
    cv2.imwrite(os.path.join(out_dir, name + "_seg.png"), (res["seg"] * 255).astype(np.uint8))
    enh = res["enh"]
    cv2.imwrite(os.path.join(out_dir, name + "_enh.png"),
                ((enh - enh.min()) / (np.ptp(enh) + 1e-9) * 255).astype(np.uint8))


def validate(net, repo_datasets):
    """Score the port against the hand-labelled samples bundled with the upstream repo."""
    for ds in ("CISL24218", "NISTSD27"):
        for img_path in glob.glob(os.path.join(repo_datasets, ds, "images", "*.bmp")):
            name = os.path.splitext(os.path.basename(img_path))[0]
            gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            res = run_image(net, gray)
            # the original multiplies mnt_s by the *raw* rounded seg in test(); deploy()
            # (used here) additionally opens it -- close enough for a sanity check.
            gt = read_mnt(os.path.join(repo_datasets, ds, "mnt_labels", name + ".mnt"))
            p, r, f = mnt_p_r_f(gt, res["mnt"])
            print("%-10s %-28s gt=%3d pred=%3d  P=%.2f R=%.2f F1=%.2f" % (ds, name, len(gt), len(res["mnt"]), p, r, f))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*")
    ap.add_argument("--out", default=os.path.join(HERE, "output"))
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--target-period", type=float, default=None,
                    help="rescale so the ridge period is this many px (~9 matches 500 DPI)")
    ap.add_argument("--scale", type=float, default=None,
                    help="fixed resize factor; overrides --target-period")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--validate", metavar="FINGERNET_REPO_DATASETS", nargs="?", const="",
                    help="path to the upstream repo's datasets/ folder")
    a = ap.parse_args()

    net = FingerNet(a.weights)
    if a.validate is not None:
        validate(net, a.validate or os.path.join(HERE, "upstream_datasets"))
        return

    paths = []
    for p in a.inputs:
        if os.path.isdir(p):
            for ext in ("bmp", "png", "jpg", "jpeg", "tif", "pgm"):
                paths += glob.glob(os.path.join(p, "*." + ext))
        else:
            paths.append(p)
    for p in sorted(set(paths)):
        gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print("skip (unreadable):", p)
            continue
        name = os.path.splitext(os.path.basename(p))[0]
        res = run_image(net, gray, a.target_period, a.thresh, a.scale)
        save_outputs(res, name, a.out)
        print("%-24s %4dx%-4d period=%s scale=%.2f minutiae=%d" % (
            name, gray.shape[1], gray.shape[0],
            "%.1f" % res["period"] if res["period"] else "n/a", res["scale"], len(res["mnt"])))


if __name__ == "__main__":
    main()
