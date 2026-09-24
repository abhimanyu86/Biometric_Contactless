# FingerNet, ported to TensorFlow 2

Inference-only port of [FingerNet](https://github.com/592692070/FingerNet) (Tang et al., IJCB 2017).
Upstream is Python 2.7 / TF 1.0.1 / Keras 2.0.2. `fingernet_tf2.py` rebuilds the deploy graph as
plain TF ops and reads the released weights directly with h5py.

**Output is a prediction, not ground truth.** Use it to pre-fill minutiae that a human then corrects.
FingerNet gives no ridge-ending/bifurcation type.

## Setup

Copy `models/released_version/Model.model` from the upstream repo into this folder. It is gitignored.

## Usage

    python fingernet_tf2.py IMG_OR_DIR --scale 1.5             # fixed resize factor
    python fingernet_tf2.py IMG_OR_DIR --target-period 9       # resize from the FFT ridge-period estimate
    python fingernet_tf2.py --validate <upstream>/datasets     # P/R/F1 against upstream's labelled samples

FingerNet expects 500 DPI contact prints, a ridge period of roughly 9 px. Tight contactless crops
of about 200-270 px have a period near 6 px, so use `--scale 1.5`. On tight crops the FFT
estimate can lock onto the finger outline (about 33 px); estimates outside 4-16 px are ignored.

## Port validation

Run against upstream's two hand-labelled samples (tolerance: 15 px, 30 degrees):

| sample | GT | pred | P | R | F1 |
|---|---|---|---|---|---|
| CISL24218 A0100003009991600022036_2 | 23 | 27 | 0.70 | 0.83 | 0.76 |
| NISTSD27 B101L9U (latent) | 15 | 12 | 0.83 | 0.67 | 0.74 |

## Hand correction (ground truth)

`../py-minutiae-viewer/` is a copy of [py-minutiae-viewer](https://github.com/IgniparousTempest/py-minutiae-viewer)
(MIT licence, last updated 2019), patched to run on current Python packages:
- replaced the `overrides` decorator with a no-op, because overrides>=4 rejects the old type hints at import
- dropped `ttkthemes`, because a source-built install lacks its themes folder
- replaced `Image.ANTIALIAS` with `Image.LANCZOS`, because Pillow 10 removed it

Round trip, tested end to end:

    python sim_convert.py to-sim "fingernet/output/scale1.5/*.json"
    python py-minutiae-viewer/py-minutiae-viewer.py
    python sim_convert.py to-json "labels/*.sim"

In the viewer, open the image, load the `.sim` file, correct the points, then export as `.sim`.
- drag: ridge ending (saved as END, green square)
- Ctrl+click: bifurcation (saved as BIF, red circle)
- right-click: delete the nearest point

The tool's on-screen legend has ridge ending and bifurcation swapped. The file output above is what's actually saved.
