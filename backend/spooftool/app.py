"""
spooftool/app.py — upload-and-test bench for spoof_detection.detect_screen_replay.

detect_screen_replay itself still needs no model weights, no TensorFlow, no
backend server -- same as tests/test_spoof_detection.py. But testing it on a
raw, uncropped phone photo is NOT representative of the real pipeline and
gives misleadingly bad results: measured on backend/../spoof_testing, running
the FFT check on full uncropped photos (nothing isolating just the fingertip)
flagged 0 of 5 real spoof attempts, all called "real". Running the SAME
photos through the actual YOLO finger crop the production pipeline uses
(app.py's detect_and_crop) caught 4 of 5. The background/off-center framing
in a full photo dilutes the signal enough to hide it entirely -- this isn't a
tuning problem, it's a "what are we even analysing" problem. So this tool
DOES need best-new.pt (the same finger-detector weights app.py loads) to give
meaningful results; without it, it falls back to a manual center-crop and
says so loudly, because that fallback is known to not be representative.

Run:
    cd backend
    streamlit run spooftool/app.py
"""

import csv
import sys
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spoof_detection import analyze_spectrum  # noqa: E402

YOLO_MODEL_PATH = Path(__file__).resolve().parent.parent / "best-new.pt"


@st.cache_resource
def load_finger_detector():
    """Loads once per server process (YOLO init is slow) -- st.cache_resource,
    not st.cache_data, since this is a live model object, not serialisable data."""
    if not YOLO_MODEL_PATH.exists():
        return None, f"best-new.pt not found at {YOLO_MODEL_PATH}"
    try:
        from ultralytics import YOLO
        return YOLO(str(YOLO_MODEL_PATH), task="detect"), None
    except Exception as e:  # ultralytics/torch missing, weights corrupt, etc.
        return None, str(e)


def yolo_crop(model, img, conf=0.10):
    """Same detect-and-crop logic as app.py's detect_and_crop_image -- take the
    highest-confidence finger box, clamp to image bounds. Returns (None, None)
    when nothing is detected."""
    results = model(img, save=False, imgsz=800, conf=conf, verbose=False)
    if not results or len(results[0].boxes) == 0:
        return None, None
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    best = int(np.argmax(confs))
    x1, y1, x2, y2 = map(int, boxes[best])
    h, w = img.shape[:2]
    x1 = max(0, min(x1, w - 1)); x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1)); y2 = max(y1 + 1, min(y2, h))
    return img[y1:y2, x1:x2], float(confs[best])

# Gitignored (see .gitignore's "Real fingerprint photos" section) -- these are
# real biometric photos of whoever tests with this tool and must never be
# committed. This is the growing dataset the calibration work in
# spoof_detection.py's docstring keeps saying we need more of.
DATASET_DIR = Path(__file__).resolve().parent.parent / "spoof_dataset"
MANIFEST_PATH = DATASET_DIR / "manifest.csv"
MANIFEST_FIELDS = ["timestamp", "label", "filename", "peak_count",
                    "strongest_peak_ratio", "chroma_diff", "score"]

st.set_page_config(page_title="Spoof Detector Test Bench", layout="wide")
st.title("Screen / Print Spoof Detector — Test Bench")
st.caption(
    "Tests spoof_detection.detect_screen_replay on the same finger crop the real "
    "pipeline sees (via YOLO, best-new.pt). Upload real finger photos alongside "
    "photos of a phone/monitor showing a finger to see how the FFT check tells "
    "them apart."
)

finger_model, finger_model_error = load_finger_detector()
if finger_model is None:
    st.warning(
        f"Could not load the finger detector ({finger_model_error}). Falling back to "
        "a manual center-crop for every image below — **results in this mode are not "
        "representative of the real pipeline** (a raw, uncropped photo dilutes the "
        "signal enough to hide spoofing entirely; that's not a tuning problem, it's a "
        "framing problem). Fix the path or install ultralytics to get real results."
    )


def dataset_counts():
    real_dir, spoof_dir = DATASET_DIR / "real", DATASET_DIR / "spoof"
    real = len(list(real_dir.glob("*"))) if real_dir.exists() else 0
    spoof = len(list(spoof_dir.glob("*"))) if spoof_dir.exists() else 0
    return real, spoof


def save_example(label, filename, raw_bytes, result):
    target_dir = DATASET_DIR / label
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_name = f"{stamp}_{Path(filename).name}"
    (target_dir / safe_name).write_bytes(raw_bytes)

    is_new = not MANIFEST_PATH.exists()
    with open(MANIFEST_PATH, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "timestamp": stamp, "label": label, "filename": safe_name,
            "peak_count": result["peak_count"],
            "strongest_peak_ratio": result["strongest_peak_ratio"],
            "chroma_diff": result["chroma_diff"], "score": result["score"],
        })


real_n, spoof_n = dataset_counts()
st.caption(f"Saved dataset so far: **{real_n} real** / **{spoof_n} spoof** "
           f"examples (backend/spoof_dataset/, gitignored, local only).")

with st.expander("How to read the result"):
    st.markdown(
        "- **peak_count** — sharp frequency peaks found. In practice, real JPEG "
        "photos rarely show more than 1 even when the finger is genuinely sharp — "
        "this signal only reliably catches a very high-quality recapture.\n"
        "- **strongest_peak_ratio** — how far the strongest peak sits above its local "
        "background; higher means a sharper, more artificial-looking peak.\n"
        "- **chroma_diff** — how differently the R and B color channels' spectra are "
        "shaped. The stronger signal on real data so far: real fingers keep similar "
        "shape across channels (same ridge shading, different channel brightness), "
        "screens diverge more per channel. Flagged above spoof_detection.CHROMA_DIFF_THRESH.\n"
        "- **score** — peak_count plus normalised peak-strength and chroma terms, for "
        "eyeballing confidence (not a probability).\n\n"
        "The threshold (2+ peaks → spoof) is **not calibrated on real photos yet** — "
        "that's what this tool is for. If a real finger gets flagged, or a screen "
        "photo slips through, note it so the constants in spoof_detection.py can be "
        "tuned against real data instead of just the synthetic test patterns."
    )

crop_pct = st.slider(
    "Fallback center-crop % (only used when YOLO can't find a finger in an image, or "
    "the detector failed to load — use 100 to fall back to the full upload)",
    min_value=20, max_value=100, value=60, step=5,
)

files = st.file_uploader(
    "Upload finger photos — real captures and screen/print spoof attempts",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
    accept_multiple_files=True,
)


def center_crop(img, pct):
    if pct >= 100:
        return img
    h, w = img.shape[:2]
    ch, cw = int(h * pct / 100), int(w * pct / 100)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return img[y0:y0 + ch, x0:x0 + cw]


def spectrum_figure(result):
    if "band" not in result:
        return None
    band, baseline, peaks = result["band"], result["baseline"], result["peaks"]
    radius = np.arange(result["band_start_radius"], result["band_start_radius"] + len(band))

    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.plot(radius, band, label="spectrum", linewidth=1)
    ax.plot(radius, baseline, label="local baseline", linewidth=1, linestyle="--")
    if len(peaks):
        ax.scatter(radius[peaks], band[peaks], color="red", zorder=3, label="flagged peaks")
    ax.set_yscale("log")
    ax.set_xlabel("spatial frequency (radial bin)")
    ax.set_ylabel("magnitude (log)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


if not files:
    st.info("Upload one or more images to test.")
else:
    rows = []
    for idx, f in enumerate(files):
        raw_bytes = f.getvalue()
        data = np.frombuffer(raw_bytes, np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            st.warning(f"Could not read {f.name} — skipping.")
            continue
        auto_crop, det_conf = (yolo_crop(finger_model, img) if finger_model is not None
                                else (None, None))
        if auto_crop is not None:
            cropped = auto_crop
            crop_note = f"YOLO finger crop (detection confidence {det_conf:.2f})"
        else:
            cropped = center_crop(img, crop_pct)
            reason = "no finger detected by YOLO" if finger_model is not None else "detector unavailable"
            crop_note = f"FALLBACK: manual {crop_pct}% center-crop ({reason}) — not representative"

        result = analyze_spectrum(cropped)
        rows.append({"file": f.name, "crop": crop_note, **{k: result[k] for k in
                     ("is_screen", "peak_count", "strongest_peak_ratio", "chroma_diff", "score", "note")}})

        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB), caption=f.name, use_container_width=True)
        with col2:
            st.caption(crop_note)
            if result["is_screen"]:
                st.error("SPOOF DETECTED — screen/print pattern found")
            elif result["note"] == "insufficient_signal":
                st.warning("Insufficient signal — image too small or too flat to analyse")
            else:
                st.success("Looks real — no screen/print pattern detected")
            st.write(
                f"peak_count: **{result['peak_count']}**  |  "
                f"strongest_peak_ratio: **{result['strongest_peak_ratio']}**  |  "
                f"chroma_diff: **{result['chroma_diff']}**  |  "
                f"score: **{result['score']}**"
            )
            with st.expander("Frequency spectrum"):
                fig = spectrum_figure(result)
                if fig is not None:
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.write("Not enough signal to plot.")

            save_col1, save_col2 = st.columns(2)
            with save_col1:
                if st.button("Save as real example", key=f"save_real_{idx}"):
                    save_example("real", f.name, raw_bytes, result)
                    st.success(f"Saved to spoof_dataset/real/")
            with save_col2:
                if st.button("Save as spoof example", key=f"save_spoof_{idx}"):
                    save_example("spoof", f.name, raw_bytes, result)
                    st.success(f"Saved to spoof_dataset/spoof/")
        st.divider()

    if len(rows) > 1:
        st.subheader("Summary")
        st.dataframe(
            [{"file": r["file"], "crop": r["crop"], "verdict": "SPOOF" if r["is_screen"] else "real",
              "peak_count": r["peak_count"], "strongest_peak_ratio": r["strongest_peak_ratio"],
              "chroma_diff": r["chroma_diff"], "score": r["score"], "note": r["note"]} for r in rows],
            use_container_width=True,
        )
