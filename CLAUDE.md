# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A contactless fingerprint authentication SDK for the UIDAI SITAA challenge (project
`SU-C1-2025-02`, YellowSense Technologies). Capture a fingerprint with a phone camera
instead of a certified contact scanner, and produce an ISO-conformant template that
Aadhaar's matching infrastructure can use.

The PoC has a **mandatory acceptance gate** that drives most design decisions:

- \>85% matching accuracy, user-wise, across at least 20 users, contactless-to-contactless
- Capture, quality control, liveness, preprocessing and segmented FIR creation must run
  **on-device**, under 5 seconds
- Templatization and matching may run in the cloud, under 5 seconds
- A single APK providing both Register and Authenticate

Nothing currently runs on-device — the entire pipeline lives in the Flask backends. That
gap is known, not an oversight.

## Commands

```bash
# Flutter app
flutter pub get
flutter run
flutter analyze --no-fatal-infos     # what CI runs
flutter build apk --debug            # what CI builds

# Backends — two separate servers, run both
cd backend
PORT=5002 python app.py              # single-finger  -> uidai.db
PORT=5010 python slap_app.py         # 4-finger slap  -> slap.db

# Tests — no model weights, database or TensorFlow required
cd backend
python -m pytest tests/ -v
python -m pytest tests/test_matching.py -v
python -m pytest tests/test_matching.py::test_rotation_invariance -v

# FAR/FRR/EER benchmark — reads the users table of backend/uidai.db,
# writes benchmark_results.csv and eer_plot.png
cd backend && python benchmark.py

# Desktop QC tool
cd backend && streamlit run qctool/app.py

# Lab bench — enrol a dataset, then walk FP-01..FP-08 on a probe photo
cd backend && streamlit run labtool/app.py

# Spoof detector test bench — upload photos, see the FFT verdict + spectrum plot.
# No backend server or model weights needed, same as the tests.
cd backend && streamlit run spooftool/app.py
```

`backend/start.sh` contains a hardcoded venv path to a former developer's machine and does
not work anywhere else. Invoke `python app.py` directly.

## Architecture

One Flutter app, two Flask backends, selected by a mode toggle:

```
Flutter app  (lib/)  — CaptureMode.single | CaptureMode.slap
      |
      +--> app.py       :5002  single finger  -> uidai.db
      +--> slap_app.py  :5010  4-finger slap  -> slap.db
```

`lib/services/api_service.dart` holds two Dio clients, one per backend, with base URLs
overridable in Settings and persisted to SharedPreferences. Both default to a remote GCP
host, so a fresh install talks to a server rather than localhost.

### The ML pipeline

Five models in sequence, in this order. `app.py` runs it inline; `slap_app.py` runs the
same chain per detected finger via `slap_core.py`.

1. **YOLO** — detect the finger, crop it (`detect_and_crop`)
2. **MobileNetV2** — liveness / anti-spoof (`check_liveness`)
3. **U²-Net TFLite** — segmentation mask (`get_segmentation_mask`)
4. **Zero-DCE + threshold** — enhancement, only fires when mean luminance < 150
   (`preprocess_fingerprint`)
5. **MinutiaeNet** — minutiae extraction, four output heads: location heatmap, cos, sin,
   type (`detect_minutiae`)

Minutiae are dicts of `{x, y, direction (radians), type ("RIG"|"BIF")}`. That structure is
the contract between extraction, matching and template export.

### Shared modules

`backend/matching.py` is the **only** matcher. `app.py`, `slap_app.py` and `benchmark.py`
all import it. It previously existed as three copies that had begun to diverge — do not
re-fork it. Notes:

- The edge descriptor is **rotation invariant**: bearings are measured relative to each
  minutia's own ridge direction, not in image space. The original measured them absolutely,
  which made a rotated finger score 0% TAR.
- `match_templates(t1, t2, legacy=True)` reproduces the original algorithm exactly, so
  before/after can be measured on identical data. Keep it working.
- The `0.25` decision threshold still hardcoded in `slap_app.py` (`MATCH_THRESH`) and the
  single-finger endpoints predates the current scoring and is **not calibrated for it**.
  Recalibrate from a measured FAR/FRR curve before relying on it.

`backend/labtool/` is a Streamlit bench for the acceptance flow: enrol a small dataset
(name + unique ID per photo), then drop in a probe and watch FP-01 → FP-08 run stage by
stage, ending in a ranked candidate table and the matched person's details. The ML stages
go to the backend over HTTP; matching, scoring and FIR decode run in-process against
`matching.py` and `fir.py`, so it works with no weights and no TensorFlow installed.
Its gallery is a separate `backend/lab_gallery.db` — `/users` does not return
`template_json`, so a ranked list can only be built from templates the bench holds itself.

`backend/fir.py` produces ISO/IEC 19794-4 **Finger Image Records** — the format the PoC
spec requires. Uncompressed records are conformant, so there is no JPEG2000 dependency.
It ships a decoder used by the tests, so conformance is demonstrated by round-trip rather
than asserted.

`backend/spoof_detection.py` is a second, independent presentation-attack signal:
`check_liveness` (the MobileNetV2 classifier) was never trained on screen/print recapture
and doesn't catch someone holding a photo of a finger up to the camera. `detect_screen_replay`
looks for the sharp multi-peak "comb" a display's pixel grid leaves in the radial FFT
spectrum, which a real fingertip's single ridge-frequency peak doesn't produce — same radial
windowing technique as `fir.estimate_ridge_period`. It's imported by `app.py` and
`slap_app.py` the same way `matching.py` is, and gates enroll/authenticate/verify alongside
`check_liveness` (OR'd together — either signal can reject). **The thresholds are an
uncalibrated starting point**, verified only against synthetic test patterns
(`tests/test_spoof_detection.py`), not real screen-recapture photos — same caveat as
`MATCH_THRESH` above, needs a measured FAR/FRR pass before being trusted alone.
`backend/spooftool/` is a Streamlit bench for exactly that: upload real/spoof photos and
see the verdict plus the actual FFT spectrum plot (`analyze_spectrum` exposes the raw
band/baseline/peaks so the tool doesn't re-derive the threshold logic itself). Like the
tests, it needs no backend server and no model weights.

Note the two template formats are different standards and are easy to confuse:

| Endpoint | Standard | Contains |
|---|---|---|
| `/export_template` | 19794-2 (FMR) | minutiae |
| `/export_fir` | 19794-4 (FIR) | the 500 DPI image |

The matcher consumes raw minutiae, not either template.

### Endpoints

- `app.py` — `/health` `/quality_check` `/check_roi` `/readiness` `/liveness_gesture`
  `/enroll` `/authenticate` `/verify` `/process` `/process_contact` `/compare_contact`
  `/users` `/history` `/export_template` `/export_fir`
- `slap_app.py` — `/health` `/quality_check` `/process_slap` `/enroll_slap`
  `/authenticate_slap` `/verify_slap` `/history`

### Camera capture

`lib/widgets/fingerprint_camera_widget.dart` runs a poll loop: frame -> `/quality_check`
-> guidance overlay -> repeat, auto-firing the shutter after N consecutive passes. It has
AUTO/MANUAL modes, an attempt cap, a request timeout and several race guards. Changes here
are easy to get subtly wrong; read the whole state machine first.

## Model weights

Seven files, all gitignored, none in the repo. **The two backends load them from different
directories** — this catches people out:

- `app.py` loads from `backend/` directly
- `slap_core.py` loads from `backend/models/` (override with `SLAP_MODELS_DIR`)
- `README.md` says `backend/models/`, which satisfies only the slap backend

Put all seven in `backend/` and set `SLAP_MODELS_DIR` to that same path to satisfy both.

Missing weights mostly degrade **silently** — `load_models()` checks `os.path.exists` and
skips. `check_liveness` is the exception: it now fails closed and reports the reason.
`ALLOW_LIVENESS_BYPASS=1` re-opens it for local pipeline work and must never be set in a
deployed environment.

`MODEL_SPECIFICATIONS.md` on the `feature/slap-preprocessing` branch documents each model's
architecture, training config and validation metrics. It is not on the working branch.

## Testing

`backend/tests/` runs without weights, database or TensorFlow, using synthetic minutiae and
synthetic ridge images. This is deliberate — the matcher and the FIR encoder can be
validated before any fingerprint data exists.

When changing the matcher, `test_separation_survives_partial_minutiae_overlap` is the test
that matters. Two real captures never yield the same minutiae set, so testing only against
a jittered copy of the same set flatters any matcher and hides the failure mode that
actually occurs in the field.

## Branches

- `feature/uidai-unified` — remote default; the unified app plus both backends
- `feature/uidai-app` — the original single-finger app
- `feature/slap-preprocessing` — IIT Bombay reference snapshot, plus `MODEL_SPECIFICATIONS.md`

CI (`.github/workflows/ci.yml`) is Flutter-only — analyze plus a debug APK build, on PRs to
`main`/`develop`. The Python tests are not wired into CI.
