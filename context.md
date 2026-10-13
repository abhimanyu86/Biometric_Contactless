# Working Context

Running record of the Claude Code sessions on this project — what was investigated, what
was decided, what was built, and what is still open. Updated on request.

**Branch:** `abhimanyu` (created from `feature/uidai-unified` @ `2aaaa9c`)
**Last updated:** 2026-08-08

---

## 1. Where the project stands

Contactless fingerprint authentication SDK for SITAA (`SU-C1-2025-02`). The PoC has
a four-condition mandatory acceptance gate:

| # | Condition | Status |
|---|---|---|
| 1 | \>85% matching accuracy, user-wise, 20+ users, cL2cL | **Not measured.** Was 0% TAR; matcher since fixed, unvalidated on real data |
| 2 | On-device capture + segmented FIR under 5 s | **Failing.** 0% on-device |
| 3 | Cloud templatization + matching under 5 s | Likely passing — 396 ms for 1:10, not formally measured |
| 4 | Both APK flows end-to-end | **Met** |

Deadline is short. Gate 2 is not reachable without a native Android + on-device ML
engineer, which the team does not have.

---

## 2. Session timeline

1. **Orientation** — explained git worktrees, the three-branch layout, and the project.
   Created working branch `abhimanyu`.
2. **PDD audit** — read `SITAA_PDD_YellowSense.pdf` (16 pp), audited the repo against it,
   produced a 62-item gap analysis PDF.
3. **Spec correction** — the actual PoC Specification (FP-01…FP-08, §2.2–§2.5) arrived and
   **supersedes the PDD**. Re-audited. Several PDD items turned out not to be required.
4. **Team and staffing** — mapped remaining work to five roles; identified the missing
   Android/on-device ML role.
5. **Matching deep-dive** — traced the matcher in code and found the rotation defect.
   Cross-checked against the team's own `matching algorithms.pdf`, which had independently
   reached the same root cause.
6. **Work allocation** — produced a status + allocation PDF for a five-person team.
7. **Implementation** — built the matcher, the FIR encoder, the test suite, and applied the
   backend fixes (see §4).
8. **Self-review** — traced the new matcher and found a defect in my own scoring; verified
   and corrected.
9. **Documentation** — model-file provenance investigation, `CLAUDE.md`, this file.

---

## 3. Key findings

### The matcher was structurally broken

`_compute_edge` stored the bearing between neighbouring minutiae as `atan2(dy, dx)` — an
angle in **absolute image space**. Rotating the finger between enrol and authenticate
shifted every bearing, edges stopped matching, and genuine pairs died. Measured on the
team's real captures: **TAR 0%** at threshold 0.25, genuine scores 0.130–0.201.

The internal report had already named this ("No rotation handling — tilted finger on auth
vs enroll"). The code confirmed it exactly.

### Scoring defect found by tracing (mine)

My first replacement blended coverage with a "confidence" term. Because relaxation
probabilities are **peak-normalised**, the top pair is always 1.0, so confidence measured
~1.0 even for impostors and contributed a constant floor. Coverage alone roughly doubles
separation:

| Scoring | drop 20% | drop 30% | drop 40% |
|---|---|---|---|
| Blend | +0.245 | +0.218 | +0.154 |
| Coverage only | **+0.458** | **+0.371** | **+0.293** |

An earlier attempt added a neighbourhood-signature prefilter; it passed every rotation test
but collapsed under realistic recapture (+0.036 at 30% minutiae loss) because the signature
depends on which minutiae were detected. Removed.

### `matching_v2.py` is not in the repository

Verified absent from the working tree, full git history, and both other branches. The
3.8× speedup and every benchmark figure derived from it exist on one machine only. The
KDTree optimisation was reimplemented in `matching.py`.

### Model provenance

`MODEL_SPECIFICATIONS.md` exists on `feature/slap-preprocessing` (not the working branch).
Models were **provided by the IIT Bombay Contactless Fingerprint Team**, not trained
in-house — which is why there are no training scripts.

| Model | Training data | Validation |
|---|---|---|
| `best-new.pt` finger detection | Custom `fingertip_segmentation.yolov8-obb` | mAP50 0.995 |
| `best_f1.pth` **MinutiaeNet** | **Not documented** | **F1 0.710** |
| `liveness_model_v3.pt` | Spoof: HDA ColFIS | AUC 1.0, FAR 0.0 |
| `bright_spot_detection.pt` | Custom Roboflow set, 25 epochs | mAP50 0.649, Precision 0.496 |
| U²-Net, Zero-DCE | Published pretrained (CVPR 2020) | — |

Three concerns:

- **MinutiaeNet's F1 of 0.710 matches the PDD's "0.71 F1 on SD27"** — NIST SD27 is *latent*
  fingerprints. If that is the same evaluation, the extractor was validated on a domain far
  from contactless phone captures, and it caps everything the matcher can achieve.
  *Unconfirmed inference — worth asking IIT Bombay directly.*
- **Liveness AUC of exactly 1.0** is a red flag; that rarely survives real data and usually
  indicates a separable-by-accident train/test split.
- **Documentation conflict**: the spec doc says the spoof set was HDA ColFIS (contactless);
  the PDD says LivDet-2023 (contact-sensor). Both cannot be true.

### Other defects found

- `check_liveness` **failed open** — returned `is_live: True` when weights were missing,
  so a broken install accepted every spoof while reporting success. Fixed.
- Active gesture liveness is **switched off** — `_livenessEnabled = false`
  ([enroll_screen.dart:29](lib/screens/enroll_screen.dart:29)). The underlying finger-count
  logic is also unreliable (Bug #3 in the internal test report).
- Model weights load from **different directories** in the two backends, and the README
  documents only one. Missing weights degrade silently.
- The single-finger schema had `UNIQUE(uid, batch)`, making the 10-finger gallery that
  FP-07's 1:10 case requires impossible. Fixed with a migration.

---

## 4. Work completed

All on `abhimanyu`, **uncommitted**. 60 tests passing, net −173 lines.

| File | What |
|---|---|
| `backend/matching.py` | New. Rotation-invariant matcher; replaces three diverging copies. `legacy=True` reproduces the original for before/after measurement |
| `backend/fir.py` | New. Ridge-frequency 500 DPI normalisation + conformant uncompressed ISO 19794-4 encoder, with an independent decoder so conformance is round-tripped, not asserted |
| `backend/tests/` | New. 60 tests, no weights/DB/TensorFlow needed |
| `backend/app.py` | Liveness fails closed; `finger_position` + migration; `/export_fir`; dead `apply_gabor_filters` removed; imports shared matcher |
| `backend/slap_app.py` | Imports shared matcher |
| `backend/benchmark.py` | Delegates to the shared matcher, so benchmark numbers reflect production |
| `CLAUDE.md` | New. Repo guidance |

Measured results (synthetic data only):

- Rotation: flat at **0.9904** from 0° to 90°. Legacy fell to 0.10 by 30° and 0.00 by 45°.
- Speed: 41 ms per match; **396 ms for a 1:10 gallery** against a 5 s budget.
- Benchmark path verified end-to-end on a synthetic 20-user × 3-capture gallery.

---

## 5. Decisions

- **The PoC Specification supersedes the PDD.** Curvelet enhancement, CLAHE, JPEG2000 and
  rectification are PDD items, not PoC requirements.
- **Cloud matching is permitted** (§2.2), so only the capture-side stages must move
  on-device. This is a much smaller migration than the PDD implied.
- **FIR only**, not FMR. Uncompressed FIR is conformant, so no JPEG2000 dependency.
- **ISO 19794-4:2005 32-byte general header** implemented. The 2011 revision differs;
  confirm which edition the authority expects.
- **Multi-finger is required** (FP-04), reversing the PDD's "out of scope" note.
- **"1:10" means the ten fingers of one person**, not a ranked top-10 list.

---

## 6. Open items

**Blocking**

1. **Obtain the seven model weight files.** Nothing can be validated against a real
   fingerprint without them. Everything measured so far is synthetic.
2. **Collect 20 users, 3+ captures each.** `benchmark.py` builds genuine pairs from users
   with 2+ enrolments — one capture per person yields zero genuine pairs.

**High value, low cost**

3. **Run bozorth3 (NBIS) as a baseline.** Distinguishes "matcher is broken" from "minutiae
   are bad". Those have completely different fixes, and it is a day's work.
4. **Recalibrate the decision threshold.** `0.25` predates the current scoring and is wrong
   for it — synthetic data puts EER near 0.45. Derive it from a measured FAR/FRR curve.
5. **Ask IIT Bombay what `best_f1.pth` was trained on.** One email; the answer determines
   whether effort belongs in matching or extraction.
6. **Commit `matching_v2.py`** if the original still exists anywhere.

**Known gaps, not yet started**

- On-device migration in full (gate 2) — needs the native Android + on-device ML role
- Distance check (FP-03) — the one missing quality gate
- Re-enable active liveness once the finger-count logic is fixed
- Surface latency and match metrics in the app journey (FP-08) — `lib/` has no timing
  references at all
- Thumbs: `SLAP_ORDER` covers 8 of 10 finger positions
- Python tests are not wired into CI

---

## 7. Artefacts produced

Generated outside the repo, delivered as files:

- **PoC Gap Analysis PDF** — 62 items vs the PDD (superseded by the item below)
- **Status & Work Allocation PDF** — audit against the current spec plus a five-person
  allocation with per-person task lists and handoffs
- WhatsApp message draft — asking the team about model training data and weight files
- Voice-note script — technical expectations for ~30 incoming candidates
