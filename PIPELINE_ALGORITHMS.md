# Pre-matching pipeline algorithms — what's actually improvable, and how far

`backend/app.py` runs five stages before any minutiae ever reach a matcher: YOLO finger
detection → MobileNetV2 liveness → U²-Net segmentation → Zero-DCE/histogram-equalisation
enhancement + adaptive threshold → MinutiaeNet extraction. Every one of the nine matching
algorithms in [MATCHING_ALGORITHMS.md](MATCHING_ALGORITHMS.md) only ever sees whatever
minutiae list this chain produces — so a weak link here caps what any matcher can achieve,
independent of which one wins the bake-off.

This doc is organised **by tier, not by stage**, because tier is what actually determines
whether something is a near-term move or a multi-week initiative. The short version: almost
nothing found is a pretrained drop-in. Every model here (`best-new.pt`, `liveness_model_v3.pt`,
`u2net_320x320_float32.tflite`, `zero_dce_model.h5`, `best_f1.pth`) was supplied by IIT
Bombay, not trained in-house, and this repo has no training scripts for any of them
(`context.md:84-107`). Real improvements to detection, liveness, segmentation or extraction
mean retraining on new data the team doesn't currently have — a different kind of project
than the matcher bake-off, which was pure code and buildable in a session.

`app.py` and `slap_core.py` duplicate all five stages line-for-line, with no shared module
the way `matching.py` is shared for matching (`CLAUDE.md` calls out the *matcher* consolidation
explicitly but this duplication was never addressed the same way). Any real change to a stage
means editing both files to stay in sync — worth fixing eventually, out of scope here.

## The single highest-leverage next step, and it costs nothing to build

`context.md` already flags a live, unresolved question: MinutiaeNet's reported F1 (0.710)
may be the same number as a vendor deck's "0.71 F1 on NIST SD27" — but SD27 is **latent**
fingerprints, a domain far from contactless phone captures. The doc's own words: *"Unconfirmed
inference — worth asking IIT Bombay directly."* Nobody has. If true, it caps every downstream
algorithm's ceiling regardless of which of the nine matchers wins — no amount of algorithm
research resolves this, and confirming or ruling it out costs one email, not a retraining
project. Ask before spending effort on anything else in this document.

## Tier 0 — no training, no new weights, buildable now

**Built and tested this session:** classical (non-ML) contactless enhancement — intrinsic
image decomposition (illumination/reflectance split in the log domain) + a hand-rolled guided
filter (`backend/labtool/experiments/classical_enhancement.py`), as a drop-in alternative to
the Zero-DCE/histogram-equalisation step. Zero training data, zero new dependencies (the
guided filter is built from `cv2.boxFilter` alone, avoiding the `opencv-contrib-python`
package `cv2.ximgproc.guidedFilter` would require).

**Result — a real negative, tested in a closed loop, not just visually:**
`backend/labtool/experiments/run_classical_comparison.py` ran the actual local pipeline
(same models, same crop, same segmentation mask — only the enhancement step differs) on 5
real team photos:

| Metric | Result |
|---|---|
| Minutiae count | Classical produces **24% more** minutiae (351 vs 282 across 5 photos) |
| Self-consistency (same photo, server-preprocessed vs. classically-preprocessed, scored via `matching.match_templates`) | **0.320 mean** — low; the two preprocessing paths extract substantially different ridge structures from the same finger |
| Classically-preprocessed probe still ranks its own gallery entry #1 (against the existing, server-preprocessed gallery) | **0/5** |

More minutiae isn't better if they're not the *same* minutiae a differently-preprocessed
capture of the same finger would find, and this experiment shows they mostly aren't.
**Interpretation:** MinutiaeNet appears tuned to the specific visual characteristics of the
current enhancement step's output (its particular adaptive-threshold binarisation style), not
to ridge structure in general — swapping the enhancement stage alone, without also adapting
or retraining the extractor to match, breaks compatibility rather than improving it. This is
itself informative: it suggests the domain gap (if the SD27 concern above is real) is more
likely to live in the extractor than in the enhancement step, since a *better* enhancement
input didn't help an extractor tuned to the old one. Re-run
`python -m labtool.experiments.run_classical_comparison` from `backend/` any time — it needs
local model weights and reads whatever's currently in `backend/labtool/dataset/`.

**Not pursued further given the result above, but noted for completeness:**
MobileSAM as a zero-shot segmentation bootstrap (useful for auto-labeling future training
masks, not as a production swap — same "needs the extractor to agree" caveat likely applies).

## Tier 1 — needs fine-tuning on the team's own captures, moderate effort

- **Contactless-specific minutiae extraction CNN** (Zhang, Liu & Liu, *Pattern Recognition*
  2021, ["A multi-task fully deep convolutional neural network for contactless fingerprint
  minutiae extraction"](https://www.sciencedirect.com/science/article/abs/pii/S0031320321003526)) —
  the closest like-for-like replacement for MinutiaeNet found: jointly regresses minutiae
  location + direction with an attention branch, and — directly relevant given the Tier-0
  result above — **operates on raw grayscale contactless images with no separate
  preprocessing step**, sidestepping the enhancement-extractor mismatch entirely rather than
  trying to patch around it. Same output shape (x/y/direction), so it's a genuine drop-in for
  `detect_minutiae`'s contract. No public pretrained weights — needs retraining on the team's
  own contactless captures.
- **Lightweight mobile segmentation** (BiSeNet — 358 FPS / CUDA, 30.5 FPS on Raspberry Pi 5;
  Mobile-SegFormer — MobileNetV3 backbone) — worth benchmarking against U²-Net if the
  on-device goal (CLAUDE.md's stated but unaddressed requirement) ever gets prioritised.
  Needs retraining on the team's finger-mask labels either way.

## Tier 2 — needs real infrastructure investment (paired datasets, GPU budget, possibly
commercial SDK licenses)

- **C2CL contact-to-contactless pipeline** (Grosz & Jain, MSU,
  [arXiv:2104.02811](https://arxiv.org/pdf/2104.02811)) — segmentation → contrast enhancement
  → ridge-frequency scaling to 500 ppi → a learned spatial transformer network for non-linear
  deformation correction. The most literal match to "turn a photo into a contact-equivalent
  image" found in the literature — the ridge-frequency normalisation piece is usable
  standalone even without the full system. Needs paired contact/contactless data to fit the
  deformation-correction network.
- **Fingerphoto deblurring GANs** (FDeblur-GAN, [arXiv:2106.11354](https://arxiv.org/abs/2106.11354);
  attention-guided multi-stage GAN variants) — target camera defocus/motion blur specifically,
  a failure mode neither Zero-DCE nor the classical Tier-0 technique addresses at all. Needs
  retraining on the team's device/lighting conditions; GAN inference is fast (ms-range) once trained.
- **Cross-device-generalising liveness models** — ColFigPhotoAttnNet
  ([arXiv:2503.05247](https://arxiv.org/html/2503.05247)) explicitly measures and targets the
  failure mode a plain MobileNetV2 classifier likely has: PAD models trained on one phone
  camera degrading on another. GRU-AUNet ([arXiv:2504.01213](https://arxiv.org/pdf/2504.01213))
  reports strong cross-dataset numbers (BPCER 0.09% / APCER 1.2% averaged across CLARKSON,
  COLFISPOOF, IIITD). Also worth noting independent of any model swap: `context.md:105-107`
  flags the *current* liveness model's reported AUC of exactly 1.0 as "a red flag... usually
  indicates a separable-by-accident train/test split," and a direct conflict between two
  internal docs about which spoof dataset (HDA ColFIS vs. LivDet-2023) it was even trained on
  — worth resolving in the same email as the MinutiaeNet/SD27 question above, before
  investing in any liveness model upgrade.
- **Ensemble ground-truth relabeling** (Mulay, Grosz & Jain, MSU PRIP, IJCB 2024,
  [PDF](http://biometrics.cse.msu.edu/docs/084.pdf)) — not a new architecture, a labeling
  strategy: ensemble multiple commercial SDK predictions (Innovatrics, Verifinger) to build
  more reliable minutiae ground truth for retraining. Directly reusable for any future
  MinutiaeNet retraining even independent of adopting their architecture. Needs commercial
  SDK licenses.

## Out of scope regardless of effort

**End-to-end embedding models** — Ridgeformer
([arXiv:2506.01806](https://arxiv.org/abs/2506.01806), code available, reportedly beats
commercial matchers on HKPolyU/RidgeBase) and AFR-Net
([arXiv:2211.13897](https://arxiv.org/pdf/2211.13897)) both output fixed-length embeddings,
not a minutiae list. Adopting either means replacing the entire nine-algorithm matcher
bake-off, not just a pipeline stage, and drops the ISO/IEC 19794-2 (FMR) minutiae-template
export requirement the PoC spec asks for. Not a same-contract swap under any circumstances.

## Reference datasets, for whenever Tier 1/2 work actually starts

**RidgeBase** ([arXiv:2307.05563](https://arxiv.org/pdf/2307.05563), cross-sensor multi-finger),
**HKPolyU Contactless2Contact**, **COLFISPOOF** (72 PAI species, presentation-attack detection),
**CLARKSON** contactless dataset, **IIITD SmartPhone** fingerphoto DB, and **SynCoLFinGer**
([arXiv:2110.09144](https://arxiv.org/pdf/2110.09144), synthesizes contactless-style
fingerprints from SFinGe ridge patterns — useful for bootstrapping training data without new
real captures).
