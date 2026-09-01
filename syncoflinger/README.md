# SynCoLFinGer workspace

Workspace for adapting SynCoLFinGer (Priesnitz, Rathgeb, Buchmann, Busch — da/sec,
Hochschule Darmstadt, *Pattern Recognition Letters* 157 (2022) 127–134) into this
project's fine-tuning pipeline.

## What this is for

SynCoLFinGer generates synthetic contactless fingerprint images on top of the SFinGe
synthetic ridge-pattern generator. Because it's built on SFinGe, every generated image
comes with exact, free minutiae ground truth (position, direction, type) — no
MINDTCT/consensus labelling needed for this data.

Planned uses (see `backend/SITAA_MinutiaeNet_FineTuning_Data_Plan.pdf`, Stage B/C):

1. **Pretraining bootstrap** — warm-start the MinutiaeNet fine-tune on synthetic data
   before real M1 data (~100 people) exists.
2. **Diversity backstop** — generate synthetic examples in skin-tone/quality ranges
   the real collection is thin on at early milestones.
3. **Fast pipeline validation** — debug Stage B's target-generator and Stage C's
   training loop against synthetic data before running them on real Stage A output,
   the same way `backend/tests/` already validates the matcher on synthetic data.

**Not for**: final accuracy claims. Real TAR/FRR/FAR at each milestone must come from
real captures only — same rule as the rest of the plan.

## Source

Real, publicly released code: https://gitlab.com/jannispriesnitz/syncolfinger

Plan is to adapt this repo directly rather than reimplement the method from the paper
— the method is fully classical/procedural (no trained weights), so risk is low, but
integration needs is verified first, not assumed.

## Open questions before porting starts

- [ ] License terms on the GitLab repo itself (the paper is CC BY-NC-ND 4.0, but that
      covers the paper text, not necessarily the code).
- [ ] Whether the repo bundles a working reimplementation of SFinGe, or depends on the
      original University of Bologna SFinGe tool (which has historically not been
      fully open-source) — this determines whether the pipeline is self-contained or
      has an external dependency to resolve first.
- [ ] Python version / dependency compatibility with this project's stack.

## Status

Not started. This README is the only file in this branch so far — next step is
cloning the upstream repo and answering the questions above before any porting work.
