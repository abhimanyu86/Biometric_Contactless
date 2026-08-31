# Building Anguli

Source vendored from the Database Systems Lab, IISc Bangalore
(`dsl.cds.iisc.ac.in/projects/Anguli/`), copyright 2012 IISc, free for research use
(no formal OSS license — see `README.txt` in this folder for the exact terms). Reimplements
SFinGe's published synthetic fingerprint generation algorithms (orientation field from
core/delta modeling, density maps, Gabor-style ridge filtering, a noise model) in C++/Qt.

**Not buildable in the Claude Code sandbox this was fetched in** — verified no C++
compiler (g++/MSVC), no Qt, no CMake, and no OpenCV present in that environment. This
needs a real dev machine. Build there, not here.

## Data assets not in this repo

`Filterbank/`, `Lib/`, `Densitymaps/`, and `noise_blob/` are gitignored here — 180MB+
across 20,000+ files of re-downloadable generation resources and prebuilt Windows
binaries, not source code (same convention this project already uses for gitignored
model weights). Get them from the same source: download `Anguli_src.tgz` from
`dsl.cds.iisc.ac.in/projects/Anguli/` and copy those four folders in alongside this
repo's `Headers/`/`Sources/`/etc. before building.

## Requirements

- Qt Creator + Qt 4.x (the project uses `QtGui`, `QDomDocument`, old-style Qt4 APIs —
  expect friction on modern Qt5/6; may need Qt4 specifically, or porting)
- OpenCV dev headers/libs (bundled `Include/opencv` headers are old — link against a
  compatible installed OpenCV, don't assume the bundled headers match your installed lib)
- Windows: .NET Framework (for the MSVC build path)
- The `qxt` library (already bundled under `Include/Qxt` — no separate install needed,
  but it's a large, mostly-abandoned Qt extension library only used by the GUI shell)

## Build

Open `Anguli.pro` in Qt Creator and build. See `README.txt` for the original instructions.

## Running headless (this is the part that matters for us)

Anguli has a real CLI mode — no GUI required when arguments are passed:

```
Anguli -num <count> -ni <impressions-per-finger> -outdir <path> \
       -cdist natural -seed <n> -numT <threads> -itype png
```

Key flags (see `Sources/main.cpp` `show_help()` for the full list):
- `-num` — number of distinct fingerprints to generate
- `-ni` — impressions (mated samples) per fingerprint — directly useful for training data
- `-cdist` — class distribution (natural/arch/tarch/right/left/dloop/whirl)
- `-outdir`, `-itype` — output location and image format
- `-impr -indir <path>` — a separate mode: generate additional noisy/rotated/translated
  impressions from an existing set of fingerprints, rather than generating from scratch

## What we actually need from it

Just the ridge-pattern generation (`-num`/`-ni` mode, no noise/scratch/rotation options —
those simulate contact-scanner artifacts we don't want). SynCoLFinGer's contactless
transformation layer (deformation warp, rotation, thinning, blur, skin color, shadow,
dirt, camera noise — see the paper, `Pattern Recognition Letters 157 (2022) 127-134`)
gets applied on top of Anguli's clean output ridge pattern, replacing Anguli's own
contact-oriented noise/degradation model rather than combining with it.

## Next step

Get this built and running on an actual dev machine (not this sandbox), verify
`-num 1 -ni 1` produces a real fingerprint image, then start on the contactless
transformation layer described in `syncoflinger/README.md`.
