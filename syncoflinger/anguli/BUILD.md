# Building Anguli

Source vendored from the Database Systems Lab, IISc Bangalore
(`dsl.cds.iisc.ac.in/projects/Anguli/`), copyright 2012 IISc, free for research use
(no formal OSS license — see `README.txt` in this folder for the exact terms). Reimplements
SFinGe's published synthetic fingerprint generation algorithms (orientation field from
core/delta modeling, density maps, Gabor-style ridge filtering, a noise model) in C++/Qt.

**Verified working**, headless, via WSL2 (Ubuntu 22.04) — Docker Desktop does not run in
the Claude Code sandbox this was built in (no interactive desktop session for it to attach
to; launches and immediately exits), but WSL works directly (`wsl -d <distro> -u root`,
which needs no sudo password since it's a Windows-side trust boundary, not a Linux one).
This doc reflects what was actually run, not the original 2012 instructions.

## Data assets not in this repo

`Filterbank/`, `Lib/`, `Densitymaps/`, and `noise_blob/` are gitignored here — 180MB+
across 20,000+ files of re-downloadable generation resources and prebuilt Windows
binaries, not source code (same convention this project already uses for gitignored
model weights). Get them from the same source: download `Anguli_src.tgz` from
`dsl.cds.iisc.ac.in/projects/Anguli/` and copy `Filterbank/`, `Densitymaps/`, and
`noise_blob/` in alongside this repo's `Headers/`/`Sources/`/etc. before building
(`Lib/` is Windows-only prebuilt binaries, not needed for the Linux/WSL build below).
**All three are required at runtime** — missing `Densitymaps/` specifically causes a
segfault in `DensityMaps::sel_n_merg_densitymap()`, not a clean error.

## What was actually required (two real gaps beyond "install packages")

1. **Qt4 is genuinely EOL** — not in Ubuntu 22.04/24.04's repos at all. Don't fight
   installing it; the code was ported to Qt5 instead (see "Source changes" below).
2. **OpenCV's legacy C API** (`IplImage`, `CvMat`, `cvLoadImage`, etc. — 225 call sites
   across the codebase) was fully removed in OpenCV 4 (the only version Ubuntu 22.04
   ships). Rather than port 225 call sites, **OpenCV 2.4.13.7 was built from source**
   instead — keeps Anguli's algorithm code completely untouched.

## Requirements (as actually installed)

```bash
apt-get install -y build-essential cmake git qtbase5-dev qt5-qmake qttools5-dev-tools \
  pkg-config libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev
```

## Build OpenCV 2.4.13.7 from source (the long pole, ~15-20 min on 12 cores)

```bash
git clone --branch 2.4.13.7 --depth 1 https://github.com/opencv/opencv.git
cd opencv && mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF -D BUILD_DOCS=OFF \
      -D WITH_CUDA=OFF -D WITH_OPENCL=OFF -D WITH_IPP=OFF -D WITH_OPENEXR=OFF \
      -D BUILD_opencv_python=OFF -D BUILD_opencv_java=OFF ..
make -j$(nproc)
make install && ldconfig
```

`WITH_OPENEXR=OFF` matters — the bundled OpenEXR 3rdparty code uses deprecated
`throw(...)` dynamic exception specs that ISO C++17 (g++ 11's default) rejects outright.
We don't need HDR/EXR support (Anguli only writes jpg/png/bmp/tif), so this is a real
fix, not a workaround. Installs both `opencv2/*.hpp` (modern) and the legacy
`opencv/cv.h`-style compat headers Anguli's code expects — confirmed at
`/usr/local/include/opencv/`.

## Source changes made to Anguli itself

Small, deliberate, and scoped — only 2 of 16 source files needed changes, because the
CLI path never touches the GUI classes:

- **`Sources/main.cpp`**: dropped `#include "mainwindow.h"` and the entire `argc==1` GUI
  branch (constructs `MainWindow`, sets `QWindowsStyle`/`QCleanlooksStyle` — neither
  exists cleanly in Qt5, and we never invoke Anguli without CLI args anyway); replaced
  with a call to `show_help()`. Added `#include "threading.h"` directly (was previously
  pulled in transitively via `mainwindow.h`). Fixed `QtGui/QApplication` and
  `QtXml/QDomDocument` → Qt5's unqualified `<QApplication>`/`<QDomDocument>` includes.
  Changed `load_filter_blobs(MainWindow* w)` → `load_filter_blobs(QWidget* w)` (the
  parameter is only ever used as a `QMessageBox` parent).
- **`Sources/threading.cpp`**: same include fixes as main.cpp. Added `#include <unistd.h>`
  for `sleep()`, which used to resolve transitively and no longer does with modern glibc
  header hygiene.
- **NOT touched**: `mainwindow.cpp`, `fingerprint_window.cpp`, `impression_window.cpp` —
  pure GUI, only ever instantiated in the branch we removed. Not compiled at all.
  `generation.cpp`, `orientmap.cpp`, `densitymap.cpp`, `noise.cpp`, `filter_image.cpp`,
  `g_cap.cpp` — the actual algorithm — untouched, using the freshly-built OpenCV 2.4
  instead of a ported API.

## Build Anguli (CLI-only, via `Sources/anguli_cli.pro`)

A minimal qmake project excluding the 3 GUI-only files (already in this repo):

```bash
cd Sources
qmake anguli_cli.pro
make
```

Links against Qt5 (`core widgets xml`) and the freshly-built OpenCV 2.4
(`-L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui`).

## Running it

```bash
./Anguli -num 1 -ni 1 -outdir ./test_output -itype png -seed 42
```

Confirmed working: produces `test_output/Fingerprints/fp_1/1.png` (a clean synthetic
ridge pattern) and `test_output/Impression_1/fp_1/1.png` (a mated impression of the same
finger with realistic capture noise applied) — real 275×400 8-bit grayscale PNGs.

Full flag reference: `Sources/main.cpp`'s `show_help()`. Key ones:
- `-num` — number of distinct fingerprints to generate
- `-ni` — impressions (mated samples) per fingerprint
- `-cdist` — class distribution (natural/arch/tarch/right/left/dloop/whirl)
- `-outdir`, `-itype` — output location and image format

## What we actually need from it

Just the ridge-pattern generation (`-num`/`-ni`, no `-noise`/`-scratch`/`-rot` options —
those simulate contact-scanner degradation we don't want). SynCoLFinGer's contactless
transformation layer (deformation warp, rotation, thinning, blur, skin color, shadow,
dirt, camera noise — see the paper, `Pattern Recognition Letters 157 (2022) 127-134`)
gets applied on top of Anguli's clean output, replacing Anguli's own contact-oriented
noise/degradation model rather than combining with it.

## Next step

Start on the contactless transformation layer described in `syncoflinger/README.md`,
taking Anguli's clean `-num`/`-ni` output as input.
