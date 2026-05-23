# YellowSense UIDAI — Team Onboarding & Work Guide

> Read this fully before touching any code.  
> This tells you exactly how to set up, where to work, and how to submit your work.

---

## Who Does What (Quick Reference)

| Person | Works On | Does NOT touch |
|--------|----------|----------------|
| **Atharv** | Flutter UI + CI/CD + GitHub | — |
| **Ishita** | Flutter UI + wiring backend into screens | — |
| **Ashish** | `backend/app.py` + `backend/qctool/` (Streamlit) | Flutter |
| **Vignesh** | `backend/app.py` + `backend/qctool/` (Streamlit) | Flutter |
| **Nandhini** | `backend/app.py` only | Flutter, Streamlit |

---

## Part 1 — Setting Up Your Laptop

### Step 1 — Install Required Tools

#### Everyone needs:
- **Git** — https://git-scm.com/downloads
- **VS Code** — https://code.visualstudio.com (recommended editor)

#### Atharv + Ishita also need (Flutter):
- **Flutter SDK 3.41.4** — https://docs.flutter.dev/get-started/install
- **Android Studio** — https://developer.android.com/studio
  - Inside Android Studio → SDK Manager → install Android SDK 34
  - Install an emulator: AVD Manager → Create Virtual Device → Pixel 6 → API 34

#### Ashish + Vignesh + Nandhini also need (Python backend):
- **Python 3.11** — https://www.python.org/downloads/release/python-3110/
  - Windows: check "Add Python to PATH" during install
  - Mac: `brew install python@3.11`

---

### Step 2 — Clone the Repository

Atharv will push the code to GitHub and add everyone as collaborators. You will get an email invite — accept it, then:

```bash
# Everyone runs this
git clone https://github.com/yellowsense/yellowsense-uidai.git
cd yellowsense-uidai
```

The folder structure you will see:
```
yellowsense-uidai/
├── uidai_app/          ← Flutter app (Atharv + Ishita work here)
│   ├── lib/
│   │   ├── screens/    ← all app screens
│   │   ├── widgets/    ← camera widget
│   │   ├── services/   ← API calls to backend
│   │   └── theme/      ← colors, fonts
│   ├── android/
│   └── pubspec.yaml
├── backend/            ← Flask server (Ashish + Vignesh + Nandhini work here)
│   ├── app.py          ← THE main file — all backend code lives here
│   ├── qctool/         ← Streamlit desktop tool (Ashish + Vignesh)
│   ├── start.sh        ← Mac/Linux: run this to start server
│   ├── requirements.txt
│   └── [model files]   ← .pt .pth .h5 .tflite — NOT in git, get separately
├── .github/
│   └── workflows/
│       └── ci.yml      ← auto build check on every PR
└── TEAM_STRATEGY.md
```

---

### Step 3 — Get the Model Files

The AI model files are too large for Git. Atharv will share them via Google Drive or WhatsApp.

Download all of these and put them inside `backend/`:

| File | Size |
|------|------|
| `best-new.pt` | 6.4 MB |
| `best_f1.pth` | 100 MB |
| `liveness_model_v3.pt` | 10 MB |
| `u2net_320x320_float32.tflite` | 4 MB |
| `zero_dce_model.h5` | 966 KB |
| `bright_spot_detection.pt` | 21 MB |

After downloading, your `backend/` folder should look like:
```
backend/
├── app.py
├── best-new.pt
├── best_f1.pth
├── liveness_model_v3.pt
├── u2net_320x320_float32.tflite
├── zero_dce_model.h5
├── bright_spot_detection.pt
├── requirements.txt
└── start.sh
```

---

### Step 4 — Backend Setup (Ashish · Vignesh · Nandhini)

#### Mac:
```bash
cd yellowsense-uidai/backend

# Create isolated Python environment
python3.11 -m venv venv

# Activate it (you must do this every time you open a new terminal)
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt

# For Mac Apple Silicon (M1/M2/M3) — replace tensorflow line with:
pip install tensorflow-macos==2.16.2 tensorflow-metal==1.2.0
```

#### Windows:
```powershell
cd yellowsense-uidai\backend

# Create isolated Python environment
python -m venv venv

# Activate it (you must do this every time you open a new terminal)
venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# If tensorflow fails on Windows, run:
pip install tensorflow-cpu
```

#### Verify everything installed correctly:
```bash
python -c "import flask, torch, cv2, ultralytics, tensorflow; print('ALL OK')"
```
You should see `ALL OK`. If any module fails, run `pip install <module-name>` again.

---

#### Starting the Backend Server:

**Mac/Linux:**
```bash
cd backend
source venv/bin/activate
bash start.sh
```

**Windows:**
```powershell
cd backend
venv\Scripts\activate
python app.py
```

You should see:
```
✓ U2Net loaded
✓ Zero-DCE loaded
✓ MinutiaeNet loaded
✓ Liveness loaded
✓ Bright-spot detector loaded
* Running on http://0.0.0.0:5001
```

**Test it's working:**
```bash
curl http://localhost:5001/health
```
Should return JSON with `"status": "ok"`.

---

### Step 5 — Flutter Setup (Atharv · Ishita)

```bash
# Check Flutter is installed correctly
flutter doctor
# Fix any issues it shows before continuing

cd yellowsense-uidai/uidai_app

# Install Flutter packages
flutter pub get

# Run on connected device or emulator
flutter run
```

**If you see "No devices found":**
- Connect your Android phone via USB
- Enable Developer Options on phone → enable USB Debugging
- Or start an emulator in Android Studio first

**Setting the server URL in the app:**
- Open the app → tap Settings
- Set URL to `http://<your-laptop-IP>:5001`
- Find your IP: Mac → `ipconfig getifaddr en0` | Windows → `ipconfig`
- Your phone and laptop must be on the same WiFi

---

## Part 2 — Git Workflow (How to Actually Work)

This is the most important section. Follow this exactly.

### The Branch Rules

```
main        ← NEVER commit here directly. Only Atharv merges here.
develop     ← NEVER commit here directly. Merge your feature branch here via PR.
feature/your-name-task  ← YOUR branch. This is where you work.
```

### Your Daily Workflow

#### Day 1 — Create your branch (do this once):

```bash
# Make sure you're up to date
git checkout develop
git pull origin develop

# Create your branch
git checkout -b feature/your-name-task

# Examples:
# git checkout -b feature/nandhini-readiness
# git checkout -b feature/vignesh-roi
# git checkout -b feature/ashish-iso
# git checkout -b feature/ishita-ui
# git checkout -b feature/atharv-autocapture
```

#### Every day — before you start coding:

```bash
# Get latest changes from develop into your branch
git checkout develop
git pull origin develop
git checkout feature/your-name-task
git merge develop
# Fix any conflicts if they appear, then continue
```

#### While coding — save your work often:

```bash
# See what you changed
git status

# Stage your changes
git add backend/app.py
# or add everything:
git add .

# Commit with a clear message
git commit -m "feat: add /readiness endpoint with 0-100 score"
# or
git commit -m "feat: add /check_roi endpoint with guidance string"
# or
git commit -m "fix: blur threshold adjusted for fingerprint images"

# Push to GitHub
git push origin feature/your-name-task
```

#### When your task is done — raise a Pull Request:

1. Go to `github.com/yellowsense/yellowsense-uidai`
2. Click **"Compare & pull request"** (appears after you push)
3. Set base branch to `develop` (NOT main)
4. Write a description of what you built
5. Tag **Atharv** as reviewer
6. Wait for CI to go green ✅
7. Wait for Atharv to approve
8. Then merge

---

### Commit Message Format

Use this format so everyone understands what changed:

```
feat: short description     ← new feature
fix: short description      ← bug fix
refactor: short description ← code cleanup, no new feature
test: short description     ← testing
docs: short description     ← documentation only
```

Examples:
```
feat: add /readiness endpoint returning 0-100 score
feat: add /check_roi endpoint with move left/right guidance
feat: add ISO 19794-4 template export function
fix: liveness model threshold corrected to 0.3
fix: blur check threshold lowered for fingerprint images
feat: add contact fingerprint preprocessing with Gabor filters
```

---

## Part 3 — Your Specific Tasks

---

### 👤 Nandhini

**You work in:** `backend/app.py` only  
**Your branch:** `feature/nandhini-backend`  
**You do NOT touch:** Flutter files, Streamlit files

#### Task 1 — `/readiness` endpoint

Open `backend/app.py`. Scroll to the bottom, just before `if __name__ == "__main__":`. Add your code there.

What you need to build:
```python
@app.route("/readiness", methods=["POST"])
def readiness():
    # 1. Accept image file from request
    # 2. Save it to uploads/
    # 3. Run quality_gate() on it → gives blur, brightness, glare
    # 4. Run detect_and_crop() → get cropped image
    # 5. Run preprocess_fingerprint() → get preprocessed image
    # 6. Run detect_minutiae() → get minutiae list
    # 7. Normalize each metric to 0-1 range
    # 8. Compute weighted score:
    #    score = blur_norm*30 + brightness_norm*25 + glare_norm*20 + minutiae_norm*25
    # 9. Return score + grade + breakdown
```

Grade thresholds:
- 80–100 → "Excellent"
- 60–79  → "Good"
- 40–59  → "Marginal"
- 0–39   → "Rejected"

Test it with curl after writing:
```bash
curl -X POST http://localhost:5001/readiness \
  -F "image=@/path/to/any/fingerprint.jpg"
```

#### Task 2 — `/liveness_gesture` endpoint

Port the gesture detection from the IIT Bombay repo (`Gesture` branch → `gesture.py`).

```python
@app.route("/liveness_gesture", methods=["POST"])
def liveness_gesture():
    # 1. Accept image file + expected_count (form field, int 1-5)
    # 2. Run MediaPipe hand detection on image
    # 3. Count extended fingers
    # 4. Compare to expected_count
    # 5. Return detected_count, expected_count, passed (bool)
```

Install mediapipe first:
```bash
pip install mediapipe
```

---

### 👤 Vignesh

**You work in:** `backend/app.py` + `backend/qctool/app.py`  
**Your branch:** `feature/vignesh-roi-benchmark`  
**You do NOT touch:** Flutter files

#### Task 1 — `/check_roi` endpoint in `backend/app.py`

```python
@app.route("/check_roi", methods=["POST"])
def check_roi():
    # 1. Accept image file
    # 2. Run YOLO detection (same as detect_and_crop but don't crop)
    # 3. Get bounding box center: bbox_cx = (x1+x2)/2, bbox_cy = (y1+y2)/2
    # 4. Get image center: img_cx = width/2, img_cy = height/2
    # 5. Compute offset: offset_x = bbox_cx - img_cx, offset_y = bbox_cy - img_cy
    # 6. Decide guidance:
    #    if abs(offset_x) < 30 and abs(offset_y) < 30 → "Good — finger centered"
    #    elif offset_x < -30 → "Move right"
    #    elif offset_x > 30  → "Move left"
    #    elif offset_y < -30 → "Move down"
    #    elif offset_y > 30  → "Move up"
    # 7. Return in_roi, offset_x, offset_y, guidance
```

#### Task 2 — `backend/benchmark.py` (new file)

Create a new file `backend/benchmark.py`:
```python
# Load all users from uidai.db
# For users with multiple enrollments → genuine pairs
# Cross-user pairs → impostor pairs
# Run match_templates() on all pairs
# Compute FAR, FRR at thresholds 0.05 to 0.95 (step 0.05)
# Find EER (where FAR ≈ FRR)
# Save results to benchmark_results.csv
# Plot FAR/FRR curve and save to eer_plot.png
```

Run it as a standalone script:
```bash
cd backend
source venv/bin/activate
python benchmark.py
```

#### Task 3 — QC Desktop Tool Tab 2 + Tab 4

In `backend/qctool/app.py`, you own:
- **Tab 2: Readiness Scoring** — upload image, call `/readiness`, show score + grade + breakdown bar chart
- **Tab 4: Contact Upload** — upload contact image, call `/process_contact` (Ashish builds this), show preprocessed result

---

### 👤 Ashish

**You work in:** `backend/app.py` + `backend/qctool/app.py`  
**Your branch:** `feature/ashish-iso-qctool`  
**You do NOT touch:** Flutter files

#### Task 1 — Contact-based preprocessing in `backend/app.py`

Port from `contactless_attendance/backend/app.py` → look for `preprocess_contact_fingerprint()`.

```python
def preprocess_contact_fingerprint(image_bgr):
    # 1. Convert to grayscale
    # 2. Otsu threshold to remove gray background
    # 3. CLAHE for local contrast enhancement
    # 4. Multi-orientation Gabor filtering (8 orientations)
    # 5. Resize to match contactless scale (256x256)
    # Return: enhanced grayscale image

@app.route("/process_contact", methods=["POST"])
def process_contact():
    # Accept image, run preprocess_contact_fingerprint()
    # Return preprocessed image as base64
```

#### Task 2 — ISO/IEC 19794-4 template export in `backend/app.py`

```python
def export_iso_template(minutiae, image_width, image_height):
    # Build ISO 19794-4 binary record
    # Header: format identifier (4 bytes) + version (4 bytes)
    # Capture equipment: 2 bytes
    # Image size: width (2 bytes) + height (2 bytes)
    # Resolution: 2 bytes each
    # Finger count: 1 byte
    # For each minutiae point:
    #   type (2 bits) + x (14 bits) + y (14 bits) + angle (8 bits) + quality (8 bits)
    # Return bytes object

@app.route("/export_template", methods=["POST"])
def export_template():
    # Accept uid + batch
    # Fetch template from DB
    # Call export_iso_template()
    # Return base64-encoded ISO binary
```

Reference: https://www.iso.org/standard/50864.html  
Also look at: https://github.com/usnistgov/NFIQ2 for format examples

#### Task 3 — QC Desktop Tool Tab 1 + Tab 3

In `backend/qctool/app.py`, you own:
- **Tab 1: Image Quality** — upload image, call `/quality_check`, show blur/brightness/glare with color indicators
- **Tab 3: Contact vs Contactless Comparison** — upload both, call `/process_contact` + `/process`, show SSIM/ORB/SIFT side by side

Copy `compare_images.py` and `annotate.py` from IIT Bombay repo (`qctool` branch) into `backend/qctool/`.

Start the QC tool:
```bash
cd backend/qctool
pip install streamlit scikit-image scikit-learn
streamlit run app.py
# Opens at http://localhost:8501
```

---

### 👤 Ishita

**You work in:** `uidai_app/lib/` only  
**Your branch:** `feature/ishita-ui`  
**You do NOT touch:** `backend/` files

#### Task 1 — Shimmer loading states

Install shimmer package — add to `pubspec.yaml`:
```yaml
shimmer: ^3.0.0
```

Then run `flutter pub get`.

On every screen that calls the backend, while `_loading == true`, show a shimmer placeholder instead of blank space. Example pattern:
```dart
if (_loading)
  Shimmer.fromColors(
    baseColor: YS.stroke,
    highlightColor: YS.cardAlt,
    child: Container(height: 80, decoration: BoxDecoration(
      color: YS.card, borderRadius: BorderRadius.circular(14))),
  )
```

Files to update: `enroll_screen.dart`, `authenticate_screen.dart`, `screens.dart` (Verify, QC, History, Settings)

#### Task 2 — Server offline error state

On every screen, if the API call throws a connection error, show:
```dart
// Instead of crashing or showing blank
Container(
  padding: EdgeInsets.all(20),
  child: Column(children: [
    Icon(Icons.wifi_off_rounded, color: YS.inkLight, size: 48),
    SizedBox(height: 12),
    Text('Cannot reach server', style: YS.label(14, color: YS.inkMid)),
    Text('Check Settings → Server URL', style: YS.label(12, color: YS.inkLight)),
    SizedBox(height: 16),
    ElevatedButton(onPressed: _retry, child: Text('Retry')),
  ]),
)
```

#### Task 3 — Wire readiness score into QC screen

After Nandhini finishes `/readiness`, update `QcScreen` in `screens.dart`:
- Add a call to `ApiService.process()` which already returns quality data
- Display the readiness score as a large number with grade badge
- Show a breakdown bar for each component (blur, brightness, glare, minutiae)

#### Task 4 — Gesture liveness screen

After Nandhini finishes `/liveness_gesture`, create `lib/screens/gesture_liveness_screen.dart`:
- Show a number (1–5) as a challenge: "Show [N] fingers"
- Use front camera
- Capture image → POST to `/liveness_gesture` with `expected_count`
- Show pass/fail result
- Wire into enroll flow: before fingerprint capture, user must pass gesture challenge

---

### 👤 Atharv

**You work in:** `uidai_app/lib/widgets/` + `uidai_app/lib/` + GitHub  
**Your branch:** `feature/atharv-autocapture`

#### Task 1 — GitHub Setup (do this first, Day 1)

```bash
cd yellowsense-uidai

# Initialize git
git init
git add .
git commit -m "feat: initial project — Flutter app + Flask backend"

# Create repo on GitHub (yellowsense org or your personal account)
# Then:
git remote add origin https://github.com/yellowsense/yellowsense-uidai.git
git branch -M main
git push -u origin main

# Create develop branch
git checkout -b develop
git push origin develop

# Add collaborators on GitHub:
# Settings → Collaborators → Add: Ishita, Ashish, Vignesh, Nandhini

# Protect main branch:
# Settings → Branches → Add rule → main
# ✅ Require pull request before merging
# ✅ Require status checks (select Flutter CI)
# ✅ Require 1 approval
```

#### Task 2 — Auto-capture in `fingerprint_camera_widget.dart`

The camera widget currently only captures when the user taps "Capture". You need to add:

1. A `Timer` that fires every 600ms while camera is live
2. On each tick, take a low-res frame and send to `/quality_check`
3. Parse the `guidance` string from response
4. Show guidance text on the viewfinder overlay
5. If `passed == true` for 3 consecutive frames → auto-trigger `_capture()`

Key additions to `_FingerprintCameraWidgetState`:
```dart
Timer? _qualityTimer;
String _guidance = 'Place finger inside oval';
int _passCount = 0;
bool _autoCapture = true; // can be toggled by user

// Start timer when camera goes live
void _startQualityPolling() {
  _qualityTimer = Timer.periodic(Duration(milliseconds: 600), (_) async {
    if (!_live || _ctrl == null) return;
    // Take frame → send to /quality_check → update _guidance
    // If passed 3 times in a row → _capture()
  });
}
```

#### Task 3 — Wire ROI guidance

After Vignesh finishes `/check_roi`, add ROI guidance to the camera overlay:
- Call `/check_roi` in the same polling loop
- Show "Move left / Move right / Move up / Move down" on the viewfinder
- Only auto-capture when BOTH quality AND ROI pass

---

## Part 4 — Testing Your Work

### Backend (Ashish · Vignesh · Nandhini)

Test every endpoint you build with curl before raising a PR:

```bash
# Test /readiness
curl -X POST http://localhost:5001/readiness \
  -F "image=@backend/uploads/any_fingerprint.jpg"

# Test /check_roi
curl -X POST http://localhost:5001/check_roi \
  -F "image=@backend/uploads/any_fingerprint.jpg"

# Test /process_contact
curl -X POST http://localhost:5001/process_contact \
  -F "image=@backend/uploads/any_fingerprint.jpg"

# Test /export_template
curl -X POST http://localhost:5001/export_template \
  -F "uid=test123" -F "batch=BATCH_A"

# Test /liveness_gesture
curl -X POST http://localhost:5001/liveness_gesture \
  -F "image=@backend/uploads/any_fingerprint.jpg" \
  -F "expected_count=3"
```

Also test that existing endpoints still work after your changes:
```bash
curl http://localhost:5001/health
```

### Flutter (Atharv · Ishita)

Before raising a PR, always run:
```bash
cd uidai_app
flutter analyze          # must show 0 errors
flutter build apk --debug  # must succeed
```

Then test on a physical device or emulator.

---

## Part 5 — Common Issues & Fixes

### "No module named X" (backend)
```bash
# Make sure venv is activated
source venv/bin/activate   # Mac
venv\Scripts\activate      # Windows
pip install X
```

### "flutter: command not found"
- Mac: add `export PATH="$PATH:/path/to/flutter/bin"` to `~/.zshrc` then `source ~/.zshrc`
- Windows: add Flutter `bin` folder to System Environment Variables → PATH

### "Merge conflict" in git
```bash
# When you run git merge develop and see conflicts:
# Open the conflicted file in VS Code
# You'll see <<<<<<< HEAD ... ======= ... >>>>>>> develop
# Keep the correct version, delete the markers
# Then:
git add .
git commit -m "fix: resolve merge conflict"
```

### Backend server crashes on startup
- Check all model files are in `backend/` folder
- Check venv is activated
- Run `python -c "import torch; print(torch.__version__)"` — if error, reinstall torch

### App says "Server unreachable"
- Make sure backend is running (`bash start.sh`)
- Check phone and laptop are on same WiFi
- In app Settings, verify URL is `http://<laptop-IP>:5001`
- Try `curl http://<laptop-IP>:5001/health` from your phone's browser

---

## Part 6 — Daily Standup Format

Every day, each person posts in the group chat:

```
✅ Done yesterday: [what you finished]
🔨 Doing today: [what you're working on]
❌ Blocked by: [anything stopping you, or "nothing"]
```

---

*YellowSense Technologies · SITAA Cohort 1 · 2025*  
*Questions? Ask Atharv.*
