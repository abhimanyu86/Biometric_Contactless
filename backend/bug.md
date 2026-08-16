# Slap (multi-finger) flow — bug tracker

Found while extending the v2 cloud-only endpoint pattern to `slap_app.py`/`slap_core.py`
and load-testing it against real data. Each item is either **fixed** in this pass or
**tracked** (real issue, not code-fixable in a single pass — needs data, a design call, or
comes later in the plan's own sequencing).

---

## 1. `/enroll_slap` never checked liveness — FIXED

**Where:** `enroll_slap_endpoint`, `slap_app.py`.

**What:** `check_liveness()` runs per finger inside `process_one_finger` and the result is
stored in `finger["liveness"]`, but the enroll handler only checked `finger["ok"]` (= "no
exception was thrown") before saving to the DB — never `liveness.is_live`.
`authenticate_slap_endpoint` and `verify_slap_endpoint` both correctly fail closed on
liveness; enroll didn't. A photo of a photo (or any other spoof) could be enrolled as
someone's slap template, no different from a real capture as far as the pipeline was
concerned.

**Fix:** enroll now rejects the whole request (422, `spoof_detected: true`) if any finger
fails liveness, mirroring the existing pattern in authenticate/verify exactly.

**Severity:** real security gap. Fixed.

---

## 2. Position-agnostic best-match + top-2 average — FIXED

**Where:** `_best_finger_match`, `slap_app.py` — shared by `/authenticate_slap`,
`/verify_slap`, and the new `/v2/authenticate_slap`, `/v2/verify_slap`.

**What:** for a given probe finger, this matched against *whichever* stored finger for that
user scored highest — the stored finger's `finger_position` was returned for display but
never used to constrain the comparison. Combined with top-2-of-N averaging, this gives an
impostor multiple independent chances to get lucky: a probe with 4 fingers gets checked
against every stored position for every candidate user, and only the best 2 of those (up to
4×stored-count) comparisons count toward the aggregate score. Confirmed empirically: an
impostor test crossed the 0.25 threshold (aggregate 0.418) using only 1 stored finger,
because the impostor's *other* 3 unrelated probe fingers were still free to search that one
stored template.

**Fix:** `_best_finger_match` now takes the probe's own `finger_position` and only compares
against stored templates at that *same* position. A probe finger with no stored counterpart
at its position simply contributes nothing for that user, rather than being free to match
anything. Applied to all four call sites (old and new authenticate/verify).

**Note — this does not fix threshold calibration (see #3).** It closes the
"more attempts than intended" gap; it does not change what a single same-position genuine-vs
-impostor comparison scores. Don't expect an impostor test built from unrelated real people's
images to cleanly fail post-fix — that's a #3 problem, not a #2 problem.

**Severity:** real correctness gap, worse than single-finger's equivalent risk because slap
aggregates across more attempts. Fixed at the matching layer; behavior of the *existing*
`/authenticate_slap`/`/verify_slap` changes too, disclosed here since it's shared code.

---

## 3. `MATCH_THRESH = 0.25` uncalibrated — TRACKED, not fixed here

Same guessed constant as single-finger's `THRESHOLD`/`MATCH_THRESH`, hardcoded in
`slap_app.py:303` and used nowhere else is it derived from a measured curve. This needs real
labelled data (the labtool Metrics-tab workstream from earlier), not a code change — picking
a new number without data would just be a different guess. Per the plan's addendum, slap's
FAR/FRR needs to be measured as its own curve (aggregate score across N fingers), not assumed
to match single-finger's eventual calibrated threshold.

**Severity:** known, already on the roadmap. Not attempted in this pass.

---

## 4. `/quality_check` for slap faked glare + ROI — FIXED

**Where:** `quality_check_endpoint`, `slap_app.py`.

**What:** the response included `"glare": {"has_glare": False}` and `"in_roi": True,
"roi_guidance": ""` — both hardcoded, never computed. No glare check ran at all (no model,
no pixel fallback), and no ROI check ran despite the response shape implying one had.
Response looked identical in shape to single-finger's real (if imperfect) version of both
fields.

**Fix:** glare now uses the same pixel-overexposure fallback ratio single-finger's
`check_glare` falls back to when no model is available (`gray > 240` fraction `> 0.05`).
ROI now actually checks that every detected finger's bounding box sits inside the frame with
a small margin (2% of width/height) rather than touching the edge, with real guidance text
when it doesn't.

**Severity:** real gap — client code trusting these fields got a false "all clear" for both
signals on every request. Fixed with the same-cost fallback single-finger already uses; not
a full parity rebuild of single-finger's model-backed glare detector or its central-ROI logic.

---

## 5. No FIR path for slap at all — FIXED

**Where:** new `POST /export_fir_slap` in `slap_app.py`.

**What:** `slap_app.py`/`slap_core.py` never imported `fir.py` at all — no FIR creation
existed for the slap flow in any form. `fir.py` itself also hardcodes
`number_of_fingers = 1` in the record it writes, so this wasn't just unwired, the on-wire
format didn't structurally support multiple fingers per record either.

**Design decision made:** one independent, standard-conformant FIR record **per finger**,
not one record extended to carry four. Reuses `fir.py`'s encoder/decoder completely
unchanged — no byte-format changes, so nothing about the already-working single-finger FIR
path is at risk. `label_fingers()`'s existing `iso_code` (1-10) maps directly onto
`encode_fir()`'s `finger_position` parameter with zero translation, since both already use
the same ISO/IEC 19794-4 finger-position codes.

**New endpoint**: `POST /export_fir_slap` — fields `image`, `hand_side`, optional
`finger_order`/`source_dpi`. Detects every finger, and for each one: segment, preprocess,
normalize to 500 DPI, encode, then self-decode to prove it round-trips (same self-check
`/export_fir` already does) — returns a `records: [...]` list, one entry per finger, plus
an `errors: [...]` list for any finger that failed independently (one bad finger doesn't
kill the whole slap, matching `process_slap`'s existing per-finger error isolation).

**Tested against real data**: ran a real dataset photo through it — produced a
standards-conformant record (500 DPI, real ridge-frequency-based normalization, scale
factor 1.159), correctly tagged `iso_code: 5` / `finger_position: "right_little"` from
`label_fingers()`, and the record decoded back through `fir.decode_fir()` without error.
Only one finger showed up because there's no real 4-finger slap photo on this machine to
test with — the per-finger loop, encoding, and round-trip decode are all verified; a real
slap capture would just produce more entries in the same `records` list.

**Severity:** was a real, structural gap (not just missing wiring). Fixed, additive only —
`fir.py`'s format and the existing single-finger `/export_fir` are untouched.

---

## 6. Model directory split between the two flows — FIXED (config only)

**Where:** `app.py` model path constants.

**What:** `app.py` loaded every weight from `BASE_DIR` (`backend/`) with no override;
`slap_core.py` already supported `SLAP_MODELS_DIR` to redirect to a shared folder. CLAUDE.md's
documented workaround was "put all seven files in `backend/` and point `SLAP_MODELS_DIR` at
that same path" — i.e. duplicate-by-convention, not real configurability.

**Fix:** `app.py` now reads `SINGLE_MODELS_DIR` (default: unchanged, `BASE_DIR`) the same way
`slap_core.py` reads `SLAP_MODELS_DIR`. Default behavior is identical for anyone not setting
the env var; a deployer can now point both flows at one real shared directory
(`SINGLE_MODELS_DIR=/shared/models SLAP_MODELS_DIR=/shared/models`) instead of keeping two
copies of ~150MB of weights in sync by hand.

**Severity:** deployment footgun, not a runtime bug. Fixed at the config level.

---

## 7. Blur/brightness thresholds differ between the two flows — TRACKED, not fixed here

Slap's `/quality_check` uses `blur<60.0` / `dark<55` / `bright>205`; single-finger's
`check_blur`/`check_brightness` use `<20.0` / `<50` / `>210`. Might be intentional (slap
frames have more content, different capture geometry); might be drift. Not changing this
without knowing which — flagging so it's a deliberate call, not silent inconsistency.

**Severity:** unclear — needs a product decision, not a code fix. Not touched.

---

## 8. `authenticate_slap` and `verify_slap` used different aggregation formulas — FIXED (consolidated)

**Where:** found while extracting `_best_finger_match` into `slap_matching.py` for
the cloud service.

**What:** `authenticate_slap`'s aggregate score averaged only the fingers that
actually matched something (`matched_positions`, zero-scores excluded before
taking top-2). `verify_slap` averaged *all* probe fingers including zero-scores
(`all_scores`), which silently dilutes a genuine match whenever any presented
finger didn't match anything — a real 1:1 verify attempt could score lower than
it should have purely because of this, not because of the person's actual
fingerprint quality.

**Fix:** `slap_matching.score_against_user()` is now the one implementation both
callers use, standardized on the less-surprising behavior (ignore non-matching
fingers, don't score them as a hard zero). This makes `verify_slap`'s effective
scoring slightly more lenient than it was before — disclosed here rather than
carried forward as an unexplained behavior change. `cloud_app.py`'s `/v2/*` slap
endpoints use this from the start; `slap_app.py`'s original `/authenticate_slap`/
`/verify_slap` still have their own two separate (still-inconsistent) copies,
since fixing this in the on-device-tier reference server wasn't in scope here —
worth doing as a follow-up so there's truly one implementation, not two-fixed-plus-
two-still-inconsistent.

**Severity:** real inconsistency, silently changed verify's behavior for years
without anyone deciding it should. Fixed in the new shared module; old copies in
`slap_app.py` untouched, flagged for follow-up.

---

## 9. Slap FMR export not wired into the cloud service — KNOWN LIMITATION

**Where:** `cloud_app.py`'s `/v2/enroll_slap`.

**What:** the on-device-tier reference server's `slap_core.export_iso_template()`
(builds the ISO 19794-2 FMR byte record) isn't importable into `cloud_app.py`
without pulling in `slap_core.py`'s entire dependency tree (torch/tensorflow/
ultralytics/mediapipe) — exactly what the cloud split exists to avoid. `slap.db`'s
`template_b64` column is written as an empty string for now; `minutiae_json` (what
matching actually reads) is populated correctly and unaffected. Not a functional
gap for enroll/authenticate/verify — a real gap if anything ever needs to export
an ISO-conformant FMR record from a cloud-side slap enrollment.

**Severity:** disclosed limitation, not a bug — matching doesn't use `template_b64`
at all. Would need a standalone FMR-encoding function with no on-device-tier
dependencies to close, mirroring how `minutiae.py` was pulled out on its own.

---

## 10. `cloud_app.py`'s DB never initialized under gunicorn — FIXED

**Where:** `cloud_app.py`, caught during the actual live deployment, not before.

**What:** `init_db()`/`init_slap_db()` were called inside `if __name__ ==
"__main__":`, matching the pattern `app.py`/`slap_app.py` already use. Those two
are always launched via `python app.py` directly, so the guard never mattered.
`cloud_app.py` is served by gunicorn (`gunicorn ... cloud_app:app`, per the
Dockerfile) — gunicorn *imports* the module rather than executing it, so
`__name__` is `"cloud_app"`, not `"__main__"`, and the guarded block silently
never ran. First live `/v2/enroll` against the deployed service failed with
`no such table: users` — no tables had ever been created.

**Fix:** moved both init calls to module level, unconditional on import. Both are
idempotent (`CREATE TABLE IF NOT EXISTS`), so calling them on every import/worker
start is safe. Confirmed by redeploying and re-running the full enroll →
authenticate → verify sequence against the live HTTPS endpoint — all three
succeeded.

**Severity:** would have silently broken the entire deployed service on first
real use — caught by testing against the actual gunicorn-served container, not
by unit tests or local `python cloud_app.py` runs (which don't hit this path).
Worth remembering for any future file served the same way.

---

## 11. `slap_app.py`'s history table never initialized under gunicorn — FIXED

**Where:** `slap_app.py`, caught deploying to `biometric`'s `slap_backend`.

**What:** same class of bug as #10, found again. `wsgi.py` (the gunicorn
entrypoint) calls `init_slap_db()` but not `init_slap_history_db()` — because
when `wsgi.py` was written, the deployed `slap_app.py` had no
`authenticate_slap`/`verify_slap` at all (confirmed: the version running on
`biometric` before this session only had `/health`, `/quality_check`,
`/process_slap`, `/enroll_slap` — no matching logic was wired in at all).
First real `/authenticate_slap` call against the redeployed code failed with
`no such table: slap_auth_history`.

**Fix:** same pattern as #10 — moved `init_slap_db()` and
`init_slap_history_db()` to module level in `slap_app.py`, unconditional on
import, rather than patching `wsgi.py` to remember a second function call.
`cloud_app.py` never had this specific gap (its own `init_slap_db()` creates
both tables in one function), but this makes `slap_app.py` self-initializing
regardless of which entrypoint runs it — matters since `biometric` runs it via
gunicorn+wsgi.py while local dev runs it via `python slap_app.py` directly.

**Severity:** would have silently broken slap authentication on first real use,
same as #10. Two occurrences of the identical bug class is a real pattern —
worth checking any future Flask module for this before assuming `if __name__`
init is sufficient.

---

## 12. Liveness gate rejects real photos on `biometric`'s slap deployment — NOT A BUG, confirms an already-known issue

**Where:** observed testing the redeployed slap backend on `biometric`.

**What:** `enroll_slap`/`verify_slap` against two real dataset photos (UID002,
UID003) were rejected with `spoof_detected: true` at liveness confidence 0.008
and 0.11. This is **expected, not a regression** — it's the liveness-model
miscalibration flagged earlier in this project's work (the same photos scored
near-zero liveness when first checked against `lab_gallery.db`). What changed
is that slap's liveness gate is now actually enforced for the first time — it
never checked anything before (see #1, #11's context) — so a pre-existing data/
calibration problem that was previously invisible on the slap path is now
visible, correctly, because the check is finally running.

**Not fixed here** — this is the liveness model calibration work already
identified as a separate, larger workstream. Recorded so it isn't mistaken for
something broken by this deployment.

---

## Final pre-integration review — 8 more findings, all fixed

Before handing this off, ran a fresh-eyes review pass (4 independent readers, one per
file group, each finding independently re-verified against the real code and — for
anything with a "this would break in production" claim — against the actual running
deployment before being trusted). Found 8 real issues, all fixed and redeployed to both
`biometric` and `sitaa-cloud-vm`, verified live afterward.

## 13. `enroll_slap_v2`'s DB connection not wrapped in try/finally — FIXED

**Where:** `cloud_app.py`. Every other DB-touching route in the file wraps its
`sqlite3.connect()` in try/finally; this one didn't. If `detect_minutiae()` or a
write raised mid-loop, the connection leaked instead of closing, risking held
locks under repeated failures. Fixed to match the file's own established pattern.

## 14. `matching.py`'s eager algorithm-registry import needs `networkx`, and the requirements comment said otherwise — FIXED (not a live outage, but a real fragility)

**Where:** `matching.py` imports the whole `algorithms/` package unconditionally
at module load (to populate the registry), and `algorithms/networkx_graph.py`
does a bare `import networkx` — so it's a real *import-time* dependency of
`matching.py` regardless of which algorithm is ever selected at runtime.
`requirements-cloud.txt`'s comment claimed networkx was safely excludable
because only the bake-off uses it — that reasoning conflates runtime selection
with import-time module loading, and is wrong.

**Important correction to how this was first reported:** an initial pass
concluded this meant the deployed cloud service was currently crashing on
startup. That's false — checked directly against the real running container,
and `networkx` was already present the whole time, pulled in transitively by
`torch` (`pip show networkx` reports `Required-by: torch`). Nothing was down.
The real issue is that this safety net was implicit and unpinned — a future
torch version could drop that dependency and silently break the service with
no warning. Fixed by pinning `networkx` explicitly in `requirements-cloud.txt`
and correcting the comment's reasoning.

**Severity:** real latent fragility, not a live incident. Fixed either way —
worth calling out because getting the actual current-state severity right
matters as much as finding the issue.

## 15. `_build_graph`'s "drop self" logic could self-loop on duplicate-coordinate minutiae — FIXED

**Where:** `matching.py`, used by the `"modern"` and `"hungarian_assignment"`
algorithms (confirmed **not** used by `"legacy"`, which has its own separate
`_legacy_build_graph` — so this doesn't touch the current production config).
`for j in idxs[i][1:]:` assumed a KD-tree query always returns a point as its
own nearest neighbor (rank 0). When two minutiae share exact coordinates,
cKDTree can return the duplicate at rank 0 instead, so the slice drops the
real duplicate and leaves the node listing itself as a neighbor — a self-loop
edge that corrupts the k-NN graph relaxation labeling depends on.

**Fix:** filter by index (`if j == i: continue`) instead of assuming position
0 is self — correct in both the normal and duplicate-coordinate case. Verified
directly with a synthetic template containing duplicate coordinates.

**Severity:** real bug, currently dormant given `"legacy"` is what's selected in
production — but a live risk the moment matching is recalibrated to `"modern"`.

## 16. `slap_core.check_liveness()` still failed open — FIXED

**Where:** `slap_core.py`. `app.py`'s equivalent was deliberately changed to
fail closed a while back ("this previously returned is_live=True when the
model was absent, which meant a deployment with missing weights silently
accepted every spoof"). `slap_core.py`'s copy was never given the same fix —
it still silently reported every capture as live if the liveness model wasn't
loaded, which would have quietly defeated this session's own new `enroll_slap`
liveness gate (#1) under exactly the failure condition that gate exists for.

**Fix:** applied the identical fail-closed pattern from `app.py`, including
the `ALLOW_LIVENESS_BYPASS` escape hatch for local dev.

**Severity:** real security gap, same class as #1 — just in the one place #1's
fix didn't reach. Fixed and redeployed.

## 17. Slap aggregation was still duplicated in FOUR places, not fixed everywhere it needed to be — FIXED

**Where:** `slap_app.py`. Turns out the #8 fix (consolidating into
`slap_matching.score_against_user()`) only ever got wired into `cloud_app.py`
— the original `/authenticate_slap`, `/verify_slap`, and this file's own
`/v2/authenticate_slap`, `/v2/verify_slap` were all still running the old
duplicated inline logic (including `/verify_slap`'s dilution bug this whole
fix was supposed to close).

**Fix:** all four now call `slap_matching.score_against_user()`. Removed the
now-fully-unused local `_best_finger_match()` and the now-unused
`from matching import match_templates` import, so this logic genuinely lives
in one place.

**Severity:** the actual live deployment (`biometric`'s reference server) was
running the un-fixed version this whole time despite the fix existing in the
codebase. Fixed and redeployed.

## 18. `export_fir_slap` could throw a raw TypeError on a 5th detected finger — FIXED

**Where:** `slap_app.py`. `label_fingers()` returns `iso_code: None` for any
detected finger beyond the 4-slot `SLAP_ORDER`/`finger_order` list (a
mis-detection, a stray thumb, etc.). Passing `None` into
`fir.encode_fir(finger_position=...)` throws `TypeError` before it even
reaches `fir.py`'s own range check — caught by the per-finger `except
Exception`, so it doesn't crash the whole request, but produces an unhelpful
raw Python error instead of a clean message for that one finger.

**Fix:** fall back to `0` (`fir.py`'s own defined "unknown" code) when
`iso_code` is `None`, matching how `/export_fir`'s own default already works.

## 19. Two stale/misleading comments — FIXED

- `app.py`'s `run_quality_gate()` docstring claimed it was "the same logic as
  /quality_check" — it's the opposite: the heavier, U²Net-masked path used at
  actual enroll/authenticate/verify time, while `/quality_check` is the
  deliberately lighter polling variant. Corrected.
- `/compare_contact` (an unrelated contact-vs-contactless comparison feature)
  was still calling `match_templates()` without `algorithm="legacy"`, silently
  using `matching.py`'s `"modern"` default — inconsistent with "legacy
  throughout" even though this endpoint isn't part of the cL2cL matching flow.
  Added for consistency.

## 20. Dead `import base64` in `cloud_app.py` — cleaned up

Genuinely unused (found and confirmed by direct grep) — removed while in the
file for the fixes above. Not a bug, just hygiene.

---

## Summary

| # | Issue | Status |
|---|---|---|
| 1 | `/enroll_slap` skips liveness | **Fixed** |
| 2 | Position-agnostic best-match + top-2 avg | **Fixed** |
| 3 | `MATCH_THRESH=0.25` uncalibrated | Tracked — needs data |
| 4 | `/quality_check` fakes glare/ROI | **Fixed** |
| 5 | No FIR path for slap | **Fixed** — `POST /export_fir_slap`, one record per finger |
| 6 | Model directory split | **Fixed** — config only |
| 7 | Blur/brightness threshold mismatch | Tracked — needs a product decision |
| 8 | authenticate_slap vs verify_slap used different aggregation formulas | **Fixed** — consolidated in slap_matching.py (cloud_app.py only; slap_app.py's originals still inconsistent) |
| 9 | Slap FMR export not wired into cloud service | Known limitation — matching unaffected, template_b64 empty for now |
| 10 | cloud_app.py DB never initialized under gunicorn | **Fixed** — caught during live deployment, moved init to module level |
| 11 | slap_app.py's slap_auth_history never initialized under gunicorn | **Fixed** — same pattern as #10, caught deploying to biometric |
| 12 | Liveness gate rejects real photos on biometric's slap deployment | Not a bug — confirms the already-known liveness calibration issue, now actually enforced |
| 13 | enroll_slap_v2 DB connection not wrapped in try/finally | **Fixed** |
| 14 | Eager networkx import undeclared in requirements-cloud.txt | **Fixed** — pinned explicitly; was never actually down (transitively present via torch) |
| 15 | _build_graph self-loop on duplicate-coordinate minutiae | **Fixed** — dormant under "legacy", live risk if ever recalibrated to "modern" |
| 16 | slap_core.check_liveness still failed open | **Fixed** — real security gap, same class as #1 |
| 17 | Slap aggregation fix (#8) never reached 3 of the 4 places it needed to | **Fixed** — biometric's live deployment was running the un-fixed version |
| 18 | export_fir_slap TypeError on a 5th detected finger | **Fixed** |
| 19 | Two stale/misleading comments (run_quality_gate docstring, compare_contact algorithm) | **Fixed** |
| 20 | Dead import base64 in cloud_app.py | Cleaned up |
