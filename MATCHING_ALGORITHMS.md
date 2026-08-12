# Matching algorithms — candidates and results

`backend/matching.py` (plus `backend/algorithms/`) is the single place every
fingerprint-matching algorithm lives. This doc catalogues what's there, what was
considered and ruled out, and the results as measured — not assumed — against both
synthetic data and the team's real gallery.

Ten algorithms are registered today, numbered `algorithm000`–`algorithm009`, all
living together on one branch (`algo/bakeoff`) rather than split across one branch
per algorithm — the point of the bake-off is comparing them side by side, and a
registry that already supports N algorithms selectable by name makes that a
one-command sweep instead of ten branch checkouts. labtool's sidebar, Identify tab,
and Metrics tab all read this same registry live.

## The data contract every algorithm has to honour

Confirmed at every call site in the repo (`app.py`, `slap_app.py`, `benchmark.py`,
`labtool/rank.py`): a matcher receives **only** two `list[dict]` minutiae sets —
`{"x": float, "y": float, "direction": float radians, "type": "RIG"|"BIF"}` — and
returns a single `float` in `[0, 1]`. No algorithm here ever sees a raw or
preprocessed image. Anything that needs pixel data (texture correlation, deep
embeddings) is out of scope for a same-registry addition; see "Considered, out of
scope" below.

## How to add one

`match_templates(t1, t2, algorithm="modern")` dispatches through a name registry in
`matching.py`:

```python
_ALGORITHMS = {
    "legacy": _match_legacy,
    "modern": _match_modern,
    "hungarian_assignment": _match_hungarian,
    "mcc_cylinder": algorithms.mcc_cylinder.match,
    ...
}
ALGORITHM_INFO = {
    "mcc_cylinder": {"code": "algorithm003", "label": "Minutia Cylinder-Code (MCC)"},
    ...
}
```

1. Write `backend/algorithms/<name>.py` exporting `match(t1, t2) -> float`, fully
   self-contained (no imports from `matching.py` or other `algorithms/*.py` modules —
   see `backend/algorithms/__init__.py`). The one exception is an algorithm that
   deliberately reuses `matching.py`'s private graph/relaxation helpers (like
   `hungarian_assignment` does) — that goes directly in `matching.py` instead.
2. Register it: one line in `_ALGORITHMS`, one entry in `ALGORITHM_INFO` with the next
   `algorithm0XX` code.
3. **Don't assume which properties it has — measure them.** Add its name to
   `CANDIDATE_ALGORITHMS` in `backend/tests/test_matching.py` (general sanity:
   identical/empty/count-ratio/translation/discrimination — every algorithm here is
   expected to pass these), then run the full suite and add it to whichever of
   `ROTATION_INVARIANT_ALGORITHMS`, `SCALE_ROBUST_ALGORITHMS`,
   `UNIFORM_SCALE_INVARIANT_ALGORITHMS`, `RECAPTURE_ROBUST_ALGORITHMS`, and
   `ROTATED_JITTER_DISCRIMINATES_ALGORITHMS` it actually passes. These five lists
   turned out to have genuinely different membership — two algorithms below pass
   plain rotation invariance but fail the harder combined partial-overlap test, and
   two others pass single-outlier robustness but fail uniform rescaling. Don't force
   a convenient grouping; a failing parametrized test for an algorithm that doesn't
   have a property is a real, useful signal.
4. `python -m pytest tests/ -v`.
5. `python benchmark.py --algorithm <name>` for FAR/FRR/EER on synthetic data.
6. It now shows up automatically everywhere in labtool: the sidebar's algorithm
   selector, the Identify tab's "compare every algorithm on this capture" table, and
   the Metrics tab's live leaderboard against the real gallery — zero UI code changes
   needed per algorithm.
7. Fill in its row in the results table below.

## Baselines

| Algorithm | Code | Mechanism |
|---|---|---|
| `legacy` | algorithm000 | Absolute-space edges, bounding-box-span scaling, node compatibility gated on absolute position (≤25px) and orientation, Dice-coefficient score. Pre-rotation-fix historical baseline, kept for before/after comparison. |
| `modern` | algorithm001 | **Current production default.** KD-tree k-NN graph (k=6), edges measured relative to each minutia's own ridge direction (rotation-invariant), median-nearest-neighbour scaling (outlier-robust), 10-iteration relaxation labeling, greedy 1:1 assignment, coverage+confidence blended score. |

Despite past references to `modern` as "MCC-style", it is **not** the Minutia
Cylinder-Code algorithm — no cylinder structures are built. It's closer to
Ratha/Jain-style local-structure matching with relaxation labeling. `algorithm003`
below is the real MCC, partly to resolve that naming mismatch directly.

## The eight candidates

All minutiae-only, matching the contract above, no image access required.

| Code | Algorithm | Mechanism | New deps |
|---|---|---|---|
| algorithm002 | Hungarian / optimal assignment | Identical front-end to `modern` (same normalisation, scaling, k-NN graph, relaxation labeling) — only the final assignment step changes, from greedy to `scipy.optimize.linear_sum_assignment` (a true globally-optimal 1:1 match). Lives directly in `matching.py`, not `backend/algorithms/`, since it reuses `modern`'s private helpers. | none |
| algorithm003 | Minutia Cylinder-Code (MCC) | Cappelli, Ferrara & Maltoni (2010). A discretized cylinder per minutia (8×8 spatial cells × 5 angular cells over a disc scaled to median nearest-neighbour distance), continuous Gaussian-weighted contributions (not the paper's binarized bit-vectors — a documented simplification), cosine local similarity, consolidated via Local Similarity Sort. | none |
| algorithm004 | Delaunay triangulation matching | `scipy.spatial.Delaunay` per template, rotation/scale-invariant per-triangle features (interior angles, side-length ratios, type multiset), hash-bucketed triangle correspondence, greedy vertex-vote assignment. | none |
| algorithm005 | RANSAC rigid alignment | Exploits that each minutia carries both position *and* direction, so a single matched pair fully determines a rigid transform (no 2-point correspondence needed). Deterministic bounded hypothesis sweep, KD-tree inlier counting. | none |
| algorithm006 | Jiang & Yau (2000) local + global structure | Two-stage: local structure similarity from each minutia's k=3 nearest neighbours (relative distance/angle/type), then global alignment seeded by the top-10 local matches (not random sampling — the paper's actual efficiency trick), inlier counting. | none |
| algorithm007 | Bozorth3-style graph consistency | Re-implements NIST NBIS Bozorth3's core mechanism in pure Python: all-pairs within-template relationships (not just k-NN), cross-template compatibility via distance-bucketed pruning, largest-connected-component consistency search (`scipy.sparse.csgraph.connected_components`) as a tractable stand-in for the paper's clique search. | none |
| algorithm008 | NetworkX graph/spectral matching | k-NN graph via `networkx`, rotation-invariant edge descriptors, Laplacian spectral similarity blended with a one-hop local-correspondence coverage score. Deliberately avoids `graph_edit_distance` (NP-hard, too slow at this scale). | **networkx** |
| algorithm009 | RANSAC v2 (scale-normalised + refit + confidence-ordered) | `algorithm005`'s successor, built after early real-gallery testing singled RANSAC out as the strongest performer. Three targeted fixes over the same core mechanism: (1) both templates pre-normalised to a canonical median-nearest-neighbour spacing before the RANSAC search runs, fixing the one property `algorithm005` measurably lacked; (2) a proper 2D Kabsch/Procrustes least-squares refit over every inlier the winning hypothesis collected, not just its seed pair; (3) candidate seed pairs ranked by combined `confidence` (the per-minutia field `detect_minutiae` emits that no other algorithm reads) in addition to local density, so the right transform is found faster within the same hypothesis budget. Kept as a separate registered algorithm rather than overwriting `algorithm005`, so the leaderboard shows a direct before/after. | none |

## Considered, out of scope

- **SourceAFIS** — excellent, proven, rotation/translation-invariant, open source. Only
  has native Java/.NET implementations, no first-party Python port. Would need JVM/CLR
  interop (`jpype`, `pythonnet`, or a subprocess wrapper around a compiled CLI).
- **DeepPrint / fixed-length deep embeddings** — very fast at scale (0.3s over a
  1.1M-record gallery per published benchmarks, vs 27s for a leading COTS matcher) but
  needs raw/aligned fingerprint images and training data, not just minutiae — breaks
  the contract above at every call site.
- **True NBIS Bozorth3 (the actual C binary)** — deferred in favour of algorithm007, a
  pure-Python reimplementation, specifically to avoid a Windows build/toolchain
  dependency. Worth revisiting if the pure-Python version underperforms and the team
  wants to rule out an implementation gap rather than an algorithmic one.
- **pyfing** (Cappelli's own library) — covers segmentation, enhancement and minutiae
  *extraction*, not matching.

## Results — measured, not assumed

Every property below was verified by actually running the algorithm through
`tests/test_matching.py`'s scenarios — not taken from any algorithm's own
self-description. Two real distinctions fell out of that which weren't obvious going
in: `mcc_cylinder` and `ransac_alignment` both pass single-outlier robustness but fail
*uniform* rescaling (RANSAC's transform model is deliberately rotation+translation
only, with no scale term — a genuine design tradeoff, not a bug); and `delaunay` /
`networkx_graph` both pass a plain rotation-invariance test but fail the harder
combined scenario (rotation *plus* dropped/spurious minutiae) — a documented,
literature-known weakness of pure triangulation matching, and of one-hop local graph
correspondence, respectively, under partial overlap.

| Algorithm | Rotation-invariant | Outlier-robust | Uniform-scale-invariant | Survives partial overlap + rotation | ms/match (~50pt) | Self-match (synthetic) |
|---|---|---|---|---|---|---|
| legacy | ❌ (by design) | ❌ (by design) | ✅ | ❌ | <1 | high |
| modern | ✅ | ✅ | ✅ | ✅ | <1 | high |
| hungarian_assignment | ✅ | ✅ | ✅ | ✅ | <1 | high (rank-1 assignment differs from `modern` rarely — 1/60 stress-tested pairs, and only slightly when it does; the relaxation-labeling front end already resolves most assignment ambiguity before the final step matters) |
| mcc_cylinder | ✅ (exact) | ✅ | ❌ (fails at 1.5×) | ✅ | 10.3 | 0.98 |
| delaunay | ✅ (exact) | ✅ | ✅ | ❌ | 6.0 | 0.99 |
| ransac_alignment | ✅ (exact) | ✅ | ❌ (no scale term) | ✅ | 1.8 | 1.00 |
| jiang_yau | ✅ (exact) | ✅ | ✅ | ✅ | 41.2 | 1.00 |
| bozorth3_style | ✅ (exact) | ✅ | ✅ | ✅ | 65.1 | 1.00 |
| networkx_graph | ✅ (exact) | ✅ | ✅ | partial — rotation+jitter yes, +drop/spurious no | 21.1 | 1.00 |
| **ransac_similarity** | ✅ (exact) | ✅ | **✅ (fixed)** | ✅ | 36.5 | 1.00 |

`ransac_similarity` (algorithm009) is the only algorithm in this table with a full
column of ✅ — every property tested here, it has. That came at two measured costs
versus `ransac_alignment`: ~20x slower per match (36.5ms vs 1.8ms — still far under
the 1-second interactive budget) and a smaller genuine/impostor separation margin
under the hardest partial-overlap scenario (+0.34/+0.34/+0.24 across 20/30/40% drop
vs. the original's +0.67/+0.60/+0.50) — still comfortably above the 0.08 pass bar, but
real: estimating scale from a template that's already missing minutiae is itself
noisier than estimating it from a clean one, and that noise partially offsets the
benefit of fixing scale invariance. On the actual real-gallery leaderboard (9 people,
36 impostor pairs, no genuine data yet), it's a wash either way that early testing
hoped to see: `ransac_similarity`'s mean impostor score (0.109) is on par with
`ransac_alignment`'s (0.105), but its *maximum* impostor score is lower (0.141 vs
0.159) — a small, real, positive signal on top of the synthetic fix.

**Real-gallery leaderboard** (labtool's Metrics tab, live against the team's growing
gallery — 9 people as of this write-up, 36 impostor pairs per algorithm, one photo per
person so no genuine pairs exist yet): all 10 algorithms currently rank purely by mean
impostor score, which is a FAR-floor proxy, not a real accuracy verdict. That becomes a
real, separation-based ranking automatically once probes are tagged with their true
identity on the Identify tab — the leaderboard recomputes from whatever data exists, no
code change needed.

## A note on what "best" means here

The properties above come from synthetic data (`tests/test_matching.py`'s
generators) plus impostor-only real-gallery data — real genuine-pair accuracy is not
yet measurable (one photo per person). Synthetic results are useful for ruling
algorithms in or out on structural grounds (rotation, scale, partial-overlap
robustness) before spending time on real-data calibration; they are not a substitute
for it. Once more captures per person exist and get tagged on the Identify tab, the
Metrics tab's leaderboard and rank-1 accuracy numbers become the real tiebreaker.
