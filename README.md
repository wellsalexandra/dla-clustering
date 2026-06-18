# dla-clustering
Describes how to measure the autocorrelation of damped lyman-alpha absorbers.

---

## Review & corrected implementation (1D LOS DLA clustering)

**Student: start at [`notes/STUDENT_FEEDBACK.md`](notes/STUDENT_FEEDBACK.md)** — one-page entry
point (what you got right, the 4 ranked fixes, and the reading order).

This adds a review of `dla_1d_clustering_updated.ipynb` and a corrected, validated
re-implementation that measures 1D line-of-sight DLA clustering from the 2LPT mock GP catalog,
calibrates purity + completeness, and fits the linear DLA bias.

### `notes/` — written analysis
- **`STUDENT_FEEDBACK.md`** — start here: what you got right, the ranked fixes, the reading order, and what's still open.
- **`coding_bugs.md`** — mechanical/programming bugs in the notebook (C1–C21), independently verified.
- **`science_bugs.md`** — methodology bugs (S1–S10), incl. adjudication of the three suspicions
  (randoms generation wrong; randoms must be ≥10× the data; units wrong) — **all confirmed**.

### `scripts/` — tested engine
- **`clustering_lib.py`** — the corrected estimator (matched DD/RR selection, count-preserving
  window-restricted randoms ≥50×, (1+z)-correct windows, truth-driven `C_pair(Δv)`, pair purity).
- **`measure_1d_dla_clustering.py`** — runs the measurement → `outputs/clustering_2lpt.png`.
- **`fit_dla_bias.py`** — fits the linear bias (EH98 ξ_matter, self-contained) → `outputs/bias_fit.png`.

### Notebooks & outputs
- **`dla_clustering_tutorial.ipynb`** — the method top-to-bottom with the physics (start here for learning).
- **`randoms_tutorial.ipynb`** — building the random catalog correctly (the hardest part), incl. the analytic dN/dX route.
- **`dla_1d_clustering_corrected.ipynb`** — the corrected analysis run end-to-end, validated truth-vs-GP.
- **`outputs/`** — `clustering_2lpt.png` (4-panel: pair counts, ξ, C_pair, purity), `bias_fit.png`, `subdla_version.png`.

### Headline results (2LPT mock-0, loa-124)
- **Method validated on truth:** the 1D estimator recovers an apparent amplitude b_app = 2.20 ± 0.25,
  calibrated to the planted **b = 2.0** (the ~10% offset is RSD + template, calibrated out) — i.e. the
  pipeline works.
- **GP catalog → upper limit only: 2σ b < 2.8.** The clustering signal lives at Δv < 1500 km/s,
  entirely below the GP close-pair sampler floor (C_pair ≈ 0), so the GP measurement cannot detect
  the signal — only bound it. This is a physical limitation of the catalog, not an analysis choice.

This work was independently reviewed from CS, statistics, cosmology, and Lyα-forest perspectives — verdict: sound with minor fixes, all applied.

Run with `conda activate gpdla` (needs numpy/scipy/astropy/fitsio; no pyigm/pycorr).
