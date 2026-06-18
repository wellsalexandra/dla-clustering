# Feedback — 1D DLA clustering

*Your single entry point — and first, nice work: this is a strong first effort. Read this first,
then follow the reading order at the bottom. The short version: the skeleton of your analysis is
right and you had genuinely good instincts about what needed attention — the remaining pieces are
a few specific concepts (mostly **randoms** and **completeness**), and they're all very fixable.*

## What you got right (keep it)
- **Velocity separation** `Δv = c·|z₁−z₂|/(1+z̄)` — correct, mean-z denominator and all. Verified.
- **Pair enumeration** — per-TARGETID, all C(n,2) pairs, no cross-sightline pairs. Correct.
- **Single-DLA purity/completeness machinery** — arithmetically sound; you even (correctly)
  resolved single-DLA completeness in z_DLA.
- **Overall plan** — load → cut → velocity separations → completeness → clustering is the right
  pipeline.

## Your three suspicions — all correct
1. *"The randoms generation is wrong"* → yes (see fix #1).
2. *"Randoms should be ≥10× the data"* → yes; you generated ~1×, so ξ was shot-noise dominated.
3. *"The units are wrong"* → yes; velocity→redshift dropped the (1+z) factor.

Really good scientific instincts. The materials below turn each into a concrete fix.

## The 4 conceptual fixes, ranked by importance

**1. Randoms — the big one.** A random catalog is your *definition of "no clustering"*: it must
match the data in **selection** (per-sightline z-window, global dN/dz, counts) but have
**independent** redshifts. Three rules that are easy to miss:
   - **Same selection for DD and RR** (your `z_max>3: continue` dropped sightlines from RR but not
     DD → ~half the sample mismatched).
   - **Enough randoms.** One realization ⇒ RR as noisy as DD. Average ~50 realizations; the
     random's error falls as 1/√n_real — *that*, not catalog size, is the point.
   - **Right redshift weight.** You drew z with weight ℓ(X) (per absorption distance) on a
     z-uniform grid, which leaves out the dX/dz Jacobian (~18%) — an easy one to overlook. Easiest
     fix: draw from the **empirical dN/dz** of the selected sample — the Jacobian is then automatic.
   → **Read `randoms_tutorial.ipynb`** (built from scratch, with a null test that *proves* the
     randoms inject no fake signal).

**2. Completeness & purity — apply once, to the data, Δv-resolved.** Use the **truth-driven**
pair completeness `C_pair(Δv)` (fraction of true pairs with *both* members recovered — bounded
≤1), not a ratio of two histograms (which can exceed 1, which is why you had to clamp it). Apply
it **once**: `DD_corr = DD_GP · purity(Δv) / C_pair(Δv)`. In the current version completeness enters on *both* sides (the randoms carry a single-DLA completeness; DD is divided by another). Folding it into the randoms is actually the right instinct — it just needs to be **reconciled** (with the single-DLA completeness in the randoms, the matching DD correction is c₁²/C_pair, or renormalize and divide by C_pair only), rather than treated as two independent completenesses. And it works best as a **Δv curve, not a scalar**: C_pair ≈ 0
below the ~850 km/s GP sampler floor, rising to ~0.8 — a single number would erase exactly the
close-pair physics.

**3. False positives, including Lyβ/Lyγ "ghosts."** A strong DLA's Lyβ/Lyγ trough can be mis-fit
as a separate Lyα DLA, creating spurious pairs at fixed Δv ≈ 50,800 / 66,600 km/s. False positives (FPs) also dilute the small-Δv signal. Remove them: `P_DLA>0.99`, `DLAFLAG==0`, `LYBETA_FLAG==False`, drop
BAL sightlines, veto the Lyγ separation — then correct residual dilution with `purity(Δv)`.

**4. Units, binning, and *validate against truth*.** Velocity→redshift carries (1+z):
Δz=(1+z)·v/c, not v/c. Use **fine/log Δv bins** below a few×10³ km/s (the single ~3000 km/s bin
ends up burying the whole signal). And always check against the **mock truth**, where the answer
(b≈2) is known.

## The bottom-line result (so you know the target)
- On the **truth** catalog the corrected pipeline recovers the planted **b ≈ 2** (this is a
  *method validation* — the RSD calibration forces it, so it's not an independent measurement).
- On the **GP catalog** (the Gaussian-process DLA finder) you can only set a **2σ upper limit, b < 2.8** — because the clustering
  signal lives at Δv < 1500 km/s, entirely below the GP close-pair sampler floor. That's a
  physical limit of the catalog, not an analysis choice. *1D LOS clustering of this catalog bounds
  the bias; it can't pin it.*

## Where this goes next — a scaffold, not a finished result

**This package is a reference, not the finished project.** The corrected notebooks and tested
scripts exist so you have a *known-good answer to check yourself against* — scaffolding for your
work, not a substitute for it. Everything below is genuinely open, and it's the interesting part.
It's yours to own.

- **Re-implement it yourself, end to end.** Reading the corrected code isn't the same as
  understanding it. Rebuild the estimator, randoms, and corrections from the concepts, then diff
  against `clustering_lib`. The null test and the "predict before you run" checkpoints are your
  self-checks.
- **The science is unfinished — the GP catalog gives only an *upper limit* on b.** The real
  clustering lives below the ~850 km/s sampler floor, so 1D LOS clustering of *this* catalog can
  only *bound* b, not measure it. Actually measuring it is open research — and *choosing the path*
  is itself a research call that's yours: (a) improve close-pair completeness at the sampler floor
  (the GP's evidence step rejects close pairs), or (b) a different probe — 3D clustering or the
  DLA×Lyα-forest cross-correlation, the route the literature uses to reach b≈2.
- **Precision refinements left open** (each turns the limit into a real measurement): resolve
  purity/completeness in **(Δv, z̄)** instead of marginalizing over z; build **per-sightline /
  SNR-conditioned** randoms; run a **full sightline bootstrap carrying C_pair and purity** through
  every resample; **re-derive the RSD calibration k per selection**; and feed pyigm's measured
  **ℓ(X, N_HI)** into the analytic randoms (`randoms_tutorial.ipynb` §4).
- **Generalize beyond truth and mock-0.** Apply the pipeline to other mocks and to real data,
  where the FP / Lyβ-Lyγ veto and the z-dependence of purity/completeness all need re-checking —
  flags that are inert on this mock won't be on real data.

None of this is leftover cleanup — this *is* the project. Pick the piece that grabs you and dig in.

## Reading order
1. **`notes/science_bugs.md`** — what to adjust and why (each point confirmed, with the fixes).
2. **`randoms_tutorial.ipynb`** — the core concept that will help most; build randoms from scratch.
3. **`dla_clustering_tutorial.ipynb`** — the whole method, top to bottom, with the physics.
4. **`dla_1d_clustering_corrected.ipynb`** — the corrected analysis run end-to-end.
- Reference as needed: `notes/coding_bugs.md` (mechanical detail), `scripts/clustering_lib.py` (the tested engine), `outputs/` (figures).

## How to actually learn it (not just read it)
- **Re-implement the randoms yourself** from the three rules above, then diff against
  `clustering_lib.random_pair_hist`. Doing beats reading.
- Use the notebooks' **"predict before you run"** checkpoints — guess, then check.
- The **null test** in the randoms tutorial (random-vs-random → ξ≈1) is your "do I really get it?"
  self-check. If you can explain *why* it must give ≈1, you've understood randoms.
