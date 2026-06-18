# DLA-clustering science review of `dla_1d_clustering_updated.ipynb`

*A methodological and physics review of your 1D DLA clustering measurement. These are conceptual
points that affect the measured ξ(Δv), and they remain after the mechanical
[`coding_bugs.md`](coding_bugs.md) are addressed. The clustering result is built in cells 19
(DD + randoms), 20 (histograms), 21 (ξ).*

The three things I flagged for you all check out; they map onto S1–S4 and S8 below. Each item: the
correct science, what the notebook currently does, why it matters, the consequence, and the fix.

---

## S1: DD and the randoms are built from different sightline populations 🔴
**Correct:** DD (data pairs) and RR (random pairs) must come from the same set of sightlines with
the same selection. ξ = DD/RR − 1 is only meaningful if RR is the unclustered expectation of the
same sample.
**Notebook (cell 19):** the real-pair loop appends `deltav_arr` for every multi-DLA sightline
first; then `if(z_max > 3.0): continue` skips the random/expected block. So sightlines with
`z_max = Z_QSO − 3000/c > 3.0` end up in DD but not RR. You spotted this yourself
("TEMP FIX FOR ERROR, ask"), which was a good catch. On the analogous earlier notebook this dropped
~46% of sightlines from RR (`~/desi_gpy_dla_notes/acwells/code_review_bugs.md` R1).
**Consequence:** ξ = DD/RR − 1 picks up a large, z-dependent population mismatch, so it isn't yet
interpretable.
**Fix:** gate DD and RR with the identical sightline mask (or remove the need for the guard by
fixing the z-window, S4). The `z_max>3` symptom is really a unit issue (S4) and a binning/coverage
issue, so I'd address it at the root rather than work around it.

---

## S2: Random redshifts need the dX/dz Jacobian in the sampling weight 🔴 *(my suggestion #1: "the randoms generation is wrong")*
**Correct:** the absorber line density `ℓ(X)=dN/dX` is per unit absorption distance X. To place
random absorbers in redshift, the sampling weight is `dN/dz = ℓ(X)·dX/dz`, not `ℓ(X)`.
**Notebook (cell 19):** `N_expected = Σ lX_complete·dX` correctly integrates in X, but the
position draw `lX_complete /= lX_complete.sum(); np.random.choice(z_bins_mid, p=lX_complete)`
uses `ℓ(X)·C` directly as a probability over a z-uniform grid. The `dX/dz` factor that is present
in the count integral isn't yet carried through to the position sampling. This is a subtle one and
easy to miss.
**Consequence:** the random z (hence random Δv) distribution is skewed. `dX/dz=(1+z)²/E(z)`
runs 2.97→3.50 over z∈[2,3], an ~18% mis-weighting across the window, which shifts RR and so shifts
ξ.
**Fix:** sample with `p = (lX_complete*dX); p/=p.sum()`. (Or sidestep pyigm entirely with the
permutation-randoms estimator, which inherits the empirical dN/dz automatically; see the
corrected notebook.)

---

## S3: Randoms are under-sampled ~1×, so RR is pure shot noise 🔴 *(my suggestion #2: "randoms should be 10× or more than the measured DLA pairs")*
**Correct:** RR sits in the denominator of ξ, so its Poisson noise propagates directly. The
random sample must have N_random ≳ 10–50× the data so RR is effectively noiseless.
**Notebook (cell 19):** exactly one Poisson realisation per sightline
(`N_sampled = np.random.poisson(N_expected)`), and pairs only form when `N_sampled > 1`. The
expected DLA count per forest is small, `N_expected ≈ ℓ(X)·ΔX ≈ 0.05–0.2`, so
`P(N_sampled ≥ 2) ≈ N_expected²/2 ≈ 0.1–2%`. Almost every sightline contributes zero random pairs,
and the total number of random pairs is of order the number of data pairs (both trace the same
close-pair rate). RR per Δv bin is then a handful of counts.
**Consequence:** `ξ = DD/RR − 1` is dominated by Poisson scatter in RR, which is exactly the
headline symptom I flagged for you. The bins where RR=0 give `inf`/`NaN` (coding C4/C9/C18).
**Fix:** generate ≥10–50× the data in randoms, i.e. `n_boot ≈ 50` permutation realisations
(reference `measure_dla_pair_clustering.py`) or ~20–50 Poisson realisations per sightline, and
average RR over them before forming ξ.

---

## S4: Units: velocity→redshift conversions drop the (1+z) factor 🔴 *(my suggestion #3: "the unit in the notebook is wrong")*
**Correct:** a velocity offset Δv maps to a redshift offset Δz = (1+z)·Δv/c, not Δv/c.
**Notebook (cells 6 and 19):** the proximity/forest-edge offsets are written as
`+ (3000./SPEED_OF_LIGHT)` and `z_max = Z_QSO − 3000/c`, i.e. a 3000 km/s velocity treated as
a Δz = 3000/c ≈ 0.010 offset. The correct value is Δz = (1+z)·3000/c ≈ 0.030–0.050 over z=2–4.
**Consequence:** the proximity and blue-edge exclusions come out too small by a factor (1+z) ≈ 3–4,
so the per-sightline z-window (which sets both which DLAs enter DD and the range the randoms are
drawn over) is off, and inconsistently between DD (uses bare `1216`/`912`) and the window helper.
This is the unit point I flagged for you. (A second, related unit detail is the ℓ(X)-per-X vs
per-z mismatch in S2.)
**Fix:** convert every velocity edge with the (1+z) factor: e.g.
`z_hi = Z_QSO − (1+Z_QSO)·3000/c`, `z_lo = max(λ_min_edge, blue_edge + (1+Z_QSO)·3000/c)`.

---

## S5: Single-DLA completeness in the randoms and the DD correction aren't reconciled 🔴
**Correct:** to compare data to randoms you apply the selection function once, consistently. Either
correct DD up by 1/C_pair(Δv) and keep RR as the true unclustered expectation (no completeness), or
impose the same completeness on the randoms and compare like with like. And the pair-completeness
must be the truth-driven recovery rate (bounded ≤1):
`C_pair(Δv)=N(true pairs with both members detected)/N(true pairs)`.
**Notebook:** completeness enters both sides, and differently:
- RR is suppressed by a z-dependent single-DLA completeness: `lX_complete = lX·c_zdla` (cell 19);
- DD is boosted by a Δv-dependent "completeness": `DD_upweighted = DD/c_vsep` (cell 21).

So `ξ = (DD/c_vsep)/(RR·c_zdla…) − 1` carries a single-DLA completeness in RR and a separate
Δv-completeness on DD without reconciling the two. Folding the single-DLA completeness into the
randoms is actually the right instinct (it sets the detected density); the matching DD correction is
then c₁²/C_pair (or renormalize and divide by C_pair only), not a second, independent completeness.
What still needs swapping out: the `c_zdla`/`c_vsep` used here are the distribution-ratio metric,
`N(finder pairs in bin)/N(truth pairs in bin)`, which is not a recovery rate and can exceed 1 (seen
up to ~23× on your data; `~/desi_gpy_dla_notes/acwells/vsep_review.md` Q3), and those >1 values are
then clamped to 0 (coding C4), which affects the very bins they touch.
**Consequence:** the completeness correction doesn't yet hang together; ξ is multiplied/divided by
ill-defined factors and goes `inf` where the ratio exceeded 1.
**Fix:** (a) replace both completeness functions with the truth-driven `C_pair(Δv)` (bounded ≤1,
method in `~/desi_gpy_dla_notes/acwells/dla_pair_completeness_method.md`); (b) apply it once,
correcting DD by 1/C_pair(Δv), with RR being the unclustered expectation under matched selection.

---

## S6: No purity handling: false positives (incl. Lyβ/Lyγ ghosts) dilute and spike ξ 🔴
**Correct:** FP detections pair at ~random Δv, adding a flat background that dilutes the small-Δv
excess (pushes ξ→0); and the Lyβ/Lyγ "ghost" FPs create sharp spurious spikes at
Δv ≈ 50,800 / 66,600 km/s (a real DLA's Lyβ/Lyγ trough mis-fit as a separate Lyα DLA; see the
ghost section of `dla_clustering_tutorial.ipynb` §6). Pair purity on the verified 2LPT catalog is
only ~0.46.
**Notebook:** DD is built from `gpp` with `SNR_REDSIDE>2 & NHI_GP>20.3 & P_DLA_GP>0.99`
(cell 3); the next cuts to add are `DLAFLAG==0`, `LYBETA_FLAG==False`, BAL removal, and the truth
match. Until those are in, every FP pair, including the Lyβ/Lyγ ghosts I want you to drop, sits in
DD.
**Consequence:** ξ is diluted at small Δv and carries non-physical spikes at ~50–67k km/s;
total pair counts (used in any normalisation) are inflated.
**Fix:** cut `DLAFLAG==0`, `LYBETA_FLAG==False`, drop BAL sightlines, veto the Lyγ separation;
then correct the residual FP dilution with the per-Δv-bin pair purity p(Δv) (DD_true≈p(Δv)·DD_obs),
or, for validation, work from truth-matched pairs.

---

## S7: DD comes from a different catalog and selection than the corrections 🟠
**Correct:** DD, RR, purity, and completeness must describe the same sample (same files, same
window/BAL/SNR cuts, same truth matching).
**Notebook:** DD uses `gpp` = `gpp_251202_new_baseline_snr_2.fits` (cell 3), to which
`make_lambda_z_BAL_qso_cuts` is never applied (no forest window, no BAL removal). The
completeness/truth use `combined_cat_all_no_bal` = `gpp_mock_snr2_combined_matched.fits`
(cell 7, windowed + BAL-removed + truth-matched). These are different catalogs with different
selections.
**Consequence:** the data pairs and their corrections refer to inconsistent populations, so the
correction isn't yet on solid footing.
**Fix:** build DD, RR, and the corrections from one catalog with one selection pipeline.

---

## S8: ξ formed from un-normalised DD and RR counts 🔴
**Correct:** ξ = DD/RR − 1 requires normalised pair counts, i.e.
ξ(Δv) = (DD(Δv)/N_DD)/(RR(Δv)/N_RR) − 1 (or scale RR by N_DD/N_RR). Otherwise the overall density
ratio N_RR/N_DD contaminates the amplitude.
**Notebook (cell 21):** `xi = cluster_1d(DD_upweighted, RR_counts)` divides raw histogram counts;
with RR a single under-sampled realisation (S3) of a different total, the amplitude of ξ is
arbitrary. (A further mismatch: random pairs scale ~N²/2 per sightline while data pairs come from
real multiplets; coding C19.)
**Fix:** normalise DD and RR by their totals (or apply the N_DD/N_RR factor) before `−1`.

---

## S9: Binning currently hides the entire signal 🟠
**Correct:** the clustering physics lives at Δv ≲ a few × 10³ km/s; use fine/log bins there.
**Notebook (cell 21):** `bins = linspace(0, dv_max, 26)` with `dv_max = max(all Δv)` ≈ 67,000 km/s
(driven by the Lyβ/Lyγ FP spikes) gives a bin width ≈ 2700 km/s, so the whole < 3000 km/s regime
falls in one bin, the bin where detected completeness is ≈0 (sampler floor). The signal ends up
invisible.
**Consequence:** even a correctly-estimated ξ would show nothing at this binning.
**Fix:** fine/log Δv bins below a few × 10³ km/s (e.g. 0,200,400,…,3000, then wider), and set the
histogram range deliberately (after removing FP spikes), not from `max(Δv)`.

---

## S10: Add a validation against truth; keep "clustering" vs "completeness" separate 🟠
**Correct:** the mock has a known answer. Measure ξ(Δv) directly from the truth catalog (where
b≈2 is planted) and require the GP-derived, purity/completeness-corrected ξ to match it. Keep the
two corrections conceptually separate: (i) measure clustering of detected DLAs against an
unclustered random model with matched selection; (ii) correct purity (FP dilution) and completeness
(close-pair recovery) using truth.
**Notebook:** the truth ξ(Δv) isn't computed yet; the random model (pyigm Poisson) and the
completeness up-weight are currently intertwined, so it's hard to tell whether the result reflects
"clustering" or "recovered completeness."
**Fix:** add a truth-ξ(Δv) measurement as the validation target; structure the GP measurement as
clustering-then-correct, and overlay GP-corrected vs truth (this is exactly what the corrected
notebook for this project does).

---

## Summary: the things I flagged, now checked

| Suggestion | Verdict | Where |
|---|---|---|
| Randoms generation needs adjusting | Confirmed: z-weight to fix (missing dX/dz), selection to match (DD≠RR sightlines), source catalog to align | S1, S2, S7 |
| Randoms should be ≥10× the data pairs | Confirmed: only ~1× generated; RR is shot-noise-dominated | S3 |
| The unit in the notebook needs the (1+z) factor | Confirmed: velocity→redshift offsets drop the (1+z) factor (off by ~3–4×); plus the ℓ(X)-per-X vs per-z detail | S4 (S2) |
| (additional) clustering calc to refine | Confirmed: completeness to apply once/consistently, ξ to normalise, FP/Lyβ-Lyγ to remove, binning to refine, truth validation to add | S5, S6, S8, S9, S10 |

The corrected re-implementation (`notebooks/` in this repo) addresses S1–S10: matched selection
for DD/RR, permutation randoms at ≥50×, (1+z)-correct windows, FP + Lyβ/Lyγ removal, truth-driven
pair-completeness applied once, normalised ξ, fine/log Δv bins, and an explicit truth-vs-GP
validation.
