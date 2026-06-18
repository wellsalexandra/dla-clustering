#!/usr/bin/env python
"""
measure_1d_dla_clustering.py
============================
Corrected 1D line-of-sight DLA clustering on the DESI 2LPT mock-0 GP catalog,
calibrated for purity + completeness and validated against the mock truth.

WHAT THIS SCRIPT MEASURES (the big picture for a student)
---------------------------------------------------------
Damped Lyman-alpha systems (DLAs) are dense neutral-hydrogen absorbers seen in
quasar (QSO) spectra. We want their *clustering*: do two DLAs along the SAME
line of sight tend to lie closer together (in velocity / redshift) than random?
If DLAs trace the same cosmic web as everything else, there is an excess of
close pairs, quantified by the two-point correlation function xi(dv). We report
it as ``1 + xi(dv) = DD/RR`` -- the ratio of observed close pairs (DD) to the
number expected if DLAs were sprinkled at random along each sightline (RR).
``1 + xi > 1`` means clustering (excess close pairs); ``= 1`` means no signal.

WHY "1D" / WHY ONLY SAME-SIGHTLINE PAIRS
----------------------------------------
A QSO spectrum samples HI along ONE pencil-beam line of sight. The sightlines on
the sky are sparse and far apart, so transverse (sightline-to-sightline) DLA
pairs are essentially never close in 3D. The only pairs we can usefully count are
two DLAs in the SAME quasar spectrum -- a purely radial (line-of-sight, hence
"1D") clustering measurement. Cross-sightline pairs are therefore never formed;
every pair below is a within-TARGETID (same-QSO) pair separated only in velocity.

THE TRUTH-VALIDATED, CALIBRATED STRATEGY
----------------------------------------
The mock has a *truth* catalog (every real DLA the simulation put in) and a
*finder* catalog (what the Gaussian-process / "GP" DLA finder actually detected).
The finder is imperfect: it misses some real DLAs (incompleteness) and reports
some spurious ones (false positives -> impurity). Both distort clustering. So we:
  - measure xi on the TRUTH (our ground-truth target / validation),
  - measure xi on the cleaned GP detections (biased by completeness + purity),
  - use the truth to CALIBRATE a per-bin pair-completeness C_pair(dv) and pair
    purity p(dv), then correct the GP pair counts and re-measure xi.
If the corrected GP curve lands on the truth curve, the correction works.

Pipeline (see ../notes/dla_clustering_science.md and ../notes/science_bugs.md):
  1. Load truth + finder + zcat + bal; apply MATCHED selection to both
     (NHI>=20.3 both members, SNR_REDSIDE>2, P_DLA>0.99, DLAFLAG==0, 2<z_QSO<4.25,
     (1+z)-correct forest/proximity window, drop BAL sightlines and Lyb/Lyg ghosts).
     "Matched" = identical cuts on truth and finder so the comparison is fair.
  2. Truth xi(dv): count-preserving, window-restricted randoms (n_real x data, >=10x).
  3. GP-clean xi(dv): same estimator on the FP/Lyb/BAL-removed detections.
  4. Calibrate: truth-driven pair-completeness C_pair(dv) and per-bin pair purity p(dv).
  5. GP-corrected pair counts DD_corr = DD_gp * p(dv) / C_pair(dv); corrected xi.
  6. Bootstrap-over-sightline error bars; save npz + a 4-panel figure.

READ-ONLY on inputs. Outputs to ../outputs/.
Run:  conda activate gpdla; python scripts/measure_1d_dla_clustering.py
"""
import os, sys, time
import numpy as np
import matplotlib
# Use the non-interactive "Agg" backend: this script runs headless (e.g. on a
# compute node with no display) and only writes a PNG, so we never open a window.
# This MUST be set before importing pyplot, or matplotlib may pick a GUI backend.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make the sibling module clustering_lib.py importable no matter what directory
# the script is launched from: prepend THIS file's own directory to sys.path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clustering_lib as L  # all the heavy lifting (loading, pairing, randoms, estimator)

# Outputs go to ../outputs/ relative to this script (not the CWD), resolved to an
# absolute path so the location is stable regardless of where we are invoked.
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
OUTDIR = os.path.abspath(OUTDIR)

# --- Monte-Carlo settings for the random catalog (RR) ---------------------
# RR is built by reshuffling redshifts many times and AVERAGING the pair
# histogram. The sampling variance of that mean falls like 1/N_REAL, so more
# realisations make RR effectively noiseless compared with the Poisson scatter of
# the data pairs DD. We use a high count for the headline curves and a cheaper
# count inside the bootstrap loop (which is already run N_BOOT times).
N_REAL = 50          # random realisations for the headline curves (RR variance ~ 1/N_REAL)
N_REAL_BOOT = 20     # randoms per bootstrap resample (cheaper; the bootstrap dominates the error budget)
N_BOOT = 60          # sightline bootstrap resamples -> error bars on 1+xi


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    t0 = time.time()  # wall-clock start; the [stage] prints below report elapsed seconds

    # ---- load + select ----
    # load_catalogs reads the truth, finder, QSO redshift (zcat) and BAL FITS tables
    # and attaches each truth DLA's host-QSO redshift. build_*_arrays repackage the
    # raw FITS columns into plain numpy dicts (TARGETID, Z, NHI, ...). select_* return
    # boolean masks applying the SAME physical cuts to each catalog (NHI>=20.3,
    # SNR>2, the (1+z)-correct forest/proximity z-window, BAL/ghost removal; the
    # finder additionally requires P_DLA>0.99 and DLAFLAG==0). Using matched cuts is
    # what makes the truth-vs-GP comparison and the calibration valid.
    loaded = L.load_catalogs()
    truth = L.build_truth_arrays(loaded)
    finder = L.build_finder_arrays(loaded)
    tmask = L.select_truth(truth)
    fmask = L.select_finder(finder)
    print(f"[load+select] {time.time()-t0:.1f}s  truth DLAs={tmask.sum()}  finder dets={fmask.sum()}")

    # ---- velocity-separation binning ----
    # default_bins(): fine 250 km/s bins below 3000 km/s (where the clustering signal
    # lives and changes fast), then progressively wider bins out to a 30000 km/s top
    # edge. The top edge deliberately stops well below the Lyb/Lyg "ghost" false-pair
    # spikes at ~50000-67000 km/s. dv_mid = bin centers, used for plotting and reporting.
    bins = L.default_bins()
    dv_mid = 0.5 * (bins[:-1] + bins[1:])

    # ---- truth clustering (the validation target) ----
    # measure_xi enumerates every same-sightline truth pair -> DD(dv) histogram, builds
    # the count-preserving window-restricted randoms -> RR(dv) (averaged over N_REAL
    # draws), renormalises RR so Sum(RR)=Sum(DD) in range (the integral constraint), and
    # returns 1+xi = DD/RR. This truth curve is the answer the corrected GP curve must
    # reproduce. seed fixes the RNG so results are reproducible.
    rt = L.measure_xi(truth["TARGETID"][tmask], truth["Z"][tmask], truth["Z_QSO"][tmask],
                      bins, n_real=N_REAL, seed=1)
    # bootstrap_xi resamples whole SIGHTLINES (not individual DLAs) with replacement to
    # get error bars: sightlines are the independent units, so this captures the real
    # sample variance. We keep only the std (rt_err); a different seed avoids reusing the
    # measurement's RNG stream.
    _, rt_err = L.bootstrap_xi(truth["TARGETID"][tmask], truth["Z"][tmask], truth["Z_QSO"][tmask],
                               bins, n_real=N_REAL_BOOT, n_boot=N_BOOT, seed=11)
    print(f"[truth xi] {time.time()-t0:.1f}s")

    # ---- GP-clean clustering (raw, biased) ----
    # Same estimator on the cleaned GP detections. "Clean" = false positives and
    # Lyb/Lyg ghosts already removed by select_finder, but the curve is still biased by
    # residual impurity (dilutes the signal toward 1) and incompleteness (we miss close
    # pairs), which steps 4-5 below correct. Different seeds from the truth run keep the
    # two random catalogs statistically independent.
    rf = L.measure_xi(finder["TARGETID"][fmask], finder["Z"][fmask], finder["Z_QSO"][fmask],
                      bins, n_real=N_REAL, seed=2)
    _, rf_err = L.bootstrap_xi(finder["TARGETID"][fmask], finder["Z"][fmask], finder["Z_QSO"][fmask],
                               bins, n_real=N_REAL_BOOT, n_boot=N_BOOT, seed=22)
    print(f"[gp xi] {time.time()-t0:.1f}s")

    # ---- calibration: completeness + purity ----
    # These two truth-driven curves quantify how the imperfect finder distorts pair
    # counts as a function of separation dv, so we can undo the distortion.
    #
    # match_recovered: greedy one-to-one NHI-ranked match of each selected truth DLA to
    #   a selected finder detection on the same sightline within a redshift tolerance.
    #   rec[i]=True means "truth DLA i was found".
    # pair_completeness -> C(dv): of all TRUE close pairs at separation dv, what fraction
    #   had BOTH members recovered? C<1 means we systematically miss close pairs. Returns
    #   NaN in bins with zero true pairs (the "sampler floor" -- no truth info there).
    # finder_pair_truth_match -> is_tp: is each finder detection a true positive (matches
    #   a real DLA) rather than a spurious one?
    # pair_purity -> p(dv): of all FINDER close pairs at dv, what fraction have BOTH
    #   members real? p<1 means false positives are diluting the measured clustering.
    # single_comp: the single-object (not pair) recovery fraction, a useful sanity scalar.
    rec = L.match_recovered(truth, tmask, finder, fmask)
    C, n_true_pairs, n_rec_pairs, _, _ = L.pair_completeness(truth, tmask, rec, bins)
    is_tp = L.finder_pair_truth_match(finder, fmask, truth, tmask)
    p, n_all_fp, n_true_fp = L.pair_purity(finder, fmask, is_tp, bins)
    single_comp = rec[tmask].mean()
    print(f"[calib] {time.time()-t0:.1f}s  single-DLA recovery={single_comp:.3f}")

    # ---- GP-corrected pair counts and xi ----
    # The correction undoes both finder biases at the level of the PAIR COUNTS, per bin:
    #   DD_corr = DD_gp * p(dv) / C_pair(dv)
    #     * multiply by purity p  -> keeps only the fraction of measured pairs that are
    #       genuinely real, removing the residual false-positive contamination, and
    #     * divide by completeness C_pair -> scales the surviving real pairs back up to
    #       the number that SHOULD exist, undoing the missed-pair incompleteness.
    # The two factors are independent (impurity removes fakes; incompleteness restores
    # missed reals), so they combine multiplicatively. C_pair appears in the denominator,
    # so the correction is only defined where C_pair>0 and finite (above the sampler floor
    # -- bins with no true pairs to calibrate from are left as NaN). Where purity is NaN
    # (no finder pairs in that bin) we fall back to p=1.0 (no correction) so a missing
    # purity estimate never zeroes out or NaNs the bin.
    DD_gp = rf["DD"]; RR_gp = rf["RR"]
    valid = (C > 0) & np.isfinite(C)
    pp = np.where(np.isfinite(p), p, 1.0)
    DD_corr = np.full_like(DD_gp, np.nan)
    DD_corr[valid] = DD_gp[valid] * pp[valid] / C[valid]
    # Corrected 1+xi = DD_corr / RR. We must RE-NORMALISE RR to the corrected-DD total,
    # NOT reuse the RR that was normalised to the raw DD_gp: changing DD's amplitude
    # (via p/C) changes the integral-constraint scaling. Count-preserving randoms impose
    # Sum(RR)=Sum(DD) over the measured range, so we recompute that single scale factor
    # against DD_corr and apply it consistently. This keeps 1+xi a pure SHAPE measurement.
    mvalid = valid & (RR_gp > 0) & np.isfinite(DD_corr)
    if mvalid.sum() > 0 and RR_gp[mvalid].sum() > 0:
        scale = DD_corr[mvalid].sum() / RR_gp[mvalid].sum()  # enforce Sum(RR_scaled)=Sum(DD_corr)
    else:
        scale = np.nan  # not enough valid bins to normalise -> leave corrected xi undefined
    opx_corr = np.full_like(DD_gp, np.nan)
    opx_corr[mvalid] = DD_corr[mvalid] / (scale * RR_gp[mvalid])

    # ---- console report ----
    # One row per dv bin so you can eyeball the whole pipeline side by side: truth pair
    # counts/randoms/clustering, the raw GP clustering, the two calibration curves
    # (C_pair, purity), and the corrected GP pair counts + clustering. NaNs appear where a
    # quantity is undefined (e.g. empty bins / sampler floor).
    print("\n  dv_bin[km/s]    DD_t  RR_t  1+xi_t   DD_gp 1+xi_gp  C_pair  purity  DD_corr 1+xi_corr")
    for k in range(len(dv_mid)):
        print(f"  {bins[k]:6.0f}-{bins[k+1]:<6.0f} {rt['DD'][k]:5.0f} {rt['RR'][k]:5.1f} "
              f"{rt['one_plus_xi'][k]:6.2f}   {rf['DD'][k]:4.0f} {rf['one_plus_xi'][k]:6.2f} "
              f"  {C[k] if np.isfinite(C[k]) else np.nan:6.3f} {p[k] if np.isfinite(p[k]) else np.nan:6.3f} "
              f"  {DD_corr[k] if np.isfinite(DD_corr[k]) else np.nan:6.1f} "
              f"{opx_corr[k] if np.isfinite(opx_corr[k]) else np.nan:7.2f}")

    # ---- save ----
    # Persist every curve and its provenance counts to a single .npz so the downstream
    # bias fit (fit_dla_bias.py) and stability tests can reload them without recomputing.
    # Keep both the raw and corrected curves, the calibration curves, the bootstrap
    # errors, and the pair/object counts that went into each (for sanity + reproducibility).
    npz = os.path.join(OUTDIR, "clustering_2lpt.npz")
    np.savez(npz, bins=bins, dv_mid=dv_mid,
             DD_truth=rt["DD"], RR_truth=rt["RR"], opx_truth=rt["one_plus_xi"], opx_truth_err=rt_err,
             DD_gp=rf["DD"], RR_gp=rf["RR"], opx_gp=rf["one_plus_xi"], opx_gp_err=rf_err,
             C_pair=C, n_true_pairs=n_true_pairs, n_rec_pairs=n_rec_pairs,
             purity=p, n_all_finder_pairs=n_all_fp, n_true_finder_pairs=n_true_fp,
             DD_corr=DD_corr, opx_corr=opx_corr, single_comp=single_comp,
             n_truth_dla=int(tmask.sum()), n_finder_dla=int(fmask.sum()))
    print(f"\n[save] {npz}")

    # ---- figure ----
    # A 4-panel summary. Panel (0,0): the raw ingredients (pair-count histograms).
    # Panel (0,1): the headline clustering comparison (truth vs raw GP vs corrected GP).
    # Panels (1,0)/(1,1): the two calibration curves that drove the correction.
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))

    # Panel (0,0): pair-count histograms vs dv. Showing DD (data pairs) and RR (random
    # pairs) for truth, plus raw and corrected GP DD, lets you see WHERE the correction
    # changes counts. "step ... where=mid" draws the histogram with steps centered on bins.
    a = ax[0, 0]
    a.step(dv_mid, rt["DD"], where="mid", color="k", lw=2, label="truth pairs (DD)")
    a.step(dv_mid, rt["RR"], where="mid", color="0.5", ls="--", lw=1.5, label="truth randoms (RR)")
    a.step(dv_mid, rf["DD"], where="mid", color="C3", lw=2, label="GP-clean pairs")
    a.step(dv_mid, np.where(np.isfinite(DD_corr), DD_corr, np.nan), where="mid",
           color="C0", lw=2, ls=":", label="GP corrected (p/C_pair)")
    # x-limit 12000 km/s: zoom onto the regime with signal/statistics (bins extend to
    # 30000 but the high-dv tail is flat and noisy).
    a.set_xlim(0, 12000); a.set_xlabel(r"$\Delta v$ [km/s]"); a.set_ylabel("pair counts")
    a.set_title("Pair-$\\Delta v$ counts"); a.legend(fontsize=8)

    # Panel (0,1): the result. 1+xi(dv) for truth (the target), raw GP (biased), and
    # corrected GP. Error bars are the sightline-bootstrap std. The horizontal line at 1
    # is "no clustering"; the shaded band <1500 km/s flags the small-dv regime that is
    # unreliable (incompleteness + fingers-of-god) and excluded from the later bias fit.
    a = ax[0, 1]
    a.errorbar(dv_mid, rt["one_plus_xi"], yerr=rt_err, color="k", marker="o", ms=4, lw=1.5,
               capsize=2, label="truth $1+\\xi$ (target)")
    a.errorbar(dv_mid, rf["one_plus_xi"], yerr=rf_err, color="C3", marker="s", ms=4, lw=1.5,
               capsize=2, label="GP-clean (raw)")
    a.plot(dv_mid, opx_corr, color="C0", marker="D", ms=4, lw=1.5, ls=":", label="GP corrected")
    a.axhline(1, color="0.6", ls=":"); a.axvspan(0, 1500, color="grey", alpha=0.12)
    a.set_xlim(0, 12000); a.set_ylim(0, 3.2); a.set_xlabel(r"$\Delta v$ [km/s]")
    a.set_ylabel(r"$1+\xi(\Delta v)$"); a.set_title("Clustering: truth vs GP"); a.legend(fontsize=8)
    a.text(150, 0.2, "sampler floor\n(C_pair~0)", fontsize=7, color="0.3")

    # Panel (1,0): pair-completeness C_pair(dv). The dotted reference line is the single-
    # DLA recovery rate; if pair finding were independent per object you'd expect C_pair to
    # sit near (single_comp)^2, and the dip at small dv shows close pairs are HARDER to
    # deblend than isolated DLAs (extra incompleteness exactly where clustering peaks).
    a = ax[1, 0]
    a.step(dv_mid, C, where="mid", color="C0", lw=2)
    a.axhline(single_comp, color="C0", ls=":", alpha=0.6, label=f"single-DLA recovery={single_comp:.2f}")
    a.axvspan(0, 1500, color="grey", alpha=0.12)
    a.set_xlim(0, 12000); a.set_ylim(-0.05, 1.05); a.set_xlabel(r"$\Delta v$ [km/s]")
    a.set_ylabel(r"pair completeness $C_{\rm pair}$"); a.set_title("Close-pair completeness (truth-driven)")
    a.legend(fontsize=8)

    # Panel (1,1): pair-purity p(dv). p=1 (top reference line) is a perfectly clean
    # sample; dips below 1 mark separations where false-positive pairs dilute the signal.
    a = ax[1, 1]
    a.step(dv_mid, p, where="mid", color="C2", lw=2)
    a.axhline(1, color="0.6", ls=":")
    a.set_xlim(0, 12000); a.set_ylim(0, 1.1); a.set_xlabel(r"$\Delta v$ [km/s]")
    a.set_ylabel("pair purity $p(\\Delta v)$"); a.set_title("GP pair purity (both members true)")

    fig.suptitle("1D DLA clustering on 2LPT mock-0 (GP catalog, calibrated + validated vs truth)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fpng = os.path.join(OUTDIR, "clustering_2lpt.png")
    fig.savefig(fpng, dpi=130)
    print(f"[save] {fpng}")
    print(f"[done] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
