#!/usr/bin/env python
"""
measure_1d_dla_clustering.py
============================
Corrected 1D line-of-sight DLA clustering on the DESI 2LPT mock-0 GP catalog,
calibrated for purity + completeness and validated against the mock truth.

Pipeline (see ../notes/dla_clustering_science.md and ../notes/science_bugs.md):
  1. Load truth + finder + zcat + bal; apply MATCHED selection to both
     (NHI>=20.3 both members, SNR_REDSIDE>2, P_DLA>0.99, DLAFLAG==0, 2<z_QSO<4.25,
     (1+z)-correct forest/proximity window, drop BAL sightlines and Lyb/Lyg ghosts).
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clustering_lib as L

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
OUTDIR = os.path.abspath(OUTDIR)
N_REAL = 50          # random realisations for the headline curves
N_REAL_BOOT = 20     # randoms per bootstrap resample (cheaper)
N_BOOT = 60          # sightline bootstrap resamples


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    t0 = time.time()
    loaded = L.load_catalogs()
    truth = L.build_truth_arrays(loaded)
    finder = L.build_finder_arrays(loaded)
    tmask = L.select_truth(truth)
    fmask = L.select_finder(finder)
    print(f"[load+select] {time.time()-t0:.1f}s  truth DLAs={tmask.sum()}  finder dets={fmask.sum()}")

    bins = L.default_bins()
    dv_mid = 0.5 * (bins[:-1] + bins[1:])

    # ---- truth clustering (the validation target) ----
    rt = L.measure_xi(truth["TARGETID"][tmask], truth["Z"][tmask], truth["Z_QSO"][tmask],
                      bins, n_real=N_REAL, seed=1)
    _, rt_err = L.bootstrap_xi(truth["TARGETID"][tmask], truth["Z"][tmask], truth["Z_QSO"][tmask],
                               bins, n_real=N_REAL_BOOT, n_boot=N_BOOT, seed=11)
    print(f"[truth xi] {time.time()-t0:.1f}s")

    # ---- GP-clean clustering (raw, biased) ----
    rf = L.measure_xi(finder["TARGETID"][fmask], finder["Z"][fmask], finder["Z_QSO"][fmask],
                      bins, n_real=N_REAL, seed=2)
    _, rf_err = L.bootstrap_xi(finder["TARGETID"][fmask], finder["Z"][fmask], finder["Z_QSO"][fmask],
                               bins, n_real=N_REAL_BOOT, n_boot=N_BOOT, seed=22)
    print(f"[gp xi] {time.time()-t0:.1f}s")

    # ---- calibration: completeness + purity ----
    rec = L.match_recovered(truth, tmask, finder, fmask)
    C, n_true_pairs, n_rec_pairs, _, _ = L.pair_completeness(truth, tmask, rec, bins)
    is_tp = L.finder_pair_truth_match(finder, fmask, truth, tmask)
    p, n_all_fp, n_true_fp = L.pair_purity(finder, fmask, is_tp, bins)
    single_comp = rec[tmask].mean()
    print(f"[calib] {time.time()-t0:.1f}s  single-DLA recovery={single_comp:.3f}")

    # ---- GP-corrected pair counts and xi ----
    # DD_corr = DD_gp * purity / C_pair  (purity removes residual FP; /C_pair undoes
    # incompleteness).  Only defined where C_pair>0 (above the sampler floor).
    DD_gp = rf["DD"]; RR_gp = rf["RR"]
    valid = (C > 0) & np.isfinite(C)
    pp = np.where(np.isfinite(p), p, 1.0)
    DD_corr = np.full_like(DD_gp, np.nan)
    DD_corr[valid] = DD_gp[valid] * pp[valid] / C[valid]
    # corrected xi: renormalise RR to the corrected-DD total over the valid range
    # (count-preserving randoms set a per-sample integral constraint; apply it consistently)
    mvalid = valid & (RR_gp > 0) & np.isfinite(DD_corr)
    if mvalid.sum() > 0 and RR_gp[mvalid].sum() > 0:
        scale = DD_corr[mvalid].sum() / RR_gp[mvalid].sum()
    else:
        scale = np.nan
    opx_corr = np.full_like(DD_gp, np.nan)
    opx_corr[mvalid] = DD_corr[mvalid] / (scale * RR_gp[mvalid])

    # ---- console report ----
    print("\n  dv_bin[km/s]    DD_t  RR_t  1+xi_t   DD_gp 1+xi_gp  C_pair  purity  DD_corr 1+xi_corr")
    for k in range(len(dv_mid)):
        print(f"  {bins[k]:6.0f}-{bins[k+1]:<6.0f} {rt['DD'][k]:5.0f} {rt['RR'][k]:5.1f} "
              f"{rt['one_plus_xi'][k]:6.2f}   {rf['DD'][k]:4.0f} {rf['one_plus_xi'][k]:6.2f} "
              f"  {C[k] if np.isfinite(C[k]) else np.nan:6.3f} {p[k] if np.isfinite(p[k]) else np.nan:6.3f} "
              f"  {DD_corr[k] if np.isfinite(DD_corr[k]) else np.nan:6.1f} "
              f"{opx_corr[k] if np.isfinite(opx_corr[k]) else np.nan:7.2f}")

    # ---- save ----
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
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
    a = ax[0, 0]
    a.step(dv_mid, rt["DD"], where="mid", color="k", lw=2, label="truth pairs (DD)")
    a.step(dv_mid, rt["RR"], where="mid", color="0.5", ls="--", lw=1.5, label="truth randoms (RR)")
    a.step(dv_mid, rf["DD"], where="mid", color="C3", lw=2, label="GP-clean pairs")
    a.step(dv_mid, np.where(np.isfinite(DD_corr), DD_corr, np.nan), where="mid",
           color="C0", lw=2, ls=":", label="GP corrected (p/C_pair)")
    a.set_xlim(0, 12000); a.set_xlabel(r"$\Delta v$ [km/s]"); a.set_ylabel("pair counts")
    a.set_title("Pair-$\\Delta v$ counts"); a.legend(fontsize=8)

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

    a = ax[1, 0]
    a.step(dv_mid, C, where="mid", color="C0", lw=2)
    a.axhline(single_comp, color="C0", ls=":", alpha=0.6, label=f"single-DLA recovery={single_comp:.2f}")
    a.axvspan(0, 1500, color="grey", alpha=0.12)
    a.set_xlim(0, 12000); a.set_ylim(-0.05, 1.05); a.set_xlabel(r"$\Delta v$ [km/s]")
    a.set_ylabel(r"pair completeness $C_{\rm pair}$"); a.set_title("Close-pair completeness (truth-driven)")
    a.legend(fontsize=8)

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
