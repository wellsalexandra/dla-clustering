#!/usr/bin/env python
"""
subdla_and_stability.py
=======================
This script runs TWO independent robustness studies on top of the 1D DLA
clustering measurement (see measure_1d_dla_clustering.py for the pipeline itself).

(A) Sub-DLA-inclusive version: re-run the 1D clustering measurement + bias fit with
    NHI>20.0 (strong sub-DLAs included) and the sub-DLA-only band (20.0-20.3), and make
    comparison figures vs the NHI>=20.3 baseline.
    WHY: the canonical DLA threshold is log10(N_HI/cm^-2) >= 20.3. "Sub-DLAs"
    (20.0-20.3) are weaker absorbers; whether you include them changes both the
    SAMPLE SIZE (more objects -> smaller error bars) and potentially the intrinsic
    bias (lower-column absorbers may live in lower-mass, less-clustered halos). This
    section quantifies how sensitive the result is to that definitional choice.

(B) Stability test of the GP-CORRECTED bias fit: vary the randoms seed, n_real, and the
    fit Delta_v range, and report how much b_app and the 2-sigma upper limit move.
    WHY: the headline number must not be an artefact of arbitrary analysis knobs.
    If b_app swings wildly when we change the random seed or the fit window, the
    "measurement" is really just noise/systematics. A small spread across all these
    variations is the evidence that the result is trustworthy.

Read-only on data. Outputs to ../outputs/.
"""
import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")  # headless backend; set before pyplot (writes PNG only)
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # make sibling modules importable
import clustering_lib as L     # the clustering engine (loading, pairs, randoms, estimator, calibration)
import fit_dla_bias as F       # the cosmology template + bias-amplitude fitter

OUT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs"))

# The mock was generated with a known input DLA bias of b=2.0. We use this to
# calibrate out the estimator's residual systematic: see the k-factor below, where
# k = B_TRUE / b_app(truth) rescales the GP-fitted amplitude onto the true scale.
B_TRUE = 2.0

# Load the catalogs ONCE at module scope (loading the FITS tables is the expensive
# step); the pipeline() function below just re-applies different selection masks to
# these same in-memory arrays, so the many re-runs in parts (A) and (B) are cheap.
loaded = L.load_catalogs()
truth = L.build_truth_arrays(loaded)
finder = L.build_finder_arrays(loaded)
bins = L.default_bins(); dv = 0.5 * (bins[:-1] + bins[1:])  # shared dv binning + bin centers
mx = F.MatterXi()  # linear matter-correlation template (EH98 + growth), reused for every fit


def pipeline(nhi_min, sub_only=False, n_real=40, seed=3):
    """Run the full truth + GP-corrected 1D clustering measurement for one NHI selection.

    This is a self-contained copy of the measure_1d_dla_clustering.py pipeline so we can
    re-run it cheaply under different column-density cuts and different RNG settings.

    Parameters
    ----------
    nhi_min : float or None
        Lower NHI threshold (log10 N_HI). 20.3 is the standard DLA cut; 20.0 also admits
        strong sub-DLAs. Ignored when sub_only=True.
    sub_only : bool
        If True, select the SUB-DLA-ONLY band 20.0 <= NHI < 20.3 (strip out the real DLAs)
        to isolate how the weak absorbers alone cluster.
    n_real, seed : Monte-Carlo controls for the randoms (RR). seed and seed+1 give the
        truth and finder random catalogs independent (but reproducible) streams.

    Returns a dict with the truth result (rt), raw GP result (rf), the two calibration
    curves (C, p), the GP-corrected 1+xi (opxc), the effective pair redshift (z_eff used
    by the cosmology template), and bookkeeping counts.
    """
    if sub_only:
        # Sub-DLA-only band: select with the 20.0 floor, then keep ONLY 20.0-20.3 by
        # also requiring NHI < 20.3. This removes the bona-fide DLAs so the remaining
        # sample is purely the weak absorbers.
        tmask = L.select_truth(truth, nhi_min=20.0) & (truth["NHI"] < 20.3)
        fmask = L.select_finder(finder, nhi_min=20.0) & (finder["NHI"] < 20.3)
    else:
        # Standard one-sided cut at nhi_min (20.3 baseline or 20.0 sub-DLA-inclusive).
        tmask = L.select_truth(truth, nhi_min=nhi_min)
        fmask = L.select_finder(finder, nhi_min=nhi_min)
    # Truth and raw-GP clustering with identical binning/estimator (see measure_xi).
    rt = L.measure_xi(truth["TARGETID"][tmask], truth["Z"][tmask], truth["Z_QSO"][tmask], bins, n_real=n_real, seed=seed)
    rf = L.measure_xi(finder["TARGETID"][fmask], finder["Z"][fmask], finder["Z_QSO"][fmask], bins, n_real=n_real, seed=seed + 1)
    # Truth-driven calibration for THIS selection: completeness C(dv) and purity p(dv).
    rec = L.match_recovered(truth, tmask, finder, fmask)
    C, _, _, _, _ = L.pair_completeness(truth, tmask, rec, bins)
    is_tp = L.finder_pair_truth_match(finder, fmask, truth, tmask)
    p, _, _ = L.pair_purity(finder, fmask, is_tp, bins)
    # Apply the same correction as the main script: DD_corr = DD_gp * purity / completeness,
    # only where C is positive/finite; fall back to purity=1 where p is undefined.
    valid = (C > 0) & np.isfinite(C); pp = np.where(np.isfinite(p), p, 1.0)
    DDc = np.full_like(rf["DD"], np.nan); DDc[valid] = rf["DD"][valid] * pp[valid] / C[valid]
    # Re-normalise RR to the corrected-DD total (integral constraint) -> corrected 1+xi.
    RRg = rf["RR"]; mv = valid & (RRg > 0) & np.isfinite(DDc)
    scale = DDc[mv].sum() / RRg[mv].sum() if mv.sum() else np.nan
    opxc = np.full_like(rf["DD"], np.nan); opxc[mv] = DDc[mv] / (scale * RRg[mv])
    # Effective redshift of the pair sample = median truth DLA redshift; the linear-growth
    # factor D(z_eff) sets the template amplitude in the bias fit, so it must match the sample.
    z_eff = float(np.median(truth["Z"][tmask]))
    return dict(rt=rt, rf=rf, C=C, p=p, opxc=opxc, z_eff=z_eff,
                ntruth=int(tmask.sum()), nfind=int(fmask.sum()),
                npair_t=int(rt["DD"].sum()), npair_g=int(rf["DD"].sum()))


def fitb(r, opx, RR, dv_lo, dv_hi, label):
    """Fit the apparent bias amplitude b_app to one 1+xi curve over [dv_lo, dv_hi].

    The error bar on each bin is taken as Poisson on the relevant pair count:
        sigma(1+xi) ~ (1+xi) / sqrt(N_pairs),
    because DD is a histogram of counts and a count of N has fractional error 1/sqrt(N).
    We use np.maximum(..., 1) so empty bins don't divide by zero, and assign inf error
    (zero weight) to bins with no pairs so they drop out of the weighted fit.

    NOTE: the FIRST `err` assignment below is immediately overwritten by the second; only
    the second takes effect. The two differ in the case test: the first keys off the
    lowercase substring "gp" (which never matches the labels passed in, e.g. "GP NHI>=20.3"
    or "truth ..."), while the EFFECTIVE second line keys off uppercase "GP". So in
    practice the Poisson reference count DDref is the GP detection counts (rf["DD"]) for
    GP fits and the truth counts (rt["DD"]) for truth fits -- the count that actually
    underlies the curve being fitted.
    """
    # use Poisson on the appropriate DD: GP fits weight by the GP pair counts, truth fits
    # by the truth pair counts.
    DDref = r["rf"]["DD"] if "GP" in label else r["rt"]["DD"]
    err = np.where(DDref > 0, opx / np.sqrt(np.maximum(DDref, 1)), np.inf)
    # fit_bapp does the weighted least-squares amplitude fit against the IC-subtracted
    # template and returns b^2, its error, b_app=sqrt(b^2), its error, chi2/dof, and #bins.
    b2, b2e, bapp, bappe, chi, n = F.fit_bapp(label, dv, opx, err, RR, mx, r["z_eff"], dv_lo, dv_hi)
    return bapp, bappe, chi


print("=" * 70); print("(A) SUB-DLA-INCLUSIVE VERSION")
# Three samples: the 20.3 baseline DLAs, the 20.0-inclusive set (DLAs + strong sub-DLAs),
# and the sub-DLA-only band. Comparing them shows whether the clustering result is robust
# to where you draw the DLA/sub-DLA line.
r03 = pipeline(20.3); r20 = pipeline(20.0); rsub = pipeline(None, sub_only=True)
for tag, r in [("NHI>=20.3", r03), ("NHI>20.0", r20), ("subDLA 20.0-20.3", rsub)]:
    # Truth fit: fit the TRUTH 1+xi down to dv_lo=250 km/s (truth is complete + pure, so we
    # can trust small separations). b_app(truth) is the estimator's apparent bias for a
    # sample we KNOW has input bias B_TRUE=2.0.
    bt, bte, _ = fitb(r, r["rt"]["one_plus_xi"], r["rt"]["RR"], 250., 20000., f"truth {tag}")
    # k = calibration factor that maps the estimator's apparent amplitude back onto the true
    # bias scale. Because the truth has known b=2.0, k=B_TRUE/b_app(truth) absorbs the
    # estimator's residual systematics (redshift-space distortions, the integral
    # constraint, template approximations) into a single empirical correction.
    k = B_TRUE / bt if bt > 0 else np.nan
    # GP fit: fit the CORRECTED GP 1+xi, but only above dv_lo=2000 km/s -- small dv is
    # triply compromised for the finder (incompleteness, fingers-of-god, linear-bias
    # breakdown), so we exclude it for the real measurement (unlike the trusted truth).
    bg, bge, _ = fitb(r, r["opxc"], r["rf"]["RR"], 2000., 20000., f"GP {tag}")
    print(f"  {tag}: truthDLA={r['ntruth']} GPdet={r['nfind']} truthpairs={r['npair_t']} GPpairs={r['npair_g']}")
    # Report the GP apparent bias, then the calibrated "real" bias k*b_app, and a
    # conservative 2-sigma upper limit k*(b_app + 2*b_app_err). Because the signal is weak,
    # the UPPER LIMIT (not a central detection) is the honest headline number. Student
    # takeaway: across all three NHI cuts the real b and the 2-sigma UL should be mutually
    # consistent -- if they were, the result does not hinge on the sub-DLA definition.
    print(f"     truth b_app={bt:.2f}+/-{bte:.2f} (k={k:.2f}); GP b_app={bg:.2f}+/-{bge:.2f} -> real b={k*bg:.2f}, 2sig UL<{k*(bg+2*bge):.2f}")

# figure: NHI>=20.3 vs NHI>20.0 clustering + bias overlay
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
# Left panel: overlay truth vs GP-corrected 1+xi for the two main cuts, so you can check
# (a) the corrected GP curve tracks truth and (b) lowering the cut to 20.0 doesn't shift
# the clustering. Log x-axis spreads out the signal-rich small-dv decade; the band <1500
# km/s flags the unreliable regime; the line at 1 is "no clustering".
a = ax[0]
for r, c, lab in [(r03, "k", "NHI>=20.3"), (r20, "C0", "NHI>20.0")]:
    a.plot(dv, r["rt"]["one_plus_xi"], color=c, marker="o", ms=3, label=f"truth {lab}")
    a.plot(dv, r["opxc"], color=c, ls=":", marker="D", ms=3, label=f"GP-corr {lab}")
a.axhline(1, color="0.6", ls=":"); a.axvspan(0, 1500, color="grey", alpha=0.12)
a.set_xscale("log"); a.set_xlim(200, 30000); a.set_ylim(0, 3.2)
a.set_xlabel(r"$\Delta v$ [km/s]"); a.set_ylabel(r"$1+\xi$"); a.set_title("Clustering: NHI>=20.3 vs NHI>20.0"); a.legend(fontsize=7)
# Right panel: truth-only clustering for all three NHI selections, to see whether the
# weaker absorbers cluster differently from full DLAs. (templ is the b=1 matter template
# evaluated at the baseline z_eff; computed here for reference even though only the truth
# points are drawn.)
a = ax[1]
templ = mx.template_1plus_xi(dv, r03["z_eff"])
for r, c, lab in [(r03, "k", "NHI>=20.3"), (r20, "C0", "NHI>20.0"), (rsub, "C3", "subDLA only")]:
    a.plot(dv, r["rt"]["one_plus_xi"], color=c, marker="o", ms=3, ls="none", label=f"truth {lab}")
a.axhline(1, color="0.6", ls=":"); a.set_xscale("log"); a.set_xlim(200, 30000); a.set_ylim(0, 3.2)
a.set_xlabel(r"$\Delta v$ [km/s]"); a.set_ylabel(r"$1+\xi$ (truth)"); a.set_title("Truth clustering by NHI cut"); a.legend(fontsize=7)
fig.suptitle("Sub-DLA-inclusive 1D DLA clustering (2LPT mock-0)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(os.path.join(OUT, "subdla_version.png"), dpi=130)
print(f"[save] {OUT}/subdla_version.png")

print("\n" + "=" * 70); print("(B) GP-CORRECTED BIAS-FIT STABILITY (NHI>=20.3)")
print("varying randoms seed, n_real, and fit range; reporting b_app and 2sigma UL")
truth_bapp_ref = None  # (unused placeholder; kept for symmetry with earlier drafts)
rows = []
# Grid over the three knobs that could artificially move the answer:
#   n_real in {30,50}  -> tests sensitivity to random-catalog SHOT NOISE (RR averaging depth);
#   seed   in {1,7,13} -> tests sensitivity to the specific random REALISATION;
#   fit range below    -> tests sensitivity to the chosen fit window.
# Re-deriving k for EACH realisation is deliberate: the truth fit uses the SAME randoms, so
# the calibration factor tracks each realisation's systematics and the comparison stays fair.
for n_real in [30, 50]:
    for seed in [1, 7, 13]:
        r = pipeline(20.3, n_real=n_real, seed=seed)
        # truth calibration for this realisation -> per-realisation calibration factor k
        bt, bte, _ = fitb(r, r["rt"]["one_plus_xi"], r["rt"]["RR"], 250., 20000., "truth")
        k = B_TRUE / bt
        # Three fit windows: nudge the lower edge up (2000->3000, drop the most
        # compromised small-dv bin) and the upper edge out (20000->30000, add the noisy
        # tail). A stable b_app under all three means the result isn't an artefact of the
        # window choice.
        for (lo, hi) in [(2000., 20000.), (3000., 20000.), (2000., 30000.)]:
            bg, bge, chi = fitb(r, r["opxc"], r["rf"]["RR"], lo, hi, "GP")
            ul = k * (bg + 2 * bge) if bg > 0 else np.nan  # calibrated 2-sigma upper limit
            rows.append((n_real, seed, f"[{int(lo)},{int(hi)}]", bg, bge, chi, k * bg if bg > 0 else np.nan, ul))
# Tabulate every configuration so the spread is visible by eye...
print(f"\n  {'n_real':>6} {'seed':>4} {'range':>14} {'b_app':>6} {'+/-':>5} {'chi2':>5} {'b_real':>6} {'UL2s':>6}")
for n_real, seed, rng, bg, bge, chi, br, ul in rows:
    print(f"  {n_real:>6} {seed:>4} {rng:>14} {bg:>6.2f} {bge:>5.2f} {chi:>5.2f} {br:>6.2f} {ul:>6.2f}")
# ...then summarise the scatter ACROSS all configurations. The STD here is the systematic
# spread from analysis choices; if it is small relative to the per-fit error bar (+/-), the
# headline b_app and 2-sigma UL are robust and the analysis-knob systematics are subdominant.
# Columns of arr: [0]=GP b_app, [1]=calibrated b_real, [2]=2-sigma UL (one row per config).
arr = np.array([(rw[3], rw[6], rw[7]) for rw in rows], float)
print(f"\n  GP b_app: mean={np.nanmean(arr[:,0]):.2f} std={np.nanstd(arr[:,0]):.2f}  range=[{np.nanmin(arr[:,0]):.2f},{np.nanmax(arr[:,0]):.2f}]")
print(f"  GP 2sig UL: mean={np.nanmean(arr[:,2]):.2f} std={np.nanstd(arr[:,2]):.2f}  range=[{np.nanmin(arr[:,2]):.2f},{np.nanmax(arr[:,2]):.2f}]")
