#!/usr/bin/env python
"""
subdla_and_stability.py
=======================
(A) Sub-DLA-inclusive version: re-run the 1D clustering measurement + bias fit with
    NHI>20.0 (strong sub-DLAs included) and the sub-DLA-only band (20.0-20.3), and make
    comparison figures vs the NHI>=20.3 baseline.
(B) Stability test of the GP-CORRECTED bias fit: vary the randoms seed, n_real, and the
    fit Delta_v range, and report how much b_app and the 2-sigma upper limit move.

Read-only on data. Outputs to ../outputs/.
"""
import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clustering_lib as L
import fit_dla_bias as F

OUT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs"))
B_TRUE = 2.0

loaded = L.load_catalogs()
truth = L.build_truth_arrays(loaded)
finder = L.build_finder_arrays(loaded)
bins = L.default_bins(); dv = 0.5 * (bins[:-1] + bins[1:])
mx = F.MatterXi()


def pipeline(nhi_min, sub_only=False, n_real=40, seed=3):
    if sub_only:
        tmask = L.select_truth(truth, nhi_min=20.0) & (truth["NHI"] < 20.3)
        fmask = L.select_finder(finder, nhi_min=20.0) & (finder["NHI"] < 20.3)
    else:
        tmask = L.select_truth(truth, nhi_min=nhi_min)
        fmask = L.select_finder(finder, nhi_min=nhi_min)
    rt = L.measure_xi(truth["TARGETID"][tmask], truth["Z"][tmask], truth["Z_QSO"][tmask], bins, n_real=n_real, seed=seed)
    rf = L.measure_xi(finder["TARGETID"][fmask], finder["Z"][fmask], finder["Z_QSO"][fmask], bins, n_real=n_real, seed=seed + 1)
    rec = L.match_recovered(truth, tmask, finder, fmask)
    C, _, _, _, _ = L.pair_completeness(truth, tmask, rec, bins)
    is_tp = L.finder_pair_truth_match(finder, fmask, truth, tmask)
    p, _, _ = L.pair_purity(finder, fmask, is_tp, bins)
    valid = (C > 0) & np.isfinite(C); pp = np.where(np.isfinite(p), p, 1.0)
    DDc = np.full_like(rf["DD"], np.nan); DDc[valid] = rf["DD"][valid] * pp[valid] / C[valid]
    RRg = rf["RR"]; mv = valid & (RRg > 0) & np.isfinite(DDc)
    scale = DDc[mv].sum() / RRg[mv].sum() if mv.sum() else np.nan
    opxc = np.full_like(rf["DD"], np.nan); opxc[mv] = DDc[mv] / (scale * RRg[mv])
    z_eff = float(np.median(truth["Z"][tmask]))
    return dict(rt=rt, rf=rf, C=C, p=p, opxc=opxc, z_eff=z_eff,
                ntruth=int(tmask.sum()), nfind=int(fmask.sum()),
                npair_t=int(rt["DD"].sum()), npair_g=int(rf["DD"].sum()))


def fitb(r, opx, RR, dv_lo, dv_hi, label):
    err = np.where(r["rt"]["DD"] > 0, opx / np.sqrt(np.maximum(r["rf"]["DD"] if "gp" in label else r["rt"]["DD"], 1)), np.inf)
    # use Poisson on the appropriate DD
    DDref = r["rf"]["DD"] if "GP" in label else r["rt"]["DD"]
    err = np.where(DDref > 0, opx / np.sqrt(np.maximum(DDref, 1)), np.inf)
    b2, b2e, bapp, bappe, chi, n = F.fit_bapp(label, dv, opx, err, RR, mx, r["z_eff"], dv_lo, dv_hi)
    return bapp, bappe, chi


print("=" * 70); print("(A) SUB-DLA-INCLUSIVE VERSION")
r03 = pipeline(20.3); r20 = pipeline(20.0); rsub = pipeline(None, sub_only=True)
for tag, r in [("NHI>=20.3", r03), ("NHI>20.0", r20), ("subDLA 20.0-20.3", rsub)]:
    bt, bte, _ = fitb(r, r["rt"]["one_plus_xi"], r["rt"]["RR"], 250., 20000., f"truth {tag}")
    k = B_TRUE / bt if bt > 0 else np.nan
    bg, bge, _ = fitb(r, r["opxc"], r["rf"]["RR"], 2000., 20000., f"GP {tag}")
    print(f"  {tag}: truthDLA={r['ntruth']} GPdet={r['nfind']} truthpairs={r['npair_t']} GPpairs={r['npair_g']}")
    print(f"     truth b_app={bt:.2f}+/-{bte:.2f} (k={k:.2f}); GP b_app={bg:.2f}+/-{bge:.2f} -> real b={k*bg:.2f}, 2sig UL<{k*(bg+2*bge):.2f}")

# figure: NHI>=20.3 vs NHI>20.0 clustering + bias overlay
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
a = ax[0]
for r, c, lab in [(r03, "k", "NHI>=20.3"), (r20, "C0", "NHI>20.0")]:
    a.plot(dv, r["rt"]["one_plus_xi"], color=c, marker="o", ms=3, label=f"truth {lab}")
    a.plot(dv, r["opxc"], color=c, ls=":", marker="D", ms=3, label=f"GP-corr {lab}")
a.axhline(1, color="0.6", ls=":"); a.axvspan(0, 1500, color="grey", alpha=0.12)
a.set_xscale("log"); a.set_xlim(200, 30000); a.set_ylim(0, 3.2)
a.set_xlabel(r"$\Delta v$ [km/s]"); a.set_ylabel(r"$1+\xi$"); a.set_title("Clustering: NHI>=20.3 vs NHI>20.0"); a.legend(fontsize=7)
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
truth_bapp_ref = None
rows = []
for n_real in [30, 50]:
    for seed in [1, 7, 13]:
        r = pipeline(20.3, n_real=n_real, seed=seed)
        # truth calibration for this realisation
        bt, bte, _ = fitb(r, r["rt"]["one_plus_xi"], r["rt"]["RR"], 250., 20000., "truth")
        k = B_TRUE / bt
        for (lo, hi) in [(2000., 20000.), (3000., 20000.), (2000., 30000.)]:
            bg, bge, chi = fitb(r, r["opxc"], r["rf"]["RR"], lo, hi, "GP")
            ul = k * (bg + 2 * bge) if bg > 0 else np.nan
            rows.append((n_real, seed, f"[{int(lo)},{int(hi)}]", bg, bge, chi, k * bg if bg > 0 else np.nan, ul))
print(f"\n  {'n_real':>6} {'seed':>4} {'range':>14} {'b_app':>6} {'+/-':>5} {'chi2':>5} {'b_real':>6} {'UL2s':>6}")
for n_real, seed, rng, bg, bge, chi, br, ul in rows:
    print(f"  {n_real:>6} {seed:>4} {rng:>14} {bg:>6.2f} {bge:>5.2f} {chi:>5.2f} {br:>6.2f} {ul:>6.2f}")
arr = np.array([(rw[3], rw[6], rw[7]) for rw in rows], float)
print(f"\n  GP b_app: mean={np.nanmean(arr[:,0]):.2f} std={np.nanstd(arr[:,0]):.2f}  range=[{np.nanmin(arr[:,0]):.2f},{np.nanmax(arr[:,0]):.2f}]")
print(f"  GP 2sig UL: mean={np.nanmean(arr[:,2]):.2f} std={np.nanstd(arr[:,2]):.2f}  range=[{np.nanmin(arr[:,2]):.2f},{np.nanmax(arr[:,2]):.2f}]")
