"""
clustering_lib.py — engine for 1D line-of-sight DLA clustering on the DESI 2LPT mock.

This is the corrected re-implementation of the logic in the student notebook
``dla_1d_clustering_updated.ipynb``. See ../notes/dla_clustering_science.md for the
science and ../notes/science_bugs.md / ../notes/coding_bugs.md for what was wrong.

Design (fixes keyed to science_bugs.md):
  * DD and RR built from the SAME catalog + SAME selection (fixes S1, S7).
  * Randoms = count-preserving, WINDOW-RESTRICTED redshift resampling, averaged over
    many realisations -> N_random pairs ~ n_real x N_data (>=10x) (fixes S2, S3).
    Count-preserving randoms keep per-sightline pair counts so SumDD ~ SumRR over the
    FULL dv axis; the histogram truncates at the top bin edge, so measure_xi EXPLICITLY
    renormalises RR to the in-range DD total (fixes S8).
  * Per-sightline z-window uses the (1+z)-correct velocity->redshift conversion (fixes S4).
  * False positives + Lyb/Lyg ghost pairs removed; residual FP dilution corrected with the
    truth-measured per-bin pair purity (fixes S6).
  * Incompleteness corrected with the truth-driven pair-completeness C_pair(dv), bounded <=1
    and applied ONCE (fixes S5).
  * Fine/log Delta_v bins; FP spikes excluded from the range (fixes S9).
  * Truth xi(dv) measured as the validation target (fixes S10).

Read-only on all input catalogs. Requires only numpy / scipy / astropy / fitsio.
"""
from __future__ import annotations
import itertools
import numpy as np

# Portable FITS table reader: prefer fitsio, fall back to astropy.io.fits so the
# notebook can run under any kernel that has one of them.
try:
    import fitsio
    def _read_table(path):
        return fitsio.read(path)
except Exception:  # pragma: no cover - environment dependent
    from astropy.io import fits as _fits
    def _read_table(path):
        with _fits.open(path, memmap=True) as h:
            for hdu in h[1:]:
                if getattr(hdu, "data", None) is not None:
                    return np.asarray(hdu.data)
        raise RuntimeError(f"no table HDU in {path}")

# ---- physical constants ----
C_KMS = 299792.458
LYA = 1215.67     # Lyman-alpha rest wavelength [Angstrom]
LYB = 1025.72     # Lyman-beta
LYG = 972.537     # Lyman-gamma
OBS_MIN = 3600.0  # DESI blue cutoff [Angstrom]

# ---- 2LPT mock-0 / loa-124 catalogs (read-only) ----
TRUTH_PATH  = "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/hcd_truth_cat.fits"
ZCAT_PATH   = "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/zcat.fits"
BAL_PATH    = "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/bal_cat.fits"
FINDER_PATH = "/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/combined_catalog/dlacat-v2.8.5-mockcat.fits"

# ---- default analysis cuts ----
NHI_MIN   = 20.3    # DLA threshold on BOTH pair members
SNR_MIN   = 2.0     # SNR_REDSIDE floor
P_DLA_MIN = 0.99    # GP detection confidence
ZQSO_MIN, ZQSO_MAX = 2.0, 4.25
LAM_RF_MIN, LAM_RF_MAX = 911.0, 1216.0   # DLA-search forest window (rest frame)
V_PROX = 3000.0     # proximity-zone / forest-edge exclusion [km/s]
ZTOL = 0.01         # truth<->finder match tolerance in |dz|/(1+z)


# =====================================================================
#  velocity separation  (the student's delta_v, which is CORRECT)
# =====================================================================
def delta_v(z1, z2):
    """Line-of-sight pair velocity separation, km/s.  c*|z1-z2|/(1+zbar)."""
    return C_KMS * np.abs(z1 - z2) / (1.0 + 0.5 * (z1 + z2))


# =====================================================================
#  per-sightline observable z_DLA window  (S4: (1+z)-correct proximity)
# =====================================================================
def zdla_window(z_qso):
    """[z_lo, z_hi] observable DLA-redshift range for a QSO at z_qso.

    Velocity offsets are converted to redshift with the (1+z) factor:
        dz = (1+z_qso) * V_PROX / C_KMS    (NOT V_PROX/C_KMS -- that was the unit bug).
    """
    z_qso = np.asarray(z_qso, float)
    dz_prox = (1.0 + z_qso) * V_PROX / C_KMS
    z_lo = np.maximum(OBS_MIN / LYA - 1.0,
                      (1.0 + z_qso) * LAM_RF_MIN / LYA - 1.0 + dz_prox)
    z_hi = np.minimum(z_qso - dz_prox,
                      (1.0 + z_qso) * LAM_RF_MAX / LYA - 1.0 - dz_prox)
    return z_lo, z_hi


# =====================================================================
#  catalog loading + selection
# =====================================================================
def load_catalogs(truth_path=TRUTH_PATH, finder_path=FINDER_PATH,
                  zcat_path=ZCAT_PATH, bal_path=BAL_PATH):
    """Load truth, finder and the QSO/BAL look-ups; attach Z_QSO to truth."""
    truth = _read_table(truth_path)
    finder = _read_table(finder_path)
    zcat = _read_table(zcat_path)
    bal = _read_table(bal_path)

    zmap = dict(zip(zcat["TARGETID"].astype(np.int64), np.asarray(zcat["Z"], float)))
    t_zqso = np.array([zmap.get(int(t), np.nan) for t in truth["TARGETID"]], float)
    bal_tids = set(bal["TARGETID"].astype(np.int64).tolist())
    return dict(truth=truth, finder=finder, t_zqso=t_zqso, bal_tids=bal_tids)


def select_truth(cat, nhi_min=NHI_MIN, snr_min=SNR_MIN, drop_bal=True):
    """Boolean mask of true DLAs passing the analysis selection.

    cat is a dict from build_truth_arrays (TARGETID, Z, NHI, SNR, Z_QSO, in_bal).
    """
    z_lo, z_hi = zdla_window(cat["Z_QSO"])
    m = cat["NHI"] >= nhi_min
    m &= cat["SNR"] > snr_min
    m &= (cat["Z_QSO"] > ZQSO_MIN) & (cat["Z_QSO"] < ZQSO_MAX)
    m &= (cat["Z"] > z_lo) & (cat["Z"] < z_hi)
    if drop_bal:
        m &= ~cat["in_bal"]
    return m


def flag_ghosts(tids, zs, nhis, dz_tol=0.006):
    """Flag Lyβ/Lyγ "ghost" detections (velocity veto).

    A detection at z_j on a sightline is a ghost if a HIGHER-NHI detection at z_i on the
    same sightline has a Lyβ- or Lyγ-implied apparent redshift
        z_ghost = (1+z_i)·λ_X/λ_Lyα − 1   (X = Lyβ or Lyγ)
    within dz_tol of z_j. This generalises the catalog's LYBETA_FLAG (which is subsumed by
    DLAFLAG==0 here) to also catch Lyγ ghosts and any unflagged residuals. The ghost pairs
    sit at Δv≈50.8k/66.6k km/s — outside the default analysis range — but the veto makes the
    clean sample faithful to the advertised behaviour.
    """
    is_ghost = np.zeros(len(tids), bool)
    grp = {}
    for i in range(len(tids)):
        grp.setdefault(int(tids[i]), []).append(i)
    for idx in grp.values():
        if len(idx) < 2:
            continue
        for j in idx:
            for i in idx:
                if i == j or nhis[i] <= nhis[j]:
                    continue
                zb = (1.0 + zs[i]) * LYB / LYA - 1.0
                zg = (1.0 + zs[i]) * LYG / LYA - 1.0
                if abs(zs[j] - zb) < dz_tol or abs(zs[j] - zg) < dz_tol:
                    is_ghost[j] = True
                    break
    return is_ghost


def select_finder(cat, nhi_min=NHI_MIN, snr_min=SNR_MIN, p_dla_min=P_DLA_MIN,
                  drop_lyb=True, drop_bal=True, drop_lyg=True):
    """Boolean mask of GP detections passing the analysis selection (the CLEAN set).

    Note: ``DLAFLAG==0`` already removes the catalog's LYBETA_FLAG rows (Lyβ ghosts carry
    DLAFLAG bit 3), so ``drop_lyb`` is belt-and-suspenders here. ``drop_lyg`` applies the
    velocity-based ghost veto (Lyβ+Lyγ) which catches residual/unflagged ghosts.
    """
    z_lo, z_hi = zdla_window(cat["Z_QSO"])
    m = cat["NHI"] >= nhi_min
    m &= cat["P_DLA"] > p_dla_min
    m &= cat["DLAFLAG"] == 0
    m &= cat["SNR_REDSIDE"] > snr_min
    m &= (cat["Z_QSO"] > ZQSO_MIN) & (cat["Z_QSO"] < ZQSO_MAX)
    m &= (cat["Z"] > z_lo) & (cat["Z"] < z_hi)
    if drop_lyb:
        m &= ~cat["LYBETA_FLAG"]
    if drop_bal:
        m &= (~cat["BAL_FLAG"]) & (~cat["in_bal"])
    if drop_lyg:
        # velocity ghost veto, computed among the quality-passed candidates
        ghost = np.zeros(len(m), bool)
        cand = np.where(m)[0]
        if cand.size:
            gflag = flag_ghosts(cat["TARGETID"][cand], cat["Z"][cand], cat["NHI"][cand])
            ghost[cand] = gflag
        m &= ~ghost
    return m


def build_truth_arrays(loaded):
    t = loaded["truth"]
    return dict(TARGETID=t["TARGETID"].astype(np.int64),
                Z=np.asarray(t["Z"], float),
                NHI=np.asarray(t["NHI"], float),
                SNR=np.asarray(t["SNR"], float),
                Z_QSO=loaded["t_zqso"],
                in_bal=np.isin(t["TARGETID"].astype(np.int64),
                               np.fromiter(loaded["bal_tids"], np.int64) if loaded["bal_tids"] else np.array([], np.int64)))


def _as_bool(col):
    """Coerce a FITS logical column to numpy bool, robust to the reader.

    fitsio returns real bool; astropy reads 'L' columns as int8 ASCII codes
    (84='T', 70='F'), where a naive .astype(bool) would make BOTH truthy. Also
    handles 0/1 ints and 'T'/'F' byte/str columns.
    """
    a = np.asarray(col)
    if a.dtype == bool:
        return a
    if a.dtype.kind in ("S", "U"):
        return np.isin(a, np.array([b"T", b"1", "T", "1"], dtype=a.dtype.kind + "1")
                       if a.dtype.kind == "S" else np.array(["T", "1"]))
    vals = set(np.unique(a).tolist())
    if vals <= {0, 1}:
        return a.astype(bool)
    return a == ord("T")   # astropy logical -> 84('T') / 70('F')


def build_finder_arrays(loaded):
    f = loaded["finder"]
    return dict(TARGETID=f["TARGETID"].astype(np.int64),
                Z=np.asarray(f["Z_DLA"], float),
                NHI=np.asarray(f["NHI"], float),
                P_DLA=np.asarray(f["P_DLA"], float),
                DLAFLAG=np.asarray(f["DLAFLAG"]),
                SNR_REDSIDE=np.asarray(f["SNR_REDSIDE"], float),
                Z_QSO=np.asarray(f["Z_QSO"], float),
                LYBETA_FLAG=_as_bool(f["LYBETA_FLAG"]),
                BAL_FLAG=_as_bool(f["BAL_FLAG"]),
                in_bal=np.isin(f["TARGETID"].astype(np.int64),
                               np.fromiter(loaded["bal_tids"], np.int64) if loaded["bal_tids"] else np.array([], np.int64)))


# =====================================================================
#  grouping + pair enumeration  (C(n,2), no double count)
# =====================================================================
def group_slices(tids):
    """Return list of (start, stop) index ranges for each sightline after sorting.
    Also returns the sort order so caller can reorder its arrays."""
    order = np.argsort(tids, kind="stable")
    st = tids[order]
    slices, i, n = [], 0, len(st)
    while i < n:
        j = i
        while j < n and st[j] == st[i]:
            j += 1
        slices.append((i, j))
        i = j
    return order, slices


def pair_dv(tids, zs, extra=None):
    """All within-sightline C(n,2) pair velocity separations.

    Returns dv array.  If ``extra`` (an array aligned with zs) is given, also returns
    the pair of extra-values (e_a, e_b) per pair (used to carry per-member 'recovered'
    or NHI). dv only otherwise.
    """
    order, slices = group_slices(tids)
    zs = zs[order]
    ex = extra[order] if extra is not None else None
    dv, ea, eb = [], [], []
    for s, e in slices:
        if e - s < 2:
            continue
        zg = zs[s:e]
        idx = range(e - s)
        for a, b in itertools.combinations(idx, 2):
            dv.append(delta_v(zg[a], zg[b]))
            if ex is not None:
                ea.append(ex[s + a]); eb.append(ex[s + b])
    dv = np.asarray(dv, float)
    if ex is None:
        return dv
    return dv, np.asarray(ea), np.asarray(eb)


# =====================================================================
#  truth <-> finder matching  -> per-truth 'recovered' flag
# =====================================================================
def match_recovered(truth, tmask, finder, fmask):
    """Greedy, one-to-one, NHI-ranked match (mirrors the production matcher).
    Returns a boolean 'recovered' array aligned with truth, True where a distinct
    eligible finder detection lies within ZTOL in (1+z)-scaled redshift."""
    recovered = np.zeros(len(truth["TARGETID"]), bool)
    # index finder detections that pass the clean selection, grouped by TARGETID
    fkeep = np.where(fmask)[0]
    fgrp = {}
    for j in fkeep:
        fgrp.setdefault(int(finder["TARGETID"][j]), []).append(j)
    # group truth (only selected truth DLAs)
    tkeep = np.where(tmask)[0]
    tgrp = {}
    for i in tkeep:
        tgrp.setdefault(int(truth["TARGETID"][i]), []).append(i)

    for tid, tidx in tgrp.items():
        fidx = fgrp.get(tid, [])
        if not fidx:
            continue
        used = np.zeros(len(fidx), bool)
        for i in tidx:
            zt, nt = truth["Z"][i], truth["NHI"][i]
            best, bestk = None, -1
            for k, j in enumerate(fidx):
                if used[k]:
                    continue
                if abs(finder["Z"][j] - zt) / (1.0 + zt) < ZTOL:
                    d = abs(finder["NHI"][j] - nt)
                    if best is None or d < best:
                        best, bestk = d, k
            if bestk >= 0:
                used[bestk] = True
                recovered[i] = True
    return recovered


# =====================================================================
#  pair-completeness  C_pair(dv)  (truth-driven, bounded <=1)
# =====================================================================
def pair_completeness(truth, tmask, recovered, bins):
    """C_pair(dv) = N(true pairs both members recovered) / N(true pairs), per dv bin."""
    dv, ra, rb = pair_dv(truth["TARGETID"][tmask], truth["Z"][tmask],
                         extra=recovered[tmask].astype(float))
    both = (ra > 0.5) & (rb > 0.5)
    n_true, _ = np.histogram(dv, bins=bins)
    n_rec, _ = np.histogram(dv[both], bins=bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.where(n_true > 0, n_rec / n_true, np.nan)
    return C, n_true, n_rec, dv, both


# =====================================================================
#  pair-purity  p(dv)  of the finder pairs (fraction that are TRUE pairs)
# =====================================================================
def finder_pair_truth_match(finder, fmask, truth, tmask):
    """For each selected finder detection, is it a true positive (matched to a
    selected truth DLA on the same sightline within ZTOL)?  Returns bool aligned
    with finder."""
    is_tp = np.zeros(len(finder["TARGETID"]), bool)
    tkeep = np.where(tmask)[0]
    tgrp = {}
    for i in tkeep:
        tgrp.setdefault(int(truth["TARGETID"][i]), []).append(i)
    for j in np.where(fmask)[0]:
        tid = int(finder["TARGETID"][j])
        zt_list = tgrp.get(tid, [])
        zf = finder["Z"][j]
        for i in zt_list:
            if abs(truth["Z"][i] - zf) / (1.0 + truth["Z"][i]) < ZTOL:
                is_tp[j] = True
                break
    return is_tp


def pair_purity(finder, fmask, is_tp, bins):
    """p(dv) = N(finder pairs with BOTH members true positives) / N(finder pairs)."""
    dv, ta, tb = pair_dv(finder["TARGETID"][fmask], finder["Z"][fmask],
                         extra=is_tp[fmask].astype(float))
    both_tp = (ta > 0.5) & (tb > 0.5)
    n_all, _ = np.histogram(dv, bins=bins)
    n_true, _ = np.histogram(dv[both_tp], bins=bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(n_all > 0, n_true / n_all, np.nan)
    return p, n_all, n_true


# =====================================================================
#  randoms: count-preserving, window-restricted  (S2, S3; renormalised in measure_xi for S8)
# =====================================================================
def random_pair_hist(tids, zs, z_qso_per_det, bins, n_real=50, seed=0):
    """Average pair-dv histogram for count-preserving, window-restricted randoms.

    For each sightline we keep its number of DLAs and draw that many redshifts from
    the GLOBAL observed-z pool restricted to that sightline's [z_lo, z_hi] window
    (so randoms follow the empirical dN/dz AND respect per-sightline selection while
    destroying line-of-sight correlation).  Averaged over n_real realisations: the RETURNED
    rr is ~1x the data per bin, but it is the mean of n_real independent draws so its sampling
    variance is reduced ~n_real-fold -- i.e. RR is effectively noiseless vs the Poisson scatter
    of DD (that, not the returned count, is the ">=10x randoms" point).

    Returns (rr_mean, rr_per_real_total).
    """
    rng = np.random.default_rng(seed)
    order, slices = group_slices(tids)
    zs = zs[order]
    zq = z_qso_per_det[order]
    global_pool = np.sort(zs)  # empirical dN/dz of the SELECTED sample
    # per-sightline window from that sightline's z_qso (constant within a sightline)
    rr = np.zeros(len(bins) - 1)
    tot = 0
    for _ in range(n_real):
        dv_all = []
        for s, e in slices:
            n_i = e - s
            if n_i < 2:
                continue
            zq_i = zq[s]
            z_lo, z_hi = zdla_window(zq_i)
            a = np.searchsorted(global_pool, z_lo, "left")
            b = np.searchsorted(global_pool, z_hi, "right")
            pool = global_pool[a:b]
            if pool.size < 2:
                # degenerate window: draw uniformly in the window (guard inverted edges)
                if z_hi > z_lo:
                    draw = rng.uniform(z_lo, z_hi, size=n_i)
                else:
                    continue
            elif pool.size >= n_i:
                draw = pool[rng.choice(pool.size, size=n_i, replace=False)]
            else:
                draw = pool[rng.integers(0, pool.size, size=n_i)]
            for x, y in itertools.combinations(range(n_i), 2):
                dv_all.append(delta_v(draw[x], draw[y]))
        h, _ = np.histogram(dv_all, bins=bins)
        rr += h
        tot += len(dv_all)
    return rr / n_real, tot


# =====================================================================
#  the estimator  1 + xi(dv) = DD / RR   (RR explicitly renormalised to in-range DD)
# =====================================================================
def measure_xi(tids, zs, z_qso_per_det, bins, n_real=50, seed=0, renormalize=True):
    """Return dict with dv_mid, DD, RR, one_plus_xi for a single catalog/selection.

    Count-preserving randoms keep per-sightline pair counts, so ΣDD≈ΣRR over the *full*
    Δv axis. But the histogram truncates at the top bin edge (30000 km/s) and DD/RR lose
    different fractions of pairs above it, so ΣDD≠ΣRR in-range. We therefore UNCONDITIONALLY
    rescale RR to the in-range DD total when ``renormalize=True`` (default):
    ``rr = rr * (dd.sum() / rr.sum())``. After this rescaling ΣRR·ξ=0 holds within the
    binned range as an algebraic consequence — i.e. ξ becomes a *shape* measurement,
    analogous to a survey integral constraint but imposed by this normalization choice,
    not by finite-volume physics. (Fixes the few-% to ~15% in-range offset.)
    """
    dd = np.histogram(pair_dv(tids, zs), bins=bins)[0].astype(float)
    rr, _ = random_pair_hist(tids, zs, z_qso_per_det, bins, n_real=n_real, seed=seed)
    if renormalize and rr.sum() > 0:
        rr = rr * (dd.sum() / rr.sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        opx = np.where(rr > 0, dd / rr, np.nan)
    return dict(dv_mid=0.5 * (bins[:-1] + bins[1:]), DD=dd, RR=rr, one_plus_xi=opx)


def bootstrap_xi(tids, zs, z_qso_per_det, bins, n_real=20, n_boot=100, seed=0):
    """Bootstrap 1+xi over SIGHTLINES (TARGETID) for error bars."""
    rng = np.random.default_rng(seed)
    order, slices = group_slices(tids)
    zs2 = zs[order]; zq2 = z_qso_per_det[order]
    multi = [(s, e) for (s, e) in slices if e - s >= 2]
    # precompute per-sightline DD contribution
    samples = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(multi), size=len(multi))
        t_b, z_b, zq_b = [], [], []
        for k, idx in enumerate(pick):
            s, e = multi[idx]
            t_b.append(np.full(e - s, k))   # relabel TARGETID so resampled copies are distinct
            z_b.append(zs2[s:e]); zq_b.append(zq2[s:e])
        t_b = np.concatenate(t_b); z_b = np.concatenate(z_b); zq_b = np.concatenate(zq_b)
        r = measure_xi(t_b, z_b, zq_b, bins, n_real=n_real, seed=int(rng.integers(1 << 30)))
        samples.append(r["one_plus_xi"])
    S = np.array(samples)
    return np.nanmean(S, axis=0), np.nanstd(S, axis=0)


# =====================================================================
#  default binning helpers
# =====================================================================
def default_bins():
    """Fine bins below 3000 km/s, then wider out to 30000 (FP spikes at ~50-67k excluded)."""
    fine = np.arange(0, 3000 + 250, 250.0)
    mid = np.array([3500, 4000, 5000, 6000, 8000, 10000, 14000, 20000, 30000.0])
    return np.unique(np.concatenate([fine, mid]))
