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
#
# WHY THE FALLBACK EXISTS: the DESI catalogs are all FITS binary tables. fitsio is the
# fast C-backed reader the DESI pipeline itself uses, but it is not always installed in a
# bare conda/Jupyter kernel. astropy.io.fits is essentially always present, so we try the
# fast path first and silently degrade to astropy if the import fails. Either way the
# caller gets back a numpy structured ("record") array, so the rest of the library never
# has to know which reader was used. The bare ``except Exception`` is intentional: an
# import can fail for many reasons (missing package, broken build, ABI mismatch) and any
# of them should trigger the fallback rather than crash.
try:
    import fitsio
    def _read_table(path):
        # fitsio.read with no HDU argument returns the first binary-table HDU as a
        # numpy structured array directly.
        return fitsio.read(path)
except Exception:  # pragma: no cover - environment dependent
    from astropy.io import fits as _fits
    def _read_table(path):
        # astropy: open the file, then walk past HDU 0 (the primary HDU never holds a
        # table) and return the first extension that actually contains data. memmap=True
        # avoids reading the whole (multi-GB) file into RAM up front.
        with _fits.open(path, memmap=True) as h:
            for hdu in h[1:]:
                if getattr(hdu, "data", None) is not None:
                    return np.asarray(hdu.data)
        raise RuntimeError(f"no table HDU in {path}")

# ---- physical constants ----
# C_KMS: the speed of light in km/s. This is the exact, defined value of c
# (299792458 m/s by SI definition) converted to km/s. It appears anywhere we turn a
# redshift difference into a velocity (delta_v) or a velocity into a redshift interval
# (the proximity-zone and ghost calculations below).
C_KMS = 299792.458
# Lyman-series REST wavelengths, in Angstrom, in vacuum. These are the wavelengths a
# hydrogen line is emitted/absorbed at in the absorber's own rest frame; the observed
# wavelength is (1+z) times these.
#   Lyman-alpha (n=2->1): the strongest H transition. DLAs are defined by their damped
#     Lyman-alpha absorption trough, and the DLA finder works in the "Lyman-alpha forest"
#     -- the stretch of QSO spectrum blueward of the QSO's own Lyman-alpha emission, which
#     is a thicket of Lyman-alpha absorbers along the line of sight. So LYA is THE
#     reference line: a feature observed at lambda_obs is interpreted as Lyman-alpha at
#     redshift z = lambda_obs/LYA - 1.
LYA = 1215.67     # Lyman-alpha rest wavelength [Angstrom]
#   Lyman-beta (n=3->1) and Lyman-gamma (n=4->1): higher Lyman-series lines of the SAME
#     absorber. A strong absorber also imprints Lyman-beta and Lyman-gamma troughs at
#     shorter rest wavelengths. When one of those higher lines lands in the forest, a
#     finder that assumes "everything is Lyman-alpha" can mis-identify it as a separate
#     Lyman-alpha absorber -- a "ghost" (see flag_ghosts). We need LYB and LYG to predict
#     exactly where those ghosts appear so we can veto them.
LYB = 1025.72     # Lyman-beta
LYG = 972.537     # Lyman-gamma
# OBS_MIN: the bluest wavelength the spectrograph records. Below ~3600 Angstrom DESI has
# no throughput, so no absorber can be SEEN bluer than this regardless of where it sits
# physically. This sets a hard floor on the lowest observable absorber redshift:
# z >= OBS_MIN/LYA - 1 (i.e. a Lyman-alpha line must redshift to at least 3600 A to be
# recorded). We impose this floor in zdla_window so the search window never includes a
# redshift the instrument literally cannot reach.
OBS_MIN = 3600.0  # DESI blue cutoff [Angstrom]

# ---- 2LPT mock-0 / loa-124 catalogs (read-only) ----
TRUTH_PATH  = "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/hcd_truth_cat.fits"
ZCAT_PATH   = "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/zcat.fits"
BAL_PATH    = "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/bal_cat.fits"
FINDER_PATH = "/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/combined_catalog/dlacat-v2.8.5-mockcat.fits"

# ---- default analysis cuts ----
# NHI_MIN: the DEFINITION of a Damped Lyman-Alpha system. A DLA is an absorber with
# neutral-hydrogen column density log10(N_HI / cm^-2) >= 20.3, i.e. N_HI >= 2e20 cm^-2
# (10^20.3 ~= 2.0e20). This is the standard threshold (Wolfe, Gawiser & Prochaska 2005,
# ARA&A) -- above it the Lyman-alpha line is "damped" (its damping wings dominate the
# profile) and the gas is self-shielded and largely neutral. We require BOTH members of a
# pair to clear this so we are correlating genuine DLAs with genuine DLAs.
NHI_MIN   = 20.3    # DLA threshold on BOTH pair members
# SNR_MIN: minimum signal-to-noise of the spectrum (measured on the red side of the
# QSO for the finder; SNR for truth). Below SNR~2 the continuum is too noisy for a DLA
# detection (or a truth injection's recoverability) to be trustworthy, so low-SNR
# sightlines are dropped to keep the data and the completeness model on equal footing.
SNR_MIN   = 2.0     # SNR_REDSIDE floor
# P_DLA_MIN: the DLA finder is a Gaussian-Process (GP) model that returns a posterior
# PROBABILITY that a given spectral feature is a real DLA. We keep only high-confidence
# detections (P_DLA > 0.99) to suppress false positives; this is the knob that trades
# completeness for purity, and 0.99 is the conservative production choice.
P_DLA_MIN = 0.99    # GP detection confidence
# ZQSO range: only use background QSOs with redshift 2.0 < z_qso < 4.25. The lower bound
# ensures the Lyman-alpha forest has redshifted far enough into the optical that an
# appreciable chunk of it sits above OBS_MIN (a low-z QSO has almost no observable forest);
# the upper bound is where this mock's QSO sample thins out and sky-line / throughput
# systematics worsen. Outside this band there are too few or too unreliable sightlines.
ZQSO_MIN, ZQSO_MAX = 2.0, 4.25
# LAM_RF_MIN / LAM_RF_MAX: the Lyman-alpha forest search window expressed in the QSO REST
# frame, in Angstrom. The red edge ~1216 A is the QSO's own Lyman-alpha emission line:
# redward of it you leave the forest. The blue edge ~911 A is the Lyman LIMIT (the
# 912 A photoelectric edge); blueward of it the spectrum is the Lyman-beta/Lyman-limit
# forest, a different and far more contaminated regime. So [911, 1216] A rest-frame is the
# stretch of QSO spectrum where Lyman-alpha absorbers (and DLAs) can be cleanly searched.
# These map to OBSERVED absorber redshifts via z = (1+z_qso)*lambda_rest/LYA - 1 (see
# zdla_window).
LAM_RF_MIN, LAM_RF_MAX = 911.0, 1216.0   # DLA-search forest window (rest frame)
# V_PROX: width of the exclusion zone, in km/s, that we trim off BOTH ends of the forest
# window. At the red end this removes the QSO "proximity zone" -- gas within ~3000 km/s of
# the QSO is ionised by the QSO's own radiation and clustered with the QSO itself, so it is
# not a fair tracer of the line-of-sight field. At the blue end it keeps us a safe margin
# off the Lyman-limit edge where Lyman-beta contamination begins. 3000 km/s is the standard
# DESI forest-edge buffer.
V_PROX = 3000.0     # proximity-zone / forest-edge exclusion [km/s]
# ZTOL: tolerance for declaring a finder detection the "same object" as a truth DLA. It is
# applied as |z_finder - z_truth| / (1 + z_truth) < ZTOL, i.e. a fractional-redshift (=
# velocity) tolerance. ZTOL=0.01 corresponds to c*ZTOL ~= 3000 km/s -- generous enough to
# absorb the finder's redshift error but tight enough that you can't accidentally match an
# unrelated neighbour. (At z~2.5 a redshift error of 0.01 IS ~3000 km/s.)
ZTOL = 0.01         # truth<->finder match tolerance in |dz|/(1+z)


# =====================================================================
#  velocity separation  (the student's delta_v, which is CORRECT)
# =====================================================================
def delta_v(z1, z2):
    """Line-of-sight pair velocity separation, in km/s, for two absorbers at z1, z2.

    Formula:  delta_v = C_KMS * |z1 - z2| / (1 + zbar),  zbar = (z1+z2)/2.

    DERIVATION (why the (1+zbar) denominator):
      Velocity and redshift are related by the relativistic Doppler/expansion relation
      1+z = sqrt((1+v/c)/(1-v/c)). Differentiating, the LOCAL conversion between a small
      redshift interval dz and a small velocity interval dv at redshift z is
          dv = c * dz / (1 + z).
      The (1+z) appears because a fixed velocity (or comoving) separation maps to a LARGER
      observed dz at higher redshift -- equivalently, the SAME dz corresponds to a SMALLER
      physical velocity separation at higher z. (Cosmological expansion stretches a given
      peculiar/comoving interval into a bigger redshift gap as z grows.) Integrating dz
      across the pair and evaluating the (1+z) at the pair's MEAN redshift gives, to first
      order in the small separation,
          delta_v ~= c * |z1 - z2| / (1 + zbar).
      This is the standard low-separation approximation used for line-of-sight clustering:
      the pairs we care about are separated by <~ a few thousand km/s, so |z1-z2| << 1 and
      the linearisation is excellent. Using zbar (rather than z1 or z2) keeps it symmetric
      in the two members. This is the SAME (1+z) physics that appears in the proximity-zone
      and ghost conversions below -- forgetting it was a real bug (see zdla_window).
    """
    return C_KMS * np.abs(z1 - z2) / (1.0 + 0.5 * (z1 + z2))


# =====================================================================
#  per-sightline observable z_DLA window  (S4: (1+z)-correct proximity)
# =====================================================================
def zdla_window(z_qso):
    """[z_lo, z_hi] observable DLA-redshift range for a QSO at redshift z_qso.

    This defines, per sightline, the band of absorber redshifts where a DLA can be both
    SEARCHED FOR (it lies in the Lyman-alpha forest) and TRUSTED (it is clear of the QSO
    proximity zone, the forest edges, and the instrument's blue cutoff). Every selection
    (truth and finder) and the randoms all use this same window so data and model match.

    HOW REST-FRAME EDGES MAP TO ABSORBER REDSHIFT:
      A QSO at z_qso emits a rest wavelength lambda_rest that we observe at
          lambda_obs = (1 + z_qso) * lambda_rest.
      An absorber seen at lambda_obs (interpreted as Lyman-alpha) has redshift
          z_abs = lambda_obs / LYA - 1 = (1 + z_qso) * lambda_rest / LYA - 1.
      So the rest-frame forest edges LAM_RF_MIN (~Lyman limit) and LAM_RF_MAX (~QSO
      Lyman-alpha emission) translate directly into a low and high absorber redshift via
      that expression -- that is the ``(1+z_qso)*LAM_RF_*/LYA - 1`` you see below.

    THE PROXIMITY / EDGE BUFFER (and the (1+z) bug it fixes):
      We trim V_PROX = 3000 km/s off both ends. A velocity offset must be converted to a
      redshift interval with the SAME (1+z) factor as delta_v:
          dz = (1 + z_qso) * V_PROX / C_KMS.
      Using plain V_PROX/C_KMS (i.e. forgetting the (1+z_qso)) was a real unit bug: at
      z_qso~2.5 it under-buffers the window by a factor of ~3.5, leaking proximity-zone
      and edge regions into the sample. The buffer is evaluated at the QSO's redshift
      because both edges live near z_qso (the high edge is essentially z_qso itself).

    THE FOUR CLAMPS:
      z_lo = max( OBS_MIN/LYA - 1 ,  blue-forest-edge + dz_prox )
        - OBS_MIN/LYA - 1 is the instrument floor: a Lyman-alpha line cannot be recorded
          below the 3600 A blue cutoff, so no absorber can be seen below this z regardless
          of the forest. We take the MORE restrictive (larger) of the two lower limits.
        - the second term is the blue forest edge pushed REDWARD (+dz_prox) to stay off the
          Lyman-limit / Lyman-beta-contaminated boundary.
      z_hi = min( z_qso - dz_prox ,  red-forest-edge - dz_prox )
        - z_qso - dz_prox excludes the proximity zone: absorbers within ~3000 km/s of the
          QSO are ionised by and clustered with the QSO, so they are not a fair tracer of
          the line-of-sight density field.
        - the second term is the red forest edge (QSO Lyman-alpha emission) pushed BLUEWARD
          (-dz_prox) for the same edge-safety reason. We take the MORE restrictive
          (smaller) of the two upper limits.
      (If z_lo ends up >= z_hi for a low-z QSO the window is empty, which the selection
      masks handle naturally by passing nothing.)
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
    """Load truth, finder and the QSO/BAL look-ups; attach Z_QSO to truth.

    The four input catalogs:
      * truth  -- the mock's injected HCD/DLA catalog (the ground truth we want to recover).
      * finder -- the GP DLA finder's output catalog run on the same mock spectra.
      * zcat   -- the QSO redshift catalog, keyed by TARGETID. The truth catalog does not
                  itself carry the BACKGROUND QSO redshift, only the absorber redshift, so
                  we have to look up each sightline's z_qso here (needed for zdla_window).
      * bal    -- list of TARGETIDs flagged as BAL (Broad Absorption Line) QSOs. BAL troughs
                  are wide intrinsic absorption features that mimic/contaminate DLAs, so any
                  sightline appearing here is dropped from both samples.
    """
    truth = _read_table(truth_path)
    finder = _read_table(finder_path)
    zcat = _read_table(zcat_path)
    bal = _read_table(bal_path)

    # Build a TARGETID -> z_qso lookup from the redshift catalog, then map it onto every
    # truth row. TARGETIDs are 64-bit -- cast to int64 so the dict keys are exact integers
    # (floats would lose precision and break the join). Missing IDs get NaN, which the
    # selection cuts then reject because NaN fails every inequality.
    zmap = dict(zip(zcat["TARGETID"].astype(np.int64), np.asarray(zcat["Z"], float)))
    t_zqso = np.array([zmap.get(int(t), np.nan) for t in truth["TARGETID"]], float)
    # A set of BAL TARGETIDs gives O(1) membership tests when we build the in_bal mask.
    bal_tids = set(bal["TARGETID"].astype(np.int64).tolist())
    return dict(truth=truth, finder=finder, t_zqso=t_zqso, bal_tids=bal_tids)


def select_truth(cat, nhi_min=NHI_MIN, snr_min=SNR_MIN, drop_bal=True):
    """Boolean mask of true DLAs passing the analysis selection.

    cat is a dict from build_truth_arrays (TARGETID, Z, NHI, SNR, Z_QSO, in_bal).

    This must apply the SAME geometric/quality cuts as select_finder so the truth sample is
    an apples-to-apples reference for completeness/purity. The cuts:
      * NHI >= NHI_MIN  -- enforce the DLA definition (log N_HI >= 20.3); sub-DLAs/LLS out.
      * SNR > SNR_MIN   -- only sightlines with usable spectra (where a finder COULD detect).
      * ZQSO in band    -- restrict to the well-behaved background-QSO redshift range.
      * Z in [z_lo,z_hi]-- absorber must lie in this sightline's clean forest window
                           (forest edges + proximity zone + blue cutoff already removed).
      * not in BAL      -- drop BAL sightlines whose intrinsic troughs contaminate the forest.
    Note ``&=`` accumulates the cuts into one boolean mask aligned with the catalog rows.
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
    """Flag Lyman-beta / Lyman-gamma "ghost" detections (velocity veto).

    WHAT A GHOST IS:
      A real absorber at redshift z_i imprints not just Lyman-alpha but the whole Lyman
      series. Its Lyman-beta trough sits at observed wavelength (1+z_i)*LYB and its
      Lyman-gamma trough at (1+z_i)*LYG. A finder that assumes "every forest feature is
      Lyman-alpha" will read those higher-series troughs as if they were Lyman-alpha lines
      of DIFFERENT, lower-redshift absorbers. Setting observed wavelengths equal,
          (1 + z_ghost) * LYA = (1 + z_i) * LYX     (X = LYB or LYG)
      gives the apparent ("ghost") redshift
          z_ghost = (1 + z_i) * LYX / LYA - 1.
      So a single strong absorber can spawn one or two spurious detections at lower z. These
      are not physical pairs -- counting them would inject a fake clustering signal.

    THE CHARACTERISTIC VELOCITY OFFSET (why ghost pairs land far out in delta_v):
      The ghost is always a fixed velocity below its parent, set purely by the line ratio:
          delta_v ~= c * ln(LYA / LYX).
      (The log form is the exact velocity for a finite wavelength ratio: 1+z = lambda_obs/
      lambda_rest, and integrating dv = c dz/(1+z) across a ratio r gives delta_v = c ln r.)
      Numerically:
          Lyman-beta:  c * ln(1215.67/1025.72) ~= 50,900 km/s
          Lyman-gamma: c * ln(1215.67/972.537) ~= 66,900 km/s
      So ghost pairs pile up near ~50.9k and ~66.9k km/s -- well OUTSIDE the default
      clustering range (top bin 30,000 km/s), which is why default_bins deliberately stops
      below them. Vetoing here keeps the clean sample honest even though those bins are
      excluded anyway.

    THE VETO LOGIC:
      Within each sightline, for every detection z_j we ask: does some HIGHER-column
      detection z_i on the same sightline predict a ghost at exactly z_j? We only let a
      HIGHER-NHI line "cast" a ghost (nhis[i] > nhis[j]) because the parent absorber is the
      stronger one and the ghost is the weaker apparent line -- this prevents two real
      absorbers from vetoing each other. ``dz_tol`` = 0.006 is the redshift match window
      (~a few hundred km/s at z~2.5), wide enough to absorb finder redshift error but narrow
      enough not to flag chance coincidences. A detection flagged True is dropped.
    """
    is_ghost = np.zeros(len(tids), bool)
    # Group detection indices by sightline (TARGETID) so we only compare lines that share a
    # line of sight -- ghosts are an intra-sightline artefact.
    grp = {}
    for i in range(len(tids)):
        grp.setdefault(int(tids[i]), []).append(i)
    for idx in grp.values():
        if len(idx) < 2:
            continue  # a single detection cannot have a parent, so no ghost is possible
        for j in idx:
            for i in idx:
                # i must be a DISTINCT, STRONGER line to be a candidate parent for ghost j.
                if i == j or nhis[i] <= nhis[j]:
                    continue
                # predicted ghost redshifts of parent i via its Lyman-beta / Lyman-gamma lines
                zb = (1.0 + zs[i]) * LYB / LYA - 1.0
                zg = (1.0 + zs[i]) * LYG / LYA - 1.0
                if abs(zs[j] - zb) < dz_tol or abs(zs[j] - zg) < dz_tol:
                    is_ghost[j] = True
                    break  # one parent is enough to condemn j; stop scanning
    return is_ghost


def select_finder(cat, nhi_min=NHI_MIN, snr_min=SNR_MIN, p_dla_min=P_DLA_MIN,
                  drop_lyb=True, drop_bal=True, drop_lyg=True):
    """Boolean mask of GP detections passing the analysis selection (the CLEAN set).

    Mirrors select_truth's geometric cuts (NHI, ZQSO band, z-window) and adds the
    finder-specific quality cuts:
      * P_DLA > P_DLA_MIN     -- keep only high-confidence GP detections (purity knob).
      * DLAFLAG == 0          -- the production "this detection is clean" flag. DLAFLAG is a
                                 bitmask; ==0 means no problem bits are set. In particular
                                 the catalog's Lyman-beta-ghost bit lives inside DLAFLAG, so
                                 DLAFLAG==0 already removes catalog-flagged Lyman-beta ghosts.
      * SNR_REDSIDE > SNR_MIN -- spectrum SNR measured on the red side of the QSO emission.
      * BAL removal           -- drop BAL-flagged detections AND any detection on a sightline
                                 in the BAL catalog (two independent BAL bookkeeping sources).

    Note: because DLAFLAG==0 already drops catalog Lyman-beta ghosts, ``drop_lyb``
    (the explicit LYBETA_FLAG cut) is belt-and-suspenders. ``drop_lyg`` runs our own
    velocity-based ghost veto (flag_ghosts), which additionally catches Lyman-GAMMA ghosts
    and any residual Lyman-beta ghosts the catalog flag missed.
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
        # Velocity ghost veto, computed ONLY among the candidates that already passed every
        # quality cut above. We run flag_ghosts on that subset (so a vetoed real DLA cannot
        # be cast as a ghost by a junk line) and scatter its result back into a full-length
        # mask before AND-ing it in.
        ghost = np.zeros(len(m), bool)
        cand = np.where(m)[0]
        if cand.size:
            gflag = flag_ghosts(cat["TARGETID"][cand], cat["Z"][cand], cat["NHI"][cand])
            ghost[cand] = gflag
        m &= ~ghost
    return m


def build_truth_arrays(loaded):
    # Repackage the raw truth FITS record array into a plain dict of typed numpy arrays
    # with the canonical key names the selection/pairing code expects. This decouples the
    # rest of the library from the catalog's exact column names and dtypes. Z here is the
    # ABSORBER redshift; Z_QSO is the background-QSO redshift we looked up in load_catalogs.
    t = loaded["truth"]
    return dict(TARGETID=t["TARGETID"].astype(np.int64),
                Z=np.asarray(t["Z"], float),
                NHI=np.asarray(t["NHI"], float),
                SNR=np.asarray(t["SNR"], float),
                Z_QSO=loaded["t_zqso"],
                # in_bal: True for any truth row whose sightline is in the BAL catalog.
                # np.isin needs an array, so convert the BAL-id set to int64; if it is empty
                # use an empty int64 array (np.isin against nothing returns all-False).
                in_bal=np.isin(t["TARGETID"].astype(np.int64),
                               np.fromiter(loaded["bal_tids"], np.int64) if loaded["bal_tids"] else np.array([], np.int64)))


def _as_bool(col):
    """Coerce a FITS logical column to numpy bool, robust to the reader.

    THE GOTCHA THIS GUARDS AGAINST:
      FITS has a logical column type ('L', values 'T'/'F'). The two readers expose it
      differently:
        * fitsio returns a genuine numpy bool array -- nothing to do.
        * astropy reads 'L' columns as the raw ASCII byte CODES: 84 for 'T', 70 for 'F'.
          Both 84 and 70 are non-zero, so a naive ``col.astype(bool)`` would map EVERY row
          to True -- a silent, catastrophic bug (every detection would look BAL-flagged or
          ghost-flagged). So we must compare against the 'T' code (ord('T')==84) instead.
      We also accept 0/1 integer columns and 'T'/'F' byte/string columns, so the same
      helper works no matter how a given catalog stored its flag.

    Returns a real numpy bool array, True only for genuine "T"/1 values.
    """
    a = np.asarray(col)
    if a.dtype == bool:
        return a  # fitsio path: already correct
    if a.dtype.kind in ("S", "U"):
        # byte ('S') or unicode ('U') strings: True for the "T"/"1" tokens. The dtype-kind
        # juggling keeps the comparison array the same kind ('S1' vs 'U') as the column.
        return np.isin(a, np.array([b"T", b"1", "T", "1"], dtype=a.dtype.kind + "1")
                       if a.dtype.kind == "S" else np.array(["T", "1"]))
    vals = set(np.unique(a).tolist())
    if vals <= {0, 1}:
        return a.astype(bool)  # genuine 0/1 integer flag: safe to cast directly
    return a == ord("T")   # astropy logical -> 84('T') / 70('F'); True only where ==84


def build_finder_arrays(loaded):
    # Same idea as build_truth_arrays, for the finder catalog. Two naming notes:
    #   * the finder's absorber redshift column is Z_DLA (not Z) -- we rename it to Z so the
    #     pairing/matching code is column-name agnostic between truth and finder.
    #   * the logical flag columns (LYBETA_FLAG, BAL_FLAG) go through _as_bool to dodge the
    #     astropy 'T'/'F'-as-ASCII-code trap documented in _as_bool.
    # Z_QSO comes straight from the finder catalog here (the finder records the QSO redshift
    # it assumed), unlike truth where we had to look it up in zcat.
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
    Also returns the sort order so caller can reorder its arrays.

    Pairs must be formed WITHIN a single sightline (TARGETID), never across sightlines, so
    we first sort by TARGETID -- which makes every sightline a contiguous block -- and then
    record each block as a (start, stop) half-open slice. A STABLE sort is used so detections
    keep their original relative order within a sightline (reproducibility, and it makes the
    accompanying ``order`` permutation deterministic). The caller applies ``order`` to its own
    z/NHI/etc. arrays so they line up with these slices.
    """
    order = np.argsort(tids, kind="stable")
    st = tids[order]
    slices, i, n = [], 0, len(st)
    # Walk the sorted ids; each inner loop advances j to the end of the current id's run.
    while i < n:
        j = i
        while j < n and st[j] == st[i]:
            j += 1
        slices.append((i, j))
        i = j
    return order, slices


def pair_dv(tids, zs, extra=None):
    """All within-sightline C(n,2) pair velocity separations.

    For a sightline with n detections there are C(n,2) = n*(n-1)/2 unordered pairs.
    itertools.combinations enumerates each unordered pair EXACTLY ONCE (a<b), so no pair is
    double-counted and no self-pair is formed -- this is the correct pair census for a
    line-of-sight pair-count clustering estimator. Sightlines with fewer than 2 detections
    contribute nothing.

    Returns dv array.  If ``extra`` (an array aligned with zs) is given, also returns
    the pair of extra-values (e_a, e_b) per pair (used to carry per-member 'recovered'
    or NHI flags through to the per-pair completeness/purity bookkeeping). dv only otherwise.
    """
    order, slices = group_slices(tids)
    zs = zs[order]
    ex = extra[order] if extra is not None else None  # reorder the payload the same way
    dv, ea, eb = [], [], []
    for s, e in slices:
        if e - s < 2:
            continue  # need >=2 detections on the sightline to form a pair
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
    eligible finder detection lies within ZTOL in (1+z)-scaled redshift.

    WHY ONE-TO-ONE (and why greedy):
      Each true DLA can be recovered by AT MOST ONE finder detection, and each finder
      detection can claim AT MOST ONE truth (the ``used`` array enforces this). Without the
      one-to-one constraint a single finder line could "recover" several nearby truths,
      inflating completeness. The match is GREEDY: we don't solve a global optimal
      assignment, we just walk the truths and let each grab its best free candidate. This
      exactly mirrors the production DESI matcher, which matters because we want our
      completeness model to reflect the SAME matching the real analysis uses.

    THE MATCH CRITERION:
      A finder detection is eligible for truth i if it is on the same sightline and within
      the velocity tolerance  |z_finder - z_truth| / (1 + z_truth) < ZTOL  (the (1+z)
      scaling makes ZTOL a fractional-redshift = velocity tolerance, consistent with
      delta_v). Among eligible free candidates we pick the one whose NHI is CLOSEST to the
      truth's NHI (min |NHI_finder - NHI_truth|) -- so when several detections sit near a
      truth, the column-density-matched one wins, which is the physically right pairing.
    """
    recovered = np.zeros(len(truth["TARGETID"]), bool)
    # Index finder detections that pass the clean selection, grouped by sightline, so each
    # truth only has to scan candidates on its own line of sight.
    fkeep = np.where(fmask)[0]
    fgrp = {}
    for j in fkeep:
        fgrp.setdefault(int(finder["TARGETID"][j]), []).append(j)
    # Group the selected truth DLAs the same way.
    tkeep = np.where(tmask)[0]
    tgrp = {}
    for i in tkeep:
        tgrp.setdefault(int(truth["TARGETID"][i]), []).append(i)

    for tid, tidx in tgrp.items():
        fidx = fgrp.get(tid, [])
        if not fidx:
            continue  # no finder detections on this sightline -> none of its truths recovered
        used = np.zeros(len(fidx), bool)  # tracks which candidates are already claimed
        for i in tidx:
            zt, nt = truth["Z"][i], truth["NHI"][i]
            best, bestk = None, -1   # best = smallest |dNHI| so far; bestk = its slot in fidx
            for k, j in enumerate(fidx):
                if used[k]:
                    continue
                if abs(finder["Z"][j] - zt) / (1.0 + zt) < ZTOL:
                    d = abs(finder["NHI"][j] - nt)
                    if best is None or d < best:
                        best, bestk = d, k
            if bestk >= 0:
                used[bestk] = True   # consume the matched candidate so no other truth reuses it
                recovered[i] = True
    return recovered


# =====================================================================
#  pair-completeness  C_pair(dv)  (truth-driven, bounded <=1)
# =====================================================================
def pair_completeness(truth, tmask, recovered, bins):
    """C_pair(dv) = N(true pairs with BOTH members recovered) / N(true pairs), per dv bin.

    This is the PAIR completeness, and it is what you must divide the observed pair counts
    by -- NOT the single-object completeness. Key facts:
      * It is measured entirely from TRUTH (we know which truths the finder recovered via
        match_recovered), so it is an unbiased model of the finder's pair-detection
        efficiency as a function of separation.
      * It is bounded <= 1 by construction (a recovered-both subset of all true pairs) and
        is applied to the data exactly ONCE.
      * Why C_pair can be SMALLER than C_single^2: if detections were independent you'd
        expect P(both) = C_single * C_single. But close pairs BLEND -- two DLAs separated by
        a small delta_v overlap spectrally and the finder tends to miss one or merge them.
        That makes recovery of the two members POSITIVELY correlated in the bad direction at
        small dv, pushing C_pair below C_single^2 in the close-pair bins. Measuring C_pair
        per dv bin captures this separation-dependent blending automatically.

    The ``extra=recovered`` payload carries each member's recovered flag (as 0.0/1.0)
    through pair_dv so that for every true pair we know whether BOTH members were recovered.
    """
    dv, ra, rb = pair_dv(truth["TARGETID"][tmask], truth["Z"][tmask],
                         extra=recovered[tmask].astype(float))
    both = (ra > 0.5) & (rb > 0.5)   # 0.5 threshold: the floats are exactly 0.0 or 1.0
    n_true, _ = np.histogram(dv, bins=bins)        # all true pairs per bin (denominator)
    n_rec, _ = np.histogram(dv[both], bins=bins)   # true pairs both recovered (numerator)
    with np.errstate(invalid="ignore", divide="ignore"):
        # empty bins (n_true==0) -> NaN rather than 0/0 warnings/garbage
        C = np.where(n_true > 0, n_rec / n_true, np.nan)
    return C, n_true, n_rec, dv, both


# =====================================================================
#  pair-purity  p(dv)  of the finder pairs (fraction that are TRUE pairs)
# =====================================================================
def finder_pair_truth_match(finder, fmask, truth, tmask):
    """For each selected finder detection, is it a true positive (matched to a
    selected truth DLA on the same sightline within ZTOL)?  Returns bool aligned
    with finder.

    This is the per-DETECTION truth-positive flag that pair_purity builds on. Unlike
    match_recovered it is NOT one-to-one: here we only ask "does this finder detection have
    ANY selected truth DLA within ZTOL on its sightline?" -- because for purity we care
    whether each detection is real, not about a unique assignment. Same velocity criterion
    |z_truth - z_finder|/(1+z_truth) < ZTOL as everywhere else.
    """
    is_tp = np.zeros(len(finder["TARGETID"]), bool)
    tkeep = np.where(tmask)[0]
    tgrp = {}
    for i in tkeep:
        tgrp.setdefault(int(truth["TARGETID"][i]), []).append(i)
    for j in np.where(fmask)[0]:
        tid = int(finder["TARGETID"][j])
        zt_list = tgrp.get(tid, [])   # selected truths on the same sightline
        zf = finder["Z"][j]
        for i in zt_list:
            if abs(truth["Z"][i] - zf) / (1.0 + truth["Z"][i]) < ZTOL:
                is_tp[j] = True   # found a real DLA this detection matches -> true positive
                break
    return is_tp


def pair_purity(finder, fmask, is_tp, bins):
    """p(dv) = N(finder pairs with BOTH members true positives) / N(finder pairs), per bin.

    The companion correction to pair_completeness. Even after the quality cuts and ghost
    veto, some finder detections are false positives; a finder PAIR is only a real pair if
    BOTH members are true positives. p(dv) measures, per separation bin, the fraction of
    finder pairs that are genuine, so the observed pair counts can be DEFLATED by p(dv) to
    remove residual false-positive dilution. Like C_pair it is in [0,1] and separation
    dependent (false positives can cluster at particular dv, e.g. near residual ghosts).
    Multiplying the data by p(dv)/C_pair(dv) removes contamination and restores
    incompleteness in one calibrated step.
    """
    dv, ta, tb = pair_dv(finder["TARGETID"][fmask], finder["Z"][fmask],
                         extra=is_tp[fmask].astype(float))
    both_tp = (ta > 0.5) & (tb > 0.5)   # both members are true positives
    n_all, _ = np.histogram(dv, bins=bins)         # all finder pairs (denominator)
    n_true, _ = np.histogram(dv[both_tp], bins=bins)  # both-true-positive pairs (numerator)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(n_all > 0, n_true / n_all, np.nan)
    return p, n_all, n_true


# =====================================================================
#  randoms: count-preserving, window-restricted  (S2, S3; renormalised in measure_xi for S8)
# =====================================================================
def random_pair_hist(tids, zs, z_qso_per_det, bins, n_real=50, seed=0):
    """Average pair-dv histogram for count-preserving, window-restricted randoms.

    THE ROLE OF RANDOMS: the clustering signal is DD/RR, where RR is the pair count you'd
    get with NO line-of-sight correlation but the SAME selection geometry. So the randoms
    must reproduce everything about the data EXCEPT the clustering: the redshift distribution
    dN/dz, the per-sightline observable window, and the number of objects per sightline --
    while scrambling their relative positions.

    HOW THESE RANDOMS ARE BUILT (the three "right things"):
      1. COUNT-PRESERVING: each sightline keeps its OWN number of detections n_i, so it
         contributes the same C(n_i,2) pairs as the data. This makes Sum DD ~ Sum RR over the
         FULL dv axis automatically and preserves the integral (see measure_xi).
      2. WINDOW-RESTRICTED: the n_i random redshifts for a sightline are drawn only from
         within THAT sightline's [z_lo, z_hi] = zdla_window(z_qso). A random pair can only
         exist where a real one could, so the selection geometry is matched exactly.
      3. EMPIRICAL dN/dz: redshifts are resampled from the GLOBAL pool of observed redshifts
         (sorted, then sliced to the window), so randoms inherit the real redshift
         distribution rather than an assumed analytic form -- but drawing INDEPENDENTLY per
         sightline destroys the line-of-sight correlation, which is exactly what RR needs.

    WHY AVERAGE OVER n_real REALISATIONS: the RETURNED rr is the MEAN over n_real draws, so
    per bin it is ~1x the data count -- NOT 10x. The point of n_real is variance reduction:
    the sampling (Poisson) variance of the mean falls like 1/n_real, so with n_real~50 the RR
    histogram is effectively NOISELESS compared to the Poisson scatter of DD. That -- "RR
    contributes negligible noise" -- is what ">=10x randoms" really buys you, achieved here
    by averaging rather than by inflating the counts.

    Returns (rr_mean, rr_per_real_total): the mean per-bin histogram, and the total pair
    count summed across all realisations (a diagnostic of how many randoms were actually drawn).
    """
    rng = np.random.default_rng(seed)
    order, slices = group_slices(tids)
    zs = zs[order]
    zq = z_qso_per_det[order]
    global_pool = np.sort(zs)  # empirical dN/dz of the SELECTED sample; sorted for searchsorted
    # per-sightline window from that sightline's z_qso (constant within a sightline)
    rr = np.zeros(len(bins) - 1)
    tot = 0
    for _ in range(n_real):
        dv_all = []
        for s, e in slices:
            n_i = e - s
            if n_i < 2:
                continue   # no pairs from this sightline; nothing to draw
            zq_i = zq[s]   # z_qso is constant within a sightline, so any row works
            z_lo, z_hi = zdla_window(zq_i)
            # Slice the sorted global pool to this window. searchsorted with "left"/"right"
            # gives the half-open index range [a,b) of pool redshifts inside [z_lo, z_hi].
            a = np.searchsorted(global_pool, z_lo, "left")
            b = np.searchsorted(global_pool, z_hi, "right")
            pool = global_pool[a:b]
            if pool.size < 2:
                # Degenerate window with too few pool members to resample from: fall back to a
                # uniform draw across the window (still respects geometry). Guard against an
                # inverted window (z_hi <= z_lo) which would make rng.uniform invalid -- skip it.
                if z_hi > z_lo:
                    draw = rng.uniform(z_lo, z_hi, size=n_i)
                else:
                    continue
            elif pool.size >= n_i:
                # Enough distinct pool redshifts: draw WITHOUT replacement so the random
                # sightline mimics n_i distinct absorbers (no artificial dv=0 pairs).
                draw = pool[rng.choice(pool.size, size=n_i, replace=False)]
            else:
                # Pool smaller than n_i: must draw WITH replacement to get n_i values.
                draw = pool[rng.integers(0, pool.size, size=n_i)]
            for x, y in itertools.combinations(range(n_i), 2):
                dv_all.append(delta_v(draw[x], draw[y]))
        h, _ = np.histogram(dv_all, bins=bins)
        rr += h
        tot += len(dv_all)
    return rr / n_real, tot   # divide by n_real -> the MEAN histogram (variance ~ 1/n_real)


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
    # DD: the actual data pair-separation histogram. RR: the noise-suppressed randoms.
    dd = np.histogram(pair_dv(tids, zs), bins=bins)[0].astype(float)
    rr, _ = random_pair_hist(tids, zs, z_qso_per_det, bins, n_real=n_real, seed=seed)
    # Renormalise RR to the in-range DD total. Count-preserving randoms balance DD over the
    # FULL axis, but the histogram truncates at the top bin edge and DD vs RR lose different
    # fractions above it -- so without this rescaling Sum DD != Sum RR within the binned
    # range and 1+xi would carry a spurious overall offset. rr.sum()>0 guards against an
    # empty-RR divide.
    if renormalize and rr.sum() > 0:
        rr = rr * (dd.sum() / rr.sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        # 1 + xi(dv) = DD/RR. Bins with RR==0 -> NaN (undefined, not a measurement).
        opx = np.where(rr > 0, dd / rr, np.nan)
    # dv_mid is the bin-center array (mid-point of each bin edge pair) for plotting.
    return dict(dv_mid=0.5 * (bins[:-1] + bins[1:]), DD=dd, RR=rr, one_plus_xi=opx)


def bootstrap_xi(tids, zs, z_qso_per_det, bins, n_real=20, n_boot=100, seed=0):
    """Bootstrap 1+xi over SIGHTLINES (TARGETID) for error bars.

    The error bars must reflect SAMPLE variance -- how much 1+xi would wobble if we'd drawn
    a different set of sightlines. So we resample the SIGHTLINES (not individual detections)
    WITH replacement, n_boot times, recomputing 1+xi each time, and take the spread.
    Resampling whole sightlines is the right unit because pairs are correlated within a
    sightline; resampling pairs would underestimate the error. Only multi-detection
    sightlines (>=2) can produce pairs, so we bootstrap over ``multi``.

    Returns (mean 1+xi across bootstraps, std 1+xi across bootstraps = the error bar).
    """
    rng = np.random.default_rng(seed)
    order, slices = group_slices(tids)
    zs2 = zs[order]; zq2 = z_qso_per_det[order]
    multi = [(s, e) for (s, e) in slices if e - s >= 2]
    # precompute per-sightline DD contribution
    samples = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(multi), size=len(multi))   # indices WITH replacement
        t_b, z_b, zq_b = [], [], []
        for k, idx in enumerate(pick):
            s, e = multi[idx]
            # CRUCIAL: relabel each picked sightline with a fresh id k. If the same sightline
            # is drawn twice, the two copies must be DISTINCT ids -- otherwise group_slices
            # would merge them and invent cross-copy pairs that never existed.
            t_b.append(np.full(e - s, k))
            z_b.append(zs2[s:e]); zq_b.append(zq2[s:e])
        t_b = np.concatenate(t_b); z_b = np.concatenate(z_b); zq_b = np.concatenate(zq_b)
        # Re-seed each measure_xi from the bootstrap rng so realisations are independent but
        # the whole run is reproducible from the top-level seed.
        r = measure_xi(t_b, z_b, zq_b, bins, n_real=n_real, seed=int(rng.integers(1 << 30)))
        samples.append(r["one_plus_xi"])
    S = np.array(samples)
    # nan-aware so empty bins (NaN in some realisations) don't poison the statistics.
    return np.nanmean(S, axis=0), np.nanstd(S, axis=0)


# =====================================================================
#  default binning helpers
# =====================================================================
def default_bins():
    """Delta_v bin edges, km/s: fine below 3000, coarsening out to 30000.

    The binning choices encode the physics we want to resolve:
      * FINE (250 km/s) bins from 0 to 3000 km/s: the clustering signal lives at small
        separations, so this is where we want resolution. (3000 km/s is also roughly the
        proximity/ZTOL velocity scale.)
      * COARSER bins out to 30,000 km/s: the signal is flat and pair counts are sparse at
        large dv, so wide bins keep S/N up without losing information.
      * TOP EDGE 30,000 km/s -- chosen DELIBERATELY below the Lyman-beta/Lyman-gamma ghost
        velocities (~50,900 and ~66,900 km/s; see flag_ghosts). Any residual ghost-induced
        false-positive spikes sit ABOVE the range and so cannot contaminate the measured xi.
    np.unique sorts and de-duplicates the merged edge list (the 3000 endpoint appears in
    both ``fine`` and just below ``mid``).
    """
    fine = np.arange(0, 3000 + 250, 250.0)
    mid = np.array([3500, 4000, 5000, 6000, 8000, 10000, 14000, 20000, 30000.0])
    return np.unique(np.concatenate([fine, mid]))
