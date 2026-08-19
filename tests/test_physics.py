#!/usr/bin/env python
"""
Physics unit tests for the 1D DLA clustering engine.

Checks that the *physics* in `scripts/clustering_lib.py` and `scripts/fit_dla_bias.py`
is sound — independent of the (cluster-only) data catalogs, so it runs anywhere with
numpy/scipy/astropy. Covers:

  COSMOLOGY (fit_dla_bias.MatterXi)
    - σ8 normalisation reproduces 0.831
    - growth factor: D(0)=1, monotone, D(2.4)≈0.37
    - EH98 transfer function limits: T(k→0)→1, suppressed at high k, monotone
    - P(k) turnover at the matter-radiation equality scale
    - ξ_matter(r): positive, monotone-declining, sensible magnitude at 8 Mpc/h
    - Δv→r = Δv(1+z)/H(z)·h  (1000 km/s ≈ 9.3 Mpc/h at z=2.5)
    - small-scale cap r_cut saturates the template at tiny Δv
    - growth rate f(z)=Ωm(z)^0.55
    - the bias fit recovers an injected amplitude b² exactly

  ESTIMATOR / SELECTION (clustering_lib)
    - delta_v = c|Δz|/(1+z̄): value, symmetry, zero on self
    - zdla_window carries the (1+z) factor (the unit fix)
    - flag_ghosts vetoes a Lyβ ghost but keeps a genuine close pair
    - pair enumeration: C(n,2) per sightline
    - measure_xi on a synthetic UNCLUSTERED catalog → 1+ξ ≈ 1 (unbiased null)
    - measure_xi on a catalog with injected close pairs → 1+ξ > 1 at small Δv

Run:  python tests/test_physics.py     (standalone; prints PASS/FAIL)
  or: pytest tests/test_physics.py
"""
import os, sys, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import clustering_lib as L
import fit_dla_bias as F

C = L.C_KMS

# ---- build the (slow) cosmology object once, lazily ----
_MX = None
def mx():
    global _MX
    if _MX is None:
        _MX = F.MatterXi()
    return _MX


# =====================================================================
#  COSMOLOGY
# =====================================================================
def test_sigma8_normalization():
    """The P(k) amplitude is set so σ8 = 0.831 in 8 Mpc/h spheres."""
    s8 = math.sqrt(mx()._sigma2(8.0, mx()._norm))
    assert abs(s8 - F.SIGMA8) < 2e-3, f"σ8={s8}, expected {F.SIGMA8}"


def test_growth_factor():
    """D(0)=1 exactly, D decreasing with z, D(2.4)≈0.37, D(2.5)≈0.36."""
    m = mx()
    assert abs(m.D(0.0) - 1.0) < 1e-3
    zs = [0.0, 0.5, 1.0, 2.0, 3.0]
    Ds = [m.D(z) for z in zs]
    assert all(Ds[i] > Ds[i+1] for i in range(len(Ds)-1)), f"D not monotone: {Ds}"
    assert abs(m.D(2.4) - 0.370) < 0.02, f"D(2.4)={m.D(2.4)}"
    assert abs(m.D(2.5) - 0.359) < 0.02, f"D(2.5)={m.D(2.5)}"


def test_transfer_function_limits():
    """T(k→0)→1; strongly suppressed at high k; monotone decreasing."""
    m = mx()
    assert m._T(1e-4) > 0.99, f"T(1e-4)={m._T(1e-4)}"
    assert m._T(10.0) < 0.02, f"T(10)={m._T(10.0)}"
    ks = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
    Ts = [float(m._T(k)) for k in ks]
    assert all(Ts[i] > Ts[i+1] for i in range(len(Ts)-1)), f"T not monotone: {Ts}"
    assert all(0.0 < t <= 1.0001 for t in Ts)


def test_Pk_turnover():
    """P(k) peaks near the matter-radiation equality scale (~0.015 h/Mpc)."""
    m = mx()
    p_low, p_eq, p_hi = float(m._Pk(1e-3)), float(m._Pk(0.015)), float(m._Pk(0.5))
    assert p_eq > p_low and p_eq > p_hi, f"no turnover: {p_low}, {p_eq}, {p_hi}"


def test_xi_matter_shape():
    """ξ_matter(r): positive on small/intermediate scales, monotone-declining,
    O(0.1–1) at 8 Mpc/h."""
    m = mx()
    r = np.array([2.0, 5.0, 8.0, 20.0, 50.0])
    xi = m.xi_matter_z0(r)
    assert np.all(xi[:-1] > xi[1:]), f"ξ_matter not declining: {xi}"
    assert np.all(xi[r <= 20.0] > 0), f"ξ_matter not positive on small scales: {xi}"
    xi8 = float(m.xi_matter_z0(8.0)[0])
    assert 0.1 < xi8 < 1.5, f"ξ_matter(8 Mpc/h)={xi8} out of expected range"


def test_dv_to_r_conversion():
    """r = Δv(1+z)/H(z)·h ; 1000 km/s ≈ 9.3 Mpc/h at z=2.5; H(2.5)≈254 km/s/Mpc."""
    m = mx()
    z = 2.5
    Hz = m.cosmo.H(z).value
    assert abs(Hz - 254.0) < 4.0, f"H(2.5)={Hz}"
    r = 1000.0 * (1 + z) / Hz * m.h
    assert abs(r - 9.3) < 0.3, f"1000 km/s -> {r} Mpc/h (expected ~9.3)"


def test_small_scale_cap():
    """Template saturates at very small Δv because r is capped at r_cut."""
    m = mx()
    t_tiny = m.template_1plus_xi(np.array([1e-3]), 2.5)[0]
    t_small = m.template_1plus_xi(np.array([1.0]), 2.5)[0]
    assert abs(t_tiny - t_small) / t_small < 1e-3, "template not capped at small Δv"


def test_growth_rate_f():
    """f(z)=Ωm(z)^0.55: f(2.5)≈0.97, increasing toward 1 with z, f(0)<f(2.5)."""
    f0, f25, f4 = F.f_growth(0.0), F.f_growth(2.5), F.f_growth(4.0)
    assert abs(f25 - 0.97) < 0.02, f"f(2.5)={f25}"
    assert f0 < f25 < f4 < 1.0, f"f not increasing: {f0}, {f25}, {f4}"


def test_bias_fit_recovers_amplitude():
    """fit_amplitude must return the injected b² exactly (linear LSQ + integral constraint)."""
    m = mx()
    dv = np.array([2000., 3000., 4000., 6000., 9000., 14000., 20000.])
    RR = np.ones_like(dv)
    template = m.template_1plus_xi(dv, 2.4)
    b2_true = 4.0
    ic = (RR * template).sum() / RR.sum()
    opx = 1.0 + b2_true * (template - ic)            # synthetic, noiseless
    err = np.full_like(dv, 0.01)
    mask = np.ones_like(dv, bool)
    b2, b2e, chi2dof, ic_fit, n = F.fit_amplitude(dv, opx, err, RR, template, mask)
    assert abs(b2 - b2_true) < 1e-6, f"recovered b²={b2}, injected {b2_true}"
    assert n == len(dv)


# =====================================================================
#  ESTIMATOR / SELECTION
# =====================================================================
def test_delta_v():
    """Δv = c|z1−z2|/(1+z̄): known value, symmetric, zero on equal z."""
    z1, z2 = 2.00, 2.01
    expect = C * 0.01 / (1 + 2.005)
    assert abs(L.delta_v(z1, z2) - expect) < 1e-6
    assert abs(L.delta_v(z1, z2) - L.delta_v(z2, z1)) < 1e-12   # symmetric
    assert L.delta_v(2.5, 2.5) == 0.0


def test_zdla_window_carries_1plusz():
    """The proximity exclusion uses Δz=(1+z)·V_PROX/c (the unit fix), not V_PROX/c."""
    zq = 3.0
    z_lo, z_hi = L.zdla_window(np.array([zq]))
    z_hi = float(z_hi[0])
    # at z_qso=3 the proximity term binds the red edge:
    expect_offset = (1 + zq) * L.V_PROX / C            # ≈ 0.040  (NOT 0.010)
    assert abs((zq - z_hi) - expect_offset) < 1e-6, f"red-edge offset {zq-z_hi}, expected {expect_offset}"
    assert (zq - z_hi) > 3.0 * (L.V_PROX / C), "missing the (1+z) factor"
    assert z_lo[0] < z_hi, "window inverted"


def test_ghost_veto():
    """flag_ghosts vetoes a Lyβ-ghost (lower-NHI absorber at the ghost z) but
    keeps a genuine close pair at the same separation if NHI ordering doesn't match."""
    z_real = 2.6
    z_ghost = (1 + z_real) * L.LYB / L.LYA - 1.0       # Lyβ ghost redshift
    tids = np.array([100, 100])
    zs = np.array([z_real, z_ghost])
    nhis = np.array([21.5, 20.4])                      # parent stronger than ghost
    flag = L.flag_ghosts(tids, zs, nhis)
    assert flag[1] and not flag[0], f"ghost not vetoed correctly: {flag}"
    # a genuine pair at an unrelated separation is NOT flagged
    zs2 = np.array([2.60, 2.62]); nhis2 = np.array([21.0, 20.8])
    flag2 = L.flag_ghosts(np.array([200, 200]), zs2, nhis2)
    assert not flag2.any(), f"genuine pair wrongly vetoed: {flag2}"


def test_pair_enumeration():
    """pair_dv enumerates C(n,2) within-sightline pairs (and none for singletons)."""
    tids = np.array([1, 1, 1, 2, 2, 3])               # sightline 1: 3 DLAs, 2: 2, 3: 1
    zs   = np.array([2.3, 2.4, 2.5, 2.6, 2.7, 2.8])
    dv = L.pair_dv(tids, zs)
    assert len(dv) == 3 + 1 + 0, f"expected C(3,2)+C(2,2)+0=4 pairs, got {len(dv)}"
    assert np.all(dv > 0)


def _synthetic_catalog(n_sight=3000, seed=0, inject_close=False):
    """Build an UNCLUSTERED toy catalog: each sightline has 2 DLAs at random z in the
    observable window of z_qso=3.0. Optionally inject a close second DLA to create
    real small-Δv clustering."""
    rng = np.random.default_rng(seed)
    zq = 3.0
    z_lo, z_hi = L.zdla_window(np.array([zq])); z_lo, z_hi = float(z_lo[0]), float(z_hi[0])
    tids, zs = [], []
    for i in range(n_sight):
        z1 = rng.uniform(z_lo, z_hi)
        if inject_close and i % 3 == 0:
            dz = rng.uniform(0.002, 0.006)            # ~150–500 km/s close companion
            z2 = min(z1 + dz, z_hi - 1e-4)
        else:
            z2 = rng.uniform(z_lo, z_hi)
        tids += [i, i]; zs += [z1, z2]
    return np.array(tids), np.array(zs), np.full(len(tids), zq)


def test_estimator_null_is_unbiased():
    """On an unclustered catalog, 1+ξ(Δv) ≈ 1 everywhere (no spurious signal)."""
    tids, zs, zq = _synthetic_catalog(n_sight=4000, seed=1, inject_close=False)
    bins = L.default_bins()
    r = L.measure_xi(tids, zs, zq, bins, n_real=40, seed=2)
    opx = r["one_plus_xi"]
    good = np.isfinite(opx)
    assert abs(np.nanmean(opx[good]) - 1.0) < 0.10, f"null mean 1+ξ={np.nanmean(opx[good])}"
    # smallest bin should also be ~1 (no injected clustering)
    assert opx[0] < 1.5, f"spurious small-Δv excess in null: 1+ξ[0]={opx[0]}"


def test_estimator_detects_injected_clustering():
    """With injected close pairs, 1+ξ rises above 1 at small Δv."""
    tids, zs, zq = _synthetic_catalog(n_sight=4000, seed=3, inject_close=True)
    bins = L.default_bins()
    r = L.measure_xi(tids, zs, zq, bins, n_real=40, seed=4)
    opx = r["one_plus_xi"]
    assert opx[0] > 1.5, f"failed to detect injected close-pair clustering: 1+ξ[0]={opx[0]}"
    # The excess is localized to small Δv: 1+ξ is much larger there than in the tail.
    # (The tail sits BELOW 1 by the integral constraint — a strong small-Δv excess is
    #  compensated by a large-Δv deficit in this count-preserving, renormalized estimator.)
    assert opx[0] > np.nanmean(opx[-4:]) + 0.8, f"no small-vs-large contrast: {opx[0]} vs tail {np.nanmean(opx[-4:])}"


# =====================================================================
#  standalone runner (pytest also discovers the test_* functions)
# =====================================================================
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    npass, fails = 0, []
    print(f"running {len(tests)} physics tests (building cosmology once)...\n")
    for t in tests:
        try:
            t(); npass += 1; print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            fails.append((t.__name__, str(e))); print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            fails.append((t.__name__, repr(e))); print(f"  ERROR {t.__name__}: {e!r}")
    print(f"\n{npass}/{len(tests)} passed")
    if fails:
        print("FAILURES:")
        for n, m in fails:
            print(f"  - {n}: {m}")
        sys.exit(1)
    print("ALL PHYSICS TESTS PASSED")
