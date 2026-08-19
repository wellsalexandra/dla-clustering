#!/usr/bin/env python
"""
Consistency of the hard-coded linear-theory equations in scripts/fit_dla_bias.py
(MatterXi: the Eisenstein & Hu 1998 *no-wiggle* transfer function, the sigma8
normalisation, and the LambdaCDM growth factor) against CAMB at the LINEAR power
spectrum level.

Why this matters: MatterXi uses an analytic EH98 fitting formula, not a Boltzmann
code.  EH98-no-wiggle is an APPROXIMATION to the true linear P(k): it reproduces the
broadband shape to a few percent but deliberately omits the BAO wiggles, so we test
the quantities that actually enter the bias fit and tolerate the wiggle-level scatter:

  - growth factor D(z)/D(0)        -> must match CAMB to <0.5%  (enters as D^2)
  - sigma(R) shape (top-hat)       -> must match CAMB to <2%    (variance integral)
  - broadband P(k) ratio           -> within ~5% per log-k bin, ~4% median
  - sigma8 self-normalisation      -> 0.831 by construction

The CAMB reference (tests/camb_reference.json) was generated once with the SAME
cosmology (tests/_gen_camb_reference.py), renormalised to sigma8=0.831, and cached as
text so this test runs WITHOUT CAMB installed.  A final test (skipped if camb is
absent) re-runs CAMB and checks the cache is still current.

Measured agreement at authoring (camb 1.6.6): D(z) <0.05%, sigma(R) <0.5%,
P(k) broadband 2.5% median / 6.7% max.

Run:  python tests/test_camb_consistency.py    or    pytest tests/test_camb_consistency.py
"""
import os, sys, json, math, warnings
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

REF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camb_reference.json")

# tolerances: set comfortably above the measured EH98-vs-CAMB agreement (see header)
GROWTH_TOL  = 0.005    # 0.5%   (measured <0.05%)
SIGMAR_TOL  = 0.02     # 2%     (measured <0.5%)
PK_BIN_TOL  = 0.05     # 5% per broadband log-k bin over [0.01,1] h/Mpc (measured <2%)
PK_MED_TOL  = 0.04     # 4% median |ratio-1| over [0.01,1]            (measured 2.5%)
PK_MAX_TOL  = 0.12     # 12% max  |ratio-1| over [0.01,2] (wiggles)   (measured 6.7%)
SIGMA8_TOL  = 0.005


def _load_ref():
    assert os.path.exists(REF_PATH), (
        f"missing {REF_PATH} -- regenerate with: python tests/_gen_camb_reference.py (needs camb)")
    return json.load(open(REF_PATH))

def _matterxi():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import fit_dla_bias as F
        return F.MatterXi()


def test_sigma8_self_normalisation():
    """MatterXi is normalised so sigma(8 Mpc/h) = 0.831 (the amplitude convention)."""
    mx = _matterxi()
    s8 = math.sqrt(mx._sigma2(8.0, mx._norm))
    assert abs(s8 - 0.831) < SIGMA8_TOL, f"sigma8={s8:.4f}, expected 0.831"


def test_growth_matches_camb():
    """D(z)/D(0) from the LambdaCDM growth integral must match CAMB's linear growth."""
    ref = _load_ref(); mx = _matterxi()
    worst = 0.0
    for z in ref["z"]:
        d_eh = mx.D(z); d_ca = ref["Dz"][f"{z:.2f}"]
        rel = abs(d_eh / d_ca - 1.0); worst = max(worst, rel)
        assert rel < GROWTH_TOL, f"D({z}): EH98={d_eh:.4f} vs CAMB={d_ca:.4f} ({rel*100:.2f}%)"
    print(f"  growth D(z) vs CAMB: worst = {worst*100:.3f}% (tol {GROWTH_TOL*100:.1f}%)")


def test_sigmaR_matches_camb():
    """sigma(R) (top-hat) shape must match CAMB across R = 4..32 Mpc/h."""
    ref = _load_ref(); mx = _matterxi()
    worst = 0.0
    for R in ref["R_mpch"]:
        s_eh = math.sqrt(mx._sigma2(R, mx._norm)); s_ca = ref["sigmaR"][f"{R:.1f}"]
        rel = abs(s_eh / s_ca - 1.0); worst = max(worst, rel)
        assert rel < SIGMAR_TOL, f"sigma(R={R}): EH98={s_eh:.4f} vs CAMB={s_ca:.4f} ({rel*100:.2f}%)"
    print(f"  sigma(R) vs CAMB: worst = {worst*100:.3f}% (tol {SIGMAR_TOL*100:.1f}%)")


def test_pk_broadband_matches_camb():
    """The EH98 no-wiggle P(k) must track CAMB's linear P(k) broadband. We test per-log-k-bin
    means over [0.01,1] h/Mpc (wiggles average out), the overall median deviation, and a loose
    max-deviation envelope over [0.01,2] that allows the BAO wiggles the no-wiggle form omits."""
    ref = _load_ref(); mx = _matterxi()
    kh = np.array(ref["k_hmpc"]); pkc = np.array(ref["pk_z0"])
    m = (kh >= 0.01) & (kh <= 2.0)
    k, pc = kh[m], pkc[m]
    pe = mx._Pk(k)                      # EH98 P(k), same sigma8 normalisation & h-units
    ratio = pe / pc

    # broadband: per log-k bin mean over [0.01,1]
    bb = (k >= 0.01) & (k <= 1.0)
    bins = np.logspace(np.log10(0.01), np.log10(1.0), 8)
    idx = np.digitize(k[bb], bins)
    worst_bin = 0.0
    for b in range(1, len(bins)):
        sel = idx == b
        if sel.sum():
            dev = abs(ratio[bb][sel].mean() - 1.0); worst_bin = max(worst_bin, dev)
            assert dev < PK_BIN_TOL, (f"P(k) bin [{bins[b-1]:.3f},{bins[b]:.3f}] mean ratio "
                                      f"{ratio[bb][sel].mean():.3f} (dev {dev*100:.1f}% > {PK_BIN_TOL*100:.0f}%)")
    med = float(np.median(np.abs(ratio[bb] - 1.0)))
    mx_dev = float(np.max(np.abs(ratio - 1.0)))
    assert med < PK_MED_TOL, f"median |P_EH98/P_CAMB - 1| = {med*100:.1f}% > {PK_MED_TOL*100:.0f}%"
    assert mx_dev < PK_MAX_TOL, f"max |P_EH98/P_CAMB - 1| = {mx_dev*100:.1f}% > {PK_MAX_TOL*100:.0f}%"
    print(f"  P(k) vs CAMB: worst bin {worst_bin*100:.1f}%, median {med*100:.1f}%, max {mx_dev*100:.1f}% "
          f"(tol {PK_BIN_TOL*100:.0f}/{PK_MED_TOL*100:.0f}/{PK_MAX_TOL*100:.0f}%)")


def test_camb_reference_is_current():
    """If camb is installed, re-run it and confirm the cached reference is still accurate
    (guards against a stale cache).  Skipped where camb is absent (e.g. the gpdla env)."""
    try:
        import camb  # noqa
    except Exception:
        try:
            import pytest; pytest.skip("camb not installed -- using cached reference")
        except Exception:
            print("  (camb not installed; skipping live re-check)"); return
    ref = _load_ref()
    h = ref["meta"]["H0"] / 100.0
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=ref["meta"]["H0"], ombh2=ref["meta"]["OB0"]*h**2,
                       omch2=(ref["meta"]["OM0"]-ref["meta"]["OB0"])*h**2,
                       mnu=0.0, omk=0.0, num_massive_neutrinos=0, nnu=3.046, TCMB=ref["meta"]["TCMB"])
    pars.InitPower.set_params(ns=ref["meta"]["NS"], As=2.1e-9)
    pars.set_matter_power(redshifts=[0.0, 2.0], kmax=5.0)
    pars.NonLinear = camb.model.NonLinear_none
    res = camb.get_results(pars)
    s8 = float(res.get_sigma8_0())
    assert abs(s8 - ref["meta"]["sigma8_camb_raw"]) < 5e-3, "cached CAMB sigma8 drifted -> regenerate"
    kh, _, pk = res.get_matter_power_spectrum(minkh=1e-3, maxkh=3.0, npoints=50)
    D2 = math.sqrt(pk[1][np.argmin(abs(kh-0.02))] / pk[0][np.argmin(abs(kh-0.02))])
    assert abs(D2 - ref["Dz"]["2.00"]) < 5e-3, "cached D(2) drifted -> regenerate reference"
    print(f"  live camb {camb.__version__}: cache current (sigma8_raw {s8:.4f}, D(2) {D2:.4f})")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    npass, fails = 0, []
    print(f"running {len(tests)} CAMB-consistency tests...\n")
    for t in tests:
        try:
            t(); npass += 1; print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            fails.append((t.__name__, str(e))); print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            fails.append((t.__name__, repr(e))); print(f"  ERROR {t.__name__}: {e!r}")
    print(f"\n{npass}/{len(tests)} passed")
    if fails:
        for n, m in fails:
            print(f"  - {n}: {m}")
        sys.exit(1)
    print("ALL CAMB-CONSISTENCY TESTS PASSED")
