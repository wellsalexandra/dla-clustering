#!/usr/bin/env python
"""
Generate a CAMB linear-P(k) reference for test_camb_consistency.py.

We run CAMB with the SAME cosmology MatterXi (scripts/fit_dla_bias.py) uses
(Planck-2015 / LyaCoLoRe: Om0=0.3156, Ob0=0.0491, H0=67.31, ns=0.9645, Tcmb=2.7255,
massless neutrinos to match the neutrino-free EH98 fitting formula), take the LINEAR
matter power spectrum at z=0, renormalise it to sigma8 = 0.831 (the value MatterXi is
normalised to), and cache:
  - k [h/Mpc] and P_lin(k, z=0) [(Mpc/h)^3], sigma8-normalised;
  - sigma(R) for several R [Mpc/h] (top-hat, same window as MatterXi._sigma2);
  - the linear growth factor D(z)/D(0) at several z (from P_lin(k_small, z)/P_lin(k_small,0)).
The cache is plain JSON (text, diff-able) so the test runs WITHOUT CAMB installed.

Run once (needs camb):  conda activate gpdla; python tests/_gen_camb_reference.py
"""
import os, json
import numpy as np
import camb

# --- cosmology: identical to scripts/fit_dla_bias.py MatterXi defaults ---
OM0, OB0, H0, NS, SIGMA8, TCMB = 0.3156, 0.0491, 67.31, 0.9645, 0.831, 2.7255
h = H0 / 100.0
ombh2 = OB0 * h**2
omch2 = (OM0 - OB0) * h**2            # total matter = cdm + baryon (no massive nu)
ZS = [0.0, 0.5, 1.0, 2.0, 2.39, 2.43, 3.0]
RS = [4.0, 8.0, 16.0, 32.0]          # Mpc/h (8 is the sigma8 sphere -> ratio 1 by construction)

pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=0.0, omk=0.0,
                   num_massive_neutrinos=0, nnu=3.046, TCMB=TCMB)
pars.InitPower.set_params(ns=NS, As=2.1e-9)          # As arbitrary; we renormalise to sigma8 below
pars.set_matter_power(redshifts=ZS, kmax=30.0)
pars.NonLinear = camb.model.NonLinear_none           # LINEAR power spectrum
res = camb.get_results(pars)

# linear P(k) on a fine grid (h-units: k in h/Mpc, P in (Mpc/h)^3)
kh, zret, pk = res.get_matter_power_spectrum(minkh=1e-4, maxkh=20.0, npoints=600)
s8_camb = float(res.get_sigma8_0())
renorm = (SIGMA8 / s8_camb) ** 2                      # rescale CAMB amplitude to sigma8 = 0.831
pk = pk * renorm

# map returned redshift rows (CAMB returns them sorted) back to our ZS
zret = np.asarray(zret)
row = {z: int(np.argmin(np.abs(zret - z))) for z in ZS}

# growth D(z)/D(0) from the large-scale (low-k) amplitude ratio (scale-independent here)
k_growth = np.argmin(np.abs(kh - 0.02))               # ~0.02 h/Mpc, deep in the linear regime
pk0 = pk[row[0.0]]
Dz = {f"{z:.2f}": float(np.sqrt(pk[row[z]][k_growth] / pk0[k_growth])) for z in ZS}

# sigma(R) with the SAME top-hat window MatterXi uses: sigma^2 = int dlnk k^3 P W^2 / (2 pi^2)
lnk = np.log(kh)
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))   # numpy>=2 renamed trapz->trapezoid
def sigmaR(R):
    x = kh * R
    W = 3.0 * (np.sin(x) - x * np.cos(x)) / x**3
    integ = pk0 * kh**3 * W**2 / (2.0 * np.pi**2)
    return float(np.sqrt(_trapz(integ, lnk)))
sig = {f"{R:.1f}": sigmaR(R) for R in RS}

ref = dict(
    meta=dict(camb_version=camb.__version__, OM0=OM0, OB0=OB0, H0=H0, NS=NS,
              SIGMA8=SIGMA8, TCMB=TCMB, sigma8_camb_raw=s8_camb,
              note="linear P(k) z=0, h-units, renormalised to sigma8=0.831; massless nu"),
    k_hmpc=kh.tolist(), pk_z0=pk0.tolist(),
    R_mpch=RS, sigmaR=sig, z=ZS, Dz=Dz,
)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camb_reference.json")
with open(out, "w") as f:
    json.dump(ref, f)
print(f"[gen] camb {camb.__version__}; sigma8_raw={s8_camb:.4f} -> renorm to {SIGMA8}")
print(f"[gen] D(z): " + ", ".join(f"{z}:{Dz[z]}" for z in list(Dz)[:4]))
print(f"[gen] sigma(R): {sig}")
print(f"[gen] wrote {out} ({len(kh)} k-points)")
