# Appendix: derivations and references for §9 of the 1D DLA clustering tutorial

This collects the formulae and references behind the bias fit so that §9 of
`dla_clustering_tutorial.ipynb` can stay short. Everything here is implemented in
`scripts/fit_dla_bias.py` (`MatterXi`), which is pure numpy/scipy/astropy (no CAMB/CLASS at
run time). The EH98 template below is checked against CAMB at the linear power-spectrum level
in `tests/test_camb_consistency.py` (agreement: growth <0.05%, σ(R) <0.5%, broadband P(k) ~2%).

## A1. The matter template ξ_matter(r)

We fit the model

$$1 + \xi_{\rm DLA}(\Delta v, z) = 1 + b^2\,D(z)^2\,\xi_{\rm matter}\big(r(\Delta v), z{=}0\big),$$

with a single free amplitude b². The pieces:

- **Transfer function and P(k).** We build the linear power spectrum as
  $P(k) = A\,k^{n_s}\,T(k)^2$ with the Eisenstein & Hu (1998) "no-wiggle" transfer function
  $T(k)$. The no-wiggle form drops the baryon acoustic oscillations but reproduces the
  broadband shape to a few percent, which is all the bias fit needs. We then Fourier-transform
  $P(k)$ to the real-space correlation function
  $\xi_{\rm matter}(r) = \int {\rm d}\ln k\,\frac{k^3 P(k)}{2\pi^2}\,\frac{\sin(kr)}{kr}.$

- **σ₈ normalisation.** We fix the amplitude $A$ so the variance in 8 Mpc/$h$ spheres equals
  $\sigma_8 = 0.831$ (the Planck-2015 / LyaCoLoRe value), using
  $\sigma^2(R) = \int {\rm d}\ln k\,\frac{k^3 P(k)}{2\pi^2}\,W(kR)^2$ with the top-hat window
  $W(x) = 3(\sin x - x\cos x)/x^3$.

- **Growth factor.** $D(z)$ is the standard ΛCDM growing-mode integral,
  $D(a) \propto H(a)\int_0^a \frac{{\rm d}a'}{(a' H(a'))^3}$, normalised so $D(0)=1$. Structure
  was less grown in the past, so $D(z)<1$; at our redshift $D(2.4)\approx0.37$, hence
  $D^2\approx0.14$.

- **Velocity to distance.** A velocity separation maps to a comoving separation by
  $r = \Delta v\,(1+z)/H(z)\cdot h$ in Mpc/$h$. We cap it at a small-scale floor
  $r_{\rm cut}=0.5$ Mpc/$h$ because the linear-bias model diverges as $r\to0$.

- **Integral constraint.** The count-preserving randoms force $\sum RR\,\xi = 0$ over the binned
  range (§3.3 of the tutorial), so we subtract the $RR$-weighted mean of the template before
  fitting. This makes the fit a measurement of the *shape* of $\xi$, with the absolute
  normalisation absorbed by the rescaling.

## A2. Redshift-space distortions: why the line-of-sight boost is modest

Our template is real-space, but the mock redshifts are in redshift space (they include
peculiar velocities). The standard worry is that linear redshift-space distortions inflate the
amplitude, so we should be careful about which quantity actually gets the textbook factor.

The Kaiser (1987) result is a statement about the **power spectrum**:
$P_s(k,\mu) = (1 + \beta\mu^2)^2\,P_r(k)$, where $\mu$ is the cosine of the angle between the
wavevector and the line of sight and $\beta = f/b$. At our redshift
$f \approx \Omega_m(z)^{0.55} \approx 0.97$ (Linder 2005), so with $b\approx2$, $\beta\approx0.49$
and $(1+\beta)^2\approx2.2$ along the line of sight ($\mu=1$).

That factor of 2.2 is the boost of the **line-of-sight power spectrum**, not of the quantity we
measure. Our 1D estimator counts same-sightline pairs, i.e. the **line-of-sight correlation
function** $\xi(s, \mu{=}1)$. The correlation function at a fixed radial separation $s$ is the
integral of $(1+\beta\mu^2)^2 P_r(k)$ over all transverse wavevectors, where $\mu$ runs from 0
to 1, not a single $(1+\beta)^2$ factor. That integral is exactly the Kaiser multipole sum
(Hamilton 1992):

$$\xi(s,\mu{=}1) = \xi_0(s) + \xi_2(s) + \xi_4(s),$$
$$\xi_0 = \Big(1 + \tfrac{2}{3}\beta + \tfrac{1}{5}\beta^2\Big)\,\xi_r(s),\quad
  \xi_2 = \Big(\tfrac{4}{3}\beta + \tfrac{4}{7}\beta^2\Big)\,[\xi_r(s) - \bar\xi(s)],$$

with $\bar\xi(s) = (3/s^3)\int_0^s \xi_r(s')\,s'^2\,{\rm d}s'$ and a small hexadecapole $\xi_4$.

For a declining $\xi_r$ we have $\xi_r < \bar\xi$, so $\xi_2 < 0$: the quadrupole **subtracts**
along the line of sight and cancels most of the monopole enhancement. Evaluated with our own
$\xi_{\rm matter}$, the net line-of-sight effect over the fit range is small and not a boost: it is
close to zero at the smallest separations and turns mildly suppressive once $r\gtrsim5$ Mpc/$h$,
because the quadrupole drives $\xi(s,\mu{=}1)$ below the real-space $\xi_r$. The monopole-only factor
$\sqrt{1 + \tfrac{2}{3}\beta + \tfrac{1}{5}\beta^2} \approx 1.17$ is an upper bound on any
line-of-sight enhancement of $b$, and the actual radial signal sits well below it. There is no
factor-of-1.5 boost on the correlation function, and the empirical calibration factor
$k = 2/b_{\rm app}({\rm truth})$ accordingly comes out slightly below 1.

Fingers-of-god, the small-scale smearing from the virial velocity dispersion, act only on
scales of a few Mpc/$h$: $\sigma_v(1+z)/H(z)\cdot h \approx 1.5$ to $3$ Mpc/$h$ for
$\sigma_v\approx 200$ to $300$ km/s. That is below our first fitted bin
($\Delta v\ge250$ km/s is $r\approx2.4$ Mpc/$h$ for the truth fit; the GP fit starts near
$r\approx19$ Mpc/$h$), so fingers-of-god touch at most the lowest bin, not the whole fit.

Putting these together: the linear line-of-sight RSD effect is small and mildly suppressive over the
fit range, so the truth fit's $b_{\rm app}\approx2.2$ (about 10% above the planted 2.0) is driven by
residual small-scale nonlinearity and the approximate template, not by an RSD boost. There is no
large Kaiser boost being cancelled by a large fingers-of-god suppression, which is why we do not
apply an analytic correction and instead calibrate empirically (tutorial §9.1 and §9.2).

## A3. References

- **Eisenstein & Hu 1998**, *ApJ* **496**, 605, "Baryonic Features in the Matter Transfer
  Function". [ADS](https://ui.adsabs.harvard.edu/abs/1998ApJ...496..605E/abstract) ·
  [arXiv:astro-ph/9709112](https://arxiv.org/abs/astro-ph/9709112). The no-wiggle transfer function.
- **Kaiser 1987**, *MNRAS* **227**, 1, "Clustering in real space and in redshift space".
  [ADS](https://ui.adsabs.harvard.edu/abs/1987MNRAS.227....1K/abstract). Linear RSD.
- **Hamilton 1992**, *ApJL* **385**, L5, "Measuring Ω and the real correlation function from the
  redshift correlation function". [ADS](https://ui.adsabs.harvard.edu/abs/1992ApJ...385L...5H/abstract).
  The configuration-space Kaiser multipoles ξ₀, ξ₂, ξ₄.
- **Linder 2005**, *Phys. Rev. D* **72**, 043529. [ADS](https://ui.adsabs.harvard.edu/abs/2005PhRvD..72d3529L/abstract)
  · [arXiv:astro-ph/0507263](https://arxiv.org/abs/astro-ph/0507263). The growth index, f ≈ Ω_m(z)^0.55.
- **Carroll, Press & Turner 1992**, *ARA&A* **30**, 499, "The Cosmological Constant".
  [ADS](https://ui.adsabs.harvard.edu/abs/1992ARA%26A..30..499C/abstract). Growth-factor formula.
- **Planck Collaboration XIII 2016**, *A&A* **594**, A13. [ADS](https://ui.adsabs.harvard.edu/abs/2016A%26A...594A..13P/abstract)
  · [arXiv:1502.01589](https://arxiv.org/abs/1502.01589). The cosmology (Ω_m=0.3156, σ₈=0.831).
- **Landy & Szalay 1993**, *ApJ* **412**, 64. [ADS](https://ui.adsabs.harvard.edu/abs/1993ApJ...412...64L/abstract).
  The (DD−2DR+RR)/RR estimator.
- **Farr et al. 2020**, *JCAP* **03**, 068, "LyaCoLoRe". [ADS](https://ui.adsabs.harvard.edu/abs/2020JCAP...03..068F/abstract)
  · [arXiv:1912.02763](https://arxiv.org/abs/1912.02763). The mock, with planted constant b_HCD=2.0.
