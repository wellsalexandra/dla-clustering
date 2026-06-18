# Coding notes for `dla_1d_clustering_updated.ipynb`

*A mechanical and programming review (not the science, which is in
[`science_bugs.md`](science_bugs.md)). Cell numbers are 0-indexed as stored in the `.ipynb`
(cell 0 = first code cell). Each item: what to adjust, why it matters, the fix. Severity:
🔴 = stops the run or produces inf/NaN; 🟠 = silently off value; 🟡 = perf;
⚪ = style/robustness. Items that are primarily scientific are tagged →SCI and detailed in
the science doc; here I note only their mechanical symptom.*

The notebook has 22 code cells. The clustering result is produced by cells 19, 20, 21, which carry
the most impactful items. Cells 0–17 are mostly the purity/completeness machinery adapted from the
`mw` notebook, and that reuse is sensible.

---

## A. Run / runtime items to address

### 🔴 C1: Hard dependencies not installed; notebook can't import yet (cell 0)
`import pyigm`, `from pyigm.fN.fnmodel import FNModel`, `import linetools`, and
`from pycorr import TwoPointCorrelationFunction, …` are all at the top. In the project env
(`conda activate gpdla`, py3.11) `pyigm`, `pycorr`, and `linetools` are not installed, so cell 0
raises `ModuleNotFoundError` and nothing else runs. `pycorr` is imported but never used anywhere;
`linetools` is never used. `pyigm.FNModel` is used (cell 19) for the expected counts.
*Fix:* drop the unused `pycorr`/`linetools` imports; either install pyigm or (better, see the
corrected notebook) replace the pyigm expected-count path with the permutation-randoms method
that needs only numpy.

### 🔴 C2: Hardcoded `./Downloads/...` paths that don't exist (cells 3, 7)
`gpp_251202_new_baseline_snr_2.fits`, `dla_cat_mock_added_gt_20.3_update-05202025.fits`,
`gpp_mock_snr2_combined_matched.fits`, `bal_cat.fits` are read by relative path from a
`./Downloads/` folder. These exist only on the machine where they were downloaded; a
from-scratch run anywhere else (including here) fails at `fitsio.read`.
*Fix:* parameterise the catalog paths at the top, pointing at the canonical mock catalogs (their paths are set at the top of `scripts/clustering_lib.py`).

### 🔴 C3: Empty-input edge case in the Δv helpers (cells 14, 17, 20)
`vsep_min = min(np.nanmin(vsep_truth), …)` and `np.max(N_sim_arr)` (cell 20) assume the lists
are non-empty. If `get_velocity_separations` returns `[]` (no multi-DLA sightlines pass the
cut) or if every sightline hit the `continue` in cell 19, `np.nanmin([])` / `max([])` raise.
*Fix:* guard for empty arrays before reducing.

---

## B. Silently-off-value items

### 🔴 C4: Completeness > 1 is clamped to 0, not 1, giving division by zero (cells 19, 21)
Three places:
- cell 19: `c_zdla_bins[c_zdla_bins > 1.0] = 0.0`, then `lX_complete = lX * c_zdla_bins`
- cell 21: `c_vsep_bins[c_vsep_bins > 1.0] = 0.0`, then `DD_upweighted = DD_counts / c_vsep_bins`

The "completeness" metric used here can exceed 1 (→SCI: it's a distribution ratio, not a recovery
rate). Clamping the offending bins to 0 isn't the floor you want here: in cell 19 it removes real
expected absorbers from those z-bins; in cell 21 it makes `DD_counts / 0 = inf`, so `xi` becomes
`inf`/`NaN` in exactly the bins where the finder over-produced pairs.
*Fix:* skip the clamp entirely. Use a recovery-rate completeness that is bounded ≤1 by
construction (→SCI), and guard divisions with `np.where(c>0, …, np.nan)`.

### 🟠 C5: `np.min` where `np.max` was meant (cells 11, 14)
- cell 11: `z_max = max(np.max(dla_cat…['Z_DLA']), np.min(combined_cat…['Z_DLA']))`: the second
  term should be `np.max`. As written, `z_max` is pulled down to the minimum combined z, so
  the z-binning for completeness is truncated.
- cell 14: `vsep_max = max(np.nanmax(vsep_truth), np.nanmin(vsep_combined))`: the second term
  should be `np.nanmax`. The histogram upper edge ends up set differently than intended. (Cell 17
  has the `np.nanmax` form, so the two figures use different ranges.)
Both look like copy-paste slips from the adjacent `*_min` line, an easy one to make.
*Fix:* use `np.max` / `np.nanmax`.
*(Verification caveat: the cell-14 case is unconditionally a wrong histogram edge; the cell-11
case is a genuine error but its downstream effect only bites when
`min(combined Z_DLA) > max(dla Z_DLA)`, otherwise the outer `max(...)` masks it.)*

### 🔴 C6: DD/RR built from sightlines selected differently (cell 19)
The `if(z_max > 3.0): continue` sits after the DD pair loop (`deltav_arr.append`) but before the
random/expected block. So a sightline with `z_max>3` contributes its real pairs to DD but generates
no randoms. →SCI (population mismatch, ~46% of sightlines). Mechanical symptom: `deltav_arr` and
`deltav_sim_arr` are tallied over different sightline sets, and your own comment flags it
("TEMP FIX FOR ERROR, ask").
*Fix:* gate DD and RR with the same condition (or fix the z-window so the guard is unnecessary).

### 🟠 C7: Random redshifts sampled with the integral's weight minus its Jacobian (cell 19)
`N_expected = np.sum(lX_complete * dX)` correctly multiplies by `dX`, but the position draw
`lX_complete /= lX_complete.sum(); np.random.choice(z_bins_mid, p=lX_complete)` omits `dX`
(equivalently the `dX/dz` Jacobian). The count and the positions therefore use **inconsistent
weights**. →SCI (skews the random Δv distribution by ~18%).
*Fix:* sample with `p = (lX_complete * dX); p /= p.sum()`.

### 🔴 C8: `xi` uses un-normalised raw counts (cell 21)
`xi = cluster_1d(DD_upweighted, RR_counts)` = `DD/RR − 1` on raw histogram counts. `deltav_arr`
(DD) and `deltav_sim_arr` (RR) have different totals (RR is one Poisson realisation; →SCI
undersampled), so the ratio folds in the overall density mismatch. ξ is not on a meaningful scale.
*Fix:* normalise: `xi = (DD/DD.sum()) / (RR/RR.sum()) − 1`, or scale RR by `N_DD/N_RR` first;
and generate enough randoms that `RR.sum() ≫ DD.sum()`.

### 🟠 C9: `c_vsep_bins` off-by-one: bin *centers* used as bin *edges* (cell 21)
```python
Ndv = 25
bins = np.linspace(0, dv_max, Ndv+1)         # 26 edges
bin_centers = 0.5*(bins[:-1] + bins[1:])     # 25 centers
c_vsep_bins = np.zeros(Ndv)                  # 25
for v_idx, vsep in enumerate(bin_centers[:-1]):   # only 24 iterations
    c_vsep_bins[v_idx] = compute_completeness_min_max_vsep(
        …, vsep, bin_centers[v_idx+1], …)         # edges are CENTERS, not bins[]
```
Three things to fix in one block: (a) it loops over `bin_centers[:-1]` (24) so `c_vsep_bins[24]`
stays 0, so later `DD_counts[24]/0 = inf`; (b) it uses consecutive centers as the bin
[lo,hi] passed to the completeness, which are offset by half a bin from the `bins[]` used for
`DD_counts`/`RR_counts` histograms, so the completeness weight is misaligned with the data
bins; (c) the completeness array length is tied to `Ndv` but filled with a center-indexed
loop.
*Fix:* compute the per-bin completeness on the same `bins` edges used for the DD/RR histograms
(`bins[k]`, `bins[k+1]`), filling all `Ndv` entries.

### 🟠 C10: `get_velocity_separations` dedupes with `np.unique` on floats (cells 14, 17)
It builds ordered pairs (each unordered pair appears twice) then
`v_diffs_los = list(np.unique(v_diffs_los))`. This happens to undo the double-count for the
common n=2 case, but: (a) `np.unique` on floats can collapse two genuinely distinct pairs that
share an identical Δv on a 3-DLA sightline (rare but possible), and (b) it's a roundabout
work-around for what `itertools.combinations(idx, 2)` would express directly. Net pair counts are
right for n≤2; the y-axis "Number of … Pairs" is otherwise trustworthy here (unlike the older
`clean` notebook which double-counted).
*Fix:* enumerate with `combinations`; don't dedupe by value.

---

## C. Performance

### 🟡 C11: Global completeness recomputed inside the per-sightline loop (cell 19)
For every sightline `i`, the code runs an inner loop over 25 z-bins, each call
`compute_completeness_min_max_zdla(combined_cat_all_no_bal, …, z_start, z_end, …)` scanning the
entire (global, sightline-independent) `combined_cat_all_no_bal`. The result does not depend on `i`
at all, so this recomputes the identical completeness curve tens of thousands of times, roughly
`O(N_sightlines × Nz × N_catalog)`. On the full catalog this makes the cell extremely slow (likely
the practical reason it's only ever run on a subset).
*Fix:* precompute `c_zdla` on a fixed z-grid once before the loop and interpolate per sightline.

---

## D. Style / fragility / latent

- ⚪ **C12: Duplicate/unused imports (cell 0):** `from pyigm.fN.fnmodel import FNModel`
  imported twice; `tempfile`, `sys`, `z_at_value`, `astropy.units as u` unused.
- ⚪ **C13: Inconsistent physical constants:** `LYA_WAVELENGTH=1215.67` (cell 4) vs literal
  `1216` (cells 5,7,19) vs `912`/`911` for the Lyman limit; `SPEED_OF_LIGHT=299792.` (cell 4)
  vs `c=const.c.to('km/s').value=299792.458` (cell 2). Differences are negligible numerically
  but the literals should reference the named constants.
- ⚪ **C14: Inconsistent column-name handling:** `gpp` uses `SNR_REDSIDE`/`NHI_GP`/`P_DLA_GP`
  (cell 3) while `combined_cat` uses `S2N_RED`/`NHI`/`P_DLA_GP`/`NHI_TRUE` (cells 9–17). The DD
  catalog and the completeness catalog are different files with different schemas, handled
  ad hoc. (→SCI: they're also different selections.)
- ⚪ **C15: `if (not cat_bal):` on an astropy Table (cell 6):** intends `if cat_bal is None`.
  For a non-empty Table it evaluates `len(cat_bal)==0`, so it works by accident; for an empty Table
  it would wrongly raise "must pass in BAL catalog".
- ⚪ **C16: Global `z_min`/`z_max`/`Nz` reassigned as loop locals (cell 19)** and elsewhere
  (cells 5,7,11,14,17,21 each redefine them). Execution order is load-bearing and undocumented;
  re-running an earlier cell after cell 19 uses clobbered values.
- ⚪ **C17: `plt.hist(deltav_arr)` / `plt.hist(deltav_sim_arr)` with default bins (cell 20):**
  the two histograms use independent default 10-bin ranges, so they can't be compared by eye.

---

### Additional items found in verification

- 🟠 **C18: `xi` contains `inf`/`NaN` that flow straight into the plot (cell 21).** C4 + C9
  produce `inf` in `DD_upweighted`; any bin with `RR_counts==0` makes `cluster_1d` divide by
  zero → `inf`/`NaN`. These propagate into the plotted `xi`; `plt.bar`/`plt.plot` silently drop
  them, so the figure looks fine while bins are missing. *Fix:* mask non-finite ξ explicitly
  and report how many bins were dropped.
- 🟠 **C19: RR pair count scales as ~N²/2 per sightline, DD does not (cell 19).** Random pairs
  come from `combinations(z_rand, 2)` on a Poisson draw `N_sampled`, so a sightline that draws
  3 randoms contributes 3 pairs, etc., a different functional construction from DD (real DLA
  multiplets). This is a second DD/RR scale mismatch on top of C8 and reinforces that DD and RR
  must be normalised consistently (→SCI S8).
- 🔴 **C20: cell 21 `dv_max = max(max(deltav_arr), max(deltav_sim_arr))`** shares the C3
  empty-list fragility (omitted from the C3 cell list): raises on an empty `deltav_sim_arr`.
- ⚪ **C21: strict `<` on both bin edges (cell 16 `compute_completeness_min_max_vsep`).** A
  value exactly on an edge is excluded from both numerator and denominator; combined with the
  center-vs-edge offset (C9) the edge handling compounds. Use half-open `[lo, hi)`.

---

## Verification status

This list was independently re-checked against the notebook source by a second (adversarial)
agent: 16/17 of C1–C17 confirmed, C5 confirmed-with-caveat (above), none refuted. The
mechanically-testable claims (C4 inf-on-divide, C9 off-by-one shapes, C10 `np.unique` collapse,
C15 `bool(Table)` behaviour) were reproduced in python. C18–C21 were added from that pass.

## E. Cross-reference to the science bugs

The mechanical fixes above (C4, C6, C7, C8, C9) are the symptoms of the deeper methodological
points documented in [`science_bugs.md`](science_bugs.md):
S1 (DD/RR selection mismatch), S2 (random redshift weighting), S3 (under-sampled randoms),
S4 (units / (1+z) factor), S5 (completeness correction to apply once/consistently), S6 (FP / Lyβ-Lyγ
removal to add), S7 (DD from a different catalog than the corrections), S8 (DD/RR normalisation to add).
Addressing the code mechanically goes hand in hand with the method fixes; both together give a sound ξ.
