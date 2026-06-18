#!/usr/bin/env python
"""
Unit tests for the truth<->finder MATCHING + completeness/purity calibration in
`scripts/clustering_lib.py`:

  match_recovered          -- greedy, one-to-one, NHI-ranked per-truth recovery flag
  pair_completeness        -- C_pair(Δv) = recovered true pairs / true pairs   (≤1)
  finder_pair_truth_match  -- per-detection true-positive flag
  pair_purity              -- p(Δv) = both-member-TP finder pairs / finder pairs (≤1)

All on hand-built synthetic catalogs (no data files). Tolerance ZTOL=0.01 in
|Δz|/(1+z_truth); at z≈2.5 that's ~0.035 in z.

Run:  python tests/test_matching.py     or     pytest tests/test_matching.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import clustering_lib as L


def cat(tid, z, nhi):
    """Minimal truth/finder catalog dict (matching uses TARGETID, Z, NHI only)."""
    return {"TARGETID": np.array(tid, np.int64),
            "Z": np.array(z, float),
            "NHI": np.array(nhi, float)}

def allT(n): return np.ones(n, bool)


# =====================================================================
#  match_recovered
# =====================================================================
def test_match_within_tolerance():
    truth = cat([1], [2.50], [21.0]);  finder = cat([1], [2.505], [21.0])
    rec = L.match_recovered(truth, allT(1), finder, allT(1))
    assert rec.tolist() == [True], f"close detection should be recovered: {rec}"

def test_no_match_outside_tolerance():
    truth = cat([1], [2.50], [21.0]);  finder = cat([1], [2.90], [21.0])  # |Δz|/(1+z)=0.11
    rec = L.match_recovered(truth, allT(1), finder, allT(1))
    assert rec.tolist() == [False], f"far detection should NOT be recovered: {rec}"

def test_no_cross_sightline_match():
    """A detection on a different TARGETID must not recover the truth DLA."""
    truth = cat([1], [2.50], [21.0]);  finder = cat([2], [2.50], [21.0])
    rec = L.match_recovered(truth, allT(1), finder, allT(1))
    assert rec.tolist() == [False], f"cross-sightline match leaked: {rec}"

def test_one_to_one_single_detection_two_truths():
    """One detection within tolerance of TWO truth DLAs recovers exactly ONE
    (the one-to-one rule) — the blending case behind C_pair < C_single²."""
    truth = cat([1, 1], [2.500, 2.505], [21.0, 20.8])
    finder = cat([1], [2.5025], [21.0])               # one broad detection
    rec = L.match_recovered(truth, allT(2), finder, allT(1))
    assert int(rec.sum()) == 1, f"one detection must not recover both truths: {rec}"

def test_two_detections_two_truths_both_recovered():
    truth = cat([1, 1], [2.50, 2.90], [21.0, 20.5])
    finder = cat([1, 1], [2.505, 2.905], [21.0, 20.5])
    rec = L.match_recovered(truth, allT(2), finder, allT(2))
    assert rec.tolist() == [True, True], f"both well-separated pairs should recover: {rec}"

def test_masks_respected():
    """A detection failing fmask, or a truth DLA failing tmask, is ignored."""
    truth = cat([1, 1], [2.50, 2.60], [21.0, 21.0])
    finder = cat([1, 1], [2.50, 2.60], [21.0, 21.0])
    # finder[1] fails quality cut -> truth[1] cannot be recovered
    rec = L.match_recovered(truth, allT(2), finder, np.array([True, False]))
    assert rec.tolist() == [True, False], f"fmask not respected: {rec}"
    # truth[0] excluded by tmask -> not considered (stays False)
    rec2 = L.match_recovered(truth, np.array([False, True]), finder, allT(2))
    assert rec2.tolist() == [False, True], f"tmask not respected: {rec2}"

def test_nhi_tiebreak_enables_both():
    """Two truths and two detections all mutually z-eligible: the NHI tie-break
    pairs like-with-like so BOTH recover (no detection wasted on the wrong truth)."""
    truth = cat([1, 1], [2.50, 2.50], [22.0, 20.5])
    finder = cat([1, 1], [2.50, 2.50], [20.5, 22.0])   # reversed NHI order
    rec = L.match_recovered(truth, allT(2), finder, allT(2))
    assert rec.tolist() == [True, True], f"NHI tie-break failed to recover both: {rec}"


# =====================================================================
#  pair_completeness  (C_pair)
# =====================================================================
def test_pair_completeness_basic():
    """One sightline with both members recovered → recovered pair;
    another with only one recovered → not. C_pair = 1 and 0 in the two bins."""
    # A: tid1 z=2.50,2.60 (Δv≈8445) both recovered;  B: tid2 z=2.50,2.70 (Δv≈16655) one recovered
    truth = cat([1, 1, 2, 2], [2.50, 2.60, 2.50, 2.70], [21, 21, 21, 21])
    recovered = np.array([True, True, True, False])
    bins = np.array([0., 12000., 30000.])
    C, n_true, n_rec, dv, both = L.pair_completeness(truth, allT(4), recovered, bins)
    assert n_true.tolist() == [1, 1], f"true-pair counts: {n_true}"
    assert n_rec.tolist() == [1, 0], f"recovered-pair counts: {n_rec}"
    assert C[0] == 1.0 and C[1] == 0.0, f"C_pair={C}"

def test_pair_completeness_bounded_and_blending():
    """C_pair ≤ 1 always; and a close pair sharing one broad detection has C_pair=0
    even though single-DLA recovery is 0.5 (the C_pair < C_single² effect)."""
    truth = cat([1, 1], [2.500, 2.505], [21.0, 20.8])   # Δv≈428 km/s close pair
    finder = cat([1], [2.5025], [21.0])
    rec = L.match_recovered(truth, allT(2), finder, allT(1))
    assert int(rec.sum()) == 1                           # single-DLA recovery = 1/2
    bins = np.array([0., 1000., 30000.])
    C, n_true, n_rec, dv, both = L.pair_completeness(truth, allT(2), rec, bins)
    assert n_true[0] == 1 and n_rec[0] == 0, f"close pair should be unrecovered: {n_true},{n_rec}"
    assert C[0] == 0.0, f"C_pair(small Δv)={C[0]} (expected 0; blending)"
    assert np.all(n_rec <= n_true), "C_pair numerator exceeds denominator (>1)!"


# =====================================================================
#  finder_pair_truth_match  +  pair_purity
# =====================================================================
def test_finder_pair_truth_match():
    """A detection is a true positive iff it matches a selected truth DLA on the
    same sightline within tolerance."""
    truth = cat([1], [2.50], [21.0])
    finder = cat([1, 1, 2], [2.505, 2.90, 2.50], [21.0, 21.0, 21.0])
    is_tp = L.finder_pair_truth_match(finder, allT(3), truth, allT(1))
    assert is_tp.tolist() == [True, False, False], f"is_tp={is_tp}"

def test_pair_purity_basic():
    """A finder pair counts toward purity only if BOTH members are true positives."""
    # A: tid1 z=2.50,2.60 both TP -> true pair;  B: tid2 z=2.50,2.70 one FP -> impure
    finder = cat([1, 1, 2, 2], [2.50, 2.60, 2.50, 2.70], [21, 21, 21, 21])
    is_tp = np.array([True, True, True, False])
    bins = np.array([0., 12000., 30000.])
    p, n_all, n_true = L.pair_purity(finder, allT(4), is_tp, bins)
    assert n_all.tolist() == [1, 1], f"finder-pair counts: {n_all}"
    assert n_true.tolist() == [1, 0], f"true-pair counts: {n_true}"
    assert p[0] == 1.0 and p[1] == 0.0, f"purity={p}"
    assert np.all(n_true <= n_all), "purity numerator exceeds denominator (>1)!"

def test_purity_all_fp_pair():
    """A pair of two false positives contributes to the denominator but not the numerator."""
    finder = cat([1, 1], [2.50, 2.60], [21, 21])
    is_tp = np.array([False, False])
    bins = np.array([0., 30000.])
    p, n_all, n_true = L.pair_purity(finder, allT(2), is_tp, bins)
    assert n_all[0] == 1 and n_true[0] == 0 and p[0] == 0.0, f"FP-FP pair: {n_all},{n_true},{p}"


# =====================================================================
#  end-to-end: match -> C_pair and purity are consistent and bounded
# =====================================================================
def test_end_to_end_bounded():
    """On a small mixed catalog, both C_pair and purity stay in [0,1]."""
    rng = np.random.default_rng(0)
    tid, zt, zf = [], [], []
    for s in range(200):
        n = rng.integers(2, 4)
        z0 = rng.uniform(2.3, 2.8)
        for k in range(n):
            tid.append(s); zt.append(z0 + 0.05 * k)
    truth = cat(tid, zt, [21.0] * len(tid))
    # finder = truth with ~70% recovered (jittered) + some pure FPs
    ftid, fz = [], []
    for i in range(len(tid)):
        if rng.random() < 0.7:
            ftid.append(tid[i]); fz.append(zt[i] + rng.normal(0, 0.001))
    for s in range(50):  # FP detections on random sightlines
        ftid.append(rng.integers(0, 200)); fz.append(rng.uniform(2.3, 2.8))
    finder = cat(ftid, fz, [21.0] * len(ftid))
    rec = L.match_recovered(truth, allT(len(tid)), finder, allT(len(ftid)))
    is_tp = L.finder_pair_truth_match(finder, allT(len(ftid)), truth, allT(len(tid)))
    bins = L.default_bins()
    C, ntc, nrc, _, _ = L.pair_completeness(truth, allT(len(tid)), rec, bins)
    p, nap, ntp = L.pair_purity(finder, allT(len(ftid)), is_tp, bins)
    assert np.all(nrc <= ntc), "C_pair > 1 somewhere"
    assert np.all(ntp <= nap), "purity > 1 somewhere"
    finite = np.isfinite(C)
    assert np.all((C[finite] >= 0) & (C[finite] <= 1.0))
    pf = np.isfinite(p)
    assert np.all((p[pf] >= 0) & (p[pf] <= 1.0))
    assert rec.sum() > 0 and is_tp.sum() > 0, "sanity: some recoveries / TPs expected"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    npass, fails = 0, []
    print(f"running {len(tests)} matching / completeness-purity tests...\n")
    for t in tests:
        try:
            t(); npass += 1; print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            fails.append((t.__name__, str(e))); print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            fails.append((t.__name__, repr(e))); print(f"  ERROR {t.__name__}: {e!r}")
    print(f"\n{npass}/{len(tests)} passed")
    if fails:
        for n, m in fails: print(f"  - {n}: {m}")
        sys.exit(1)
    print("ALL MATCHING TESTS PASSED")
