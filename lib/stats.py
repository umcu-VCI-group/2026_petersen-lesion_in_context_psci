"""Statistical helpers: BH-FDR, paired permutation, and Nadeau-Bengio corrected paired t-test."""

import numpy as np
from scipy.stats import t as _t_dist


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values (step-up procedure)."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order] * n / (np.arange(n) + 1)
    adj_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    adj_sorted = np.minimum(adj_sorted, 1.0)
    adj = np.empty_like(adj_sorted)
    adj[order] = adj_sorted
    return adj


def paired_permutation_test(
    deltas, n_perm=10_000, alternative="two-sided", random_state=0
):
    """Sign-flip permutation test on paired per-fold deltas.

    Drops NaNs. Returns 1.0 if all values are NaN.
    """
    d = np.asarray(deltas, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return 1.0
    observed = float(np.mean(d))
    rng = np.random.RandomState(random_state)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    perm_means = (signs * d).mean(axis=1)
    if alternative == "two-sided":
        p = float(np.mean(np.abs(perm_means) >= np.abs(observed)))
    elif alternative == "greater":
        p = float(np.mean(perm_means >= observed))
    else:
        p = float(np.mean(perm_means <= observed))
    return min(1.0, (p * (n_perm + 1) + 1) / (n_perm + 1))


def nadeau_bengio_test(
    focal_scores, ref_scores, n_splits=10, test_size=0.1, alternative="greater"
):
    """Nadeau-Bengio corrected paired t-test for repeated K-fold CV scores.

    Variance is inflated by (1/n + test_size / (1 - test_size)) to account for
    train-set overlap across folds (Nadeau & Bengio 2003). NaN-aligned: pairs
    where either side is NaN are dropped before testing.

    Parameters
    ----------
    focal_scores, ref_scores : array-like
        Per-fold AUPRCs (or any per-fold metric) for two competing models,
        already paired by fold index.
    n_splits : int
        Number of splits per repeat (default 10 for 10x10 CV).
    test_size : float
        Fraction of samples held out per split (default 0.1).
    alternative : {"greater", "less", "two-sided"}
        H1 direction.

    Returns
    -------
    dict with keys: t_stat, p, mean_diff, n_paired.
    """
    focal = np.asarray(focal_scores, dtype=float)
    ref = np.asarray(ref_scores, dtype=float)
    if focal.shape != ref.shape:
        raise ValueError(f"shape mismatch: {focal.shape} vs {ref.shape}")
    diffs = focal - ref
    diffs = diffs[~np.isnan(diffs)]
    n = len(diffs)
    if n < 2:
        return {"t_stat": np.nan, "p": np.nan, "mean_diff": np.nan, "n_paired": n}

    mean_diff = float(np.mean(diffs))
    var_diff = float(np.var(diffs, ddof=1))
    correction = (1.0 / n) + (test_size / (1.0 - test_size))
    se = np.sqrt(correction * var_diff)
    t_stat = mean_diff / (se + 1e-12)

    if alternative == "greater":
        p = float(_t_dist.sf(t_stat, n - 1))
    elif alternative == "less":
        p = float(_t_dist.cdf(t_stat, n - 1))
    else:
        p = float(2.0 * _t_dist.sf(abs(t_stat), n - 1))

    return {"t_stat": float(t_stat), "p": p, "mean_diff": mean_diff, "n_paired": n}
