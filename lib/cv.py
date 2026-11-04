"""Cross-validation helpers and paired/sample-size runners over stackers."""

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from analysis.lib.stacking import process_stacking_fold

POOLED_N_SPLITS = 10
POOLED_N_REPEATS = 10


def pooled_cv(seed=42):
    """Return a RepeatedStratifiedKFold(10x10) for the pooled outer CV."""
    return RepeatedStratifiedKFold(
        n_splits=POOLED_N_SPLITS,
        n_repeats=POOLED_N_REPEATS,
        random_state=seed,
    )


def loco_cv(cohort_series):
    """Leave-one-cohort-out splits.

    For each unique non-NaN cohort label, yield (train_idx, test_idx) where
    test contains positional indices belonging to that cohort and train is
    the rest. Rows with NaN cohort always go into train.
    """
    cohort = pd.Series(cohort_series).reset_index(drop=True)
    positions = np.arange(len(cohort))
    valid = cohort.dropna()
    cohorts = valid.unique()
    for c in cohorts:
        test_mask = (cohort == c).to_numpy()
        test_idx = positions[test_mask]
        train_idx = positions[~test_mask]
        yield train_idx, test_idx


def paired_fold_runner(splits, arm_a_views, arm_b_views, y, n_demo_features):
    """Run two stacker arms on identical (train, test) splits.

    Returns {"a": [fold_dict, ...], "b": [fold_dict, ...]}.
    """
    results = {"a": [], "b": []}
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        r_a = process_stacking_fold(
            fold_idx, train_idx, test_idx, arm_a_views, y, n_demo_features
        )
        r_b = process_stacking_fold(
            fold_idx, train_idx, test_idx, arm_b_views, y, n_demo_features
        )
        results["a"].append(r_a)
        results["b"].append(r_b)
    return results


def sample_size_curve(
    arm_a_views,
    arm_b_views,
    y,
    n_demo,
    fractions=(0.1, 0.25, 0.5, 0.75, 1.0),
    n_outer_folds=10,
    n_bootstraps=10,
    seed=42,
):
    """Learning curve via train-set subsampling within fixed outer CV folds.

    - Outer CV (StratifiedKFold, default 10 folds) splits the data into
      train/test partitions once per fold. The test fold is fixed for that
      fold's row of the learning curve.
    - For each fraction f: subsample the train fold stratified to fraction f,
      fit on the subsample, evaluate on the (full, untouched) test fold.
    - Repeat n_bootstraps times per (fold, fraction) to reduce subsampling noise.

    Returns long-form DataFrame with columns: frac, fold, boot, arm, AUPRC.
    """
    from joblib import Parallel, delayed

    y = np.asarray(y)
    n_total = len(y)
    outer_cv = StratifiedKFold(
        n_splits=n_outer_folds, shuffle=True, random_state=seed
    )

    def _stratified_train_subsample(train_idx, frac, rng):
        if frac >= 1.0:
            return train_idx
        pos = train_idx[y[train_idx] == 1]
        neg = train_idx[y[train_idx] == 0]
        n_pos = max(2, int(round(len(pos) * frac)))
        n_neg = max(2, int(round(len(neg) * frac)))
        return np.concatenate([
            rng.choice(pos, size=n_pos, replace=False),
            rng.choice(neg, size=n_neg, replace=False),
        ])

    def _one_run(arm_label, views, train_sub, test_idx, fold_i, frac, boot):
        try:
            out = process_stacking_fold(
                fold_i, train_sub, test_idx, views, y, n_demo
            )
            auprc = out["AUPRC"]
        except Exception:
            auprc = float("nan")
        return {
            "frac": frac, "fold": fold_i, "boot": boot,
            "arm": arm_label, "AUPRC": auprc,
        }

    jobs = []
    for fold_i, (train_full, test_idx) in enumerate(outer_cv.split(np.zeros(n_total), y)):
        for frac in fractions:
            # frac=1.0 is deterministic (no rng), so 1 boot suffices; smaller
            # fractions benefit from multiple bootstraps to average out the
            # subsampling draw.
            n_boots = 1 if frac >= 1.0 else n_bootstraps
            for boot in range(n_boots):
                rng = np.random.RandomState(seed + fold_i * 10_000 + boot)
                train_sub = _stratified_train_subsample(train_full, frac, rng)
                for arm_label, views in [("a", arm_a_views), ("b", arm_b_views)]:
                    jobs.append((arm_label, views, train_sub, test_idx,
                                  fold_i, frac, boot))

    records = Parallel(n_jobs=-1)(
        delayed(_one_run)(*job) for job in jobs
    )
    return pd.DataFrame(records)


def sample_size_curve_baselines(
    y,
    X_base,
    fractions=(0.1, 0.25, 0.5, 0.75, 1.0),
    n_outer_folds=10,
    n_bootstraps=10,
    seed=42,
    splits=None,
):
    """Permuted + Majority arms for the sample-size curve, sharing the same
    outer-CV folds as `sample_size_curve` (or `sample_size_curve_loco` if
    `splits` is supplied).

    Both arms are cheap relative to the imaging + stacker arms, so by this we
    add reference baselines to a sample-size figure without re-running anything else.

    Arms produced (long-form columns: frac, fold, boot, arm, AUPRC):
    - "majority": AUPRC = positive-class prevalence in the held-out test fold.
      Independent of training data and training fraction → constant across
      `frac` per (target, fold). One row per (frac, fold) with boot=0.
    - "permuted": baseline-only Ridge fit on shuffled `y_train` (subsampled to
      `frac`), scored on the *real* `y_test`. Expected AUPRC near the
      positive-class prevalence; per-fold variance reflects shuffle noise.

    Parameters
    ----------
    y : 1-D numpy array
    X_base : (n_subjects, n_demo) demographics design matrix
    fractions, n_outer_folds, n_bootstraps, seed : same semantics as
        `sample_size_curve` — pass the same values to get aligned folds.
    splits : optional iterable of (train_idx, test_idx) — pass LOCO splits
        from `loco_cv` (or its valid-mask remap) to mirror
        `sample_size_curve_loco`. When supplied, n_outer_folds is ignored.
    """
    from joblib import Parallel, delayed
    from sklearn.metrics import average_precision_score
    from analysis.lib.pipelines import make_baseline_only

    y = np.asarray(y)
    n_total = len(y)

    if splits is None:
        outer_cv = StratifiedKFold(
            n_splits=n_outer_folds, shuffle=True, random_state=seed
        )
        splits = list(outer_cv.split(np.zeros(n_total), y))
    else:
        splits = list(splits)

    def _stratified_train_subsample(train_idx, frac, rng):
        if frac >= 1.0:
            return train_idx
        pos = train_idx[y[train_idx] == 1]
        neg = train_idx[y[train_idx] == 0]
        n_pos = max(2, int(round(len(pos) * frac)))
        n_neg = max(2, int(round(len(neg) * frac)))
        return np.concatenate([
            rng.choice(pos, size=n_pos, replace=False),
            rng.choice(neg, size=n_neg, replace=False),
        ])

    records = []

    # --- Majority: AUPRC = test-fold positive prevalence (training-invariant) ---
    for fold_i, (_, test_idx) in enumerate(splits):
        if len(np.unique(y[test_idx])) < 2:
            continue
        prevalence = float(np.mean(y[test_idx]))
        for frac in fractions:
            records.append({
                "frac": frac, "fold": fold_i, "boot": 0,
                "arm": "majority", "AUPRC": prevalence,
            })

    # --- Permuted: baseline-only Ridge on shuffled y_train, real y_test --------
    def _permuted_one(train_sub, test_idx, fold_i, frac, boot, perm_seed):
        if len(np.unique(y[test_idx])) < 2:
            return {"frac": frac, "fold": fold_i, "boot": boot,
                    "arm": "permuted", "AUPRC": float("nan")}
        try:
            rng_p = np.random.RandomState(perm_seed)
            y_train_perm = rng_p.permutation(y[train_sub])
            pipe = make_baseline_only()
            pipe.fit(X_base[train_sub], y_train_perm)
            scores = pipe.decision_function(X_base[test_idx])
            auprc = float(average_precision_score(y[test_idx], scores))
        except Exception:
            auprc = float("nan")
        return {"frac": frac, "fold": fold_i, "boot": boot,
                "arm": "permuted", "AUPRC": auprc}

    jobs = []
    for fold_i, (train_full, test_idx) in enumerate(splits):
        for frac in fractions:
            n_boots = 1 if frac >= 1.0 else n_bootstraps
            for boot in range(n_boots):
                rng = np.random.RandomState(seed + fold_i * 10_000 + boot)
                train_sub = _stratified_train_subsample(train_full, frac, rng)
                # Use a separate RNG seed for the label permutation so it is
                # not coupled to the subsampling seed.
                perm_seed = seed + fold_i * 10_000 + boot + 9999
                jobs.append((train_sub, test_idx, fold_i, frac, boot, perm_seed))

    perm_records = Parallel(n_jobs=-1)(
        delayed(_permuted_one)(*job) for job in jobs
    )
    records.extend(perm_records)

    return pd.DataFrame(records)


def sample_size_curve_loco(
    arm_a_views,
    arm_b_views,
    y,
    splits,
    n_demo,
    fractions=(0.1, 0.25, 0.5, 0.75, 1.0),
    n_bootstraps=10,
    seed=42,
):
    """LOCO learning curve via train-fold subsampling.

    Mirrors `sample_size_curve` but uses caller-supplied (train_idx, test_idx)
    splits instead of inner StratifiedKFold. Typical usage: pass LOCO splits
    reindexed for the target's valid mask via `splits_for(...)`. Each outer
    fold's test set is a fixed held-out cohort; within the fold, the train
    portion is stratified-subsampled to fraction f and the stacker evaluated
    on the (full, untouched) test cohort. Train-fold size varies across folds
    (full ≈ N - N_cohort), so the long-form output records `n_train` per row.

    Returns long-form DataFrame: frac, fold, boot, arm, AUPRC, n_train.
    """
    from joblib import Parallel, delayed

    y = np.asarray(y)
    splits = list(splits)

    def _stratified_train_subsample(train_idx, frac, rng):
        if frac >= 1.0:
            return train_idx
        pos = train_idx[y[train_idx] == 1]
        neg = train_idx[y[train_idx] == 0]
        n_pos = max(2, int(round(len(pos) * frac)))
        n_neg = max(2, int(round(len(neg) * frac)))
        return np.concatenate([
            rng.choice(pos, size=n_pos, replace=False),
            rng.choice(neg, size=n_neg, replace=False),
        ])

    def _one_run(arm_label, views, train_sub, test_idx, fold_i, frac, boot, n_train):
        try:
            out = process_stacking_fold(
                fold_i, train_sub, test_idx, views, y, n_demo
            )
            auprc = out["AUPRC"]
        except Exception:
            auprc = float("nan")
        return {
            "frac": frac, "fold": fold_i, "boot": boot,
            "arm": arm_label, "AUPRC": auprc, "n_train": n_train,
        }

    jobs = []
    for fold_i, (train_full, test_idx) in enumerate(splits):
        for frac in fractions:
            n_boots = 1 if frac >= 1.0 else n_bootstraps
            for boot in range(n_boots):
                rng = np.random.RandomState(seed + fold_i * 10_000 + boot)
                train_sub = _stratified_train_subsample(train_full, frac, rng)
                for arm_label, views in [("a", arm_a_views), ("b", arm_b_views)]:
                    jobs.append((arm_label, views, train_sub, test_idx,
                                  fold_i, frac, boot, len(train_sub)))

    records = Parallel(n_jobs=-1)(
        delayed(_one_run)(*job) for job in jobs
    )
    return pd.DataFrame(records)
