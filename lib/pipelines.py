"""Sklearn pipeline factories and per-fold runners for predictive modeling.

Three pipeline shapes:
    - baseline-only (demographics-only Ridge)
    - base learner (demographics passthrough + imaging Ridge)
    - Nonlinear variants (same preprocessor with XGBoost / TabPFN classifiers)
"""

import warnings

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RIDGE_ALPHAS = np.logspace(-3, 3, 10)
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "reg_alpha": 0.5,
    "n_estimators": 500,
    "early_stopping_rounds": 50,
    "verbosity": 0,
}


# Custom XGB wrapper: stratified internal val split + scale_pos_weight + early stopping.
try:
    import xgboost as _xgboost_for_wrapper

    class XGBClassifierWithEarlyStopping(_xgboost_for_wrapper.XGBClassifier):
        """XGBClassifier with built-in train/val split + per-fit scale_pos_weight."""

        def __init__(self, *, val_size=0.15, val_seed=42, **kwargs):
            # Default val_seed aligned with the project-wide seed=42 convention.
            super().__init__(**kwargs)
            self.val_size = val_size
            self.val_seed = val_seed

        def fit(self, X, y, **kwargs):
            from sklearn.model_selection import train_test_split

            y = np.asarray(y)
            try:
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X, y, test_size=self.val_size,
                    stratify=y, random_state=self.val_seed,
                )
            except ValueError:
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X, y, test_size=self.val_size, random_state=self.val_seed,
                )

            n_pos = int((y_tr == 1).sum())
            n_neg = int((y_tr == 0).sum())
            if n_pos > 0 and n_neg > 0:
                self.set_params(scale_pos_weight=n_neg / n_pos)

            kwargs.setdefault("eval_set", [(X_val, y_val)])
            kwargs.setdefault("verbose", False)
            return super().fit(X_tr, y_tr, **kwargs)

    _HAS_XGB_WRAPPER = True
except ImportError:
    _HAS_XGB_WRAPPER = False


def make_baseline_only():
    """Demographics-only Ridge pipeline (impute -> scale -> RidgeClassifierCV).
    """
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("clf", RidgeClassifierCV(alphas=RIDGE_ALPHAS, class_weight="balanced")),
    ])


def _make_preprocessor(n_demo, n_total):
    """ColumnTransformer: demographics impute+scale, imaging impute+scale.

    Returns None if there are no imaging features (n_total <= n_demo).
    """
    if n_total <= n_demo:
        return None
    demo_cols = list(range(n_demo))
    img_cols = list(range(n_demo, n_total))
    demo_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
    ])
    img_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
    ])
    return ColumnTransformer([
        ("demo", demo_pipe, demo_cols),
        ("img", img_pipe, img_cols),
    ])


def make_base_learner(n_demo, n_total):
    """Demo passthrough + imaging impute+scale -> RidgeClassifierCV.

    Falls back to make_baseline_only() if there are no imaging features.
    """
    pre = _make_preprocessor(n_demo, n_total)
    if pre is None:
        return make_baseline_only()
    clf = RidgeClassifierCV(alphas=RIDGE_ALPHAS, class_weight="balanced")
    return Pipeline([("prep", pre), ("clf", clf)])


def make_base_learner_xgboost(n_demo, n_total, gpu=False):
    """XGBoost classifier with the demo+imaging preprocessor."""
    if not _HAS_XGB_WRAPPER:
        raise ImportError("xgboost not installed; cannot construct XGBoost base learner.")

    params = dict(XGB_PARAMS)
    if gpu:
        params["device"] = "cuda"
    clf = XGBClassifierWithEarlyStopping(**params)
    pre = _make_preprocessor(n_demo, n_total)
    if pre is None:
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("clf", clf),
        ])
    return Pipeline([("prep", pre), ("clf", clf)])


def make_base_learner_tabpfn(n_demo, n_total, device="cuda"):
    """TabPFN classifier with the demo+imaging preprocessor."""
    from tabpfn import TabPFNClassifier

    clf = TabPFNClassifier(device=device, ignore_pretraining_limits=True)
    pre = _make_preprocessor(n_demo, n_total)
    if pre is None:
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("clf", clf),
        ])
    return Pipeline([("prep", pre), ("clf", clf)])


def process_fold(
    fold_idx, train_idx, test_idx, X, y, model_name, target, n_demo_features,
):
    """Run one outer fold: fit Ridge base learner, score on test fold."""
    try:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        n_features = X_train.shape[1]
        n_imaging = n_features - n_demo_features

        pipe = make_base_learner(n_demo_features, n_features) \
            if n_imaging > 0 else make_baseline_only()
        # Inner-CV alpha selection emits UndefinedMetricWarning when a fold
        # happens to be single-class. The lib already handles single-class
        # outer folds via NaN. Suppress the inner-fold noise here so notebook
        # output stays readable.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UndefinedMetricWarning)
            pipe.fit(X_train, y_train)

        scores_test = pipe.decision_function(X_test)
        scores_train = pipe.decision_function(X_train)
        single_test  = len(np.unique(y_test))  < 2
        single_train = len(np.unique(y_train)) < 2
        return {
            "Target": target,
            "Model": model_name,
            "Fold": fold_idx,
            "AUPRC_train": np.nan if single_train else average_precision_score(y_train, scores_train),
            "AUPRC":       np.nan if single_test  else average_precision_score(y_test,  scores_test),
            "AUROC_train": np.nan if single_train else roc_auc_score(y_train, scores_train),
            "AUROC":       np.nan if single_test  else roc_auc_score(y_test,  scores_test),
        }
    except Exception as exc:
        warnings.warn(
            f"process_fold failed for target={target} model={model_name} "
            f"fold={fold_idx}: {exc}"
        )
        return {
            "Target": target,
            "Model": model_name,
            "Fold": fold_idx,
            "AUPRC_train": np.nan,
            "AUPRC": np.nan,
            "AUROC_train": np.nan,
            "AUROC": np.nan,
        }


def process_fold_majority(fold_idx, train_idx, test_idx, y, target):
    """Majority/prevalence baseline — constant predictor scored via sklearn.

    Fits a constant predictor that emits the training-fold positive-class
    prevalence for every subject, and scores it through the same
    `average_precision_score` / `roc_auc_score` pipeline used by every other
    model in this file. Expected values on a constant score vector:

      * AUROC = 0.5 exactly. This is a mathematical identity — a fully-tied
        score vector produces a ROC curve that collapses to the diagonal,
        and any correctly-implemented ROC-AUC integrates it to 0.5.
      * AUPRC = evaluation-fold positive prevalence, in practice, under
        sklearn's `average_precision_score`. This uses the step-function
        formula `AP = Σ (R_n − R_{n-1}) · P_n`, and with a single big tie
        the entire curve collapses to one step from recall 0 to recall 1
        at precision = prevalence, integrating to prevalence exactly. Other
        AP definitions (trapezoidal / interpolated PR area) could yield a
        marginally different number for the same data — flagged here so the
        baseline can be reproduced faithfully by a reviewer using a
        different library.

    Single-class folds return NaN for the affected side.
    """
    y_train = y[train_idx]
    y_test  = y[test_idx]
    prevalence_train = float(np.mean(y_train))
    scores_train = np.full(len(y_train), prevalence_train, dtype=float)
    scores_test  = np.full(len(y_test),  prevalence_train, dtype=float)
    single_train = len(np.unique(y_train)) < 2
    single_test  = len(np.unique(y_test))  < 2
    return {
        "Target": target,
        "Model":  "Majority",
        "Fold":   fold_idx,
        "AUPRC_train": np.nan if single_train else float(average_precision_score(y_train, scores_train)),
        "AUPRC":       np.nan if single_test  else float(average_precision_score(y_test,  scores_test)),
        "AUROC_train": np.nan if single_train else float(roc_auc_score(y_train, scores_train)),
        "AUROC":       np.nan if single_test  else float(roc_auc_score(y_test,  scores_test)),
    }


N_PERMS_PER_FOLD = 5


def process_fold_permuted(
    fold_idx, train_idx, test_idx, X, y, target, n_demo_features, rng_seed,
):
    """Permutation control: shuffle y_train only; evaluate against real y_test.

    Mean over N_PERMS_PER_FOLD shuffles per fold (high-variance single shuffle on
    small LOCO held-out cohorts → average to stabilize per-fold AUPRCs).
    """
    rng = np.random.RandomState(rng_seed + fold_idx)
    auprc_list, auprc_train_list = [], []
    auroc_list, auroc_train_list = [], []
    for _ in range(N_PERMS_PER_FOLD):
        y_perm = np.array(y, copy=True)
        y_perm[train_idx] = rng.permutation(y[train_idx])
        result = process_fold(
            fold_idx, train_idx, test_idx, X, y_perm, "Permuted", target,
            n_demo_features,
        )
        if not np.isnan(result["AUPRC"]):
            auprc_list.append(result["AUPRC"])
        if not np.isnan(result["AUPRC_train"]):
            auprc_train_list.append(result["AUPRC_train"])
        if not np.isnan(result.get("AUROC", np.nan)):
            auroc_list.append(result["AUROC"])
        if not np.isnan(result.get("AUROC_train", np.nan)):
            auroc_train_list.append(result["AUROC_train"])

    def _mean_or_nan(xs):
        return float(np.mean(xs)) if xs else np.nan

    return {
        "Target": target,
        "Model": "Permuted",
        "Fold": fold_idx,
        "AUPRC_train": _mean_or_nan(auprc_train_list),
        "AUPRC":       _mean_or_nan(auprc_list),
        "AUROC_train": _mean_or_nan(auroc_train_list),
        "AUROC":       _mean_or_nan(auroc_list),
    }


def process_fold_proba(
    fold_idx,
    train_idx,
    test_idx,
    X,
    y,
    model_name,
    target,
    n_demo_features,
    classifier_factory,
):
    """Fold runner for classifiers exposing predict_proba.

    classifier_factory: callable (n_demo, n_total) -> sklearn estimator.
    """
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    n_features = X_train.shape[1]

    try:
        pipe = classifier_factory(n_demo_features, n_features)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UndefinedMetricWarning)
            pipe.fit(X_train, y_train)

        proba_test  = pipe.predict_proba(X_test)[:, 1]
        proba_train = pipe.predict_proba(X_train)[:, 1]
        single_test  = len(np.unique(y_test))  < 2
        single_train = len(np.unique(y_train)) < 2
        return {
            "Target": target,
            "Model": model_name,
            "Fold": fold_idx,
            "AUPRC_train": np.nan if single_train else average_precision_score(y_train, proba_train),
            "AUPRC":       np.nan if single_test  else average_precision_score(y_test,  proba_test),
            "AUROC_train": np.nan if single_train else roc_auc_score(y_train, proba_train),
            "AUROC":       np.nan if single_test  else roc_auc_score(y_test,  proba_test),
        }
    except Exception as exc:
        warnings.warn(
            f"process_fold_proba failed for target={target} model={model_name} "
            f"fold={fold_idx}: {exc}"
        )
        return {
            "Target": target,
            "Model": model_name,
            "Fold": fold_idx,
            "AUPRC_train": np.nan,
            "AUPRC": np.nan,
            "AUROC_train": np.nan,
            "AUROC": np.nan,
        }
