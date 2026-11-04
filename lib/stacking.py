"""Generalized stacking fold runner over arbitrary view configurations.

A stacker fits one base learner per view (i.e, feature subdomain), generates OOF predictions on the
training fold via inner CV, then fits a logistic regression meta learner over
the standardized OOF predictions.

view_configs is a list of `(label, X_view)` 2-tuples. The "baseline" label
selects the clinical-only Ridge pipeline; every other label uses the
demographics + imaging Ridge base learner.
"""

import inspect
import warnings

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from analysis.lib.pipelines import make_base_learner, make_baseline_only

INNER_FOLDS = 10
META_CV_FOLDS = 10  # aligned with INNER_FOLDS for a single project-wide CV depth.


_LRCV_SIG = inspect.signature(LogisticRegressionCV.__init__).parameters
if "use_legacy_attributes" in _LRCV_SIG:
    _META_LR_KWARGS = {"l1_ratios": (0.0,), "use_legacy_attributes": True}
else:
    _META_LR_KWARGS = {"penalty": "l2"}


def _build_learner(label, X_view, n_demo_features, learner_factory=None):
    """Build a per-view base learner.

    - "baseline" view → demographics-only Ridge (no imaging features).
    - any other view → `learner_factory(n_demo, n_total)`. Defaults to
      `make_base_learner` (Ridge); HPC notebooks pass `make_base_learner_tabpfn`
      or `make_base_learner_xgboost` (curried for their device/gpu kwargs) so
      the stacker uses the requested backend per imaging view.
    """
    if label == "baseline":
        return make_baseline_only()
    if learner_factory is None:
        learner_factory = make_base_learner
    return learner_factory(n_demo_features, X_view.shape[1])


def _learner_method(model, score_method):
    """Resolve which prediction method to use for a learner."""
    if score_method == "decision_function":
        return "decision_function"
    if score_method == "predict_proba":
        return "predict_proba"
    return "decision_function" if hasattr(model, "decision_function") else "predict_proba"


def _score(model, X, method):
    """Apply the resolved method, returning a 1-D score vector."""
    if method == "decision_function":
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def process_stacking_fold(
    fold_idx,
    train_idx,
    test_idx,
    view_configs,
    y,
    n_demo_features,
    random_state=42,
    score_method="auto",
    learner_factory=None,
    return_test_scores=False,
    meta_scoring="average_precision",
):
    """Run one outer fold of a generalized stacker over view_configs.

    view_configs: list of (label, X_view).
    score_method: "auto" (default), "decision_function", or "predict_proba".
    learner_factory: callable (n_demo, n_total) -> sklearn estimator used for
        every non-baseline imaging view. Defaults to `make_base_learner`
        (Ridge). HPC notebooks pass TabPFN/XGBoost factories here so the
        stacker actually uses those backends per imaging view (not just for
        the single-view `all_scores.csv` runs).
    return_test_scores: when True, the returned dict additionally carries
        per-subject test-fold predictions (`test_idx`, `y_test`,
        `meta_decision_test` raw meta-LR decision, and `meta_proba_test`
        sigmoid-converted probability), plus the per-view base-learner
        decisions (`base_decisions_test` — shape (n_test, n_views), same
        column order as `view_labels`). Default False keeps the dict
        backward-compatible.
    meta_scoring: sklearn scoring string used to select the L2 regularization
        constant `C` for the LogisticRegressionCV meta learner. Default
        "average_precision" preserves prior behaviour. Set to "roc_auc" to
        compare metric-selection sensitivity. The base learners always use
        RidgeClassifierCV's default GCV criterion regardless of meta_scoring
        — custom scoring on RidgeClassifierCV interacts poorly with its
        internal ±1 label encoding and produces over-fit alphas.
    """
    y_train = y[train_idx]
    y_test = y[test_idx]
    n_train = len(train_idx)
    n_test = len(test_idx)
    n_views = len(view_configs)

    oof_train = np.zeros((n_train, n_views))
    test_preds = np.zeros((n_test, n_views))

    inner_cv = StratifiedKFold(
        n_splits=INNER_FOLDS, shuffle=True, random_state=random_state
    )

    for j, (label, X_view) in enumerate(view_configs):
        try:
            learner = _build_learner(
                label, X_view, n_demo_features,
                learner_factory=learner_factory,
            )
            method = _learner_method(learner, score_method)
            X_view_train = X_view[train_idx]
            X_view_test = X_view[test_idx]
            # Inner-CV alpha selection with scoring="roc_auc" emits an
            # UndefinedMetricWarning on folds that happen to contain only one
            # class (stratification can't guarantee both classes at large k
            # for very small N). The lib already handles single-class outer
            # folds by setting AUROC = NaN; the inner-fold warnings are
            # noise we suppress here.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UndefinedMetricWarning)
                oof = cross_val_predict(
                    learner,
                    X_view_train,
                    y_train,
                    cv=inner_cv,
                    method=method,
                    n_jobs=1,
                )
                if method == "predict_proba":
                    oof = oof[:, 1]
                oof_train[:, j] = oof
                learner.fit(X_view_train, y_train)
            test_preds[:, j] = _score(learner, X_view_test, method)
        except Exception as exc:
            warnings.warn(
                f"process_stacking_fold view={label} fold={fold_idx} "
                f"failed: {exc}"
            )
            oof_train[:, j] = 0.0
            test_preds[:, j] = 0.0

    scaler = StandardScaler()
    oof_scaled = np.nan_to_num(scaler.fit_transform(oof_train), nan=0.0, posinf=0.0, neginf=0.0)
    test_scaled = np.nan_to_num(scaler.transform(test_preds), nan=0.0, posinf=0.0, neginf=0.0)

    meta = LogisticRegressionCV(
        Cs=np.logspace(-4, 1, 10),
        cv=META_CV_FOLDS,
        scoring=meta_scoring,
        max_iter=2000,
        n_jobs=1,
        **_META_LR_KWARGS,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        meta.fit(oof_scaled, y_train)

    decisions_test  = meta.decision_function(test_scaled)
    decisions_train = meta.decision_function(oof_scaled)
    single_test  = len(np.unique(y_test))  < 2
    single_train = len(np.unique(y_train)) < 2

    out = {
        "Fold": fold_idx,
        "AUPRC":       np.nan if single_test  else average_precision_score(y_test,  decisions_test),
        "AUPRC_train": np.nan if single_train else average_precision_score(y_train, decisions_train),
        "AUROC":       np.nan if single_test  else roc_auc_score(y_test,  decisions_test),
        "AUROC_train": np.nan if single_train else roc_auc_score(y_train, decisions_train),
        "meta_coefs": meta.coef_.ravel().tolist(),
        "meta_intercept": float(meta.intercept_[0]),
        "meta_C": float(meta.C_[0]),
        "meta_scoring": meta_scoring,
        "view_labels": [label for label, _ in view_configs],
    }
    if return_test_scores:
        # Sigmoid → calibrated-enough probability for calibration / DCA work.
        # (Meta-LR's predict_proba would do the same.) The base-view scores
        # are kept un-sigmoided so a downstream consumer can apply their own
        # transform if needed; they're the standardized scaler-space scores
        # the meta-LR actually saw.
        out["test_idx"] = np.asarray(test_idx)
        out["y_test"] = np.asarray(y_test)
        out["meta_decision_test"] = decisions_test
        out["meta_proba_test"] = 1.0 / (1.0 + np.exp(-decisions_test))
        out["base_decisions_test"] = test_scaled  # (n_test, n_views)
    return out
