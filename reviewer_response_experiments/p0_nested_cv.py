"""Leakage-free P0 evaluation for the reviewer response.

The script evaluates full and empirically selected top-10 models with repeated
stratified nested CV. All feature selection, scaling, tuning, and calibration
are fitted inside training data only. Probabilities are reported as reentry
(class 1) probabilities; withdrawal risk is 1 - p(reentry).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SEEDS = [42, 123, 2025, 6, 255]
LABEL = "School Withdrawal/ Reentry Status"


class EmpiricalTop10Selector(BaseEstimator, TransformerMixin):
    """Select top features using LR/RF importance, fitted only on the fold."""

    def __init__(self, k: int = 10, random_state: int = 42):
        self.k = k
        self.random_state = random_state

    def fit(self, X, y):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
            X_fit = X.to_numpy(dtype=float)
        else:
            X_fit = np.asarray(X, dtype=float)
            self.feature_names_in_ = np.asarray([f"x{i}" for i in range(X_fit.shape[1])], dtype=object)
        # Coefficient magnitudes are meaningful for ranking only after fitting
        # the LR selector on a common within-fold scale.  The scaler is fitted
        # here (inside this transformer), never on an outer test fold.
        selector_scaler = StandardScaler()
        X_lr = selector_scaler.fit_transform(X_fit)
        lr = LogisticRegression(
            C=1.0,
            max_iter=2000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=self.random_state,
        )
        rf = RandomForestClassifier(
            n_estimators=100, random_state=self.random_state, n_jobs=1
        )
        lr.fit(X_lr, y)
        rf.fit(X_fit, y)
        lr_rank = np.argsort(np.abs(lr.coef_[0]))[::-1]
        rf_rank = np.argsort(rf.feature_importances_)[::-1]
        counts = np.zeros(X_fit.shape[1], dtype=int)
        # Frequency in each model's top-k list mirrors the manuscript's
        # empirical selection while ensuring the feature set is fold-specific.
        counts[lr_rank[: self.k]] += 1
        counts[rf_rank[: self.k]] += 1
        mean_rank = (np.abs(lr.coef_[0]) / (np.abs(lr.coef_[0]).max() + 1e-12)) + (
            rf.feature_importances_ / (rf.feature_importances_.max() + 1e-12)
        )
        order = np.lexsort((-mean_rank, -counts))
        # np.lexsort uses the last key as primary; explicitly sort for clarity.
        order = sorted(range(len(counts)), key=lambda i: (counts[i], mean_rank[i]), reverse=True)
        required = [
            i for i, name in enumerate(self.feature_names_in_)
            if name in {"HEI_TS", "CSES_TS"}
        ]
        selected = required + [i for i in order if i not in required]
        self.selected_indices_ = np.asarray(selected[: min(self.k, X_fit.shape[1])], dtype=int)
        self.selected_features_ = self.feature_names_in_[self.selected_indices_].tolist()
        return self

    def transform(self, X):
        if hasattr(X, "iloc"):
            return X.iloc[:, self.selected_indices_]
        return np.asarray(X)[:, self.selected_indices_]


def make_estimator(model: str, selector: bool, seed: int) -> Pipeline:
    if model == "LR":
        clf = LogisticRegression(max_iter=3000, solver="lbfgs", random_state=seed)
    elif model == "RF":
        clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)
    elif model == "SVM":
        clf = SVC(kernel="rbf", probability=True, random_state=seed)
    else:
        raise ValueError(model)
    steps = []
    if selector:
        steps.append(("selector", EmpiricalTop10Selector(k=10, random_state=seed)))
    steps.extend([("scaler", StandardScaler()), ("clf", clf)])
    return Pipeline(steps)


def grid_for(model: str, selector: bool) -> dict[str, list[Any]]:
    prefix = "clf__"
    if model == "LR":
        grid = {prefix + "C": [0.1, 1.0, 10.0], prefix + "class_weight": [None, "balanced"]}
    elif model == "RF":
        grid = {
            prefix + "max_depth": [None, 5],
            prefix + "class_weight": [None, "balanced"],
        }
    else:
        grid = {
            prefix + "C": [0.5, 2.0],
            prefix + "gamma": ["scale"],
            prefix + "class_weight": [None, "balanced"],
        }
    return grid


def safe_metric(fn, y_true, score):
    try:
        return float(fn(y_true, score))
    except ValueError:
        return float("nan")


def fold_metrics(y_true: np.ndarray, p_reentry: np.ndarray) -> dict[str, float]:
    y_withdraw = (y_true == 0).astype(int)
    p_withdraw = 1.0 - p_reentry
    pred_withdraw = (p_withdraw >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_withdraw, pred_withdraw, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_withdraw, pred_withdraw)),
        "balanced_accuracy": float(balanced_accuracy_score(y_withdraw, pred_withdraw)),
        "withdrawal_sensitivity": float(recall_score(y_withdraw, pred_withdraw, zero_division=0)),
        "withdrawal_specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "withdrawal_precision": float(precision_score(y_withdraw, pred_withdraw, zero_division=0)),
        "withdrawal_f1": float(f1_score(y_withdraw, pred_withdraw, zero_division=0)),
        "withdrawal_auroc": safe_metric(roc_auc_score, y_withdraw, p_withdraw),
        "withdrawal_pr_auc": safe_metric(average_precision_score, y_withdraw, p_withdraw),
        "reentry_brier": float(brier_score_loss(y_true, p_reentry)),
        "n_test": int(len(y_true)),
        "n_withdrawal_test": int(y_withdraw.sum()),
    }


def _append_or_replace(existing: pd.DataFrame, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Keep existing configurations while replacing exactly rerun configurations."""
    if existing.empty:
        return new
    if new.empty:
        return existing
    combined = pd.concat([existing, new], ignore_index=True)
    return combined.drop_duplicates(subset=keys, keep="last")


def _read_if_present(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def run_group(
    path: Path,
    out_dir: Path,
    n_splits: int,
    n_repeats: int,
    calibrate: bool,
    models: list[str],
    feature_sets: list[str],
    reset_configs: bool,
) -> None:
    df = pd.read_csv(path)
    X = df.drop(columns=[LABEL])
    y = df[LABEL].astype(int).to_numpy()
    group = path.stem.replace("clean_", "")
    group_dir = out_dir / group
    group_dir.mkdir(parents=True, exist_ok=True)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=20260710)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=20260710)
    existing_metrics = _read_if_present(group_dir / "nested_fold_metrics.csv")
    existing_predictions = _read_if_present(group_dir / "nested_oof_predictions.csv")
    existing_selection = _read_if_present(group_dir / "nested_feature_selection.csv")
    existing_params = _read_if_present(group_dir / "nested_best_params.csv")
    target = lambda d: (
        d["model"].isin(models) & d["feature_set"].isin(feature_sets)
        if not d.empty else pd.Series(dtype=bool)
    )
    if reset_configs:
        existing_metrics = existing_metrics.loc[~target(existing_metrics)].copy()
        existing_predictions = existing_predictions.loc[~target(existing_predictions)].copy()
        existing_selection = existing_selection.loc[~target(existing_selection)].copy()
        existing_params = existing_params.loc[~target(existing_params)].copy()
    all_metrics = existing_metrics.to_dict("records")
    all_predictions = existing_predictions.to_dict("records")
    selection_rows = existing_selection.to_dict("records")
    param_rows = existing_params.to_dict("records")
    completed = {
        (str(r["model"]), str(r["feature_set"]), int(r["fold"]))
        for _, r in existing_metrics.iterrows()
    }
    for model in models:
        for feature_set, selector in [("full", False), ("top10", True)]:
            if feature_set not in feature_sets:
                continue
            for fold_id, (train_idx, test_idx) in enumerate(cv.split(X, y)):
                if (model, feature_set, fold_id) in completed:
                    continue
                seed = int(SEEDS[fold_id % len(SEEDS)])
                estimator = make_estimator(model, selector, seed)
                search = GridSearchCV(
                    estimator,
                    grid_for(model, selector),
                    scoring="roc_auc",
                    cv=inner,
                    n_jobs=-1,
                    refit=True,
                    error_score="raise",
                )
                search.fit(X.iloc[train_idx], y[train_idx])
                fitted = search.best_estimator_
                if calibrate:
                    calibrated = CalibratedClassifierCV(
                        estimator=fitted, method="sigmoid", cv=3, n_jobs=-1
                    )
                    calibrated.fit(X.iloc[train_idx], y[train_idx])
                    fitted_for_test = calibrated
                else:
                    fitted_for_test = fitted
                p = fitted_for_test.predict_proba(X.iloc[test_idx])[:, 1]
                metrics = fold_metrics(y[test_idx], p)
                metrics.update(
                    {
                        "group": group,
                        "model": model,
                        "feature_set": feature_set,
                        "fold": fold_id,
                        "repeat": fold_id // n_splits,
                        "best_inner_auc": float(search.best_score_),
                    }
                )
                all_metrics.append(metrics)
                all_predictions.extend(
                    {
                        "group": group,
                        "model": model,
                        "feature_set": feature_set,
                        "fold": fold_id,
                        "repeat": fold_id // n_splits,
                        "row_index": int(i),
                        "y": int(y[i]),
                        "p_reentry": float(pp),
                    }
                    for i, pp in zip(test_idx, p)
                )
                param_rows.append(
                    {
                        "group": group,
                        "model": model,
                        "feature_set": feature_set,
                        "fold": fold_id,
                        "best_inner_auc": float(search.best_score_),
                        "best_params": json.dumps(search.best_params_, sort_keys=True, default=str),
                    }
                )
                if selector:
                    sel = fitted.named_steps["selector"]
                    selection_rows.append(
                        {
                            "group": group,
                            "model": model,
                            "feature_set": feature_set,
                            "fold": fold_id,
                            "selected_features": json.dumps(sel.selected_features_, ensure_ascii=False),
                        }
                    )
                # Persist after every outer fold.  This makes a long nested
                # run resumable without ever mixing a partial test fold into
                # a completed estimate.
                pd.DataFrame(all_metrics).to_csv(group_dir / "nested_fold_metrics.csv", index=False, encoding="utf-8-sig")
                pd.DataFrame(all_predictions).to_csv(group_dir / "nested_oof_predictions.csv", index=False, encoding="utf-8-sig")
                pd.DataFrame(selection_rows).to_csv(group_dir / "nested_feature_selection.csv", index=False, encoding="utf-8-sig")
                pd.DataFrame(param_rows).to_csv(group_dir / "nested_best_params.csv", index=False, encoding="utf-8-sig")
    metrics_df = pd.DataFrame(all_metrics)
    pred_df = pd.DataFrame(all_predictions)
    selection_df = pd.DataFrame(selection_rows)
    params_df = pd.DataFrame(param_rows)
    selection_df.to_csv(group_dir / "nested_feature_selection.csv", index=False, encoding="utf-8-sig")
    params_df.to_csv(group_dir / "nested_best_params.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(group_dir / "nested_fold_metrics.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(group_dir / "nested_oof_predictions.csv", index=False, encoding="utf-8-sig")
    summary = (
        metrics_df.groupby(["group", "model", "feature_set"], as_index=False)
        .agg({c: ["mean", "std"] for c in metrics_df.columns if c in {
            "accuracy", "balanced_accuracy", "withdrawal_sensitivity", "withdrawal_specificity",
            "withdrawal_precision", "withdrawal_f1", "withdrawal_auroc", "withdrawal_pr_auc",
            "reentry_brier", "best_inner_auc"
        }})
    )
    summary.columns = ["_".join(x).strip("_") if isinstance(x, tuple) else x for x in summary.columns]
    summary.to_csv(group_dir / "nested_summary.csv", index=False, encoding="utf-8-sig")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="reviewer_response_experiments/results/p0_nested_cv")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-repeats", type=int, default=10)
    ap.add_argument("--no-calibration", action="store_true")
    ap.add_argument("--groups", nargs="+", choices=["adults", "teens", "children"], default=["adults", "teens", "children"])
    ap.add_argument("--models", nargs="+", choices=["LR", "RF", "SVM"], default=["LR", "RF", "SVM"])
    ap.add_argument("--feature-sets", nargs="+", choices=["full", "top10"], default=["full", "top10"])
    ap.add_argument("--reset-configs", action="store_true", help="Discard prior rows for the requested model/feature-set configurations before a resumable rerun.")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.groups:
        run_group(
            Path(args.data_dir) / f"clean_{name}.csv",
            out,
            args.n_splits,
            args.n_repeats,
            not args.no_calibration,
            args.models,
            args.feature_sets,
            args.reset_configs,
        )
    print(f"P0 nested CV complete: {out.resolve()}")


if __name__ == "__main__":
    main()
