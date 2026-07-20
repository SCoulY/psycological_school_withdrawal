"""Fixed-default baseline on the same outer folds as the revised nested CV.

This is a descriptive sensitivity comparison, not a replacement for nested
model selection.  It uses the revised fold-specific selector, preprocessing,
outer splits, and sigmoid calibration, but skips the inner GridSearchCV and
fits each estimator at its documented default settings.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RepeatedStratifiedKFold

from reviewer_response_experiments.p0_nested_cv import (
    LABEL,
    SEEDS,
    fold_metrics,
    make_estimator,
)


BASE = Path("reviewer_response_experiments/results/p0_nested_cv_scaled_selector")
OUT_FOLDS = BASE / "default_baseline_fold_metrics.csv"
OUT_SUMMARY = BASE / "default_baseline_summary.csv"
OUT_COMPARE = BASE / "tuned_vs_default_summary.csv"


def main() -> None:
    rows: list[dict[str, float | int | str]] = []
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=20260710)
    for group in ("adults", "teens", "children"):
        df = pd.read_csv(Path("data") / f"clean_{group}.csv")
        X = df.drop(columns=[LABEL])
        y = df[LABEL].astype(int).to_numpy()
        for model in ("LR", "RF", "SVM"):
            for feature_set, selector in (("full", False), ("top10", True)):
                for fold_id, (train_idx, test_idx) in enumerate(cv.split(X, y)):
                    seed = int(SEEDS[fold_id % len(SEEDS)])
                    estimator = make_estimator(model, selector, seed)
                    estimator.fit(X.iloc[train_idx], y[train_idx])
                    calibrated = CalibratedClassifierCV(
                        estimator=estimator, method="sigmoid", cv=3, n_jobs=-1
                    )
                    calibrated.fit(X.iloc[train_idx], y[train_idx])
                    p = calibrated.predict_proba(X.iloc[test_idx])[:, 1]
                    metrics = fold_metrics(y[test_idx], p)
                    metrics.update(
                        {
                            "group": group,
                            "model": model,
                            "feature_set": feature_set,
                            "fold": fold_id,
                            "repeat": fold_id // 5,
                        }
                    )
                    rows.append(metrics)

    fold_df = pd.DataFrame(rows)
    OUT_FOLDS.parent.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(OUT_FOLDS, index=False, encoding="utf-8-sig")
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "withdrawal_sensitivity",
        "withdrawal_specificity",
        "withdrawal_precision",
        "withdrawal_f1",
        "withdrawal_auroc",
        "withdrawal_pr_auc",
        "reentry_brier",
    ]
    summary = fold_df.groupby(["group", "model", "feature_set"], as_index=False).agg(
        **{f"{m}_mean": (m, "mean") for m in metric_names},
        **{f"{m}_std": (m, "std") for m in metric_names},
    )
    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")

    tuned_rows = []
    for group in ("adults", "teens", "children"):
        tuned = pd.read_csv(BASE / group / "nested_fold_metrics.csv")
        tuned = tuned.assign(group=group)
        tuned_rows.append(tuned)
    tuned_df = pd.concat(tuned_rows, ignore_index=True)
    tuned_df = tuned_df[
        ["group", "model", "feature_set", "fold"] + metric_names
    ].rename(columns={m: f"tuned_{m}" for m in metric_names})
    default_df = fold_df[
        ["group", "model", "feature_set", "fold"] + metric_names
    ].rename(columns={m: f"default_{m}" for m in metric_names})
    comparison = tuned_df.merge(
        default_df, on=["group", "model", "feature_set", "fold"], how="inner"
    )
    for m in metric_names:
        comparison[f"delta_{m}"] = comparison[f"tuned_{m}"] - comparison[f"default_{m}"]
    compare_summary = comparison.groupby(
        ["group", "model", "feature_set"], as_index=False
    ).agg(
        **{
            f"tuned_{m}_mean": (f"tuned_{m}", "mean")
            for m in metric_names
        },
        **{
            f"tuned_{m}_std": (f"tuned_{m}", "std")
            for m in metric_names
        },
        **{
            f"default_{m}_mean": (f"default_{m}", "mean")
            for m in metric_names
        },
        **{
            f"default_{m}_std": (f"default_{m}", "std")
            for m in metric_names
        },
        **{
            f"delta_{m}_mean": (f"delta_{m}", "mean")
            for m in metric_names
        },
        **{
            f"delta_{m}_std": (f"delta_{m}", "std")
            for m in metric_names
        },
    )
    compare_summary.to_csv(OUT_COMPARE, index=False, encoding="utf-8-sig")
    print(f"Wrote {OUT_FOLDS}")
    print(f"Wrote {OUT_SUMMARY}")
    print(f"Wrote {OUT_COMPARE}")


if __name__ == "__main__":
    main()
