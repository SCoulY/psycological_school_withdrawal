"""Sensitivity of KDE anomaly discrimination to thresholds, n, and bandwidth."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from reviewer_response_experiments.p1_kde_cohort import LABEL, select_reference_cohort
from reviewer_response_experiments.p1_kde_independent_validation import (
    FEATURE_DROP, fit_models, kde_quantiles, quantile_scores,
)


OUT = Path("reviewer_response_experiments/results/p1_kde")
FIG = Path("reviewer_response_experiments/figures/p1_kde")


def auc(y, score):
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan


def ap(y, score):
    return float(average_precision_score(y, score)) if np.sum(y) else np.nan


def main():
    rows = []
    thresholds = [(0.2, 0.8), (0.3, 0.7), (0.4, 0.6)]
    min_samples = [10, 20]
    bandwidths = [("scott", None), ("silverman", "silverman"), ("bw05", 0.5)]
    for group in ["adults", "teens", "children"]:
        df = pd.read_csv(Path("data") / f"clean_{group}.csv")
        X, y = df.drop(columns=[LABEL]), df[LABEL].astype(int).to_numpy()
        features = [c for c in X.columns if c not in FEATURE_DROP]
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=20260710)
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
            ytr, yte = y[train_idx], y[test_idx]
            lr, rf, _ = fit_models(Xtr, ytr)
            p_lr_tr, p_rf_tr = lr.predict_proba(Xtr)[:, 1], rf.predict_proba(Xtr)[:, 1]
            p_lr_te, p_rf_te = lr.predict_proba(Xte)[:, 1], rf.predict_proba(Xte)[:, 1]
            train_aug = Xtr.copy()
            train_aug[LABEL] = ytr
            train_aug["LogisticRegression"] = p_lr_tr
            train_aug["RandomForest"] = p_rf_tr
            test_features = Xte[features]
            for low_thr, high_thr in thresholds:
                for min_n in min_samples:
                    high_ref, hm = select_reference_cohort(train_aug, high=True, low_threshold=low_thr, high_threshold=high_thr, min_samples=min_n)
                    low_ref, lm = select_reference_cohort(train_aug, high=False, low_threshold=low_thr, high_threshold=high_thr, min_samples=min_n)
                    high_ref = high_ref[features]
                    low_ref = low_ref[features]
                    for bw_label, bw in bandwidths:
                        estimator = lambda x, bw=bw: kde_quantiles(x, bandwidth=bw)
                        score = np.abs(quantile_scores(test_features, high_ref, low_ref, features, estimator)).mean(axis=1)
                        y_withdraw = (yte == 0).astype(int)
                        p_ensemble = (p_lr_te + p_rf_te) / 2
                        error = ((p_ensemble < 0.5).astype(int) != yte).astype(int)
                        rows.append({
                            "group": group, "fold": fold, "low_threshold": low_thr, "high_threshold": high_thr,
                            "min_samples": min_n, "bandwidth": bw_label,
                            "withdrawal_auroc": auc(y_withdraw, score), "withdrawal_pr_auc": ap(y_withdraw, score),
                            "error_auroc": auc(error, score), "error_pr_auc": ap(error, score),
                            "high_stage": hm["stage"], "low_stage": lm["stage"], "high_n": hm["n_selected"], "low_n": lm["n_selected"],
                        })
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "p1_sensitivity_fold_results.csv", index=False, encoding="utf-8-sig")
    summary = d.groupby(["group", "low_threshold", "high_threshold", "min_samples", "bandwidth"], as_index=False).agg({
        "withdrawal_auroc": ["mean", "std"], "withdrawal_pr_auc": ["mean", "std"],
        "error_auroc": ["mean", "std"], "error_pr_auc": ["mean", "std"],
        "high_n": "mean", "low_n": "mean",
    })
    summary.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in summary.columns]
    summary.to_csv(OUT / "p1_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    # Plot withdrawal AUROC by threshold and bandwidth, faceted by group.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, group in zip(axes, ["adults", "teens", "children"]):
        q = summary[summary.group == group]
        for bw in ["scott", "silverman", "bw05"]:
            z = q[q.bandwidth == bw].groupby("low_threshold").withdrawal_auroc_mean.mean()
            ax.plot(z.index, z.values, marker="o", label=bw)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_title(group)
        ax.set_xlabel("Withdrawal threshold")
    axes[0].set_ylabel("Mean withdrawal AUROC")
    axes[-1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "kde_sensitivity_auroc.png", dpi=220)
    plt.close()
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
