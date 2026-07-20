"""Information-preserving aggregations of feature-level KDE scores.

The original analysis reduced a vector of feature anomalies to mean absolute
anomaly. This experiment keeps the full vector in each outer fold and compares
direction-aware density evidence, top-k evidence, importance-weighted evidence,
and a training-fold meta-logistic aggregator.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from reviewer_response_experiments.p1_kde_cohort import LABEL, select_reference_cohort
from reviewer_response_experiments.p1_kde_independent_validation import (
    FEATURE_DROP, fit_models, interval_score, kde_quantiles, quantile_scores,
)


OUT = Path("reviewer_response_experiments/results/p1_kde")
FIG = Path("reviewer_response_experiments/figures/p1_kde")
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def safe_auc(y, score):
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan


def safe_ap(y, score):
    return float(average_precision_score(y, score)) if np.sum(y) else np.nan


def density_matrix(X, high_ref, low_ref, features):
    """Log f_withdrawal(x) - log f_reentry(x), feature by feature."""
    out = np.zeros((len(X), len(features)), float)
    for j, feature in enumerate(features):
        h = pd.Series(high_ref[feature]).dropna().to_numpy(float)
        l = pd.Series(low_ref[feature]).dropna().to_numpy(float)
        values = X[feature].to_numpy(float)
        if len(h) >= 5 and np.ptp(h) > 0:
            kh = gaussian_kde(h)
            fh = kh(values)
        else:
            # Degenerate folds use a narrow Gaussian around the empirical
            # mean, preserving a finite density ratio.
            fh = np.exp(-0.5 * ((values - np.mean(h)) / max(np.std(h), 1e-3)) ** 2)
        if len(l) >= 5 and np.ptp(l) > 0:
            kl = gaussian_kde(l)
            fl = kl(values)
        else:
            fl = np.exp(-0.5 * ((values - np.mean(l)) / max(np.std(l), 1e-3)) ** 2)
        out[:, j] = np.clip(np.log(fh + 1e-12) - np.log(fl + 1e-12), -12, 12)
    return out


def aggregate_interval(A, weights):
    """Return information-preserving aggregations of interval scores."""
    abs_a = np.abs(A)
    n_features = A.shape[1]
    out = {
        "kde_abs_mean": abs_a.mean(axis=1),
        "kde_abs_median": np.median(abs_a, axis=1),
        "kde_abs_max": abs_a.max(axis=1),
        "kde_abs_top3_mean": np.sort(abs_a, axis=1)[:, -min(3, n_features):].mean(axis=1),
        "kde_abs_top5_mean": np.sort(abs_a, axis=1)[:, -min(5, n_features):].mean(axis=1),
        "kde_abs_top10_mean": np.sort(abs_a, axis=1)[:, -min(10, n_features):].mean(axis=1),
        "kde_signed_mean": A.mean(axis=1),
        "kde_signed_top5_net": A[np.arange(len(A))[:, None], np.argsort(abs_a, axis=1)[:, -min(5, n_features):]].mean(axis=1),
        "kde_weighted_signed": (A * weights[None, :]).sum(axis=1) / max(weights.sum(), 1e-12),
        "kde_weighted_abs": (abs_a * weights[None, :]).sum(axis=1) / max(weights.sum(), 1e-12),
    }
    return out


def aggregate_logratio(R, weights):
    abs_r = np.abs(R)
    n_features = R.shape[1]
    top_idx = np.argsort(abs_r, axis=1)[:, -min(5, n_features):]
    top_values = np.take_along_axis(R, top_idx, axis=1)
    return {
        "kde_logratio_mean": R.mean(axis=1),
        "kde_logratio_sum": R.sum(axis=1),
        "kde_logratio_top5_net": top_values.mean(axis=1),
        "kde_weighted_logratio": (R * weights[None, :]).sum(axis=1) / max(weights.sum(), 1e-12),
    }


def fit_meta(X_train, y_train, X_test):
    """Fit a supervised aggregator only on the outer training fold."""
    scaler = StandardScaler()
    tr = scaler.fit_transform(X_train)
    te = scaler.transform(X_test)
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=42)
    model.fit(tr, y_train)
    return model.predict_proba(te)[:, 1]


def main():
    fold_rows = []
    score_rows = []
    feature_rows = []
    for group in ["adults", "teens", "children"]:
        df = pd.read_csv(Path("data") / f"clean_{group}.csv")
        X = df.drop(columns=[LABEL])
        y = df[LABEL].astype(int).to_numpy()
        features = [c for c in X.columns if c not in FEATURE_DROP]
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=20260710)
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
            ytr, yte = y[train_idx], y[test_idx]
            lr, rf, lr_unwrapped = fit_models(Xtr, ytr)
            p_lr_tr, p_rf_tr = lr.predict_proba(Xtr)[:, 1], rf.predict_proba(Xtr)[:, 1]
            p_lr_te, p_rf_te = lr.predict_proba(Xte)[:, 1], rf.predict_proba(Xte)[:, 1]
            train_aug = Xtr.copy()
            train_aug[LABEL] = ytr
            train_aug["LogisticRegression"] = p_lr_tr
            train_aug["RandomForest"] = p_rf_tr
            high_ref, high_meta = select_reference_cohort(train_aug, high=True)
            low_ref, low_meta = select_reference_cohort(train_aug, high=False)
            high_ref = high_ref.drop(columns=[LABEL, "LogisticRegression", "RandomForest"], errors="ignore")[features]
            low_ref = low_ref.drop(columns=[LABEL, "LogisticRegression", "RandomForest"], errors="ignore")[features]
            Xtr_f, Xte_f = Xtr[features], Xte[features]
            interval_tr = quantile_scores(Xtr_f, high_ref, low_ref, features, kde_quantiles)
            interval_te = quantile_scores(Xte_f, high_ref, low_ref, features, kde_quantiles)
            ratio_tr = density_matrix(Xtr_f, high_ref, low_ref, features)
            ratio_te = density_matrix(Xte_f, high_ref, low_ref, features)
            # Absolute LR coefficient weights are fitted on the training fold.
            lr_base = lr_unwrapped.fit(Xtr, ytr)
            coef = np.abs(lr_base.named_steps["clf"].coef_[0])
            coef = coef[[Xtr.columns.get_loc(f) for f in features]]
            weights = coef / max(coef.mean(), 1e-12)
            aggregations = {}
            aggregations.update(aggregate_interval(interval_te, weights))
            aggregations.update(aggregate_logratio(ratio_te, weights))
            # Meta aggregators use the complete feature-level vector and are
            # fitted on training-fold representations only.
            aggregations["kde_meta_lr_signed"] = fit_meta(interval_tr, (ytr == 0).astype(int), interval_te)
            aggregations["kde_meta_lr_logratio"] = fit_meta(ratio_tr, (ytr == 0).astype(int), ratio_te)
            y_withdraw = (yte == 0).astype(int)
            p_ensemble = (p_lr_te + p_rf_te) / 2
            errors = ((p_ensemble < 0.5).astype(int) != yte).astype(int)
            for name, score in aggregations.items():
                score = np.asarray(score, float)
                fold_rows.append({
                    "group": group, "fold": fold, "method": name,
                    "withdrawal_auroc": safe_auc(y_withdraw, score),
                    "withdrawal_pr_auc": safe_ap(y_withdraw, score),
                    "error_auroc": safe_auc(errors, score), "error_pr_auc": safe_ap(errors, score),
                    "n_test": len(yte), "withdrawal_test_n": int(y_withdraw.sum()), "error_n": int(errors.sum()),
                    "high_stage": high_meta["stage"], "low_stage": low_meta["stage"],
                })
                for i, row_idx in enumerate(test_idx):
                    score_rows.append({"group": group, "fold": fold, "row_index": int(row_idx), "y": int(yte[i]), "error": int(errors[i]), "method": name, "score": float(score[i])})
            for j, feature in enumerate(features):
                for i, row_idx in enumerate(test_idx):
                    feature_rows.append({"group": group, "fold": fold, "row_index": int(row_idx), "y": int(yte[i]), "feature": feature, "interval_signed": float(interval_te[i, j]), "interval_abs": float(abs(interval_te[i, j])), "logratio": float(ratio_te[i, j])})
    folds = pd.DataFrame(fold_rows)
    scores = pd.DataFrame(score_rows)
    features = pd.DataFrame(feature_rows)
    folds.to_csv(OUT / "p1_kde_aggregation_fold_metrics.csv", index=False, encoding="utf-8-sig")
    scores.to_csv(OUT / "p1_kde_aggregation_scores.csv", index=False, encoding="utf-8-sig")
    features.to_csv(OUT / "p1_kde_feature_level_scores.csv", index=False, encoding="utf-8-sig")
    # Pool the two repeated test predictions to one participant for outcome
    # discrimination. Error can differ between repeats, so retain its mean
    # rate and use a majority-error indicator for the participant-level error
    # analysis instead of silently duplicating participants.
    pooled = scores.groupby(["group", "method", "row_index", "y"], as_index=False).agg(
        score=("score", "mean"), error_rate=("error", "mean")
    )
    summaries = []
    for (group, method), d in pooled.groupby(["group", "method"]):
        yw = (d.y.to_numpy() == 0).astype(int)
        score = d.score.to_numpy(float)
        error_label = (d.error_rate.to_numpy(float) >= 0.5).astype(int)
        summaries.append({"group": group, "method": method, "n": len(d), "withdrawal_auroc": safe_auc(yw, score), "withdrawal_pr_auc": safe_ap(yw, score), "error_auroc": safe_auc(error_label, score), "error_pr_auc": safe_ap(error_label, score), "mean_error_rate": float(d.error_rate.mean())})
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "p1_kde_aggregation_summary.csv", index=False, encoding="utf-8-sig")
    # Plot outcome AUROC to reveal whether information-preserving aggregation
    # changes the conclusion relative to mean absolute anomaly.
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    for ax, group in zip(axes, ["adults", "teens", "children"]):
        d = summary[summary.group == group].sort_values("withdrawal_auroc")
        ax.barh(d.method, d.withdrawal_auroc, color=["#4c72b0" if x != "kde_abs_mean" else "#dd8452" for x in d.method])
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_title(group)
        ax.set_xlabel("Withdrawal AUROC")
    plt.tight_layout()
    plt.savefig(FIG / "kde_aggregation_auroc.png", dpi=220)
    plt.close()
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
