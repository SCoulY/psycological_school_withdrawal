"""Independent outer-fold validation of KDE anomaly scores.

Reference cohorts are selected from each training fold using training-fold
predictions only. Test-fold anomaly scores are then evaluated against labels
that were not used to fit KDEs. The script also reports empirical-quantile,
robust-z, regularised Mahalanobis, and LR additive attribution baselines.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

from reviewer_response_experiments.p1_kde_cohort import LABEL, select_reference_cohort


OUT = Path("reviewer_response_experiments/results/p1_kde")
FIG = Path("reviewer_response_experiments/figures/p1_kde")
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
FEATURE_DROP = {"Age", "Gender"}


def kde_quantiles(x, q=(0.05, 0.95), bandwidth=None):
    x = pd.Series(x).dropna().to_numpy(float)
    if len(x) < 5 or np.ptp(x) == 0:
        return np.quantile(x, q)
    kde = gaussian_kde(x, bw_method=bandwidth)
    pad = max(0.1 * np.ptp(x), 1e-6)
    grid = np.linspace(x.min() - pad, x.max() + pad, 512)
    density = kde(grid)
    cdf = np.concatenate([[0.0], np.cumsum((density[1:] + density[:-1]) * np.diff(grid) / 2)])
    cdf = cdf / cdf[-1]
    return np.interp(q, cdf, grid)


def interval_score(value, q_high, q_low):
    l5, l95 = float(q_low[0]), float(q_low[1])
    h5, h95 = float(q_high[0]), float(q_high[1])
    low_bound, high_bound = min(l5, h5), max(l95, h95)
    overlap_low, overlap_high = max(l5, h5), min(l95, h95)
    union_scale = max(abs(high_bound - low_bound), abs(l95 - l5), abs(h95 - h5), 1e-6)
    if overlap_low > overlap_high:
        # Non-overlapping reference intervals have no shared ``normal``
        # region. Use the gap midpoint as zero and a data-derived scale;
        # never divide by a zero-width boundary.
        if h95 < l5:
            center = (h95 + l5) / 2
        else:
            center = (l95 + h5) / 2
        return float(np.clip((value - center) / union_scale, -10.0, 10.0))
    if overlap_low <= value <= overlap_high:
        half = max((overlap_high - overlap_low) / 2, 1e-12)
        return 0.5 * (value - (overlap_low + overlap_high) / 2) / half
    if value < overlap_low:
        scale = max(overlap_low - low_bound, 0.1 * union_scale, 1e-6)
        score = -0.5 - 0.5 * (overlap_low - value) / scale if value >= low_bound else -1.0 - (low_bound - value) / scale
        return float(np.clip(score, -10.0, 10.0))
    scale = max(high_bound - overlap_high, 0.1 * union_scale, 1e-6)
    score = 0.5 + 0.5 * (value - overlap_high) / scale if value <= high_bound else 1.0 + (value - high_bound) / scale
    return float(np.clip(score, -10.0, 10.0))


def quantile_scores(test, high_ref, low_ref, features, estimator):
    out = np.zeros((len(test), len(features)), float)
    for j, feature in enumerate(features):
        qh = estimator(high_ref[feature])
        ql = estimator(low_ref[feature])
        out[:, j] = [interval_score(v, qh, ql) for v in test[feature].to_numpy(float)]
    return out


def safe_auc(y, score):
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan


def safe_ap(y, score):
    return float(average_precision_score(y, score)) if np.sum(y) > 0 else np.nan


def fit_models(X_train, y_train):
    lr = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", random_state=42))])
    rf = RandomForestClassifier(n_estimators=150, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=1)
    # Calibrate inside training data; no test observation enters calibration.
    lr_c = CalibratedClassifierCV(lr, method="sigmoid", cv=3, n_jobs=-1)
    rf_c = CalibratedClassifierCV(rf, method="sigmoid", cv=3, n_jobs=-1)
    lr_c.fit(X_train, y_train)
    rf_c.fit(X_train, y_train)
    return lr_c, rf_c, lr


def main():
    rows = []
    fold_rows = []
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
            # Remove model/label columns from the distribution reference.
            high_ref = high_ref.drop(columns=[LABEL, "LogisticRegression", "RandomForest"], errors="ignore")
            low_ref = low_ref.drop(columns=[LABEL, "LogisticRegression", "RandomForest"], errors="ignore")
            te_features = Xte[features].copy()
            high_ref = high_ref[features]
            low_ref = low_ref[features]
            kde = quantile_scores(te_features, high_ref, low_ref, features, kde_quantiles)
            empirical = quantile_scores(te_features, high_ref, low_ref, features, lambda x: np.quantile(x.dropna(), [0.05, 0.95]))
            kde_abs = np.abs(kde)
            empirical_abs = np.abs(empirical)
            # Reentry reference robust-z and regularised Mahalanobis baselines.
            low_arr = low_ref.to_numpy(float)
            te_arr = te_features.to_numpy(float)
            med = np.nanmedian(low_arr, axis=0)
            mad = np.nanmedian(np.abs(low_arr - med), axis=0)
            q25, q75 = np.nanpercentile(low_arr, [25, 75], axis=0)
            robust_scale = np.maximum.reduce([1.4826 * mad, 0.1 * (q75 - q25), np.full_like(mad, 1e-3)])
            robust_z = np.nanmean(np.abs((te_arr - med) / robust_scale), axis=1)
            center = np.nanmean(low_arr, axis=0)
            cov = np.cov(np.nan_to_num(low_arr, nan=center), rowvar=False) if len(low_arr) > 2 else np.eye(len(features))
            cov = np.atleast_2d(cov) + 0.1 * np.eye(len(features))
            inv = np.linalg.pinv(cov)
            diff = np.nan_to_num(te_arr - center, nan=0.0)
            mahal = np.sqrt(np.einsum("ij,jk,ik->i", diff, inv, diff))
            # Analytical linear-SHAP magnitude: for a standardized LR model,
            # coefficient * (x - training mean) is the exact additive SHAP
            # value for the logit output under a mean background.
            lr_base = lr_unwrapped.fit(Xtr, ytr)
            scaled_te = lr_base.named_steps["scaler"].transform(Xte)
            coef = np.abs(lr_base.named_steps["clf"].coef_[0])
            lr_attr = np.mean(np.abs(scaled_te * coef), axis=1)
            p_ensemble = (p_lr_te + p_rf_te) / 2
            y_withdraw = (yte == 0).astype(int)
            error = ((p_ensemble < 0.5).astype(int) != yte).astype(int)
            score_dict = {
                "kde_abs_mean": np.nanmean(kde_abs, axis=1),
                "kde_abs_max": np.nanmax(kde_abs, axis=1),
                "empirical_abs_mean": np.nanmean(empirical_abs, axis=1),
                "robust_z_mean": robust_z,
                "mahalanobis_reentry": mahal,
                "lr_shap_linear_abs_mean": lr_attr,
            }
            train_kde = np.abs(quantile_scores(Xtr[features], high_ref, low_ref, features, kde_quantiles)).mean(axis=1)
            train_empirical = np.abs(quantile_scores(Xtr[features], high_ref, low_ref, features, lambda x: np.quantile(x.dropna(), [0.05, 0.95]))).mean(axis=1)
            train_robust_z = np.nanmean(np.abs((np.nan_to_num(Xtr[features].to_numpy(float), nan=med) - med) / robust_scale), axis=1)
            train_diff = np.nan_to_num(Xtr[features].to_numpy(float) - center, nan=0.0)
            train_mahal = np.sqrt(np.einsum("ij,jk,ik->i", train_diff, inv, train_diff))
            scaled_tr = lr_base.named_steps["scaler"].transform(Xtr)
            train_lr_attr = np.mean(np.abs(scaled_tr * coef), axis=1)
            train_score_dict = {
                "kde_abs_mean": train_kde, "kde_abs_max": np.abs(quantile_scores(Xtr[features], high_ref, low_ref, features, kde_quantiles)).max(axis=1),
                "empirical_abs_mean": train_empirical, "robust_z_mean": train_robust_z,
                "mahalanobis_reentry": train_mahal, "lr_shap_linear_abs_mean": train_lr_attr,
            }
            kde_feature_mean = np.nanmean(np.abs(kde), axis=0)
            empirical_feature_mean = np.nanmean(np.abs(empirical), axis=0)
            rank_corr = float(spearmanr(kde_feature_mean, empirical_feature_mean).statistic)
            for i, row_idx in enumerate(test_idx):
                row = {"group": group, "fold": fold, "row_index": int(row_idx), "y": int(yte[i]), "p_reentry": float(p_ensemble[i]), "error": int(error[i])}
                row.update({k: float(v[i]) for k, v in score_dict.items()})
                rows.append(row)
            for name, score in score_dict.items():
                cal = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
                tr_score = np.asarray(train_score_dict[name], float)
                if np.ptp(tr_score) > 1e-8 and len(np.unique(ytr)) == 2:
                    cal.fit(tr_score.reshape(-1, 1), (ytr == 0).astype(int))
                    cal_prob = cal.predict_proba(np.asarray(score).reshape(-1, 1))[:, 1]
                    anomaly_brier = float(brier_score_loss((yte == 0).astype(int), cal_prob))
                else:
                    anomaly_brier = np.nan
                fold_rows.append({
                    "group": group, "fold": fold, "score": name,
                    "withdrawal_auroc": safe_auc(y_withdraw, score),
                    "withdrawal_pr_auc": safe_ap(y_withdraw, score),
                    "error_auroc": safe_auc(error, score),
                    "error_pr_auc": safe_ap(error, score),
                    "withdrawal_brier": anomaly_brier,
                    "kde_empirical_rank_spearman": rank_corr,
                    "n_test": len(yte), "withdrawal_test_n": int(y_withdraw.sum()), "error_n": int(error.sum()),
                    "high_stage": high_meta["stage"], "low_stage": low_meta["stage"], "high_n": high_meta["n_selected"], "low_n": low_meta["n_selected"],
                })
    scores = pd.DataFrame(rows)
    folds = pd.DataFrame(fold_rows)
    scores.to_csv(OUT / "p1_independent_oof_scores.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(OUT / "p1_independent_fold_metrics.csv", index=False, encoding="utf-8-sig")
    summaries = []
    score_names = ["kde_abs_mean", "kde_abs_max", "empirical_abs_mean", "robust_z_mean", "mahalanobis_reentry", "lr_shap_linear_abs_mean"]
    for (group,), d in scores.groupby(["group"]):
        y = (d.y.to_numpy() == 0).astype(int)
        error = d.error.to_numpy(int)
        for score in score_names:
            s = d[score].to_numpy(float)
            summaries.append({"group": group, "score": score, "n_rows": len(d), "n_unique_participants": d.row_index.nunique(), "withdrawal_auroc": safe_auc(y, s), "withdrawal_pr_auc": safe_ap(y, s), "error_auroc": safe_auc(error, s), "error_pr_auc": safe_ap(error, s), "mean_score": float(np.nanmean(s)), "score_sd": float(np.nanstd(s))})
    summary = pd.DataFrame(summaries)
    fold_summary = folds.groupby(["group", "score"], as_index=False).agg({"withdrawal_brier": "mean", "kde_empirical_rank_spearman": "mean"})
    summary = summary.merge(fold_summary, on=["group", "score"], how="left")
    summary.to_csv(OUT / "p1_independent_summary.csv", index=False, encoding="utf-8-sig")
    plt.figure(figsize=(11, 5))
    primary = summary[summary.score.isin(["kde_abs_mean", "empirical_abs_mean", "robust_z_mean", "mahalanobis_reentry", "lr_shap_linear_abs_mean"])]
    for score, d in primary.groupby("score"):
        plt.plot(d.group, d.withdrawal_auroc, marker="o", label=score)
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
    plt.ylabel("Withdrawal AUROC (independent outer folds)")
    plt.title("KDE anomaly and normative/model-centric baselines")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "independent_baseline_auroc.png", dpi=220)
    plt.close()
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
