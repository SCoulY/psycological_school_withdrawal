"""Participant-level uncertainty intervals for the revised performance table."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import average_precision_score, roc_auc_score


BASE = Path("reviewer_response_experiments/results/p0_nested_cv_scaled_selector")
OUT = BASE / "p2_pooled_performance_intervals.csv"


def wilson(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    z = norm.ppf(1 - alpha / 2)
    p = successes / total
    den = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / den
    half = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / den
    return float(centre - half), float(centre + half)


def qci(values: np.ndarray) -> tuple[float, float]:
    return tuple(float(x) for x in np.quantile(values, [0.025, 0.975]))


def one_configuration(y_reentry: np.ndarray, p_reentry: np.ndarray, seed: int, n_boot: int) -> dict:
    y_w = (y_reentry == 0).astype(int)
    p_w = 1 - p_reentry
    pred_w = (p_w >= 0.5).astype(int)
    tp = int(np.sum((pred_w == 1) & (y_w == 1)))
    fn = int(np.sum((pred_w == 0) & (y_w == 1)))
    tn = int(np.sum((pred_w == 0) & (y_w == 0)))
    fp = int(np.sum((pred_w == 1) & (y_w == 0)))
    accuracy_ci = wilson(tp + tn, len(y_w))
    sensitivity_ci = wilson(tp, tp + fn)
    specificity_ci = wilson(tn, tn + fp)
    precision_ci = wilson(tp, tp + fp)

    rng = np.random.default_rng(seed)
    auc, ap, ba, f1, brier = [], [], [], [], []
    n = len(y_w)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy, pp, pred = y_w[idx], p_w[idx], pred_w[idx]
        if np.unique(yy).size == 2:
            auc.append(roc_auc_score(yy, pp))
            ap.append(average_precision_score(yy, pp))
        btp = np.sum((pred == 1) & (yy == 1))
        bfn = np.sum((pred == 0) & (yy == 1))
        btn = np.sum((pred == 0) & (yy == 0))
        bfp = np.sum((pred == 1) & (yy == 0))
        sens = btp / (btp + bfn) if btp + bfn else np.nan
        spec = btn / (btn + bfp) if btn + bfp else np.nan
        ba.append((sens + spec) / 2)
        f1.append(2 * btp / (2 * btp + bfp + bfn) if 2 * btp + bfp + bfn else np.nan)
        brier.append(np.mean((p_reentry[idx] - y_reentry[idx]) ** 2))

    return {
        "accuracy_ci95_low": accuracy_ci[0], "accuracy_ci95_high": accuracy_ci[1],
        "withdrawal_sensitivity_ci95_low": sensitivity_ci[0], "withdrawal_sensitivity_ci95_high": sensitivity_ci[1],
        "withdrawal_specificity_ci95_low": specificity_ci[0], "withdrawal_specificity_ci95_high": specificity_ci[1],
        "withdrawal_precision_ci95_low": precision_ci[0], "withdrawal_precision_ci95_high": precision_ci[1],
        "withdrawal_f1_ci95_low": qci(np.asarray(f1))[0], "withdrawal_f1_ci95_high": qci(np.asarray(f1))[1],
        "balanced_accuracy_ci95_low": qci(np.asarray(ba))[0], "balanced_accuracy_ci95_high": qci(np.asarray(ba))[1],
        "withdrawal_auroc_ci95_low": qci(np.asarray(auc))[0], "withdrawal_auroc_ci95_high": qci(np.asarray(auc))[1],
        "withdrawal_pr_auc_ci95_low": qci(np.asarray(ap))[0], "withdrawal_pr_auc_ci95_high": qci(np.asarray(ap))[1],
        "reentry_brier_ci95_low": qci(np.asarray(brier))[0], "reentry_brier_ci95_high": qci(np.asarray(brier))[1],
        "n_boot": n_boot,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="+", choices=["adults", "teens", "children"], default=["adults", "teens", "children"])
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()
    point = pd.read_csv(BASE / "p0_pooled_oof_metrics_calibration.csv")
    prior = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
    new_rows = []
    for group in args.groups:
        pred = pd.read_csv(BASE / group / "nested_oof_predictions.csv")
        pooled = pred.groupby(["group", "model", "feature_set", "row_index", "y"], as_index=False).p_reentry.mean()
        for i, ((model, feature_set), d) in enumerate(pooled.groupby(["model", "feature_set"], sort=True)):
            p_row = point.loc[point.group.eq(group) & point.model.eq(model) & point.feature_set.eq(feature_set)].iloc[0].to_dict()
            p_row.update(one_configuration(d.y.to_numpy(int), d.p_reentry.to_numpy(float), 20260710 + i + 100 * ["adults", "teens", "children"].index(group), args.n_boot))
            new_rows.append(p_row)
    new = pd.DataFrame(new_rows)
    if not prior.empty:
        prior = prior.loc[~prior.group.isin(args.groups)]
        new = pd.concat([prior, new], ignore_index=True)
    new.sort_values(["group", "model", "feature_set"]).to_csv(OUT, index=False, encoding="utf-8-sig")
    print(new.groupby("group").size().to_string())


if __name__ == "__main__":
    main()
