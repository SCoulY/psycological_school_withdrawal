"""Paired participant-level comparison of full and nested top-10 models."""

from pathlib import Path
import argparse
import ast
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score, recall_score

RNG = np.random.default_rng(20260710)


def metric(y, p, name):
    yw = (y == 0).astype(int)
    pw = 1 - p
    pred = (p < 0.5).astype(int)
    if name == "withdrawal_auroc": return roc_auc_score(yw, pw)
    if name == "withdrawal_pr_auc": return average_precision_score(yw, pw)
    if name == "balanced_accuracy": return balanced_accuracy_score(yw, pred)
    if name == "withdrawal_sensitivity": return recall_score(yw, pred, zero_division=0)
    if name == "withdrawal_f1": return f1_score(yw, pred, zero_division=0)
    if name == "reentry_brier": return brier_score_loss(y, p)
    raise ValueError(name)


def bootstrap_delta(y, p_full, p_top, name, n_boot=2000):
    observed = metric(y, p_top, name) - metric(y, p_full, name)
    deltas = np.empty(n_boot)
    n = len(y)
    for b in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        deltas[b] = metric(y[idx], p_top[idx], name) - metric(y[idx], p_full[idx], name)
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    # Two-sided bootstrap sign probability; this is a descriptive paired test,
    # with the CI as the primary inferential quantity.
    pvalue = 2 * min(float(np.mean(deltas <= 0)), float(np.mean(deltas >= 0)))
    return observed, float(lo), float(hi), min(1.0, pvalue)


def selection_stability(base, group):
    s = pd.read_csv(base / group / "nested_feature_selection.csv")
    rows = []
    for model, d in s.groupby("model"):
        sets = [set(json.loads(x)) for x in d.selected_features]
        pairwise = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                pairwise.append(len(sets[i] & sets[j]) / max(1, len(sets[i] | sets[j])))
        counts = {}
        for st in sets:
            for f in st:
                counts[f] = counts.get(f, 0) + 1
        for f, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            rows.append({"group": group, "model": model, "feature": f, "selection_frequency": count / len(sets), "n_folds": len(sets), "mean_pairwise_jaccard": np.mean(pairwise)})
    return rows


def main(base: Path, fig: Path):
    fig.mkdir(parents=True, exist_ok=True)
    rows = []
    stability = []
    metrics = ["withdrawal_auroc", "withdrawal_pr_auc", "balanced_accuracy", "withdrawal_sensitivity", "withdrawal_f1", "reentry_brier"]
    for group in ["adults", "teens", "children"]:
        pred = pd.read_csv(base / group / "nested_oof_predictions.csv")
        pooled = pred.groupby(["group", "model", "feature_set", "row_index", "y"], as_index=False).p_reentry.mean()
        for model in ["LR", "RF", "SVM"]:
            wide = pooled[pooled.model == model].pivot(index=["row_index", "y"], columns="feature_set", values="p_reentry").dropna().reset_index()
            y = wide.y.to_numpy(int)
            for name in metrics:
                observed, lo, hi, pvalue = bootstrap_delta(y, wide.full.to_numpy(float), wide.top10.to_numpy(float), name)
                rows.append({"group": group, "model": model, "metric": name, "top10_minus_full": observed, "ci95_low": lo, "ci95_high": hi, "paired_bootstrap_p": pvalue, "n_participants": len(y), "n_boot": 2000})
        stability.extend(selection_stability(base, group))
    comp = pd.DataFrame(rows)
    comp.to_csv(base / "p0_paired_full_vs_top10.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(stability).to_csv(base / "p0_selection_stability.csv", index=False, encoding="utf-8-sig")

    # Compact forest plot for the primary discrimination metrics.
    plot = comp[comp.metric.isin(["withdrawal_auroc", "withdrawal_pr_auc"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for ax, metric_name in zip(axes, ["withdrawal_auroc", "withdrawal_pr_auc"]):
        d = plot[plot.metric == metric_name].copy()
        d["label"] = d.group + " " + d.model
        yloc = np.arange(len(d))
        ax.errorbar(d.top10_minus_full, yloc, xerr=[d.top10_minus_full - d.ci95_low, d.ci95_high - d.top10_minus_full], fmt="o", capsize=3)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_yticks(yloc)
        ax.set_yticklabels(d.label)
        ax.set_xlabel("Top-10 minus full (95% paired bootstrap CI)")
        ax.set_title(metric_name)
    plt.tight_layout()
    plt.savefig(fig / "paired_full_vs_top10_forest.png", dpi=220)
    plt.close()
    print(comp.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="reviewer_response_experiments/results/p0_nested_cv")
    ap.add_argument("--fig", default="reviewer_response_experiments/figures/p0_comparison")
    args = ap.parse_args()
    main(Path(args.base), Path(args.fig))
