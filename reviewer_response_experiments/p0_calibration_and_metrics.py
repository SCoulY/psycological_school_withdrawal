"""Summarise P0 OOF predictions and create calibration figures."""

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score,
)


def calibration_logistic(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    y = np.asarray(y, float)

    def objective(beta):
        eta = beta[0] + beta[1] * z
        # stable Bernoulli negative log likelihood
        return np.sum(np.logaddexp(0, eta) - y * eta)

    result = minimize(objective, x0=np.array([0.0, 1.0]), method="BFGS")
    return float(result.x[0]), float(result.x[1]), bool(result.success)


def metrics(y, p):
    # Main probability is p(reentry); withdrawal is class 0 and is the
    # screening target, so all discrimination metrics use 1-p(reentry).
    yw = (y == 0).astype(int)
    pw = 1 - p
    pred = (p < 0.5).astype(int)  # withdrawal prediction
    return {
        "n": int(len(y)),
        "withdrawal_n": int(yw.sum()),
        "withdrawal_prevalence": float(yw.mean()),
        "majority_accuracy_baseline": float(max(y.mean(), 1 - y.mean())),
        "accuracy": float(accuracy_score(yw, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(yw, pred)),
        "withdrawal_sensitivity": float(recall_score(yw, pred, zero_division=0)),
        "withdrawal_specificity": float(recall_score(1 - yw, 1 - pred, zero_division=0)),
        "withdrawal_precision": float(precision_score(yw, pred, zero_division=0)),
        "withdrawal_f1": float(f1_score(yw, pred, zero_division=0)),
        "withdrawal_auroc": float(roc_auc_score(yw, pw)),
        "withdrawal_pr_auc": float(average_precision_score(yw, pw)),
        "reentry_brier": float(brier_score_loss(y, p)),
    }


def main(base: Path, fig: Path):
    fig.mkdir(parents=True, exist_ok=True)
    all_rows = []
    # Keep the same pooled predictions available for a single, three-panel
    # reliability figure used in the response letter.
    pooled_by_group = {}
    for group in ["adults", "teens", "children"]:
        pred = pd.read_csv(base / group / "nested_oof_predictions.csv")
        # Average the ten independent test predictions per participant. This
        # yields one repeated-CV ensemble probability per participant/config.
        pooled = (
            pred.groupby(["group", "model", "feature_set", "row_index", "y"], as_index=False)
            .p_reentry.mean()
        )
        pooled_by_group[group] = pooled
        for (g, model, feature_set), d in pooled.groupby(["group", "model", "feature_set"]):
            y = d.y.to_numpy(int)
            p = d.p_reentry.to_numpy(float)
            row = {"group": g, "model": model, "feature_set": feature_set}
            row.update(metrics(y, p))
            intercept, slope, ok = calibration_logistic(y, p)
            row.update({"calibration_intercept": intercept, "calibration_slope": slope, "calibration_fit_success": ok})
            # Quantile-bin calibration error, with 10 bins where possible.
            frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
            row["calibration_abs_error_mean"] = float(np.mean(np.abs(frac - mean_pred)))
            row["calibration_abs_error_max"] = float(np.max(np.abs(frac - mean_pred)))
            all_rows.append(row)

            frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
            plt.plot(mean_pred, frac, marker="o", linewidth=1.5, label=f"{model} {feature_set}")
        plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
        plt.xlabel("Mean predicted probability of reentry")
        plt.ylabel("Observed reentry frequency")
        plt.title(f"Repeated nested-CV calibration: {group}")
        plt.legend(fontsize=8, ncol=2)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(fig / f"{group}_calibration.png", dpi=220)
        plt.close()

    # Combine the three cohort-specific reliability plots into one figure.
    # The panels share the probability axes, while one common legend avoids
    # repeating the six model/feature-set labels three times.
    fig_combined, axes = plt.subplots(1, 3, figsize=(15.0, 5.1), sharex=True, sharey=True)
    palette = {
        ("LR", "full"): "#1f77b4",
        ("LR", "top10"): "#ff7f0e",
        ("RF", "full"): "#2ca02c",
        ("RF", "top10"): "#d62728",
        ("SVM", "full"): "#9467bd",
        ("SVM", "top10"): "#8c564b",
    }
    for ax, group in zip(axes, ["adults", "teens", "children"]):
        pooled = pooled_by_group[group]
        for (model, feature_set), d in pooled.groupby(["model", "feature_set"]):
            y = d.y.to_numpy(int)
            p = d.p_reentry.to_numpy(float)
            frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
            label = f"{model} {feature_set}"
            ax.plot(
                mean_pred,
                frac,
                marker="o",
                linewidth=1.5,
                markersize=4,
                color=palette[(model, feature_set)],
                label=label,
            )
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration")
        ax.set_title(group.title())
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability of reentry")
        ax.grid(alpha=0.15, linewidth=0.5)
    axes[0].set_ylabel("Observed reentry frequency")
    handles, labels = axes[0].get_legend_handles_labels()
    fig_combined.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=8, frameon=True)
    fig_combined.tight_layout(rect=[0, 0, 1, 0.94])
    fig_combined.savefig(fig / "calibration_combined.png", dpi=240, bbox_inches="tight")
    plt.close(fig_combined)

    out = pd.DataFrame(all_rows)
    out.to_csv(base / "p0_pooled_oof_metrics_calibration.csv", index=False, encoding="utf-8-sig")
    with open(base / "p0_calibration_metadata.json", "w", encoding="utf-8") as f:
        json.dump({"probability_definition": "p_reentry = predict_proba[:, 1]", "withdrawal_score": "1-p_reentry", "pooling": "mean of 10 repeated-CV test predictions per participant/config", "bins": "10 quantile bins"}, f, indent=2)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="reviewer_response_experiments/results/p0_nested_cv")
    ap.add_argument("--fig", default="reviewer_response_experiments/figures/p0_calibration")
    args = ap.parse_args()
    main(Path(args.base), Path(args.fig))
