"""Summarise feature-level KDE evidence rather than collapsing it prematurely."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

OUT = Path("reviewer_response_experiments/results/p1_kde")
sys.stdout.reconfigure(encoding="utf-8")


def auc(y, score):
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan


def ap(y, score):
    return float(average_precision_score(y, score)) if np.sum(y) else np.nan


def main():
    raw = pd.read_csv(OUT / "p1_kde_feature_level_scores.csv")
    raw["abs_rank"] = raw.groupby(["group", "fold", "row_index"]).interval_abs.rank(method="first", ascending=False)
    raw["top10"] = raw.abs_rank <= 10
    # Average repeated outer-test scores to one participant/feature.
    pooled = raw.groupby(["group", "row_index", "y", "feature"], as_index=False).agg(
        interval_signed=("interval_signed", "mean"), interval_abs=("interval_abs", "mean"),
        logratio=("logratio", "mean"), top10_frequency=("top10", "mean")
    )
    rows = []
    for (group, feature), d in pooled.groupby(["group", "feature"]):
        y = (d.y.to_numpy() == 0).astype(int)
        rows.append({
            "group": group, "feature": feature, "n_participants": d.row_index.nunique(),
            "mean_signed_interval": float(d.interval_signed.mean()),
            "mean_abs_interval": float(d.interval_abs.mean()),
            "mean_logratio": float(d.logratio.mean()),
            "withdrawal_direction_consistency": float((d.logratio > 0).mean()),
            "top10_frequency": float(d.top10_frequency.mean()),
            "logratio_auroc": auc(y, d.logratio.to_numpy(float)),
            "logratio_pr_auc": ap(y, d.logratio.to_numpy(float)),
            "abs_interval_auroc": auc(y, d.interval_abs.to_numpy(float)),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "p1_kde_feature_level_summary.csv", index=False, encoding="utf-8-sig")
    # Top ten features by directional evidence per group.
    top = out.sort_values(["group", "logratio_auroc"], ascending=[True, False]).groupby("group", as_index=False).head(10)
    top.to_csv(OUT / "p1_kde_top_directional_features.csv", index=False, encoding="utf-8-sig")
    print(top.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
