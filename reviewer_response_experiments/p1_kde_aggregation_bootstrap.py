"""Bootstrap CIs for information-preserving KDE aggregations."""

from pathlib import Path
import numpy as np
import pandas as pd

from reviewer_response_experiments.p1_kde_aggregation import safe_auc, safe_ap

OUT = Path("reviewer_response_experiments/results/p1_kde")


def main():
    fold = pd.read_csv(OUT / "p1_kde_aggregation_fold_metrics.csv")
    rows = []
    rng = np.random.default_rng(20260710)
    for (group, method, metric), d in fold.melt(
        id_vars=["group", "fold", "method"],
        value_vars=["withdrawal_auroc", "withdrawal_pr_auc", "error_auroc", "error_pr_auc"],
        var_name="metric", value_name="value",
    ).groupby(["group", "method", "metric"]):
        vals = d.value.to_numpy(float)
        boot = np.array([vals[rng.integers(0, len(vals), size=len(vals))].mean() for _ in range(2000)])
        rows.append({
            "group": group, "method": method, "metric": metric,
            "n_outer_folds": len(vals), "mean": float(vals.mean()), "sd": float(vals.std(ddof=1)),
            "bootstrap_ci_low": float(np.quantile(boot, 0.025)), "bootstrap_ci_high": float(np.quantile(boot, 0.975)),
            "n_boot": 2000,
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "p1_kde_aggregation_bootstrap.csv", index=False, encoding="utf-8-sig")
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
