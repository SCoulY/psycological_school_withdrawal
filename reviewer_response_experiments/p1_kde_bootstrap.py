"""Bootstrap stability intervals across KDE outer-fold estimates."""

from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("reviewer_response_experiments/results/p1_kde")


def main():
    d = pd.read_csv(OUT / "p1_sensitivity_fold_results.csv")
    d = d[(d.low_threshold == 0.3) & (d.high_threshold == 0.7) & (d.min_samples == 10) & (d.bandwidth == "scott")]
    rng = np.random.default_rng(20260710)
    rows = []
    for (group, metric), g in d.melt(id_vars=["group", "fold"], value_vars=["withdrawal_auroc", "withdrawal_pr_auc", "error_auroc", "error_pr_auc"], var_name="metric", value_name="value").groupby(["group", "metric"]):
        vals = g.value.to_numpy(float)
        boot = np.array([vals[rng.integers(0, len(vals), size=len(vals))].mean() for _ in range(2000)])
        rows.append({"group": group, "metric": metric, "n_outer_folds": len(vals), "mean": float(vals.mean()), "sd": float(vals.std(ddof=1)), "bootstrap_ci_low": float(np.quantile(boot, 0.025)), "bootstrap_ci_high": float(np.quantile(boot, 0.975)), "n_boot": 2000})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "p1_sensitivity_bootstrap.csv", index=False, encoding="utf-8-sig")
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
