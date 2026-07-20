"""Persist the primary full-versus-top-10 paired AUROC comparisons.

This compact companion avoids rerunning secondary metrics when only the
primary AUROC table changes after the scale-aware selector audit.
"""

from pathlib import Path

import pandas as pd

from reviewer_response_experiments.p0_paired_comparison import bootstrap_delta


BASE = Path("reviewer_response_experiments/results/p0_nested_cv_scaled_selector")


def main() -> None:
    rows = []
    for group in ["adults", "teens", "children"]:
        pred = pd.read_csv(BASE / group / "nested_oof_predictions.csv")
        pooled = pred.groupby(["group", "model", "feature_set", "row_index", "y"], as_index=False).p_reentry.mean()
        for model in ["LR", "RF", "SVM"]:
            wide = (
                pooled[pooled.model.eq(model)]
                .pivot(index=["row_index", "y"], columns="feature_set", values="p_reentry")
                .dropna()
                .reset_index()
            )
            observed, lo, hi, pvalue = bootstrap_delta(
                wide.y.to_numpy(int),
                wide.full.to_numpy(float),
                wide.top10.to_numpy(float),
                "withdrawal_auroc",
                n_boot=2000,
            )
            rows.append(
                {
                    "group": group,
                    "model": model,
                    "metric": "withdrawal_auroc",
                    "top10_minus_full": observed,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "paired_bootstrap_p": pvalue,
                    "n_participants": len(wide),
                    "n_boot": 2000,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(BASE / "p0_scaled_selector_auroc_paired_bootstrap.csv", index=False, encoding="utf-8-sig")
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
