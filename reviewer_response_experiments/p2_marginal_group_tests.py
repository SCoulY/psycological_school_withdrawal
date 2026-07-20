"""Welch marginal group contrasts with within-stratum BH-FDR adjustment.

These tests are descriptive only.  They are deliberately separate from model
selection and feature attribution, which use nested resampling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, t


LABEL = "School Withdrawal/ Reentry Status"


def welch_ci(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return withdrawal-minus-reentry mean difference and a Welch 95% CI."""
    diff = float(np.mean(x) - np.mean(y))
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    se2 = vx / len(x) + vy / len(y)
    se = np.sqrt(se2)
    df = se2**2 / ((vx / len(x)) ** 2 / (len(x) - 1) + (vy / len(y)) ** 2 / (len(y) - 1))
    critical = t.ppf(0.975, df)
    return diff, float(diff - critical * se), float(diff + critical * se)


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    pooled = np.sqrt(((len(x) - 1) * np.var(x, ddof=1) + (len(y) - 1) * np.var(y, ddof=1)) / (len(x) + len(y) - 2))
    return float((np.mean(x) - np.mean(y)) / pooled) if pooled > 0 else float("nan")


def benjamini_hochberg(pvalues: list[float], alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Dependency-free BH adjusted p-values and rejection decisions."""
    p = np.asarray(pvalues, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    adjusted = np.empty(n, float)
    adjusted[order] = np.clip(adjusted_ranked, 0, 1)
    rejected = adjusted <= alpha
    return rejected, adjusted


def main() -> None:
    out = Path("reviewer_response_experiments/results/p2_marginal_tests")
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str | int]] = []
    for group in ["adults", "teens", "children"]:
        df = pd.read_csv(Path("data") / f"clean_{group}.csv")
        withdrawal = df.loc[df[LABEL].eq(0)]
        reentry = df.loc[df[LABEL].eq(1)]
        group_rows = []
        for feature in df.columns.drop(LABEL):
            x = withdrawal[feature].dropna().to_numpy(float)
            y = reentry[feature].dropna().to_numpy(float)
            stat, pvalue = ttest_ind(x, y, equal_var=False)
            diff, lo, hi = welch_ci(x, y)
            group_rows.append(
                {
                    "group": group,
                    "feature": feature,
                    "withdrawal_n": len(x),
                    "reentry_n": len(y),
                    "withdrawal_minus_reentry_mean": diff,
                    "mean_difference_ci95_low": lo,
                    "mean_difference_ci95_high": hi,
                    "welch_t": float(stat),
                    "welch_p": float(pvalue),
                    "cohen_d": cohen_d(x, y),
                }
            )
        rejected, adjusted_p = benjamini_hochberg([row["welch_p"] for row in group_rows])
        for row, p_fdr, reject in zip(group_rows, adjusted_p, rejected):
            row["bh_fdr_p"] = float(p_fdr)
            row["bh_fdr_reject_0_05"] = bool(reject)
        rows.extend(group_rows)
    result = pd.DataFrame(rows)
    result.to_csv(out / "welch_bh_feature_contrasts.csv", index=False, encoding="utf-8-sig")
    print(result.groupby("group").size().to_string())
    print(f"Wrote {len(result)} descriptive feature contrasts to {out.resolve()}")


if __name__ == "__main__":
    main()
