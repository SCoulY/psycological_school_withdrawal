"""Leakage-aware KDE reference cohort selection and fallback provenance.

This module intentionally does not overwrite anomaly_quantile.py. It provides
the corrected selection used by the reviewer-response experiments.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd


LABEL = "School Withdrawal/ Reentry Status"


def select_reference_cohort(
    df: pd.DataFrame,
    prob_cols=("LogisticRegression", "RandomForest"),
    high: bool = True,
    min_samples: int = 10,
    withdrawal_label: int = 0,
    reentry_label: int = 1,
    low_threshold: float = 0.3,
    high_threshold: float = 0.7,
):
    """Select a reference cohort and report exactly which fallback was used.

    p columns are p(reentry). For withdrawal, low joint probability is the
    maximum of the two model probabilities; for reentry, high joint
    probability is the minimum. Ranking is performed within the correct true
    class so each fallback can actually reach ``min_samples`` when available.
    """
    d = df.copy()
    p1, p2 = prob_cols
    if high:
        eligible = d[LABEL].eq(withdrawal_label)
        primary = eligible & d[p1].le(low_threshold) & d[p2].le(low_threshold)
        joint = d[[p1, p2]].max(axis=1)
        direction = "lowest"
        cohort_name = "withdrawal"
    else:
        eligible = d[LABEL].eq(reentry_label)
        primary = eligible & d[p1].ge(high_threshold) & d[p2].ge(high_threshold)
        joint = d[[p1, p2]].min(axis=1)
        direction = "highest"
        cohort_name = "reentry"

    candidates = d.loc[eligible].copy()
    candidates["_joint_probability"] = joint.loc[candidates.index]
    n_primary = int(primary.sum())
    if n_primary >= min_samples:
        selected = d.loc[primary].copy()
        stage = "primary_threshold"
    else:
        ranked = candidates.sort_values("_joint_probability", ascending=(high), kind="mergesort")
        selected = ranked.head(min_samples).drop(columns="_joint_probability")
        stage = "rank_top10"
        n_rank10 = len(selected)
        if n_rank10 < min_samples:
            selected = ranked.head(min_samples * 2).drop(columns="_joint_probability")
            stage = "rank_top20"
        if len(selected) < min_samples:
            selected = ranked.copy().drop(columns="_joint_probability")
            stage = "final_available_class"

    meta = {
        "cohort": cohort_name,
        "stage": stage,
        "n_primary": n_primary,
        "n_selected": int(len(selected)),
        "n_eligible_class": int(eligible.sum()),
        "min_samples": int(min_samples),
        "low_threshold": float(low_threshold),
        "high_threshold": float(high_threshold),
        "joint_score": "max(p_reentry) for withdrawal / min(p_reentry) for reentry",
        "rank_direction": direction,
    }
    return selected, meta


def pooled_oof_probabilities(group: str, feature_set: str) -> pd.DataFrame:
    """Join raw features and repeated OOF probabilities for one configuration."""
    data = pd.read_csv(Path("data") / f"clean_{group}.csv")
    pred = pd.read_csv(Path("reviewer_response_experiments/results/p0_nested_cv") / group / "nested_oof_predictions.csv")
    pred = pred[pred.feature_set.eq(feature_set)]
    wide = pred.pivot_table(index=["row_index", "y"], columns="model", values="p_reentry", aggfunc="mean").reset_index()
    wide = wide.rename(columns={"y": LABEL, "row_index": "_row_index", "LR": "LogisticRegression", "RF": "RandomForest"})
    wide[LABEL] = wide[LABEL].astype(int)
    data = data.reset_index(drop=True)
    data["_row_index"] = np.arange(len(data))
    joined = data.merge(wide, on=["_row_index", LABEL], how="inner", validate="one_to_one")
    return joined


def main():
    rows = []
    out = Path("reviewer_response_experiments/results/p1_kde")
    out.mkdir(parents=True, exist_ok=True)
    for group in ["adults", "teens", "children"]:
        for feature_set in ["full", "top10"]:
            joined = pooled_oof_probabilities(group, feature_set)
            joined.to_csv(out / f"{group}_{feature_set}_pooled_oof_input.csv", index=False, encoding="utf-8-sig")
            for high in [True, False]:
                selected, meta = select_reference_cohort(joined, high=high)
                meta.update({"group": group, "feature_set": feature_set})
                rows.append(meta)
                selected.to_csv(out / f"{group}_{feature_set}_{meta['cohort']}_reference.csv", index=False, encoding="utf-8-sig")
    prov = pd.DataFrame(rows)
    prov.to_csv(out / "p1_fallback_provenance.csv", index=False, encoding="utf-8-sig")
    prov["triggered_fallback"] = prov.stage.ne("primary_threshold")
    prov.to_csv(out / "p1_fallback_provenance.csv", index=False, encoding="utf-8-sig")
    print(prov.to_string(index=False))


if __name__ == "__main__":
    main()
