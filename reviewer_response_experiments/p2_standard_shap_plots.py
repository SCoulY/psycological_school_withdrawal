"""Generate conventional SHAP beeswarm and bar plots for the revised paper.

The plots are descriptive refits after nested validation.  For each age group,
the fixed display set comprises the ten most frequently selected features
across the 50 scale-aware outer folds.  LR SHAP values are on the log-odds
scale; RF SHAP values are on the reentry-probability scale.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


LABEL = "School Withdrawal/ Reentry Status"
BASE = Path("reviewer_response_experiments/results/p0_nested_cv_scaled_selector")
OUT = Path("reviewer_response_experiments/figures/p2_standard_shap")
DPI = 600


def selected_top10(group: str) -> list[str]:
    records = pd.read_csv(BASE / group / "nested_feature_selection.csv")
    counts: Counter[str] = Counter()
    for value in records.loc[records.feature_set.eq("top10"), "selected_features"]:
        counts.update(json.loads(value))
    return [name for name, _ in counts.most_common(10)]


def modal_params(group: str, model: str) -> dict:
    records = pd.read_csv(BASE / group / "nested_best_params.csv")
    records = records.loc[records.model.eq(model) & records.feature_set.eq("top10")]
    values = records.best_params.map(json.loads)
    canonical = values.map(lambda x: json.dumps(x, sort_keys=True))
    return json.loads(canonical.value_counts().index[0])


def positive_class_explanation(group: str, model_name: str) -> shap.Explanation:
    df = pd.read_csv(Path("data") / f"clean_{group}.csv")
    features = selected_top10(group)
    X = df[features].astype(float)
    y = df[LABEL].astype(int).to_numpy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    params = modal_params(group, model_name)
    rng = np.random.default_rng(20260710)
    background_idx = rng.choice(len(Xs), size=min(100, len(Xs)), replace=False)

    if model_name == "LR":
        model = LogisticRegression(
            C=float(params["clf__C"]),
            class_weight=params["clf__class_weight"],
            solver="lbfgs",
            max_iter=3000,
            random_state=20260710,
        ).fit(Xs, y)
        raw = shap.LinearExplainer(model, Xs[background_idx])(Xs)
        values = raw.values
        base_values = raw.base_values
    else:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=params["clf__max_depth"],
            class_weight=params["clf__class_weight"],
            random_state=20260710,
            n_jobs=-1,
        ).fit(Xs, y)
        explainer = shap.TreeExplainer(
            model,
            data=Xs[background_idx],
            feature_perturbation="interventional",
            model_output="probability",
        )
        raw = explainer(Xs, check_additivity=False)
        if raw.values.ndim == 3:
            values = raw.values[:, :, 1]
            base_values = raw.base_values[:, 1]
        else:
            values = raw.values
            base_values = raw.base_values

    return shap.Explanation(
        values=values,
        base_values=base_values,
        data=X.to_numpy(),
        feature_names=features,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    explanations: dict[tuple[str, str], shap.Explanation] = {}
    for group in ["adults", "teens", "children"]:
        for model in ["LR", "RF"]:
            explanation = positive_class_explanation(group, model)
            explanations[(group, model)] = explanation

            shap.plots.beeswarm(explanation, max_display=10, show=False)
            plt.title(f"{group.title()} {model}: SHAP toward reentry")
            plt.tight_layout()
            plt.savefig(OUT / f"{group}_{model.lower()}_top10_beeswarm.png", dpi=DPI, bbox_inches="tight")
            plt.close()

            shap.plots.bar(explanation, max_display=10, show=False)
            plt.title(f"{group.title()} {model}: mean absolute SHAP")
            plt.tight_layout()
            plt.savefig(OUT / f"{group}_{model.lower()}_top10_bar.png", dpi=DPI, bbox_inches="tight")
            plt.close()

    # Save a compact provenance table used by the figure caption/supplement.
    rows = []
    for (group, model), explanation in explanations.items():
        mean_abs = np.abs(explanation.values).mean(axis=0)
        for feature, value in zip(explanation.feature_names, mean_abs):
            rows.append({"group": group, "model": model, "feature": feature, "mean_abs_shap": float(value)})
    pd.DataFrame(rows).to_csv(OUT / "standard_shap_mean_abs.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(3, 2, figsize=(20, 20))
    for row, group in enumerate(["adults", "teens", "children"]):
        for col, model in enumerate(["LR", "RF"]):
            shap.plots.beeswarm(
                explanations[(group, model)],
                max_display=10,
                show=False,
                ax=axes[row, col],
                plot_size=None,
                color_bar=(col == 1),
                s=10,
            )
            scale = "log-odds" if model == "LR" else "probability"
            axes[row, col].set_title(f"{group.title()} {model} ({scale}; toward reentry)")
    fig.tight_layout()
    fig.savefig(OUT / "Fig6_standard_shap_beeswarm.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Fig6_standard_shap_beeswarm.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(18, 17))
    for row, group in enumerate(["adults", "teens", "children"]):
        for col, model in enumerate(["LR", "RF"]):
            shap.plots.bar(
                explanations[(group, model)],
                max_display=10,
                show=False,
                ax=axes[row, col],
                show_data=False,
            )
            axes[row, col].set_title(f"{group.title()} {model}: mean absolute SHAP")
    fig.tight_layout()
    fig.savefig(OUT / "FigS_standard_shap_bar.pdf", bbox_inches="tight")
    fig.savefig(OUT / "FigS_standard_shap_bar.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {len(explanations) * 2} conventional SHAP plots in {OUT.resolve()}")


if __name__ == "__main__":
    main()
