## Setup and usage guide

This guide shows how to set up the Python environment and run the full workflow: train models, plot importances and SHAP, generate risk probabilities, compute anomalies, and render anomaly heatmaps. Paths below assume you run commands from the repo root `/psycology`.

---

## 1) Install environment (mamba/Conda + pip)

Create and activate a clean Conda/Mamba environment, then install Python packages from `requirements.txt`.

```powershell
# Create env (Python 3.11 matches cached files in this repo)
mamba create -n school-withdrawal python=3.11 -y

# Activate it
conda activate school-withdrawal

# Install dependencies
pip install -r requirements.txt
```

Notes
- Works on Windows PowerShell. If you don’t have Mamba, use `conda` instead of `mamba` for the first line.
- The requirements include numpy, pandas, scikit-learn, shap, seaborn, matplotlib, scipy, joblib, tqdm, etc.

---

## 2) Train classifiers

Run with defaults (adults dataset; all features since top-10 selection is disabled by default):

```bash
python ./classifier.py
```

Arguments (defaults shown)
- `--file_path` (str, default `data/clean_adults.csv`): Input CSV/XLSX of a cohort: `clean_adults.csv`, `clean_teens.csv`, or `clean_children.csv`.
- `--out_path` (str, default `ckpt/adults`): Folder to save trained models and a scaler bundle (`*_scaler*.pkl`). Created if missing.
- `--ckpt_path` (str|None, default `None`): If set to a `.pkl` checkpoint, the script loads it and reports metrics instead of training.
- `--classifier` (list[str], default `["SVM","RandomForest","LogisticRegression"]`): One or more models to train.
- `--disable_top10` (flag, default `True`): If True, use all features; if False, train with cohort-specific top-10 features. Note: current CLI flag is `store_true` with default True, so toggling to False requires editing the script or changing the default.
- `--scaler_path` (str|None, default `None`): Optional path to a saved scaler bundle to reuse scaling; otherwise a new scaler is fit and saved under `out_path`.

Outputs
- One model file per run per classifier (5 runs using seeds) saved in `out_path`.
- A scaler bundle `*_scaler.pkl` or `*_scaler_top10.pkl` saved in `out_path`.
- Console metrics: accuracy, precision/recall/F1, and AUC (mean ± std across runs).

Tips
- To train on teens or children, pass `--file_path data/clean_teens.csv` or `--file_path data/clean_children.csv` and adjust `--out_path` accordingly.

---

## 3) Plot feature importances

Generates averaged feature-importance plots across 5-run checkpoints per model and cohort.

Run with portable relative paths:
```bash
python ./imp_plot.py --file_path ./data --ckpt_path ./ckpt_5runs/children --plot_path ./plot/
```

Default arguments (in the script; absolute paths are already set to this repo on Windows)
- `--file_path` (str, default `psycology/data`): Folder containing the cohort CSVs. The script infers which CSV to read from checkpoint filenames.
- `--ckpt_path` (str, default `psycology/ckpt_5runs/children/`): Folder with 5-run model `.pkl` files (LogisticRegression/RandomForest; SVM is skipped).
- `--plot_path` (str, default `psycology/plot/`): Output folder for the PDF figure.

Notes
- Checkpoints are grouped by naming; ensure you have 5 runs per model so means/SDs are meaningful.
- Output is saved as `importance_{cohort}_avg.pdf` in `plot_path`.

---

## 4) Plot SHAP explanations 

Computes and aggregates SHAP values across checkpoints, then renders circular SHAP summary plots per model and feature set.

Run with portable relative paths:
```bash
python ./shap_explain.py --file_path ./data --ckpt_path ./ckpt_5runs/children --plot_path ./plot/shap_plot_children
```

Default arguments
- `--file_path` (str, default `/psycology/data`): Folder containing cohort CSVs.
- `--ckpt_path` (str, default `psycology/ckpt_5runs/children/`): Folder with trained checkpoints (LogisticRegression/RandomForest).
- `--plot_path` (str, default `psycology/plot/shap_plot_children`): Output folder for SHAP SVG files.

Outputs
- SVG plots named like `{table}_{model}_{top10|}.svg` in `plot_path`.

---

## 5) Predict risk probabilities 
Generates per-student withdrawal risk probabilities by averaging predictions across 5-run checkpoints, then saves an Excel file.

Run with defaults (children, top-10 enabled because the default `--disable_top10` is False):
```bash
python ./risk_prob_pred.py
```

Arguments (defaults shown)
- `--file_path` (str, default `data/clean_children.csv`): Input cohort CSV/XLSX.
- `--ckpt_path` (str, default `ckpt_5runs/children`): Folder containing trained model `.pkl` files.
- `--output_path` (str, default `risk_prob/top10`): Output folder for the Excel with predictions.
- `--disable_top10` (flag, default `False`): If present, use all features; if omitted, use the cohort-specific top-10 features.
- `--scaler_path` (str, default `ckpt_5runs/children/clean_children_scaler_top10.pkl`): Path to a scaler bundle to apply consistent scaling at inference.

Outputs
- `{file_stem}_risk_prob.xlsx` with two new columns: `LogisticRegression` and `RandomForest` (class-1 probabilities), saved to `output_path`.

Tips
- Ensure checkpoints in `ckpt_path` match the feature setting: the script filters files by `top10` in the filename depending on `--disable_top10`.

---

## 6) Compute anomaly scores and quantiles

Builds KDE-based 5th/95th quantiles for “high-risk” and “low-risk” groups (split by predicted probabilities + ground truth), then assigns a signed anomaly score per feature.

Run (children example using defaults):
```bash
python ./anomaly_quantile.py
```

Arguments (defaults shown)
- `--file_path` (str, default `risk_prob/full/clean_children_risk_prob.xlsx`): Input Excel/CSV that includes ground-truth `School Withdrawal/ Reentry Status` and model probability columns from step 5.
- `--kde_q_high_path` (str, default `children_kde_q_high.json`): Filename to read/write high-risk quantiles (used under `output_path/stats`).
- `--kde_q_low_path` (str, default `children_kde_q_low.json`): Filename to read/write low-risk quantiles (used under `output_path/stats`).
- `--output_path` (str, default `risk_prob/full/`): Base folder to write `stats` JSONs and final anomaly Excel.

Outputs
- `output_path/stats/{cohort}_kde_q_high.json` and `..._low.json` with per-feature 5th/95th quantiles.
- `{cohort}_anomaly.xlsx` in `output_path` containing signed anomaly scores for all features.

---

## 7) Plot anomaly heatmaps

Renders ordered anomaly heatmaps for Adults/Teens/Children side-by-side for each model. Note: the filename is `sort_anomaly_plot.py` (not “ploy”).

Run (set paths to your anomaly Excel files from step 6):
```bash
python ./sort_anomaly_plot.py --uncert_path ./risk_prob/top10 --output_path ./risk_prob/top10/anomaly_plot
```

Arguments (defaults in script use macOS-style absolute paths; overriding is recommended on Windows)
- `--uncert_path` (str): Folder containing `adults_anomaly.xlsx`, `teens_anomaly.xlsx`, `children_anomaly.xlsx`.
- `--output_path` (str): Output folder for the combined PDF figure.
- `--disable_top10` (flag, default `False`): If present, plot “All features”; if omitted, plot “Top 10 features”.

Outputs
- `top10_anomaly.pdf` or `all_anomaly.pdf` in `output_path`.

---

## 8) Interactive single-sample prediction

Run the notebook UI in IDE to interactively predict and explain one sample.

Steps
1) Open `single_sample_inference.ipynb` in IDE.
2) Select the Python kernel from the `school-withdrawal` env (Command Palette → “Python: Select Interpreter”). If prompted, install the IPython kernel into the env.
3) In the first cells, update any variables that point to your trained checkpoints (from step 2) and scaler bundle (saved in `ckpt_5runs/...`).
4) Run cells top-to-bottom. Provide feature values or select a row as instructed in the notebook. The notebook will scale inputs, load the model(s), predict risk probabilities, and render explanations (e.g., SHAP).

---

Troubleshooting
- Paths: Prefer running from the repo root so relative paths like `data/clean_*.csv` resolve correctly. Otherwise, pass explicit `--file_path`, `--ckpt_path`, etc.
- Fonts/plots: If labels don’t render, ensure matplotlib/seaborn fonts are installed or use default settings.
- Top-10 switch: In `classifier.py`, `--disable_top10` defaults to True and uses a `store_true` flag. To train with top-10 features, you may need to change the default in the script or add a complementary flag.
