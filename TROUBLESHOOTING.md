# Troubleshooting Guide

## Common Errors and Solutions

### Error: FileNotFoundError - Quantile files not found

**Error Message:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'risk_prob/full/stats/adults_kde_q_low.json'
```

**Cause:** The notebook cannot find the required quantile JSON files. This usually happens when:
1. The repository wasn't downloaded properly
2. The working directory is incorrect
3. Running in Colab but didn't clone the repository

**Solutions:**

#### For Google Colab Users:

1. **Verify you ran the download cell (Cell 5)**
   - Make sure the git clone completed successfully
   - Look for "✓ Repository downloaded successfully!" message

2. **Check working directory**
   ```python
   import os
   print(os.getcwd())
   ```
   - Should show: `/content/psycological_school_withdrawal`
   - If not, run the download cell again

3. **Run the verification cell (Cell 6)**
   - This checks if all required files exist
   - Follow the instructions if files are missing

4. **If still not working:**
   - Go to `Runtime` → `Restart runtime`
   - Run all cells from top to bottom again
   - Ensure each cell completes before moving to next

#### For Local Users:

1. **Check you're in the repository root**
   ```bash
   pwd
   ls -la
   ```
   - You should see folders: `ckpt/`, `risk_prob/`, `data/`, etc.

2. **If not in correct directory:**
   ```bash
   cd /path/to/psycological_school_withdrawal
   ```

3. **Verify files exist:**
   ```bash
   ls -R risk_prob/full/stats/
   ```
   - Should show: `adults_kde_q_low.json`, `adults_kde_q_high.json`, etc.

4. **If files are missing:**
   - Re-clone the repository
   - Make sure to clone the full repository, not just download specific files

---

## File Structure Requirements

The notebook expects this directory structure:

```
psycological_school_withdrawal/
├── ckpt/
│   ├── adults/
│   │   ├── clean_adults_LogisticRegression_acc_0.91_run_123.pkl
│   │   ├── clean_adults_scaler.pkl
│   │   └── ... (other model files)
│   ├── teens/
│   └── children/
├── risk_prob/
│   ├── full/
│   │   └── stats/
│   │       ├── adults_kde_q_low.json
│   │       ├── adults_kde_q_high.json
│   │       ├── teens_kde_q_low.json
│   │       ├── teens_kde_q_high.json
│   │       ├── children_kde_q_low.json
│   │       └── children_kde_q_high.json
│   └── top10/
│       └── stats/
│           └── ... (same structure)
├── data/
└── single_sample_inference.ipynb
```

---

## Other Common Issues

### Issue: "Module not found" errors

**Solution:**
```python
# Run in a cell:
!pip install -q numpy pandas scikit-learn scipy matplotlib seaborn joblib ipywidgets
```

### Issue: Widgets not displaying in Colab

**Solution:**
1. Enable widgets: `Runtime` → `Change runtime type` → ensure widgets are enabled
2. Or try: `!jupyter nbextension enable --py widgetsnbextension`

### Issue: Model predictions seem incorrect

**Checklist:**
- [ ] Using correct age group in configuration (`adults`, `teens`, or `children`)
- [ ] Model path matches the age group
- [ ] Feature mode (top10 vs full) matches the scaler and model
- [ ] All features have reasonable values (not extreme outliers)

### Issue: Cannot save updated scaler

**Cause:** Write permissions or path issues

**Solution:**
- Check you have write permissions in the directory
- For Colab: Files will be saved in session storage (lost after runtime ends)
- To persist: Save to Google Drive by modifying the save path

---

## Getting Help

If you're still experiencing issues:

1. **Check the GitHub repository**: https://github.com/SCoulY/psycological_school_withdrawal
2. **Verify all files are present** in the repository
3. **Open an issue** on GitHub with:
   - Error message (full traceback)
   - Which cell you're running
   - Your environment (Colab vs local)
   - Output from: `print(os.getcwd())` and `print(os.listdir('.'))`

---

## Quick Diagnostic Commands

Run these in a notebook cell to diagnose issues:

```python
import os
import sys

print("Python version:", sys.version)
print("Working directory:", os.getcwd())
print("\nDirectory contents:")
for item in sorted(os.listdir('.')):
    print(f"  {'[DIR]' if os.path.isdir(item) else '[FILE]'} {item}")

print("\nChecking required paths:")
paths_to_check = [
    'ckpt/adults',
    'risk_prob/full/stats',
    'ckpt/adults/clean_adults_scaler.pkl',
    'risk_prob/full/stats/adults_kde_q_low.json'
]
for path in paths_to_check:
    status = "✓" if os.path.exists(path) else "✗"
    print(f"  {status} {path}")
```
