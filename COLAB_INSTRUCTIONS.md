# Google Colab Instructions for Single Sample Inference

## 🚀 Quick Start for Colab Users

This notebook has been converted to be fully compatible with Google Colab and is completely self-contained.

### How to Use in Google Colab

1. **Open in Colab**: 
   - Upload `single_sample_inference.ipynb` to your Google Drive
   - Open it with Google Colaboratory
   - Or use: `File` → `Upload notebook` in Colab

2. **Run All Cells in Order**:
   - Click `Runtime` → `Run all` 
   - Or press `Ctrl+F9` (Windows) / `Cmd+F9` (Mac)

3. **Follow the Setup**:
   - Cell 1: Environment check (detects Colab automatically)
   - Cell 2: Mounts Google Drive (click "Connect to Google Drive")
   - Cell 3: Installs required packages (~30 seconds)
   - Cell 4: Downloads repository from GitHub
   - Remaining cells: Load models and create interactive UI

### What's Included

The notebook now includes everything needed to run:
- ✅ Automatic package installation
- ✅ Repository cloning from GitHub
- ✅ Helper functions embedded directly
- ✅ Interactive widget-based UI
- ✅ Anomaly detection and visualization
- ✅ Model loading and prediction

### Changes from Original

**Self-Contained Design:**
- All helper functions from `anomaly_quantile.py` are now embedded
- No external file dependencies within the notebook
- Automatic detection of Colab vs local environment

**Automatic Setup:**
- Detects if running in Colab
- Installs packages only when needed
- Downloads repository files automatically
- Configures paths correctly for both environments

**Enhanced User Experience:**
- Clear section headers with emojis
- Step-by-step instructions
- Configuration examples
- Troubleshooting guide

### Configuration

Before running predictions, configure these parameters in the Configuration cell:

```python
group_name = 'adults'  # Options: 'adults', 'teens', 'children'
model_name = 'LogisticRegression'  # Options: 'LogisticRegression', 'RandomForest'
top10 = False  # True: use top 10 features, False: use all features
clf_path = 'ckpt/adults/clean_adults_LogisticRegression_acc_0.91_run_123.pkl'
```

### Using the Interactive Interface

1. **Expand the Feature Inputs accordion** to see all sliders
2. **Adjust feature values** using sliders OR click on values to enter custom numbers
3. **Set demographic information** (age and gender) if using full feature mode
4. **Click "Predict"** to see:
   - Risk probabilities
   - Feature values (partial or full view)
   - Anomaly analysis with heatmap
5. **Use "Reset Means"** to return all features to their mean values

### Tips for Best Results

- **First Run**: Always run all cells from top to bottom the first time
- **Custom Values**: Click on the value display (right of slider) to enter values outside the slider range
- **Anomaly Toggle**: Enable "Show Anomaly Analysis" to see detailed feature-level insights
- **Different Models**: To switch models, update the configuration cell and re-run from there

### Troubleshooting

**Problem**: "FileNotFoundError: risk_prob/full/stats/adults_kde_q_low.json"
**Solution**: 
- Verify the download cell (Cell 5) ran successfully
- Run the verification cell (Cell 6) to check file structure
- Check your working directory with `print(os.getcwd())`
- In Colab, should be: `/content/psycological_school_withdrawal`
- See detailed solutions in [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

**Problem**: "Module not found" errors
**Solution**: Re-run the package installation cell (Cell 3)

**Problem**: "File not found" errors  
**Solution**: 
- Ensure the repository download cell (Cell 5) completed successfully
- Look for "✓ Repository downloaded successfully!" message
- Run the verification cell (Cell 6) to diagnose issues

**Problem**: Widgets not displaying
**Solution**: In Colab, go to `Runtime` → `Restart runtime` then run all cells again

**Problem**: Out of memory
**Solution**: In Colab, go to `Runtime` → `Change runtime type` → Select "High-RAM"

For detailed troubleshooting steps, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

### Running Locally

If you're running locally (not in Colab):
1. Clone the repository manually
2. Install packages: `pip install -r requirements.txt`
3. Ensure you're in the repository root directory
4. Run the notebook in Jupyter or VS Code

### Next Steps

- Experiment with different age groups and models
- Try both top-10 and full feature modes
- Use the "Update Scaler" feature to incrementally update statistics
- Export results by copying output or taking screenshots

### Support

For issues or questions:
- Check the troubleshooting section in the notebook
- Visit the [GitHub repository](https://github.com/SCoulY/psycological_school_withdrawal)
- Review the main README.md for detailed documentation

---

**Enjoy exploring psychological school withdrawal risk prediction! 🎓**
