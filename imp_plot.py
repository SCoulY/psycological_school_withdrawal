


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
# from preprocess.ch2en import column_name2eng
from matplotlib.patches import Patch
from collections import OrderedDict, defaultdict

# plt.rcParams['font.family'] = 'SimHei'
# print(plt.style.available)

'''Empirical top 10 features from the analysis'''
top10_features_adults = ['HEI_TS', 'SCL-90 DEP', 'CSES_TS', 'SCL-90 PSY', 'SCL-90 ANX', 'EMBU-F EW', 'DES-Ⅱ_TS', 'DES-Ⅱ_ABS', 'DES-Ⅱ_AMN', 'EMBU-M FS']

top10_features_teens = ['SCL-90 DEP', 'A-DES-Ⅱ_PI', 'SCL-90 ANX', 'CSES_TS', 'EMBU-F EW', 'HEI_TS', 'SCL-90 PSY', 'A-DES-Ⅱ_DPDR', 'A-DES-Ⅱ_AII', 'EMBU-M FS']

top10_features_children = ['CSES_TS', 'HEI_TS', 'A-DES-Ⅱ_PI', 'A-DES-Ⅱ_AII', 'A-DES-Ⅱ_DPDR', 'EMBU-F EW', 'EMBU-F PUN', 'EMBU-M FS', 'CSQ_REP', 'A-SSRS_OS']




def get_avg_feature_importance(ckpt_group, args):
    """
    Calculates the average feature importance and standard deviation from a group of model checkpoints (e.g., 5 runs).
    """
    # Use the first checkpoint to determine model name and file path
    first_ckpt = ckpt_group[0]
    if '.pkl' not in first_ckpt or 'SVM' in first_ckpt:
        return None, None, None

    # --- Data Loading (done once for the group) ---
    model_name = first_ckpt.split('acc')[0][:-1].split('_')[-1] # LogisticRegression
    file_name = first_ckpt.split(model_name)[0][:-1] # clean_adults
    file_path = os.path.normpath(os.path.join(args.file_path, file_name+'.csv'))

    if 'xlsx' in file_path:
        df = pd.read_excel(file_path)
    elif 'csv' in file_path:
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format for {file_path}")

    # drop unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    if 'Name' in df.columns:
        df = df.drop(columns=['Name'])

    if 'top10' not in first_ckpt:
        X = df.drop(columns=['School Withdrawal/ Reentry Status'])
    else:
        if 'teens' in  file_path:
            top10_features_set = top10_features_teens
        elif 'children' in file_path:
            top10_features_set = top10_features_children
        elif 'adults' in file_path:
            top10_features_set = top10_features_adults
        X = df[top10_features_set]

    Y = df['School Withdrawal/ Reentry Status']

    # Get original feature names before converting to English
    original_feature_names = X.columns.tolist()

    # --- Importance Calculation across all runs ---
    importances_by_feature = defaultdict(list)

    for ckpt in ckpt_group:
        clf = joblib.load(os.path.normpath(os.path.join(args.ckpt_path, ckpt)))

        if hasattr(clf, 'coef_'):
            importances = np.abs(clf.coef_).flatten()
        elif hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
        else:
            # Skip models without standard importance attributes for simplicity
            continue
        # Align importances with feature names for this run
        for feature_name, importance_val in zip(original_feature_names, importances):
            importances_by_feature[feature_name].append(importance_val)

    if not importances_by_feature:
         raise ValueError("Could not extract importances from the model group.")

    # --- Aggregation and Sorting ---
    # Calculate mean and standard deviation for each feature
    mean_importances = {name: np.mean(vals) for name, vals in importances_by_feature.items()}
    std_importances = {name: np.std(vals) for name, vals in importances_by_feature.items()}


    # Convert Chinese feature names to English for the final output DataFrame
    X_eng = X[list(mean_importances.keys())]
    eng_feature_names = X_eng.columns.tolist()
    ch_to_eng_map = dict(zip(mean_importances.keys(), eng_feature_names))

    # Create a DataFrame for easy sorting and plotting
    importance_df = pd.DataFrame({
        'feature': [ch_to_eng_map[name] for name in mean_importances.keys()],
        'mean_importance': [mean_importances[name] for name in mean_importances.keys()],
        'std_importance': [std_importances[name] for name in std_importances.keys()]
    })
    
    # Sort features by mean importance (descending)
    importance_df = importance_df.sort_values(by='mean_importance', ascending=False).reset_index(drop=True)

    return importance_df, model_name, file_path

# --- Plotting function updated to handle error bars ---
def plot_importance(ax, data, title, xlabel):
    
    # Create the bar plot with error bars using the 'xerr' parameter
    plot_data = data.groupby('feature', as_index=False).agg(
    mean_importance=('mean_importance','mean'),
    std_importance=('std_importance','mean')
)
    # sns.barplot(x='mean_importance', y='feature', data=plot_data, palette=bar_colors.tolist(), 
    #             ax=ax, width=0.8, xerr=plot_data['std_importance'], capsize=0.2)
    
    # sort the data by mean importance for better visualization
    plot_data = plot_data.sort_values(by='mean_importance', ascending=False)
    # Map features to categories to get the color for each bar
    bar_colors = plot_data['feature'].map(feature_to_category).map(category_color_map).fillna('grey')

    sns.barplot(
    x='mean_importance',
    y='feature',
    data=plot_data,                # raw data with multiple rows per feature
    palette=bar_colors.tolist(),
    ax=ax,
    width=0.8,
    capsize=0.2,
    ci=None,  # Disable built-in confidence intervals
    )
    # Add error bars separately
    ax.errorbar(
        x=plot_data['mean_importance'],
        y=np.arange(len(plot_data)),  # positions of bars
        xerr=plot_data['std_importance'],
        fmt='none',                   # no line, just error bars
        ecolor='black',
        capsize=2
    )

    
    # --- Aesthetics for a 'Nature' look ---
    ax.set_title(title, loc='left', weight='bold')
    ax.set_xlabel(xlabel, weight='bold')
    ax.set_ylabel("") # We don't need a "Features" y-label
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', direction='out', width=1)
    ax.tick_params(axis='y', length=0, pad=2)
    ax.grid(axis='x', color='grey', linestyle=':', linewidth=0.5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, help='Path to the excel file', default='C://Users//SCoulY//Desktop//psycology//data')
    parser.add_argument('--ckpt_path', type=str, help='Path to the checkpoint file', default='C://Users//SCoulY//Desktop//psycology//ckpt_5runs//children_correct/full_wo_gender/children/')
    parser.add_argument('--plot_path', type=str, help='Path to save the plot', default='C://Users//SCoulY//Desktop//psycology//plot//ver7_children_correct')
    args = parser.parse_args()

    if not os.path.exists(args.plot_path):
        os.makedirs(args.plot_path)
    
    # --- Group checkpoints by model type and data ---
    # This assumes a naming convention like 'dataname_modelname_..._run_X.pkl'
    ckpt_groups = defaultdict(list)
    for ckpt in os.listdir(args.ckpt_path):
        if '.pkl' in ckpt and 'SVM' not in ckpt:
            ckpt_groups[ckpt].append(ckpt)

    # --- Process each group of models ---
    importance_dfs = []
    model_names = []
    file_paths = []

    # Sort groups to maintain the desired plot order
    sorted_group_keys = sorted(ckpt_groups.keys(), key=lambda x: ('top10' in x, 'LogisticRegression' not in x, 'RandomForest' not in x))

    for idx in range(0, len(sorted_group_keys), 5):
        group = sorted_group_keys[idx:idx+5]
        importance_df, model_name, file_path = get_avg_feature_importance(group, args)
        if importance_df is None:
            continue
        
        # Only show top N features
        N = min(100, len(importance_df))
        importance_dfs.append(importance_df.head(N))
        model_names.append(model_name)
        file_paths.append(file_path)

    # --- Feature-to-category mapping (unchanged) ---
    feature_to_category = OrderedDict({
        'HEI_TS': 'Emotional Distress', 'CSES_TS': 'Self-Evaluation', 'SCL-90 DEP': 'SCL-90 Symptoms', 'SCL-90 ANX': 'SCL-90 Symptoms', 'SCL-90 HOS': 'SCL-90 Symptoms', 'SCL-90 PHOB': 'SCL-90 Symptoms', 'SCL-90 PAR': 'SCL-90 Symptoms', 'SCL-90 PSY': 'SCL-90 Symptoms', 'SCL-90 SOM': 'SCL-90 Symptoms', 'SCL-90 OC': 'SCL-90 Symptoms', 'SCL-90 IS': 'SCL-90 Symptoms', 'SCL-90 PST': 'SCL-90 Symptoms', 'SCL-90 NST': 'SCL-90 Symptoms', 'SCL-90 GSI': 'SCL-90 Symptoms', 'SCL-90 PSDI': 'SCL-90 Symptoms', 'SCL-90 ADD': 'SCL-90 Symptoms', 'SCL-90 TS': 'SCL-90 Symptoms', 'DES-Ⅱ_TS': 'Dissociative Exp.', 'DES-Ⅱ_DPDR': 'Dissociative Exp.', 'DES-Ⅱ_AMN': 'Dissociative Exp.', 'DES-Ⅱ_ABS': 'Dissociative Exp.', 'A-DES-Ⅱ_TS': 'Dissociative Exp.', 'A-DES-Ⅱ_AII': 'Dissociative Exp.', 'A-DES-Ⅱ_PI': 'Dissociative Exp.', 'A-DES-Ⅱ_DPDR': 'Dissociative Exp.', 'A-DES-Ⅱ_DA': 'Dissociative Exp.', 'EMBU-F EW': 'Family Dynamics', 'EMBU-F REJ': 'Family Dynamics', 'EMBU-F OI': 'Family Dynamics', 'EMBU-F PUN': 'Family Dynamics', 'EMBU-F FS': 'Family Dynamics', 'EMBU-M EW': 'Family Dynamics', 'EMBU-M REJ': 'Family Dynamics', 'EMBU-M OI': 'Family Dynamics', 'EMBU-M PUN': 'Family Dynamics', 'EMBU-M FS': 'Family Dynamics', 'EMBU-F OP': 'Family Dynamics', 'SSRS_TS': 'Social Support', 'SSRS_SS': 'Social Support', 'SSRS_OS': 'Social Support', 'SSRS_SU': 'Social Support', 'A-SSRS_TS': 'Social Support', 'A-SSRS_SS': 'Social Support', 'A-SSRS_OS': 'Social Support', 'A-SSRS_SU': 'Social Support', 'CSQ_RAT': 'Coping Strategies', 'CSQ_PS': 'Coping Strategies', 'CSQ_HS': 'Coping Strategies', 'CSQ_FAN': 'Coping Strategies', 'CSQ_SB': 'Coping Strategies', 'CSQ_REP': 'Coping Strategies', 'Age': 'Demographics', 'Gender': 'Demographics'
    })
    ordered_categories = ['Emotional Distress', 'Self-Evaluation', 'SCL-90 Symptoms', 'Dissociative Exp.', 'Family Dynamics', 'Social Support', 'Coping Strategies', 'Demographics']
    colors = sns.color_palette("muted", len(ordered_categories))
    category_color_map = OrderedDict(zip(ordered_categories, colors))

    # --- Plotting Setup (unchanged) ---
    plt.style.use('default') 
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": "DejaVu Sans", "axes.unicode_minus": False,
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 12, "xtick.labelsize": 9, "ytick.labelsize": 9, "figure.dpi": 300,
    })

    # --- Plotting Loop ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 16))

    for i, (model_name, importance_df) in enumerate(zip(model_names, importance_dfs)):
        importance_str = "Mean Importance Score (Gini)" if model_name == 'RandomForest' else "Mean Importance Score (Abs. Coeff.)"
        plot_importance(axes[i//2, i%2], 
                        data=importance_df, 
                        title="",
                        xlabel=importance_str)

    # --- Titles, Labels, and Legend (unchanged) ---
    fig.text(0.3, 0.90, 'LogisticRegression', ha='center', va='center', fontsize=12, weight='bold')
    fig.text(0.8, 0.90, 'RandomForest', ha='center', va='center', fontsize=12, weight='bold')
    fig.text(0.1, 0.72, 'Full model', ha='center', va='center', fontsize=12, weight='bold', rotation=90)
    fig.text(0.1, 0.30, 'Partial model', ha='center', va='center', fontsize=12, weight='bold', rotation=90)
    
    panel_labels = ['a', 'c', 'b', 'd']
    for i, ax in enumerate(axes.flatten()):
        ax.text(-0.25, 1.05, panel_labels[i], transform=ax.transAxes, fontsize=13, fontweight='bold', va='top')

    legend_elements = [Patch(facecolor=color, edgecolor='k', label=cat) for cat, color in category_color_map.items()]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.01), 
               ncol=len(ordered_categories)//2, frameon=False, fontsize=12, handletextpad=0.5)

    if 'adults' in args.ckpt_path:
        cohort_name = "adults"
    elif 'teens' in args.ckpt_path:
        cohort_name = "teens"
    else:
        cohort_name = "children"
        
    plt.subplots_adjust(left=0.2, bottom=0.1, right=0.95, top=0.88, wspace=0.6, hspace=0.2)
    plt.savefig(os.path.join(args.plot_path, f"importance_{cohort_name}_avg.pdf"), bbox_inches='tight', format='pdf')
    # plt.show()