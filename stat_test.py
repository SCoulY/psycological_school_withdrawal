import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
from scipy.stats import ttest_ind
from preprocess.ch2en import column_name2eng

plt.style.use('default') 
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": "DejaVu Sans", "axes.unicode_minus": False,
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 12, "xtick.labelsize": 9, "ytick.labelsize": 9, "figure.dpi": 300,
})


# Define significance levels
def significance_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''

def plot_stat_test(file_path, plot_path):
    df = pd.read_excel(file_path)
    # drop first column
    df = df.drop(df.columns[0], axis=1)
    df_name = os.path.basename(file_path).split('.')[0]

    df = column_name2eng(df)

    # Define groups based on '状态'
    group1 = df[df['School Withdrawal/ Reentry Status'] == df['School Withdrawal/ Reentry Status'].unique()[0]]
    group2 = df[df['School Withdrawal/ Reentry Status'] == df['School Withdrawal/ Reentry Status'].unique()[1]]

    # List of feature columns (excluding '状态')
    features = df.columns.difference(['School Withdrawal/ Reentry Status'])



    # Perform t-tests
    p_values = []
    t_stats = []

    for feature in features:
        t_stat, p_value = ttest_ind(group1[feature], group2[feature], nan_policy='omit')
        p_values.append(p_value)
        t_stats.append(t_stat)


    # Convert results into a DataFrame
    results_df = pd.DataFrame({
        'Feature': features,
        'T-stat': t_stats,
        'P-value': p_values
    }).set_index('Feature')

    # Apply significance labels
    results_df['Significance'] = results_df['P-value'].apply(significance_stars)

    # Plot heatmap
    plt.figure(figsize=(8, 2.5))
    heatmap_data = results_df[['T-stat']].T  # Transpose for better visualization
    vmin = np.min(heatmap_data.values)
    vmax = np.max(heatmap_data.values)
    vmid = (vmin + vmax) / 2
    cbar_ticks = [vmin, vmid, vmax]
    ax = sns.heatmap(heatmap_data, 
                     fmt='',
                     cmap='coolwarm', 
                     center=0, 
                     linewidths=0.5, 
                     cbar_kws={'label': '', 'ticks': cbar_ticks, 'format': '%.2f', 'pad': 0.01},
                     )

    
    # Manually add vertical annotations
    for text, (j, feature) in zip(results_df['Significance'], enumerate(results_df.index)):
        ax.text(j + 0.75, 0.5, text, ha='center', va='center', rotation=90, fontsize=10, color='black', fontweight='bold')

    ax.set_xticks(np.arange(len(features)) + 0.5)
    ax.set_xticklabels(features, rotation=90, ha="center", fontsize=10)
    # Improve aesthetics
    # plt.title("T-test Heatmap with Significance Levels", fontsize=14)
    # plt.xticks(rotation=90, fontsize=10)
    plt.ylabel('', fontsize=12)
    # plt.yticks(fontsize=12)
    plt.xlabel('', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, f'{df_name}_t_test_heatmap.svg'), format='svg', transparent=True)

    


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--file_path', type=str, help='Path to the excel file', nargs='+', default=['C:/Users/SCoulY/Desktop/psycology/data/clean_teens_wo_scl.xlsx', 
                                                                                                  'C:/Users/SCoulY/Desktop/psycology/data/clean_teens.xlsx', 
                                                                                                  'C:/Users/SCoulY/Desktop/psycology/data/clean_adults.xlsx'])
    args.add_argument('--plot_path', type=str, help='Path to save the plot', default='C:/Users/SCoulY/Desktop/psycology/plot')

    args = args.parse_args()

    os.makedirs(args.plot_path, exist_ok=True)

    for file_path in args.file_path:
        if os.path.exists(file_path):
            plot_stat_test(file_path, args.plot_path)
        else:
            raise ValueError(f"File not found: {file_path}")

