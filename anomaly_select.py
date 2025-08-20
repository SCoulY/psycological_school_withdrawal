import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import numpy as np
from preprocess.ch2en import column_name2eng

adults_importance = ['HEI_TS', 'CSES_TS', 'SCL-90 DEP', 'SCL-90 GSI', 'SCL-90 PST', 'SCL-90 TS', 'SCL-90 ANX', 'SCL-90 PSY', 'SCL-90 NST', 'SCL-90 ADD', 'SCL-90 PSDI', 'SCL-90 PAR', 'EMBU-M OI', 'SCL-90 SOM', 'EMBU-F OP', 'EMBU-F EW', 'EMBU-M EW', 'SSRS_TS', 'DES-Ⅱ_AMN', 'DES-Ⅱ_TS', 'SCL-90 OC', 'SSRS_SS', 'SCL-90 IS', 'DES-Ⅱ_ABS', 'EMBU-F OI', 'SCL-90 HOS', 'DES-Ⅱ_DPDR', 'CSQ_FAN', 'SSRS_OS', 'EMBU-F REJ', 'CSQ_RAT', 'EMBU-F PUN', 'CSQ_HS', 'SSRS_SU', 'CSQ_REP', 'SCL-90 PHOB', 'CSQ_PS', 'EMBU-M REJ', 'EMBU-F FS', 'EMBU-M PUN',  'CSQ_SB', 'EMBU-M FS']

teens_importance = ['CSES_TS', 'SCL-90 DEP', 'HEI_TS', 'SCL-90 ANX', 'A-DES-Ⅱ_TS', 'A-DES-Ⅱ_PI', 'SCL-90 GSI', 'SCL-90 NST', 'SCL-90 PSY', 'EMBU-F EW', 'A-SSRS_SS', 'SCL-90 ADD', 'A-DES-Ⅱ_DPDR', 'SCL-90 PSDI', 'A-SSRS_TS', 'A-SSRS_SU', 'A-DES-Ⅱ_DA', 'SCL-90 HOS', 'EMBU-F OI', 'SCL-90 SOM', 'EMBU-M EW', 'EMBU-F PUN', 'SCL-90 TS', 'SCL-90 PHOB', 'EMBU-F OP', 'EMBU-M OI', 'SCL-90 PST', 'CSQ_FAN', 'A-SSRS_OS', 'EMBU-M PUN', 'CSQ_REP', 'SCL-90 IS', 'CSQ_PS', 'SCL-90 PAR', 'SCL-90 OC', 'CSQ_HS', 'A-DES-Ⅱ_AII', 'CSQ_SB', 'CSQ_RAT', 'EMBU-M REJ', 'EMBU-F REJ', 'EMBU-M FS', 'EMBU-F FS']

children_importance = ['CSES_TS', 'HEI_TS', 'A-DES-Ⅱ_TS', 'A-SSRS_TS', 'A-DES-Ⅱ_PI', 'A-SSRS_SU', 'A-SSRS_OS', 'A-DES-Ⅱ_DA', 'A-DES-Ⅱ_AII', 'A-DES-Ⅱ_DPDR', 'CSQ_PS', 'EMBU-M PUN', 'A-SSRS_SS', 'EMBU-M OI', 'EMBU-F EW', 'EMBU-M EW', 'EMBU-M REJ', 'CSQ_SB', 'CSQ_HS', 'EMBU-F OP', 'CSQ_REP', 'EMBU-F FS', 'EMBU-F REJ', 'CSQ_RAT', 'EMBU-F PUN', 'EMBU-F OI', 'CSQ_FAN', 'EMBU-M FS']

top10_features_adults = ['HEI_TS', 'SCL-90 DEP', 'CSES_TS', 'SCL-90 PSY', 'SCL-90 ANX', 'EMBU-F EW', 'DES-Ⅱ_TS', 'DES-Ⅱ_ABS', 'DES-Ⅱ_AMN', 'EMBU-M FS']

top10_features_teens = ['SCL-90 DEP', 'A-DES-Ⅱ_PI', 'SCL-90 ANX', 'CSES_TS', 'EMBU-F EW', 'HEI_TS', 'SCL-90 PSY', 'A-DES-Ⅱ_DPDR', 'A-DES-Ⅱ_AII', 'EMBU-M FS']

top10_features_children = ['CSES_TS', 'HEI_TS', 'A-DES-Ⅱ_PI', 'A-DES-Ⅱ_AII', 'A-DES-Ⅱ_DPDR', 'EMBU-F EW', 'EMBU-F PUN', 'EMBU-M FS', 'CSQ_REP', 'A-SSRS_OS']


# plt.rcParams['font.family'] = 'SimHei'

def single_plot(anomaly_df, axarr=None, table_name=''):
    labels = anomaly_df['status'].values
    random_forest_scores = anomaly_df['RandomForest'].values

    anomaly_df['FN'] = labels - random_forest_scores
    anomaly_df['FP'] = random_forest_scores - labels
    anomaly_df['TP'] = random_forest_scores * labels
    anomaly_df['TN'] =  (1 - random_forest_scores) * (1 - labels)

    top_FN = anomaly_df.sort_values(by='FN', ascending=False).head(5)
    top_FP = anomaly_df.sort_values(by='FP', ascending=False).head(5)
    top_TP = anomaly_df.sort_values(by='TP', ascending=False).head(5)
    top_TN = anomaly_df.sort_values(by='TN', ascending=False).head(5)

    if axarr is None:
        fig, axes = plt.subplots(2, 2, figsize=(30, 8))
    else:
        axes = axarr
        fig = axes

    if 'adults' in table_name.lower() or 'teens' in table_name.lower():
        heatmap_kwargs = dict(cmap='coolwarm', annot=True, annot_kws={"size": 8}, fmt=".2f", linewidths=.5, cbar=False)
    else:
        heatmap_kwargs = dict(cmap='coolwarm', annot=True, annot_kws={"size": 11}, fmt=".2f", linewidths=.5, cbar=False)

    for ax, samples, title in zip(
        axes.flat,
        [top_FN, top_TN, top_FP, top_TP],
        ['Top 5 FN Samples', 'Top 5 TN Samples', 'Top 5 FP Samples', 'Top 5 TP Samples']
    ):
        if not samples.empty:
            # Prepare dataframe to display (remove intermediate metric columns)
            display_df = samples.drop(columns=['FN', 'FP', 'TP', 'TN'], errors='ignore').copy()

            # Ensure the first three columns (if present) are ordered: status, RandomForest, LogisticRegression
            binary_order = [c for c in ['status', 'RandomForest', 'LogisticRegression'] if c in display_df.columns]
            other_cols = [c for c in display_df.columns if c not in binary_order]
            display_df = display_df[binary_order + other_cols]

            # Build RGBA image manually so binary columns use independent categorical palette
            bin_cmap_hex = {1: "#a6d96a", 0: "#1a9641"}
            # Convert hex to RGBA
            bin_cmap = {k: tuple(int(bin_cmap_hex[k][i:i+2], 16)/255 for i in (1,3,5)) + (1.0,) for k in bin_cmap_hex}

            vmin, vmax = -2, 2
            cont_cmap = plt.cm.get_cmap('coolwarm')
            n_rows, n_cols = display_df.shape
            img = np.zeros((n_rows, n_cols, 4))
            denom = (vmax - vmin) if vmax != vmin else 1.0

            for j, col in enumerate(display_df.columns):
                col_vals = display_df[col].values
                if col in binary_order:
                    for i, v in enumerate(col_vals):
                        img[i, j] = bin_cmap.get(int(v) if pd.notna(v) else 0, (0.5, 0.5, 0.5, 1.0))
                else:
                    norm_vals = (np.clip(col_vals.astype(float), vmin, vmax) - vmin) / denom
                    img[:, j] = cont_cmap(norm_vals)

            ax.imshow(img, aspect='auto', interpolation='nearest')
            ax.set_xticks(range(n_cols))
            ax.set_xticklabels(display_df.columns, rotation=90, fontsize=8 if ('adults' in table_name.lower() or 'teens' in table_name.lower()) else 10)
            ax.set_yticks([])

            # Add value annotations similar to seaborn heatmap when annot=True
            for i in range(n_rows):
                for j, col in enumerate(display_df.columns):
                    val = display_df.iloc[i, j]
                    if col in binary_order:
                        txt = f"{int(val)}" if pd.notna(val) else ''
                    else:
                        txt = f"{val:.2f}" if pd.notna(val) else ''
                    if txt:
                        ax.text(j, i, txt, ha='center', va='center', fontsize=6 if ('adults' in table_name.lower() or 'teens' in table_name.lower()) else 7, color='black')

            # Draw grid lines for clearer separation
            for gcol in range(n_cols + 1):
                ax.axvline(gcol - 0.5, color='white', linewidth=0.5)
            for grow in range(n_rows + 1):
                ax.axhline(grow - 0.5, color='white', linewidth=0.2)
        ax.set_title(title, fontsize=16, fontweight='bold')
        # ax.set_ylabel('Samples', fontsize=14, fontweight='bold')
        ax.set_yticks([])

    return fig, axes

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--anomaly_path', type=str, nargs='+', help='Path to the original file', default=['C:/Users/SCoulY/Desktop/psycology/data/risk_uncertainty_top10_children_correct/full/adults_anomaly.xlsx', 'C:/Users/SCoulY/Desktop/psycology/data/risk_uncertainty_top10_children_correct/full/teens_anomaly.xlsx', 'C:/Users/SCoulY/Desktop/psycology/data/risk_uncertainty_top10_children_correct/full/children_anomaly.xlsx'])
    args.add_argument('--output_path', type=str, help='Path to save the plot', default='C:/Users/SCoulY/Desktop/psycology/data/anomaly_select/correct_children')
    args = args.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    # Create a figure with 6 rows and 2 columns for all subplots
    fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(30, 26))

    for i, anomaly_path in enumerate(args.anomaly_path):
        anomaly_df = pd.read_excel(anomaly_path)
        anomaly_df.columns = anomaly_df.columns.str.replace('SCl', 'SCL', regex=True)
        anomaly_df.columns = anomaly_df.columns.str.replace('SCQ', 'CSQ', regex=True)
        anomaly_df = column_name2eng(anomaly_df)
        table_name = os.path.basename(anomaly_path).replace('.xlsx', '')
        for col in ['Gender', 'name', 'Age']:
            if col in anomaly_df.columns:
                anomaly_df.drop(columns=[col], inplace=True)

        # Pass the axes for the current row to single_plot
        _, row_axes = single_plot(anomaly_df, axarr=axes[i*2:(i+1)*2, :], table_name=table_name)

        # Add a colorbar in the middle of every two rows
        if i==0:
            start_y = 0.1
            axes[i, 0].set_ylabel('Adults', fontsize=16, fontweight='bold')
            axes[i+1, 0].set_ylabel('Adults', fontsize=16, fontweight='bold')

        elif i==1:
            start_y = 0.43
            axes[i*2, 0].set_ylabel('Teens', fontsize=16, fontweight='bold')
            axes[i*2+1, 0].set_ylabel('Teens', fontsize=16, fontweight='bold')
        elif i==2:
            start_y = 0.76
            axes[i*2, 0].set_ylabel('Children', fontsize=16, fontweight='bold')
            axes[i*2+1, 0].set_ylabel('Children', fontsize=16, fontweight='bold')

        cbar_ax = fig.add_axes([0.97, start_y, 0.01, 0.18])
        norm = plt.Normalize(vmin=-2, vmax=2)
        sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax)

    plt.tight_layout(rect=[0, 0, 0.97, 1])
    fig.savefig(os.path.join(args.output_path, 'top_FN_FP_TP_TN_samples_heatmap_all.pdf'), format='pdf')
    # plt.show()