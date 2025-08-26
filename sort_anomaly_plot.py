import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import numpy as np
from matplotlib.patches import Patch
from matplotlib import gridspec

adults_importance = ['HEI_TS', 'CSES_TS', 'SCL-90 DEP', 'SCL-90 GSI', 'SCL-90 PST', 'SCL-90 TS', 'SCL-90 ANX', 'SCL-90 PSY', 'SCL-90 NST', 'SCL-90 ADD', 'SCL-90 PSDI', 'SCL-90 PAR', 'EMBU-M OI', 'SCL-90 SOM', 'EMBU-F OP', 'EMBU-F EW', 'EMBU-M EW', 'SSRS_TS', 'DES-Ⅱ_AMN', 'DES-Ⅱ_TS', 'SCL-90 OC', 'SSRS_SS', 'SCL-90 IS', 'DES-Ⅱ_ABS', 'EMBU-F OI', 'SCL-90 HOS', 'DES-Ⅱ_DPDR', 'CSQ_FAN', 'SSRS_OS', 'EMBU-F REJ', 'CSQ_RAT', 'EMBU-F PUN', 'CSQ_HS', 'SSRS_SU', 'CSQ_REP', 'SCL-90 PHOB', 'CSQ_PS', 'EMBU-M REJ', 'EMBU-F FS', 'EMBU-M PUN',  'CSQ_SB', 'EMBU-M FS']
teens_importance = ['CSES_TS', 'SCL-90 DEP', 'HEI_TS', 'SCL-90 ANX', 'A-DES-Ⅱ_TS', 'A-DES-Ⅱ_PI', 'SCL-90 GSI', 'SCL-90 NST', 'SCL-90 PSY', 'EMBU-F EW', 'A-SSRS_SS', 'SCL-90 ADD', 'A-DES-Ⅱ_DPDR', 'SCL-90 PSDI', 'A-SSRS_TS', 'A-SSRS_SU', 'A-DES-Ⅱ_DA', 'SCL-90 HOS', 'EMBU-F OI', 'SCL-90 SOM', 'EMBU-M EW', 'EMBU-F PUN', 'SCL-90 TS', 'SCL-90 PHOB', 'EMBU-F OP', 'EMBU-M OI', 'SCL-90 PST', 'CSQ_FAN', 'A-SSRS_OS', 'EMBU-M PUN', 'CSQ_REP', 'SCL-90 IS', 'CSQ_PS', 'SCL-90 PAR', 'SCL-90 OC', 'CSQ_HS', 'A-DES-Ⅱ_AII', 'CSQ_SB', 'CSQ_RAT', 'EMBU-M REJ', 'EMBU-F REJ', 'EMBU-M FS', 'EMBU-F FS']
children_importance = ['CSES_TS', 'HEI_TS', 'A-DES-Ⅱ_TS', 'A-SSRS_TS', 'A-DES-Ⅱ_PI', 'A-SSRS_SU', 'A-SSRS_OS', 'A-DES-Ⅱ_DA', 'A-DES-Ⅱ_AII', 'A-DES-Ⅱ_DPDR', 'CSQ_PS', 'EMBU-M PUN', 'A-SSRS_SS', 'EMBU-M OI', 'EMBU-F EW', 'EMBU-M EW', 'EMBU-M REJ', 'CSQ_SB', 'CSQ_HS', 'EMBU-F OP', 'CSQ_REP', 'EMBU-F FS', 'EMBU-F REJ', 'CSQ_RAT', 'EMBU-F PUN', 'EMBU-F OI', 'CSQ_FAN', 'EMBU-M FS']

top10_features_adults = ['HEI_TS', 'SCL-90 DEP', 'CSES_TS', 'SCL-90 PSY', 'SCL-90 ANX', 'EMBU-F EW', 'DES-Ⅱ_TS', 'DES-Ⅱ_ABS', 'DES-Ⅱ_AMN', 'EMBU-M FS']
top10_features_teens = ['SCL-90 DEP', 'A-DES-Ⅱ_PI', 'SCL-90 ANX', 'CSES_TS', 'EMBU-F EW', 'HEI_TS', 'SCL-90 PSY', 'A-DES-Ⅱ_DPDR', 'A-DES-Ⅱ_AII', 'EMBU-M FS']
top10_features_children = ['CSES_TS', 'HEI_TS', 'A-DES-Ⅱ_PI', 'A-DES-Ⅱ_AII', 'A-DES-Ⅱ_DPDR', 'EMBU-F EW', 'EMBU-F PUN', 'EMBU-M FS', 'CSQ_REP', 'A-SSRS_OS']


def single_plot(ax, args, uncert_path, method):
    '''Plot a single heatmap on a given Axes and return the ordered status data.'''
    # Determine feature list and table name
    if 'adults' in os.path.basename(uncert_path):
        uncertainty_list = top10_features_adults if not args.disable_top10 else adults_importance

    elif 'teens' in os.path.basename(uncert_path):
        uncertainty_list = top10_features_teens if not args.disable_top10 else teens_importance

    elif 'children' in os.path.basename(uncert_path):
        uncertainty_list = top10_features_children if not args.disable_top10 else children_importance
    else:
        raise ValueError("Unknown dataset in the file path. Please check the file name.")

        
    df_ori = pd.read_excel(uncert_path)
    df_uncert = pd.read_excel(uncert_path)

    # Clean column names
    for df in [df_ori, df_uncert]:
        df.columns = df.columns.str.replace('SCl', 'SCL', regex=True)
        df.columns = df.columns.str.replace('SCQ', 'CSQ', regex=True)

    # Sort rows by the specified method
    index_risk = df_ori.sort_values(by=method, ascending=False).index
    df_uncert = df_uncert.loc[index_risk]
    df_ori = df_ori.loc[index_risk]

    # Filter and sort columns by the importance list
    uncertainty_list = [col for col in uncertainty_list if col in df_ori.columns]
    df_uncert = df_uncert[uncertainty_list]

    # Plotting the main heatmap
    sns.heatmap(
        df_uncert,
        ax=ax,
        cmap="coolwarm",
        xticklabels=True,
        yticklabels=False,
        cbar=False
    )

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=14, rotation=90)
    # ax.set_title(f'{table_name.title()} - {method}', fontsize=16)
    
    # Return the correctly ordered status column for the side plot
    return df_ori['School Withdrawal/ Reentry Status']


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--uncert_path', type=str, help='Path to the anomaly files of three age groups', default='/Users/colin/Desktop/psycological_school_withdrawaw/risk_prob/top10')
    args.add_argument('--output_path', type=str, help='Path to save the plot', default='/Users/colin/Desktop/psycological_school_withdrawaw/risk_prob/top10/anomaly_plot')
    args.add_argument('--disable_top10', default=False, action='store_true', help='Whether to use top 10 features')
    args = args.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    tables = ['adults_anomaly.xlsx', 'teens_anomaly.xlsx', 'children_anomaly.xlsx']


    methods = ['LogisticRegression', 'RandomForest']

    # --- KEY CHANGE 1: Modify subplot creation ---
    # Create a 2-row, 6-column grid.
    fig = plt.figure(figsize=(36, 24), constrained_layout=False)
    gs = gridspec.GridSpec(
        nrows=2, 
        ncols=6, 
        figure=fig,
        width_ratios=[1, 30, 1, 30, 1, 30]
    )

    full_str = 'Full' if args.disable_top10 else 'Partial'
    gs.update(wspace=0)
    # Manually adjust spacing between columns 1-2 and 3-4

    axes = np.empty((2, 6), dtype=object)
    seq = ['g', 'h', 'i', 'j', 'k', 'l'] if full_str == 'Partial' else ['a', 'b', 'c', 'd', 'e', 'f']
    for row_idx in range(2):
        for col_idx in range(6):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            axes[row_idx, col_idx] = ax

            if col_idx % 2 == 0:  # Status columns
            # Move right by a small amount
                pos = ax.get_position()
                axes[row_idx, col_idx].set_position([
                    pos.x0 + 0.0018, pos.y0, pos.width, pos.height
                ])
            # if col_idx % 2 == 1:  add title 'Adults', 'Teens', 'Children' to the center of the heatmap
            if col_idx == 1 and row_idx == 0:
                ax.set_title('Adults', fontsize=16, pad=2, weight='bold')
            elif col_idx == 3 and row_idx == 0:
                ax.set_title('Teens', fontsize=16, pad=2, weight='bold')
            elif col_idx == 5 and row_idx == 0:
                ax.set_title('Children', fontsize=16, pad=2, weight='bold')

            ## add sequence labels to the top-left corner of status columns
            if col_idx % 2 == 0:
                ax.text(
                    0.45, 0.985, seq[row_idx * 3 + col_idx // 2],
                    transform=ax.transAxes,
                    fontsize=22, weight='bold',
                    ha='center', va='center'
                )

        #if row_idx == 0: add a title for the methods
        pos = axes[row_idx, 0].get_position()
        y_center = pos.y0 + pos.height / 2
        x_center = pos.x0 + pos.width / 2
        label_text = 'LogisticRegression' if row_idx == 0 else 'RandomForest'
        fig.text(
                pos.x0 - 0.005,                 # X-position: Adjusted to the left of the status column
                y_center,                      # Y-position: The calculated vertical center
                f'{full_str}  {label_text}',    # The label text
                va='center',                   # Vertical alignment: center
                ha='center',                   # Horizontal alignment: center
                rotation='vertical',           # Rotate the text
                fontsize=16,
                weight='bold'                  # Make the text bold
            )

    for col_idx, table in enumerate(tables):
        uncert_path = os.path.join(args.uncert_path, table)
        if not os.path.exists(uncert_path):
            continue

        for row_idx, method in enumerate(methods):
            status_ax = axes[row_idx, col_idx * 2]
            heatmap_ax = axes[row_idx, col_idx * 2 + 1]

            status_data = single_plot(heatmap_ax, args, uncert_path=uncert_path, method=method)

            status_map = {0: "#a6d96a", 1:"#1a9641" }
            status_numeric = status_data.map({0:1, 1:0}).to_numpy().reshape(-1, 1) #withdrawal:0->1, reentry:1->0

            sns.heatmap(
                status_numeric,
                ax=status_ax,
                cmap=list(status_map.values()),
                yticklabels=False,
                xticklabels=False,
                cbar=False,
                annot=False
            )
            status_ax.set_xlabel('Status', fontsize=14, rotation=90)

    # if not args.disable_top10: ### add legend only if top10 is enabled
    pos = axes[1, 5].get_position()
    legend_handles = [
        Patch(color="#1a9641", label='Withdrawal'),
        Patch(color="#a6d96a", label='Reentry')
    ]
    #place the legend at the right side of the last heatmap
    fig.legend(handles=legend_handles, fontsize=16, title='Status', bbox_to_anchor=(pos.x1 + 0.06, pos.y1 + 0.4))

    ## Customize the colorbar (cbar)
    
    # Set custom ticks and labels
    # Get colorbar limits
    # Add a general colorbar for the entire figure, not tied to any axes

    # Create a pseudo colorbar using a blank Axes and imshow
    # Place it below the heatmaps, not affecting any subplot axes

    # Add a new axes for the pseudo colorbar
    pseudo_cbar_ax = fig.add_axes([pos.x1 + 0.02, pos.y0 + 0.2, 0.02, 0.4])

    # Create a gradient image for the colorbar
    gradient = np.linspace(2, -2, 256).reshape(1, -1)
    pseudo_cbar_ax.imshow(
        gradient.T,
        aspect='auto',
        cmap='coolwarm',
        extent=[0, 1, -2, 2]
    )

    # Set custom ticks and labels for y-axis (not x-axis)
    pseudo_cbar_ax.set_yticks([-2, 0, 2])
    pseudo_cbar_ax.set_yticklabels(
        ['Negative Anomaly', 'Normal', 'Positive Anomaly'],
        fontsize=16,
        va='center',
        rotation=90
    )

    # Remove x-axis
    pseudo_cbar_ax.get_xaxis().set_visible(False)

    # Remove frame
    for spine in pseudo_cbar_ax.spines.values():
        spine.set_visible(False)

    # Make tick lines shorter
    pseudo_cbar_ax.tick_params(axis='y', length=6)

    # Add a main title
    top10_str = 'All Features' if args.disable_top10 else 'Top 10 Features'
    # fig.suptitle(f'Anomaly Heatmaps ({top10_str})', fontsize=24, y=1.02)

    # Save the complete figure
    output_filename = f'{"all" if args.disable_top10 else "top10"}_anomaly.pdf'
    output_path = os.path.join(args.output_path, output_filename)

    plt.savefig(output_path, bbox_inches='tight', format='pdf')    
    print(f"Figure saved to {output_path}")
