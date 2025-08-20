from turtle import width
import shap
import os
from sklearn.model_selection import train_test_split
import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import numpy as np
import matplotlib.pyplot as plt
from preprocess.ch2en import column_name2eng   
import matplotlib.colors as mcolors
from sklearn.preprocessing import robust_scale
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
import matplotlib.patches as mpatches
from matplotlib.transforms import Affine2D
from collections import OrderedDict


# --- Complete feature-to-category mapping based on your Figure 1 ---
feature_to_category = OrderedDict({
# Emotional Distress
'HEI_TS': 'Emotional Distress',

# Self-Evaluation
'CSES_TS': 'Self-Evaluation',

# SCL-90 Symptoms
'SCL-90 DEP': 'SCL-90 Symptoms',
'SCL-90 ANX': 'SCL-90 Symptoms',
'SCL-90 HOS': 'SCL-90 Symptoms',
'SCL-90 PHOB': 'SCL-90 Symptoms',
'SCL-90 PAR': 'SCL-90 Symptoms',
'SCL-90 PSY': 'SCL-90 Symptoms',
'SCL-90 SOM': 'SCL-90 Symptoms',
'SCL-90 OC': 'SCL-90 Symptoms',
'SCL-90 IS': 'SCL-90 Symptoms',
'SCL-90 PST': 'SCL-90 Symptoms', # Positive Symptom Total
'SCL-90 NST': 'SCL-90 Symptoms', # Negative Symptom Total
'SCL-90 GSI': 'SCL-90 Symptoms', # Global Severity
'SCL-90 PSDI': 'SCL-90 Symptoms', # Positive Symptom Distress Index
'SCL-90 ADD': 'SCL-90 Symptoms', # Additional Symptoms
'SCL-90 TS': 'SCL-90 Symptoms', # Total Score

# Dissociative Experiences
'DES-Ⅱ_TS': 'Dissociative Exp.',
'DES-Ⅱ_DPDR': 'Dissociative Exp.',
'DES-Ⅱ_AMN': 'Dissociative Exp.', # Amnesia
'DES-Ⅱ_ABS': 'Dissociative Exp.', # Absorption
'A-DES-Ⅱ_TS': 'Dissociative Exp.',
'A-DES-Ⅱ_AII': 'Dissociative Exp.',
'A-DES-Ⅱ_PI': 'Dissociative Exp.',
'A-DES-Ⅱ_DPDR': 'Dissociative Exp.',
'A-DES-Ⅱ_DA': 'Dissociative Exp.', # Dissociative Amnesia

# Family Dynamics (EMBU)
'EMBU-F EW': 'Family Dynamics', # Father Emotional Warmth
'EMBU-F REJ': 'Family Dynamics', # Father Rejection
'EMBU-F OI': 'Family Dynamics',  # Father Over-interference
'EMBU-F PUN': 'Family Dynamics', # Father Punishment
'EMBU-F FS': 'Family Dynamics',  # Father Favoritism
'EMBU-M EW': 'Family Dynamics', # Mother Emotional Warmth
'EMBU-M REJ': 'Family Dynamics', # Mother Rejection
'EMBU-M OI': 'Family Dynamics',  # Mother Over-interference
'EMBU-M PUN': 'Family Dynamics', # Mother Punishment
'EMBU-M FS': 'Family Dynamics',  # Mother Favoritism
'EMBU-F OP': 'Family Dynamics', # Father Over-protection

# Social Support (SSRS)
'SSRS_TS': 'Social Support',
'SSRS_SS': 'Social Support',    # Subjective Support
'SSRS_OS': 'Social Support',    # Objective Support
'SSRS_SU': 'Social Support',    # Support Utilization
'A-SSRS_TS': 'Social Support',
'A-SSRS_SS': 'Social Support',  # Subjective Support
'A-SSRS_OS': 'Social Support',  # Objective Support
'A-SSRS_SU': 'Social Support',  # Support Utilization

# Coping Strategies (CSQ)
'CSQ_RAT': 'Coping Strategies', # Rationalization
'CSQ_PS': 'Coping Strategies',  # Problem Solving
'CSQ_HS': 'Coping Strategies',# Seeking Help
'CSQ_FAN': 'Coping Strategies',# Fantasy
'CSQ_SB': 'Coping Strategies',
'CSQ_REP': 'Coping Strategies',

# Demographics
'Age': 'Demographics',
'Gender': 'Demographics'}
)

# Create a color mapping for the categories
ordered_categories = [
'Emotional Distress',
'Self-Evaluation',
'SCL-90 Symptoms',
'Dissociative Exp.',
'Family Dynamics',
'Social Support',
'Coping Strategies',
'Demographics'
]

'''Empirical top 10 features from the analysis'''
top10_features_adults = ['HEI_TS', 'SCL-90 DEP', 'CSES_TS', 'SCL-90 PSY', 'SCL-90 ANX', 'EMBU-F EW', 'DES-Ⅱ_TS', 'DES-Ⅱ_ABS', 'DES-Ⅱ_AMN', 'EMBU-M FS']

top10_features_teens = ['SCL-90 DEP', 'A-DES-Ⅱ_PI', 'SCL-90 ANX', 'CSES_TS', 'EMBU-F EW', 'HEI_TS', 'SCL-90 PSY', 'A-DES-Ⅱ_DPDR', 'A-DES-Ⅱ_AII', 'EMBU-M FS']

top10_features_children = ['CSES_TS', 'HEI_TS', 'A-DES-Ⅱ_PI', 'A-DES-Ⅱ_AII', 'A-DES-Ⅱ_DPDR', 'EMBU-F EW', 'EMBU-F PUN', 'EMBU-M FS', 'CSQ_REP', 'A-SSRS_OS']



def circular_shap_plot(shap_values, 
                       feature_values, 
                       feature_names, 
                       max_display=20,
                       plot_path='shap_summary_plot.png',
                       category_map=None,
                       category_order=None,
                       category_colors=None):
    """
    Creates a circular SHAP summary plot that combines a bar plot of mean absolute SHAP values
    with a swarm plot of individual SHAP values for each feature.

    Args:
        shap_values (np.ndarray): A 2D numpy array of SHAP values (n_samples, n_features).
        feature_values (np.ndarray): A 2D numpy array of feature values (n_samples, n_features).
        feature_names (list): A list of feature names.
        max_display (int): The maximum number of features to display.
        plot_path (str): The path where the plot will be saved.
        category_map (dict, optional): A mapping of feature names to categories.
        category_order (list, optional): The order of categories to display.
        category_colors (dict, optional): A mapping of categories to colors.
    """
    # Data Preparation ---
    num_samples, num_features = shap_values.shape
    df_shap = pd.DataFrame(shap_values, columns=feature_names)
    df_features = pd.DataFrame(feature_values, columns=feature_names)
    mean_abs_shap = df_shap.abs().mean()

    # Calculate mean absolute SHAP and sort features
    # mean_abs_shap = np.abs(shap_values).mean(axis=0)
    # feature_order = np.argsort(mean_abs_shap)
    if category_map and category_order:
        # Limit features to those in the map and data
        all_mapped_features = list(category_map.keys())
        features_to_plot = [f for f in feature_names if f in all_mapped_features][:max_display]
        # print(f"Features to plot: {features_to_plot}, total {len(features_to_plot)} features")

        # Sort the selected features by importance within each category
        # Group features by category
        features_by_category = {cat: [] for cat in category_order}
        for f in features_to_plot:
            cat = category_map[f]
            features_by_category[cat].append(f)
        # Sort features within each category by mean_abs_shap (descending)
        sorted_features = []
        for cat in category_order:
            feats = features_by_category[cat]
            feats_sorted = sorted(feats, key=lambda f: mean_abs_shap[f], reverse=True)
            sorted_features.extend(feats_sorted)
        features_to_plot = sorted_features
        mean_abs_shap = mean_abs_shap[features_to_plot]
        # reorder feature_names
        feature_names = features_to_plot
        
        # Create a list of colors for the bars based on category
        bar_colors = [category_colors.get(category_map[f], 'grey') for f in features_to_plot]
        
    else: # Fallback to default sorting by importance
        features_to_plot = mean_abs_shap.sort_values(ascending=True).index.tolist()[:max_display]
        feature_names = features_to_plot
        bar_colors = '#4682B4'

    # Limit to top features
    if num_features > max_display:
        feature_order = feature_order[-max_display:]


    # Re-order all data based on the final feature list
    feature_values = df_features[features_to_plot].values
    mean_abs_shap = mean_abs_shap[features_to_plot]
    num_features = len(features_to_plot)
    shap_values = df_shap[features_to_plot].values

    # Setup Polar Plot ---
    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Set plot direction and start point
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # Define Inner and Outer Plotting Radii ---
    # The maximum possible radius, based on the largest SHAP value
    max_radius = np.abs(shap_values).max() 
    
    #  Define the radius for the central hole
    inner_hole_radius = max_radius * 0.05 # e.g., 5% of the total plot radius

    # Define the outer edge of the bar ring
    bar_ring_outer_edge = max_radius * 0.25
    
    # Calculate the available height within the ring for the bars
    bar_ring_height = bar_ring_outer_edge - inner_hole_radius

    # Define where the outer scatter plot begins, leaving a gap
    scatter_start_radius = bar_ring_outer_edge * 1.5

    # Scale Bar Heights ---
    # Scale the mean SHAP values to fit perfectly within the available bar_ring_height
    scaled_bar_heights = np.interp(
        mean_abs_shap,
        (0, mean_abs_shap.max()),
        (0, bar_ring_height*1.5) # Scale to the available height
    )

    # Calculate angles for each feature
    angles = np.linspace(0, 2 * np.pi, num_features, endpoint=False)
    
    if num_features <= 20:
        bar_width = 0.5  # Wider bars for fewer features
    else:
        bar_width = 0.1
    # Update Plotting Code ---
    # Plot the SCALED bars with the new width and bottom parameters
    ax.bar(
        angles,
        scaled_bar_heights,
        width=bar_width,  
        bottom=inner_hole_radius,  
        color=bar_colors,
        alpha=0.6,
        label='Mean |SHAP value|'
    )
    
    # Plot the scatter points, starting OUTSIDE the inner ring.
    # A baseline circle is drawn to show where SHAP=0 for the scatter points.
    ax.plot(np.linspace(0, 2 * np.pi, 100), np.full(100, scatter_start_radius), 
            color='gray', linestyle='--', linewidth=1)
    
    # # Use robust scaling for feature values to handle outliers for better color mapping
    # scaled_features = robust_scale(feature_values, quantile_range=(5, 95))
    norm = mcolors.Normalize(vmin=-1.5, vmax=1.5) # Clip color range for clarity
    cmap = plt.get_cmap('coolwarm') # Red for high, Blue for low

    for i in range(num_features):
        jitter = np.random.normal(0, 0.02, num_samples)
        feature_angles = angles[i] + jitter
        
        # Radial position is now the SHAP value ADDED to the scatter plot's start radius
        radial_positions = scatter_start_radius + shap_values[:, i]
        
        sc = ax.scatter(feature_angles, radial_positions,
                        c=feature_values[:, i], cmap=cmap, norm=norm,
                        s=11, alpha=0.75, zorder=10)

    # Finally, adjust the plot's outer limit to fit everything
    ax.set_ylim(0, scatter_start_radius + shap_values.max() * 1.1)


    # Remove grid lines and radial axis labels for a cleaner look
    ax.grid(False)
    ax.set_yticklabels([])

    # Hide the original tick labels that are causing overlaps
    ax.set_xticklabels([])
    radius_label = ax.get_rmax() * 1.2  # Set a radius for the labels to be drawn at

    # Loop through your features and draw each label manually
    for i, (angle, label_text) in enumerate(zip(angles, feature_names)):
        # Set rotation to +90 for the left side and -90 for the right side
        angle_deg = np.degrees(angle)  # Convert radians to degrees for easier handling
        if 0 <= angle_deg < 180:
            rotation = 90 - angle_deg
        elif 180 <= angle_deg <= 360:
            rotation = 270 - angle_deg
        else:
            rotation = 0

        # Alternate the radius for each label to prevent collision
        # offset = 0.10 * ((i % 2) * 2 - 1) * shap_values.max() * 0.08  # alternate +/- small offset
        ax.text(angle,
                radius_label,
                label_text,
                rotation=rotation,
                ha='center',          # Horizontally center the text
                va='center',          # Vertically center the text
                fontsize=14,
                fontweight='bold')


    # Add Group Separators and Legend ---
    if category_map and category_order:
        # Draw separator lines between groups
        group_boundaries = [category_map[f] for f in features_to_plot]
        separator_angles = []
        for i in range(num_features):
            if not i and group_boundaries[0] != group_boundaries[1] and group_boundaries[0] != group_boundaries[-1]: #place the first separator at the start
                separator_angle = angles[i] - np.pi / num_features
                ax.plot([separator_angle, separator_angle], [inner_hole_radius, ax.get_rmax()],
                        color='grey', linestyle=':', linewidth=1.2, zorder=11)
                
            elif i and group_boundaries[i] != group_boundaries[i-1]:
                # Angle is halfway between two features
                separator_angle = (angles[i] + angles[i-1]) / 2.0
                ax.plot([separator_angle, separator_angle], [inner_hole_radius, ax.get_rmax()],
                        color='grey', linestyle=':', linewidth=1.2, zorder=11)
            separator_angles.append(separator_angle)
        for i, sep in enumerate(separator_angles):
            # add a background color in between the separators
            if i < len(separator_angles) - 1:
                start_angle = sep
                end_angle = separator_angles[i + 1]
                # Fill the area between the separators with the category color as a circular sector
            if i == len(separator_angles) - 1:
                start_angle = sep
                end_angle = 2 * np.pi - np.pi / num_features

            # Calculate the bar's midpoint angle and its angular width
            midpoint_angle = (start_angle + end_angle) / 2
            angular_width = end_angle - start_angle

            # Get the color for the current group
            sector_color = category_colors.get(group_boundaries[i], 'grey')

            rmax = ax.get_rmax()

            # Define the radial height and starting point for the background bars
            bar_height = rmax - inner_hole_radius
            bar_bottom = inner_hole_radius
            # Draw a bar for the background sector
            ax.bar(
                x=midpoint_angle,
                height=bar_height,
                width=angular_width,
                bottom=bar_bottom,
                color=sector_color,
                alpha=0.2, 
                zorder=0   
            )


        # Create and display the legend for categories
        # legend_patches = [mpatches.Patch(color=color, label=label) for label, color in category_colors.items()]
        # ax.legend(
        #     handles=legend_patches,
        #     title="Feature Groups",
        #     bbox_to_anchor=(1.15, 1.05),  # Position legend outside the plot
        #     fontsize=14, title_fontsize=16
        # )

        # Add Color Bar Legend
        # Place the colorbar at the same x as the legend (x=1.15)
        cbar = fig.colorbar(
        sc, ax=ax, shrink=0.3, pad=0.08,
        location='right', anchor=(1.15, 0.5)
        )
        cbar.set_label('Feature Value', size=14, weight='bold')
        # Set colorbar ticks to actual min, 0, and max feature values
        vmin = np.min(feature_values)
        vmax = np.max(feature_values)
        cbar.set_ticks([-1.5, 0, 1.5])
        cbar.set_ticklabels([f"{vmin:.2f}", "0", f"{vmax:.2f}"])
        cbar.outline.set_visible(False)


    # --- Add the Custom Arc & Arrow Legend ---

    # 1. Define the legend's position and size using axes coordinates.
    #    This places it in the bottom-right corner of the plot.
    legend_center_x = 0.9 # Position 95% from the left edge of the figure
    legend_center_y = 0.25  # Position at the vertical center of the figure
    arc_height = 0.1       # The height of the arc

    # 3. Create the arc and arrow patches.
    arc = mpatches.Arc(
        (legend_center_x, legend_center_y-arc_height/2),  # Center of the arc
        width=0.1,
        height=arc_height,
        angle=90,
        theta1=-90,
        theta2=90,
        linewidth=2.5,
        edgecolor='dimgray',
        linestyle='--',  # Uncomment if you want a dashed arc
    )

    arrow = mpatches.FancyArrowPatch(
        (legend_center_x, legend_center_y-arc_height/2),  # Arrow tail
        (legend_center_x, legend_center_y+arc_height/2),  # Arrow head
        mutation_scale=5,
        arrowstyle='<->,head_length=3,head_width=3',
        color='dimgray'
    )

    # 4. Add the patches directly to the FIGURE, not the axes.
    fig.add_artist(arc)
    fig.add_artist(arrow)

    # 5. Add the text labels for the legend.
    fig.text(
        legend_center_x - 0.07,  
        legend_center_y - arc_height / 1.5,
        'Toward Withdrawal',
        ha='left',
        va='center',
        fontsize=14,
        fontweight='bold'
        )

    fig.text(
        legend_center_x - 0.05,
        legend_center_y + arc_height / 1.5,
        'Toward Reentry',
        ha='left',
        va='center',
        fontsize=14,
        fontweight='bold'
        )

    # Save the plot
    # plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor='white', transparent=True)
    plt.savefig(plot_path, format='svg', bbox_inches="tight", facecolor='white', transparent=True)
    plt.close()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--file_path', type=str, help='Path to the excel file', default='C:/Users/SCoulY/Desktop/psycology/data')
    args.add_argument('--ckpt_path', type=str, help='Path to the checkpoint files', default='C:/Users/SCoulY/Desktop/psycology/ckpt_5runs/children_correct/full_wo_gender/children/')
    args.add_argument('--plot_path', type=str, help='Path to save the plot', default='C:/Users/SCoulY/Desktop/psycology/plot/shap_plot_avg_children_wo_gender')
    args = args.parse_args()


    os.makedirs(args.plot_path, exist_ok=True)

    LR = [] #list to store 5-fold logistic regression SHAP values, file_path, model_name, if_top10
    LR_top10 = [] #list to store 5-fold logistic regression SHAP values, file_path, model_name, if_top10
    RF = [] #list to store 5-fold random forest SHAP values, file_path, model_name, if_top10
    RF_top10 = [] #list to store 5-fold random forest SHAP values, file_path, model_name, if_top10
    LR_features = [] #list to store feature names
    RF_features = [] #list to store feature names
    LR_top10_features = [] #list to store top10 feature names
    RF_top10_features = [] #list to store top10 feature names

    # print(f"Checkpoint files found: {os.listdir(args.ckpt_path)}")
    for ckpt in os.listdir(args.ckpt_path):
        if '.pkl' not in ckpt or 'SVM' in ckpt:
            continue
        model_name = ckpt.split('acc')[0][:-1].split('_')[-1] #LogisticRegression

        file_name = ckpt.split(model_name)[0][:-1] #clean_adults
        file_path = os.path.normpath(os.path.join(args.file_path, file_name+'.csv'))

        df = pd.read_csv(file_path)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        # df = column_name2eng(df)
        if 'top10' not in ckpt:
            X = df.drop(columns=['School Withdrawal/ Reentry Status'])
        else:
            # Your logic for selecting top 10 features
            current_top10 = top10_features_adults
            if 'teens' in file_path:
                current_top10 = top10_features_teens
            if 'children' in file_path:
                current_top10 = top10_features_children
            X = df[current_top10]

        Y = df['School Withdrawal/ Reentry Status']

        model = joblib.load(os.path.normpath(os.path.join(args.ckpt_path, ckpt)))

        feature_names = X.columns.tolist()

        scaler = StandardScaler()
        feat = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(feat, Y, test_size=0.2, random_state=42)

        if 'LogisticRegression' in ckpt:
            explainer = shap.Explainer(model, X_test)
            shap_values_obj = explainer(X_test)

        else: # tree-based models
            explainer = shap.Explainer(model)
            shap_values_obj = explainer(X_test)[:,:,1]


        # Extract the raw 2D numpy arrays needed for the plot
        # For binary classification, we are interested in the SHAP values for the positive class (e.g., '复学' which is 1)
        if len(shap_values_obj.values.shape) == 3: # For tree-based models with 2 outputs
            shap_values_np = shap_values_obj.values[:,:,1]
        else: # For linear models or other models with 1 output
            shap_values_np = shap_values_obj.values
        
        # The feature values are simply X_test
        X_test_np = X_test

        # --- VISUALIZATION ---
        # Define save path
        if_top10 = True if "top10" in ckpt else False

        if 'LogisticRegression' in ckpt:
            if if_top10:
                LR_top10.append((shap_values_np, file_path, model_name, if_top10))
                LR_top10_features.append((feature_names, X_test_np))
            else:
                LR.append((shap_values_np, file_path, model_name, if_top10))
                LR_features.append((feature_names, X_test_np))
        else:
            if if_top10:
                RF_top10.append((shap_values_np, file_path, model_name, if_top10))
                RF_top10_features.append((feature_names, X_test_np))
            else:
                RF.append((shap_values_np, file_path, model_name, if_top10))
                RF_features.append((feature_names, X_test_np))

    # Combine SHAP values across folds

    LR_shaps = [shap_values for shap_values, _, _, _ in LR]
    RF_shaps = [shap_values for shap_values, _, _, _ in RF]
    LR_top10_shaps = [shap_values for shap_values, _, _, _ in LR_top10]
    RF_top10_shaps = [shap_values for shap_values, _, _, _ in RF_top10]
    # Average the SHAP values across folds
    LR_shap_values = np.mean(LR_shaps, axis=0)
    RF_shap_values = np.mean(RF_shaps, axis=0)
    LR_top10_shap_values = np.mean(LR_top10_shaps, axis=0)
    RF_top10_shap_values = np.mean(RF_top10_shaps, axis=0)

    table_names = [table_name.split('.')[0].split(os.sep)[-1] for _, table_name, _, _ in [LR[0], RF[0], LR_top10[0], RF_top10[0]]]
    model_names = [model_name for _, _, model_name, _ in [LR[0], RF[0], LR_top10[0], RF_top10[0]]]
    if_top10s = [if_top10 for _, _, _, if_top10 in [LR[0], RF[0], LR_top10[0], RF_top10[0]]]
    shaps_dict = {
                    'LogisticRegression': LR_shap_values,
                    'RandomForest': RF_shap_values,
                    'LogisticRegression_top10': LR_top10_shap_values,
                    'RandomForest_top10': RF_top10_shap_values
                }
    
    feature_dict = {
                    'LogisticRegression': LR_features[0],
                    'RandomForest': RF_features[0],
                    'LogisticRegression_top10': LR_top10_features[0],
                    'RandomForest_top10': RF_top10_features[0]
                }

    for table_name, model_name, if_top10 in zip(table_names, model_names, if_top10s):
        top10_str = '_top10' if if_top10 else ''
        shap_key = f"{model_name}{top10_str}"
        shap_np = shaps_dict[shap_key]
        feature_names, X_test_np = feature_dict[shap_key]

        plot_save_path = os.path.join(args.plot_path, f"{table_name}_{model_name}_{top10_str}.svg")

        circular_shap_plot(shap_np, 
                            X_test_np, 
                            feature_names, 
                            max_display=50, 
                            plot_path=plot_save_path,
                            category_map=feature_to_category,
                            category_order=ordered_categories,
                            category_colors={
                                'Emotional Distress': '#4878d0',  
                                'Self-Evaluation': '#ee854a',      # Steel Blue
                                'SCL-90 Symptoms': '#6acc64',      # Lime Green
                                'Dissociative Exp.': '#d65f5f',    # Gold
                                'Family Dynamics': '#956cb4',      # Blue Violet
                                'Social Support': '#8c613c',       # Hot Pink
                                'Coping Strategies': '#dc7ec0',    # Light Sea Green
                                'Demographics': '#797979'          # Light Gray
                            })

        print(f"Saved circular SHAP plot to: {plot_save_path}")