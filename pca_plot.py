from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import argparse
from preprocess.ch2en import column_name2eng
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

import seaborn as sns
import numpy as np
import os
from sklearn.model_selection import train_test_split

# --- Plotting Setup (unchanged) ---
plt.style.use('default') 
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": "DejaVu Sans", "axes.unicode_minus": False,
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 12, "xtick.labelsize": 9, "ytick.labelsize": 9, "figure.dpi": 300,
})


def plot_single_table(file_path: str, save_path: str):
    # Data Loading and Preprocessing 
    # This section remains largely the same
    input_file = file_path #'C:/Users/SCoulY/Desktop/psycology/data/clean_adults.xlsx'
    df = pd.read_csv(input_file)

    table_name = os.path.basename(input_file).split('.')[0]

    # Features and Labels
    feat = df.drop(columns=['School Withdrawal/ Reentry Status'])
    label = df['School Withdrawal/ Reentry Status']
    label.name = 'Status' # Simplified for legend

    feat = column_name2eng(feat)

    # Scale features
    scaler = StandardScaler()
    feat_scaled = scaler.fit_transform(feat)

    # Train-test split with a random_state for reproducibility
    # X_train, X_test, y_train, y_test = train_test_split(
    #     feat_scaled, label, test_size=0.2, random_state=42, stratify=label
    # )
    X_train = feat_scaled
    y_train = label

    # Plotting
    # Define a professional color palette
    colors = {0: '#D81B60', 1: '#1E88E5'} # Vivid pink and blue
    label_map = {0: 'Withdrawal', 1: 'Reentry'}

    fig = plt.figure(figsize=(12, 12))

    ari_scores = []
    silhouette_scores = []
    score_texts = []

    for dim in range(2, 11):

        pca = PCA(n_components=dim)
        pca.fit(X_train)
        embedding = pca.transform(X_train)

        # Run K-Means on the low-dimensional data
        kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(embedding)

        # --- Performance Metrics ---
        # Compare K-Means clusters to true labels
        ari = adjusted_rand_score(y_train, cluster_labels)
        
        # --- Subplot Configuration ---
        plot_index = dim - 1
        # Compute and display silhouette score
        score = silhouette_score(embedding[:, :min(embedding.shape[1], 10)], y_train)
        ari_scores.append(ari)
        silhouette_scores.append(score)

        if dim == 2:
            # For 2 components, create a 2D scatter plot
            ax = fig.add_subplot(3, 3, plot_index)
            ax.scatter(
                embedding[:, 0], embedding[:, 1],
                c=[colors[x] for x in y_train],
                s=25,
                alpha=0.7,
                edgecolors='k',
                linewidth=0.3
            )
            ax.set_xlabel('PCA 1')
            ax.set_ylabel('PCA 2')
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            score_text = ax.text(0.05, 0.95, f"Silhouette: {score:.2f}  ARI: {ari:.2f}", transform=ax.transAxes, fontsize=13)

        else:
            # For 3+ components, create a 3D scatter plot of the first 3 dimensions
            ax = fig.add_subplot(3, 3, plot_index, projection='3d')
            ax.scatter(
                embedding[:, 0], embedding[:, 1], embedding[:, 2],
                c=[colors[x] for x in y_train],
                s=25,
                alpha=0.6,
                edgecolors='k',
                linewidth=0.2
            )
            ax.set_xlabel('PCA 1', labelpad=-4)
            ax.set_ylabel('PCA 2', labelpad=-4)
            ax.set_zlabel('PCA 3', labelpad=-4)
            # Clean up the 3D view
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.zaxis.set_ticklabels([])
            ax.grid(False)
            score_text = ax.text2D(0.05, 0.95, f"Silhouette: {score:.2f}  ARI: {ari:.2f}", transform=ax.transAxes, fontsize=13)

        score_texts.append(score_text)
        # --- Aesthetics for all subplots ---
        ax.set_title(f"n_components = {dim}", fontsize=14)
        # Remove top and right borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Style ticks
        ax.tick_params(axis='both', which='major', labelsize=8, width=0.75)


    # --- 4. Legend and Final Touches ---
    # Create a single, clear legend for the entire figure
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                        markerfacecolor=colors[k], markersize=8,
                        label=label_map[k]) for k in label_map]
    fig.legend(handles=handles, title=label.name, loc='upper right', bbox_to_anchor=(0.95, 0.95), fontsize=10, title_fontsize=13)

    # if highest ari_score highlight the subplot title and score text using bold font
    max_ari = max(ari_scores)
    max_index = ari_scores.index(max_ari) 
    ax = fig.axes[max_index]  # Get the subplot with the highest
    score_texts[max_index].remove()  # Remove the previous text
    ax.set_title(f"n_components = {max_index+2}", fontsize=13, fontweight='bold')
    if max_index == 0:
        ax.text(0.05, 0.95, f"Silhouette: {silhouette_scores[max_index]:.2f}  ARI: {max_ari:.2f}", transform=ax.transAxes, fontsize=14, fontweight='bold')
    else:
        ax.text2D(0.05, 0.95, f"Silhouette: {silhouette_scores[max_index]:.2f}  ARI: {max_ari:.2f}", transform=ax.transAxes, fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make space for the legend
    plt.savefig(f'{save_path}/pca_{table_name}.svg', format='svg', bbox_inches='tight')
    # plt.show()
    plt.close(fig)  # Close the figure to free memory

    ### plot PCA loads as heatmap
    # # Get the components (eigenvectors)
    feature_names = feat.columns
    components = pd.DataFrame(
        pca.components_[:3],
        columns=[f"{feature_names[i]}" for i in range(X_train.shape[1])],
        index=[f"PC{i+1}" for i in range(3)]
    )

    # print(components)


    vmin = np.min(components.values)
    vmax = np.max(components.values)
    vmid = (vmin + vmax) / 2
    cbar_ticks = [vmin, vmid, vmax]
    plt.figure(figsize=(20, 4)) # width, height in inches
    ax = sns.heatmap(
        components, 
        cmap='coolwarm', 
        annot=True, 
        fmt=".2f", 
        xticklabels=True,
        cbar_kws={'pad': 0.01, 'ticks': cbar_ticks, 'format': '%.2f'} # Adjust this value to control distance
    )
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout()
    plt.savefig(f'{save_path}/pca_loads_{table_name}.svg', format="svg")

if __name__ == "__main__":
    file_list = [
        '/Users/colin/Desktop/psycological_school_withdrawaw/data/clean_adults.csv',
        '/Users/colin/Desktop/psycological_school_withdrawaw/data/clean_children.csv',
        '/Users/colin/Desktop/psycological_school_withdrawaw/data/clean_teens.csv'
        ]
    output_dir = '/Users/colin/Desktop/psycological_school_withdrawaw/plot_correct_pca'
    os.makedirs(output_dir, exist_ok=True)
    for file_path in file_list:
        plot_single_table(file_path, save_path=output_dir)


