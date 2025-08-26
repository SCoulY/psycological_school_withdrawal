import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
import os
import umap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import hdbscan

# --- 1. Setup & Style Configuration ---
# Use a professional, clean font common in publications
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False


def plot_single_table(file_path: str):
    # --- 2. Data Loading and Preprocessing ---
    # This section remains largely the same
    input_file = file_path #'C:/Users/SCoulY/Desktop/psycology/data/clean_adults.xlsx'
    df = pd.read_excel(input_file)
    df = df.drop(df.columns[0], axis=1)
    table_name = os.path.basename(input_file).split('.')[0]

    # Features and Labels
    feat = df.drop(columns=['状态'])
    label = df['状态'].map({"复学": 1, "休学": 0})
    label.name = 'Status' # Simplified for legend

    # This is a placeholder for your actual conversion function
    def column_name2eng(df):
        # In a real scenario, you would have your column name mapping here
        # For this example, we'll assume the columns are already in English
        # or we can create dummy English names.
        df.columns = [f'Feature_{i+1}' for i in range(df.shape[1])]
        return df

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

    # --- 3. Plotting ---
    # Define a professional color palette
    colors = {0: '#D81B60', 1: '#1E88E5'} # Vivid pink and blue
    label_map = {0: 'Withdrawal', 1: 'Reentry'}

    fig = plt.figure(figsize=(12, 12))

    ari_scores = []
    silhouette_scores = []
    score_texts = []

    for dim in range(2, 11):
        # Ensure reproducibility for UMAP
        reducer = umap.UMAP(n_components=dim, random_state=42)
        embedding = reducer.fit_transform(X_train)

        #Run K-Means on the low-dimensional data
        kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(embedding)

        #run HDBSCAN on the low-dimensional data
        # hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=1, cluster_selection_epsilon=0.01)
        # cluster_labels = hdbscan_clusterer.fit_predict(embedding)
        # clustered_points = (cluster_labels != -1)
        # ari = adjusted_rand_score(y_train[clustered_points], cluster_labels[clustered_points])

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
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            score_text =ax.text(0.05, 0.95, f"Silhouette: {score:.2f}  ARI: {ari:.2f}", transform=ax.transAxes, fontsize=13)

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
            ax.set_xlabel('UMAP 1', labelpad=-4)
            ax.set_ylabel('UMAP 2', labelpad=-4)
            ax.set_zlabel('UMAP 3', labelpad=-4)
            # Clean up the 3D view
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.zaxis.set_ticklabels([])
            ax.grid(False)
            score_text = ax.text2D(0.05, 0.95, f"Silhouette: {score:.2f}  ARI: {ari:.2f}", transform=ax.transAxes, fontsize=13)

        score_texts.append(score_text)

        # --- Aesthetics for all subplots ---
        ax.set_title(f"n_components = {dim}", fontsize=12)
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
    # plt.savefig(f'C:/Users/SCoulY/Desktop/psycology/plot/umap_publication_quality.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'C:/Users/SCoulY/Desktop/psycology/plot/umap_{table_name}.svg', format='svg', bbox_inches='tight')
    # plt.show()
    plt.close(fig)  # Close the figure to free memory

if __name__ == "__main__":
    file_list = [
        'C:/Users/SCoulY/Desktop/psycology/data/clean_adults.xlsx',
        'C:/Users/SCoulY/Desktop/psycology/data/clean_teens_wo_scl.xlsx',
        'C:/Users/SCoulY/Desktop/psycology/data/clean_teens.xlsx'
        ]
    for file_path in file_list:
        plot_single_table(file_path)