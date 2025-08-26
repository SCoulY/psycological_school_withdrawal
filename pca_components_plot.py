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

# --- 1. Setup & Style Configuration ---
# Use a professional, clean font common in publications
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False


def plot_single_table(file_path: str, save_path: str):
    # Data Loading and Preprocessing 
    # This section remains largely the same
    input_file = file_path #'C:/Users/SCoulY/Desktop/psycology/data/clean_adults.xlsx'
    df = pd.read_excel(input_file)
    df = df.drop(df.columns[0], axis=1)
    table_name = os.path.basename(input_file).split('.')[0]

    # Features and Labels
    feat = df.drop(columns=['状态'])
    label = df['状态'].map({"复学": 1, "休学": 0})
    label.name = 'Status' # Simplified for legend

    feat = column_name2eng(feat)

    # Scale features
    scaler = StandardScaler()
    feat_scaled = scaler.fit_transform(feat)

    # Train-test split with a random_state for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        feat_scaled, label, test_size=0.2, random_state=42, stratify=label
    )


    pca = PCA(n_components=3)
    pca.fit(X_train)
    # embedding = pca.transform(X_train)

    feature_names = feat.columns.tolist()
    components = pd.DataFrame(
        pca.components_[:3],
        columns=[f"{feature_names[i]}" for i in range(X_train.shape[1])],
        index=[f"PC{i+1}" for i in range(3)]
    )

    
    ### find the top-10 features that has the largest absolute mean loadings
    top_features = components.abs().mean(axis=0).nlargest(10)
    print(f'table:{table_name} top 10: {top_features}')



if __name__ == "__main__":
    file_list = [
        'C:/Users/SCoulY/Desktop/psycology/data/clean_adults.xlsx',
        'C:/Users/SCoulY/Desktop/psycology/data/clean_teens_wo_scl.xlsx',
        'C:/Users/SCoulY/Desktop/psycology/data/clean_teens.xlsx'
        ]
    for file_path in file_list:
        plot_single_table(file_path, save_path='C:/Users/SCoulY/Desktop/psycology/plot/')


