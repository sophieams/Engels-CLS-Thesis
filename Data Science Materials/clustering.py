"""
This script is responsible for the clustering parts of the thesis.
Here we can both choose our features of interest or generate them from a grid search.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import type_metric, distance_metric
from pyclustering.cluster.center_initializer import random_center_initializer
from pyclustering.cluster.silhouette import silhouette
from matplotlib.ticker import MaxNLocator
from itertools import combinations

# Plotting preferences
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 28,
    "axes.titlesize": 28,
    "axes.labelsize": 28,
    "xtick.labelsize": 28,
    "ytick.labelsize": 28,
    "legend.fontsize": 28,
    "image.cmap": "Blues",
    "lines.linewidth": 1.5,
    "lines.markersize": 10,
    "text.usetex": True, "mathtext.fontset": "cm",
    "pgf.preamble": r"\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage{cmbright}"
})

def extract_column_features(df, unique_id):
    """
    This function extracts features of a particular unique_id's data.
    Input:
    - df: the dataframe with all of the data
    - unique_id: the particular subject ID for filtering.
    Output:
    - a dataframe with the calculated features for that individual.
    """
    individual_data = df[df["Nr"] == unique_id].copy()
    individual_data = individual_data.apply(pd.to_numeric, errors='coerce')
    exclude_cols = ["Nr"]
    feature_columns = [col for col in individual_data.columns if col not in exclude_cols]
    individual_data = individual_data[feature_columns]
    features = {"ID": unique_id}
    for col in feature_columns:
        col_data = individual_data[col].dropna()
        if col_data.empty:
            continue

        features[f"{col}_Variance"] = np.var(col_data)
        features[f"{col}_Skewness"] = col_data.skew()
        features[f"{col}_ZeroCrossing"] = np.count_nonzero(np.diff(np.sign(col_data - col_data.mean())))
        features[f"{col}_RMS"] = np.sqrt(np.mean((col_data - col_data.mean())**2))
        features[f"{col}_Mean"] = np.mean(col_data)
        features[f"{col}_Rolloff"] = np.percentile(col_data, 85)
        features[f"{col}_Kurtosis"] = col_data.kurtosis()
        features[f"{col}_Irregularity"] = np.std(np.diff(col_data)).mean()

    return pd.DataFrame([features])

def kmeans_clustering(data, selected_features, k, plot_pca=False):    
    """
    This function performs the k-means clustering with our desired features.
    Input:
    - data: the dataframe with our subject's data
    - selected_features: the features we want to cluster from.
    - k: the number of clusters we want
    - plot_pca: boolean for plotting the clusters or not.
    Output:
    - silhouette score: measure of quality of clustering
    - clustered data: data with cluster assignment
    """
    X = data[selected_features]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    custom_metric = distance_metric(type_metric.EUCLIDEAN)
    initial_centers = random_center_initializer(X_scaled, k).initialize()
    kmeans_instance = kmeans(X_scaled, initial_centers, metric=custom_metric)
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    final_sil_score = np.mean(np.array(silhouette(X_scaled, clusters).process().get_score()))

    labels = np.zeros(len(X_scaled))
    for cluster_idx, cluster_points in enumerate(clusters):
        labels[cluster_points] = cluster_idx

    clustered_data = data.copy()
    clustered_data["Cluster"] = labels
    
    if plot_pca:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        plot_df = pd.DataFrame(X_pca, columns=["PCA1", "PCA2"])
        plot_df["Cluster"] = labels

        plt.figure(figsize=(15, 7.5))
        sns.scatterplot(x="PCA1", y="PCA2", hue="Cluster", data=plot_df, palette="Set2")
        plt.legend().remove()
        plt.title(f"K-Means Clustering - Grid Search Feature Selection")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        # Uncomment the following line to show legend for clusters
        # plt.legend(title="Cluster", bbox_to_anchor=(1, 1.015), loc='upper left')
        plt.tight_layout()
        plt.show()

    return final_sil_score, clustered_data

def analyze_clusters(clustered_data, selected_features, significance_threshold=0.05, max_plots=2):
    """
    Function to analyze the statistical properties of the data in each cluster. 
    For non-normal data, we use the Mann-Whitney U test to determine differences between two distributions.
    If the data is normal, we use ANOVA.
    Finally the feature distributions are plotted for each cluster.
    Input: 
    - clustered_data: from kmeans_clustering function; data with cluster assignments
    - selected_features: the features we want to analyze.
    - significance_threshold: the p-value we will determine normality from
    - max_plots: the number of feature distributions we want to plot.
    Returns nothing.
    """
    exclude_cols = ["ID", "Cluster"]
    feature_columns = [col for col in clustered_data.columns if col in selected_features and col not in exclude_cols]
    test_results = {}
    unique_clusters = clustered_data["Cluster"].unique()
    cluster1, cluster2 = unique_clusters

    for feature in feature_columns:
        group1 = clustered_data[clustered_data["Cluster"] == cluster1][feature].dropna()
        group2 = clustered_data[clustered_data["Cluster"] == cluster2][feature].dropna()

        # Normality check
        normal1 = stats.shapiro(group1)[1] > significance_threshold if len(group1) > 2 else False
        normal2 = stats.shapiro(group2)[1] > significance_threshold if len(group2) > 2 else False

        if normal1 and normal2:
            test_type = "ANOVA"
            _, p_value = stats.f_oneway(group1, group2)
        else: 
            test_type = "Mann-Whitney"
            _, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')

        test_results[feature] = {"p-value": p_value, "Test": test_type}

    test_results_df = pd.DataFrame.from_dict(test_results, orient="index")
    test_results_df = test_results_df.sort_values("p-value")
    significant_features = test_results_df[test_results_df["p-value"] < significance_threshold].index.tolist()

    print("\nStatistical Test Results")
    print(test_results_df[test_results_df["p-value"] < significance_threshold])

    features_to_plot = significant_features[:max_plots]
    num_plots = len(features_to_plot)
    _, axes = plt.subplots(1, num_plots, figsize=(15, 5), sharey=True)
    if num_plots == 1:
        axes = [axes]
    for ax, feature in zip(axes, features_to_plot):
        sns.histplot(data=clustered_data, x=feature, hue="Cluster", bins=30, 
                     kde=False, element="step", stat="density", alpha=0.5, ax=ax, palette="Set2")
        ax.set_title(f"{feature} by Cluster")
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
        ax.get_legend().remove()
    axes[0].set_title("Skewness by Cluster")
    axes[0].set_xlabel("Skewness")
    axes[1].set_title("Kurtosis by Cluster")
    axes[1].set_xlabel("Kurtosis")
    # The following lines do not work with the fake data, but would perhaps with different data.
    # axes[2].set_title("Zerocross by Cluster")
    # axes[2].set_xlabel("Zerocross")
    cluster_labels = clustered_data["Cluster"].unique()
    cluster_labels.sort()
    plt.tight_layout()
    plt.show()
    
def cluster_by_symptom(df_original, clustered_data, significance_threshold=0.05):
    """
    This function analyzes differences between data from different cluster assignments for each symptom.
    Input: 
    - df_original: the full original dataframe.
    - clustered_data: the data with cluster assignments.
    - significance_threshold: normality p-value
    Returns nothing.
    """
    symptom_labels = ['Energie', 'Valenz', 'Ruhe', 'PA', 'Angst', 'Depr', 
                       'PB', 'TB', 'Hoff', 'suge']

    avg_per_id = df_original.groupby("Nr")[symptom_labels].mean().reset_index()
    avg_per_id.rename(columns={"Nr": "ID"}, inplace=True) 
    avg_per_id = avg_per_id.merge(clustered_data[["ID", "Cluster"]], on="ID", how="left")
    df_long = avg_per_id.melt(id_vars=["ID", "Cluster"], value_vars=symptom_labels, var_name="Symptom", value_name="Average Value")

    test_results = {}
    for symptom in symptom_labels:
        group1 = df_long[(df_long["Cluster"] == 0) & (df_long["Symptom"] == symptom)]["Average Value"]
        group2 = df_long[(df_long["Cluster"] == 1) & (df_long["Symptom"] == symptom)]["Average Value"]
        
        # Normality check
        normal1 = stats.shapiro(group1)[1] > significance_threshold if len(group1) > 2 else False
        normal2 = stats.shapiro(group2)[1] > significance_threshold if len(group2) > 2 else False
        if normal1 and normal2:
            test_type = "T-Test"
            _, p_value = stats.ttest_ind(group1, group2, equal_var=False)
        else:
            test_type = "Mann-Whitney"
            _, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')

        test_results[symptom] = {"p-value": p_value, "Test": test_type}

    test_results_df = pd.DataFrame.from_dict(test_results, orient="index")
    test_results_df = test_results_df.sort_values("p-value")

    # Print the statistical test results
    print("\nStatistical Test Results:")
    print(test_results_df[test_results_df["p-value"] < significance_threshold])
    plt.figure(figsize=(15, 7.5))
    sns.stripplot(
        x="Symptom", y="Average Value", hue="Cluster",
        data=df_long, palette="Set2", alpha=0.7, jitter=True, dodge=True, zorder=1
    )
    sns.boxplot(
        x="Symptom", y="Average Value", hue="Cluster", data=df_long, 
        palette="Set2", showcaps=True, boxprops={'alpha': 0.4}, 
        whiskerprops={'alpha': 0.4}, flierprops={'marker': 'o', 'markersize': 2, 'alpha': 0.1},
        linewidth=1.5, dodge=True, showfliers=False, zorder=2
    )
    plt.xlabel("")
    plt.xticks(rotation=45, ticks=range(len(symptom_labels)), labels=['Energy', 'Valence', 'Calmness', 'PA', 'Anxiety',
                                                                       'Depression', 'PB', 'TB', 'Hopelessness', 'SI']) 
    plt.ylabel("Mean Symptom Value")
    plt.title("Mean Symptom Values by Cluster")
    ax = plt.gca()
    ax.legend_.remove()
    plt.tight_layout()
    plt.show()

def cluster_by_error(clustered_data, gnn_type):
    """
    This function analyzes differences between GNN errors for each cluster and symptom.
    Input: 
    - clustered_data: the data with cluster assignments.
    - gnn_type: either "AGCRN" or "MTGNN"
    Returns nothing.
    """
    df_errors_mtgnn = pd.read_csv("analysis_data/gnn_learned_seq_length_5_10_nodes.csv")
    df_errors_agcrn = pd.read_csv("analysis_data/agcrn_seq_length_5_predictions_10_nodes.csv")
    df_errors_mtgnn["Errors"] = df_errors_mtgnn["Normalized_Errors"].apply(eval)
    df_errors_agcrn["Errors"] = df_errors_agcrn["Normalized_Errors"].apply(eval)

    if gnn_type == "MTGNN":
        df_errors = df_errors_mtgnn
    elif gnn_type == "AGCRN":
        df_errors = df_errors_agcrn

    df_errors = df_errors.merge(clustered_data[["ID", "Cluster"]], on="ID", how="left")

    symptom_labels = ['Energy', 'Valence', 'Calmness', 'PA', 'Anxiety', 'Depression', 
                      'PB', 'TB', 'Hopelessness', 'SI']
    error_data = []
    for _, row in df_errors.iterrows():
        unique_id = row["ID"]
        error_values = row["Errors"]
        cluster_label = row["Cluster"]
        for i, symptom in enumerate(symptom_labels):
            error_data.append({
                "ID": unique_id,
                "Symptom": symptom,
                "Error": np.abs(error_values[i]),
                "Cluster": cluster_label
            })

    # Average across the ten runs for each individual and symptom
    df_plot = pd.DataFrame(error_data)
    df_plot = df_plot.groupby(["ID", "Symptom", "Cluster"], as_index=False)["Error"].mean()
    plt.figure(figsize=(15, 7.5))
    sns.stripplot(
        x="Symptom", y="Error", hue="Cluster", data=df_plot,
        palette="Set2", alpha=0.7, jitter=True, dodge=True, zorder=1, order=symptom_labels
    )
    sns.boxplot(
        x="Symptom", y="Error", hue="Cluster", data=df_plot,
        palette="Set2", showcaps=True, boxprops={'alpha': 0.4},
        whiskerprops={'alpha': 0.4}, flierprops={'marker': 'o', 'markersize': 2, 'alpha': 0.1},
        linewidth=1.5, dodge=True, showfliers=False, zorder=2, order=symptom_labels
    )

    plt.xticks(rotation=45)
    plt.xlabel("")
    plt.ylabel("Absolute Prediction Error")
    plt.title(f"{gnn_type} Mean Symptom Errors per Cluster")
    ax = plt.gca()
    ax.legend_.remove()
    plt.tight_layout()
    plt.show()

def grid_search(df, feature_list, sequence_lengths, k_values, top_n=10):
    """
    Performs a grid search with suicidal ideation features, size of cluster, and sequence length
    to optimize the silhouette score.
    Input:
    - df: the original dataframe
    - feature_list: the features we want to combine
    - sequence_lengths: the different time-series lengths we want to combine. (numpy array)
    - k_values: the different number s of clusters we want to look at. (numpy  array)
    - top_n: the number of combinations we want to display.
    Output:
    - top_results: the top_n combinations of features with the best silhouette scores.
    """
    feature_combinations = []
    combination_sizes = [3]
    for k in combination_sizes:
        feature_combinations.extend(combinations(feature_list, k))
    
    results = []
    for seq_length in sequence_lengths:
        print(f"\nProcessing Sequence Length: {seq_length}")
        feature_data = pd.concat([
            extract_column_features(df[df["Nr"] == uid].tail(seq_length), uid)
            for uid in df["Nr"].unique()
        ], ignore_index=True)
        
        for feature_combo in feature_combinations:
            feature_names = list(feature_combo)
            
            for k in k_values:
                print(f"Feature Combination: {feature_names} with K = {k}")
                sil_score, _ = kmeans_clustering(feature_data, feature_names, k)
                results.append({
                    "Sequence Length": seq_length,
                    "Features": ', '.join(feature_names),
                    "K": k,
                    "Silhouette Score": sil_score
                })

    results_df = pd.DataFrame(results)
    top_results = results_df.sort_values(by="Silhouette Score", ascending=False).head(top_n)
    print(top_results)

    return top_results
    
def clustering_analysis(df_original, feature_data, selected_features, k=2, significance_threshold=0.05, plot_pca=False):
    """
    Helper function to combine all the analyses at once.
    Input:
    - df_original: full original dataframe
    - feature_data: statistical feature dataframe calculated from df_original
    - selected_features: the features we are using to cluster
    - k: the number of clusters we are interested in.
    - significance_threshold: our normality p-value
    - plot_pca: boolean for plotting the clusters or not.
    Returns nothing.
    """
    sil_score, clustered_data = kmeans_clustering(feature_data, selected_features, k, plot_pca=plot_pca)
    print(f"Silhouette Score: {sil_score:.4f}")

    analyze_clusters(clustered_data, selected_features, significance_threshold=significance_threshold)

    cluster_by_symptom(df_original, clustered_data, significance_threshold=significance_threshold)

    for gnn_type in ["MTGNN", "AGCRN"]:
        cluster_by_error(clustered_data, gnn_type)


# Example usage
def main(): 
    df = pd.read_csv("data.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')
    relevant_columns = ['Nr', 'Energie', 'Valenz', 'Ruhe', 'PA', 'Angst', 'Depr', 
                        'PB', 'TB', 'Hoff', 'suge']
    df = df[relevant_columns]
    df = df.dropna(axis=1, how='all')
    df = df.dropna()
    feature_data = pd.concat(
        [extract_column_features(df[df["Nr"] == uid].tail(55), uid) for uid in df["Nr"].unique()],
        ignore_index=True
    )
    # Example of features for clustering
    selected_features = ['suge_Skewness', 'suge_Kurtosis', 'suge_ZeroCrossing']
    clustering_analysis(df_original=df, feature_data=feature_data, selected_features=selected_features, k=2, plot_pca=True)

main()