"""
Script for plotting the results from the GNNs.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyreadstat

# Plotting preferences
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 28,
    "axes.titlesize": 28,
    "axes.labelsize": 28,
    "xtick.labelsize": 28,
    "ytick.labelsize": 28,
    "legend.fontsize": 24,
    "image.cmap": "Blues",
    "lines.linewidth": 1.5,
    "lines.markersize": 10,
    "text.usetex": True, "mathtext.fontset": "cm",
    "pgf.preamble": r"\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage{cmbright}"
})

def plot_individual_predictions(id):
    """
    This function plots the next step predicted and true values of symptoms.
    The plot is then saved to the plots folder. 
    Input:
    - id: the id of the subject we want to plot the values for.
    """
    df_mtgnn = pd.read_csv("analysis_data/gnn_learned_seq_length_5_predictions_1.csv")
    df_agcrn = pd.read_csv("analysis_data/agcrn_seq_length_5_predictions.csv")

    # Convert string lists back to Python lists
    df_mtgnn["Predicted_Values"] = df_mtgnn["Predicted_Values"].apply(eval)
    df_mtgnn["Errors"] = df_mtgnn["Errors"].apply(eval)
    df_mtgnn["True_Values"] = df_mtgnn["True_Values"].apply(eval)
    
    df_agcrn["Predicted_Values"] = df_agcrn["Predicted_Values"].apply(eval)
    df_agcrn["Errors"] = df_agcrn["Errors"].apply(eval)

    predictions_mtgnn = df_mtgnn.loc[df_mtgnn["ID"] == id, "Predicted_Values"].values[0]
    true_values = df_mtgnn.loc[df_mtgnn["ID"] == id, "True_Values"].values[0]
    predictions_agcrn = df_agcrn.loc[df_agcrn["ID"] == id, "Predicted_Values"].values[0]

    plt.figure(figsize=(15, 7.5))
    plt.scatter(range(len(true_values)), true_values, label="True Value", color="green", marker="o")
    plt.scatter(range(len(predictions_mtgnn)), predictions_mtgnn, label="MTGNN Predicted", color="red", marker="^")
    plt.scatter(range(len(predictions_agcrn)), predictions_agcrn, label="AGCRN Predicted", color="blue", marker="X")
    node_labels = ['Energy', 'Valence', 'Calmness', 'PA', 'Anxiety', 'Depression', 
                       'PB', 'TB', 'Hopelessness', 'SI']
    plt.xticks(ticks=range(len(true_values)), labels=node_labels, rotation=45)
    plt.yticks()
    plt.ylabel("Value")
    plt.title(f"Last Step Predictions for all Symptoms")
    plt.legend()
    # If you want the legend outside of the plot, uncomment the next line.
    #plt.legend(bbox_to_anchor=(1, 1.015), loc='upper left', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"plots/{id}_GNN_comparison.png")
    plt.close()

def plot_error_distributions():
    """
    Plots violin plots of normalized prediction error for each GNN type and each symptom. 
    The data for each plot is directly inputted in the function.
    No input.
    Output:
    - summary_df: dataframe with each symptom's and GNN's associated MSE and standard error.
    """
    df_mtgnn = pd.read_csv("analysis_data/gnn_learned_seq_length_5_10_nodes.csv")
    df_agcrn = pd.read_csv("analysis_data/agcrn_seq_length_5_predictions_10_nodes.csv")
    df_mtgnn["Errors"] = df_mtgnn["Normalized_Errors"].apply(eval)
    df_agcrn["Errors"] = df_agcrn["Normalized_Errors"].apply(eval)
    symptom_labels = ['Energy', 'Valence', 'Calmness', 'PA', 'Anxiety', 'Depression', 
                      'PB', 'TB', 'Hopelessness', 'SI']
    error_data = []
    for _, row in df_agcrn.iterrows():
        for i, error_value in enumerate(row["Errors"]):
            error_data.append({
                "Symptom": symptom_labels[i],
                "Error": error_value,
                "GNN": "AGCRN"
            })
            
    for _, row in df_mtgnn.iterrows():
        for i, error_value in enumerate(row["Errors"]):
            error_data.append({
                "Symptom": symptom_labels[i],
                "Error": error_value,
                "GNN": "MTGNN"
            })
            
    error_df = pd.DataFrame(error_data)
    plt.figure(figsize=(15, 7.5))
    sns.violinplot(
        x="Symptom", y="Error", hue="GNN", data=error_df, 
        split=True, palette={"AGCRN": "blue", "MTGNN": "red"}
    )
    plt.xticks(rotation=45)
    plt.xlabel("")
    plt.ylabel("Normalized Prediction Error")
    plt.title("GNN Error Distributions for Each Symptom")
    plt.legend(bbox_to_anchor=(0.5, -0.4), loc='upper center', ncol=2)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.35)
    plt.show()
    
    summary_df = error_df.groupby(['GNN', 'Symptom'])['Error'].agg(['mean', 'sem']).reset_index()
    summary_df['Mean ± SE'] = summary_df.apply(
        lambda row: f"{row['mean']:.3f} ± {row['sem']:.3f}", axis=1
    )
    latex_table = summary_df.pivot(index='Symptom', columns='GNN', values='Mean ± SE').reset_index()
    print("\nSummary Statistics Table:")
    print(latex_table.to_latex(index=False, caption="GNN Mean Prediction Errors", label="tab:error_stats"))
    return summary_df

def plot_mse_for_nodes():
    """
    Small function for plotting the mean squared error for each subgraph size in the MTGNN.
    No input or output.
    """
    new_df = pd.DataFrame()
    new_df['num_nodes'] = list(range(1, 11))
    avg_mse_list = []
    var_mse_list = []
    for i in range(1, 11):
        df_mse_values = pd.read_csv(f"analysis_data/gnn_learned_seq_length_5_{i}_nodes.csv")
        avg_mse_list.append(df_mse_values['MSE'].mean())
        var_mse_list.append(df_mse_values['MSE'].sem())

    new_df['avg_mse'] = avg_mse_list
    new_df['var_mse'] = var_mse_list
    plt.figure(figsize=(15, 7.5))
    plt.plot(new_df['num_nodes'], new_df['avg_mse'], marker='o')
    plt.fill_between(new_df['num_nodes'], new_df['avg_mse'] - new_df['var_mse'], new_df['avg_mse'] + new_df['var_mse'], alpha=0.2)
    plt.xlabel('Subgraph Size')
    plt.ylabel('Average MTGNN MSE ± SE')
    plt.tight_layout()
    plt.show()
 
def plot_variance_error_per_symptom():
    """
    Plotting function to create subplots for each symptom with variance on the x-axis and error for each
    GNN type on the y-axis. 
    Data is directly specified in the function. 
    No input or output.
    """
    df_errors_mtgnn = pd.read_csv("analysis_data/gnn_learned_seq_length_5_10_nodes.csv")
    df_errors_agcrn = pd.read_csv("analysis_data/agcrn_seq_length_5_predictions_10_nodes.csv")
    df_errors_mtgnn["Errors"] = df_errors_mtgnn["Normalized_Errors"].apply(eval)
    df_errors_agcrn["Errors"] = df_errors_agcrn["Normalized_Errors"].apply(eval)
    df_errors_mtgnn["GNN"] = "MTGNN"
    df_errors_agcrn["GNN"] = "AGCRN"
    df_errors = pd.concat([df_errors_mtgnn, df_errors_agcrn], ignore_index=True)

    # Original data for variances
    df_original, _ = pyreadstat.read_sav("original_data/data.sav")
    symptom_labels = ['Energy', 'Valence', 'Calmness', 'PA', 'Anxiety', 
                      'Depression', 'PB', 'TB', 'Hopelessness', 'SI']
    variance_data = []
    for uid in df_original["Nr"].unique():
        df = df_original[df_original["Nr"] == uid].copy()
        df = df.sort_values(by="date")
        relevant_columns = ['Energie', 'Valenz', 'Ruhe', 'PA', 'Angst', 'Depr', 
                            'PB', 'TB', 'Hoff', 'suge']
        df = df[relevant_columns].dropna()
        last_5_variance = df.iloc[-5:, :].var().tolist()
        variance_data.append({"ID": uid, "Variance_Last_5": last_5_variance})

    df_variance = pd.DataFrame(variance_data)
    error_variance_data = []
    for _, row in df_errors.iterrows():
        unique_id = row["ID"]
        gnn_type = row["GNN"]
        error_values = row["Errors"]
        variance_values = df_variance.loc[df_variance["ID"] == unique_id, "Variance_Last_5"].values
        variance_values = variance_values[0]

        # Looking at each symptom
        for i, symptom in enumerate(symptom_labels):
            error_variance_data.append({
                "ID": unique_id,
                "Symptom": symptom,
                "Variance": variance_values[i],
                "Error": np.abs(error_values[i]),
                "GNN": gnn_type
            })

    df_plot = pd.DataFrame(error_variance_data)
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 7.5), sharey=True)

    # Because legend handles are annoying
    handles = []
    labels = []
    for i, symptom in enumerate(symptom_labels):
        ax = axes[i // 5, i % 5]

        sns.scatterplot(
            x="Variance", y="Error",
            data=df_plot[(df_plot["Symptom"] == symptom) & (df_plot["GNN"] == "AGCRN")],
            color="blue", alpha=0.5, label="AGCRN" if i == 0 else "", ax=ax, markers=0.1
        )
        sns.scatterplot(
            x="Variance", y="Error",
            data=df_plot[(df_plot["Symptom"] == symptom) & (df_plot["GNN"] == "MTGNN")],
            color="red", alpha=0.8, label="MTGNN" if i == 0 else "", ax=ax, markers=0.1
        )
        ax.set_title(symptom)
        ax.set_xlabel("")
        if i % 5 == 0:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("")
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()

    axes[0,0].get_legend().remove()
    fig.legend(handles, labels, bbox_to_anchor=(0.5, 0), loc='lower center', ncols=2)
    fig.text(0.5, 0.13, "Variance (Last 5 Time Steps)", ha='center')
    fig.text(0.01, 0.56, "Absolute, Normalized Prediction Error", va='center', rotation='vertical', fontsize=24)
    plt.tight_layout(rect=[0.02, 0.02, 0.99, 0.99]) 
    fig.subplots_adjust(bottom=0.25)
    plt.show()

# Example usage
plot_individual_predictions(id=1)
plot_error_distributions()
plot_variance_error_per_symptom()
plot_mse_for_nodes()