"""
This script actually sets up the MTGNN with Pytorch.
The original data is available upon request. 
"""
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric_temporal.nn.attention import MTGNN
from process_data import *
from sklearn.metrics import mean_squared_error
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)

class SymptomMTGNN(torch.nn.Module):
    def __init__(self, num_nodes, seq_length, static_graph=False):
        super(SymptomMTGNN, self).__init__()
        self.static_graph = static_graph

        self.mtgnn = MTGNN(
            gcn_true=True,
            build_adj=not static_graph, 
            gcn_depth=2,
            num_nodes=num_nodes,
            kernel_set=[2, 3],
            kernel_size=3,
            dropout=0.3,
            subgraph_size=5,
            node_dim=40,
            dilation_exponential=1,
            conv_channels=32,
            residual_channels=32,
            skip_channels=64,
            end_channels=128,
            seq_length=seq_length,
            in_dim=1,
            out_dim=1,
            layers=3,
            propalpha=0.05,
            tanhalpha=3,
            layer_norm_affline=True 
        ) # Unless otherwise specified in the thesis, these parameter values come from the original implementation on GitHub.

    def forward(self, x, A_tilde=None):
        if self.static_graph:
            assert A_tilde is not None, "Static graph selected, but no adjacency matrix provided."
        return self.mtgnn.forward(x, A_tilde=A_tilde)

def train_model(model, x_train, y_train, x_test, y_test, A_tilde, unique_id, graph_type, seq_length, run, epochs=300):
    """
    This function sets up the model training, taking in the model initialization and prepared data.
    A_tilde refers to a predetermined static graph, if provided.
    graph_type refers to either a pre-provided static graph or the learned graph option, which we will use in this thesis.
    seq_length refers to the time-series input length.
    The number of epochs, learning rate, and accuracy metric was determined by Ntekouli et al. (2024). 
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        if model.static_graph: 
            output = model(x_train, A_tilde=A_tilde)
        else:
            output = model(x_train)
        
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    
    with torch.no_grad():
        test_output = model(x_test, A_tilde=A_tilde) if model.static_graph else model(x_test)
        test_pred = test_output.cpu().numpy()
        test_true = y_test.cpu().numpy()

    if np.isnan(test_pred).all():
        print("Warning: All test predictions are NaNs. Skipping MSE calculation for this iteration.")
        return np.nan, None, None

    test_pred_reshaped = test_pred.reshape(-1)
    test_true_reshaped = test_true.reshape(-1)
    mse = mean_squared_error(test_true_reshaped, test_pred_reshaped)
    print(f"MSE Value for this iteration: {mse}")

    # Rescale predictions and errors back to original scale
    df = pd.read_csv('data.csv')
    df = df[df["Nr"] == unique_id].copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')
    relevant_columns = ['date', 'Energie', 'Valenz', 'Ruhe', 'PA', 'Angst', 'Depr', 
                        'PB', 'TB', 'Hoff', 'suge']
    df = df[relevant_columns]
    df = df.dropna()
    num_time_steps = df.shape[0]
    split_index = int(num_time_steps * 0.7)
    train_df = df.iloc[:split_index, 1:]
    train_max, train_min = train_df.max(), train_df.min()
    train_range = train_max - train_min
    rescaled_pred = (test_pred_reshaped * train_range) + train_min  
    rescaled_true = (test_true_reshaped * train_range) + train_min
    rescaled_error = rescaled_pred - rescaled_true
    normalized_error = test_pred_reshaped - test_true_reshaped
    
    if not model.static_graph:
        learned_A_tilde = model.mtgnn._graph_constructor(model.mtgnn._idx.to(x_train.device)).detach().cpu().numpy()
        print(f"Learned Adjacency Matrix (A_tilde) for ID {unique_id}, Run {run}:\n", learned_A_tilde)

    return mse, rescaled_pred.tolist(), rescaled_true.tolist(), rescaled_error.tolist(), normalized_error.tolist()

def run_experiments(save_csv=True, repetitions=10):
    """
    This function runs the training for the designated number of repetitions.
    The output is a dataframe with the particular subject ID, the graph type, the run and sequence length associated 
    with that training, the true values and predicted values, and the rescaled and normalized errors.
    The different versions of the rescaled and normalized errors are for plotting purposes.
    """
    df = pd.read_csv('data.csv')
    unique_ids = df['Nr'].unique()
    
    graph_types = ["gnn_learned"]
    sequence_lengths = [5]
    results = []

    for seq_length in sequence_lengths:
        for graph_type in graph_types:
            print(f"\nTraining with Graph: {graph_type}, Sequence Length: {seq_length}")
            for unique_id in unique_ids:
                print(f"Processing ID: {unique_id}")

                for i in range(repetitions):
                    print(f"Starting run {i+1}...")
                    num_nodes, seq_length, x_train, y_train, x_test, y_test, A_tilde = process_data_mtgnn(
                        df, unique_id, graph_type=graph_type, seq_length=seq_length
                    ) 

                    if graph_type == "gnn_learned":
                        model = SymptomMTGNN(num_nodes, seq_length, static_graph=False)
                        mse, predictions, true_values, rescaled_error, normalized_error = train_model(model, x_train, y_train, x_test, y_test, None, unique_id, graph_type, seq_length, run=i+1, epochs=300)
                    else:
                        model = SymptomMTGNN(num_nodes, seq_length, static_graph=True)
                        mse, predictions, true_values, rescaled_error, normalized_error = train_model(model, x_train, y_train, x_test, y_test, A_tilde, unique_id, graph_type, seq_length, run=i+1, epochs=300)

                    results.append([unique_id, i+1, seq_length, graph_type, mse, str(predictions), str(true_values), str(rescaled_error), str(normalized_error)])

        results_df = pd.DataFrame(results, columns=["ID", "Run", "Sequence_Length", "Graph_Type", "MSE", "Predicted_Values", "True_Values", "Rescaled_Errors", "Normalized_Errors"])
        
        if save_csv:
            results_df.to_csv(f"analysis_data/{graph_type}_seq_length_5_10_nodes.csv", index=False)

    return results

# Example Usage
run_experiments(save_csv=True, repetitions=10)
