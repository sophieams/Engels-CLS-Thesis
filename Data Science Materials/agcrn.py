"""
This script actually sets up the AGCRN with Pytorch.
The original data is available upon request. 
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from torch_geometric_temporal.nn.recurrent import AGCRN
from process_data import process_data_agcrn

class SymptomAGCRN(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, embedding_dim, K=3):
        super(SymptomAGCRN, self).__init__()

        self.agcrn = AGCRN(
            number_of_nodes=num_nodes,
            in_channels=in_channels,
            out_channels=out_channels,
            K=K,
            embedding_dimensions=embedding_dim,
        )

    def forward(self, x, node_embeddings, H=None):  
        return self.agcrn(x, node_embeddings, H)


def train_model(model, x_train, y_train, unique_id, node_embeddings, x_test, y_test, epochs=300, learning_rate=0.01):
    """
    This function sets up the model training, taking in the model initialization and prepared data.
    The number of epochs, learning rate, and accuracy metric was determined by Ntekouli et al. (2024). 
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(x_train, node_embeddings) 
        loss = criterion(output, y_train)  
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_output = model(x_test, node_embeddings)
        test_pred = test_output.numpy()
        test_true = y_test.numpy() 
        
    mse = mean_squared_error(test_true.squeeze(), test_pred.squeeze())
    print(f"Test MSE: {mse}")
    
    test_pred_last = test_pred[:, :, -1].reshape(-1)
    test_true_last = test_true[:, :, -1].reshape(-1) 
    
    # Rescale predictions and errors back to original scale for plotting
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
    rescaled_pred = (test_pred_last * train_range) + train_min  
    rescaled_true = (test_true_last * train_range) + train_min
    rescaled_error = rescaled_pred - rescaled_true
    normalized_error = test_pred_last - test_true_last

    return mse, rescaled_pred.tolist(), rescaled_true.tolist(), rescaled_error.tolist(), normalized_error.tolist() 

def run_experiments(save_csv=True, repetitions=10):
    """
    This function runs the training for the designated number of repetitions.
    The output is a dataframe with the particular subject ID, the run and sequence length associated 
    with that training, the true values and predicted values, and the rescaled and normalized errors.
    The different versions of the rescaled and normalized errors are for plotting purposes.
    """
    df = pd.read_csv('data.csv')
    unique_ids = df['Nr'].unique()
    sequence_lengths = [5] 
    results = []

    for seq_length in sequence_lengths:
        for unique_id in unique_ids:
            print(f"\nProcessing ID: {unique_id}")

            for i in range(repetitions):
                print(f"Starting run {i+1}...")
                num_nodes, node_embeddings, x_train, y_train, x_test, y_test = process_data_agcrn(
                    df, unique_id, seq_length=seq_length
                )
                model = SymptomAGCRN(
                    in_channels=x_train.shape[-1],  
                    out_channels=1,
                    num_nodes=num_nodes,
                    embedding_dim=node_embeddings.shape[1], 
                )
                mse, predictions, true_values, rescaled_error, normalized_error = train_model(model, x_train, y_train, unique_id, node_embeddings, x_test, y_test, epochs=300)
                results.append([unique_id, i+1, seq_length, mse, str(predictions), str(true_values), str(rescaled_error), str(normalized_error)])

        results_df = pd.DataFrame(results, columns=["ID", "Run", "Sequence_Length", "MSE", "Predicted_Values", "True_Values", "Rescaled_Errors", "Normalized_Errors"])
        
        if save_csv:
            results_df.to_csv(f"analysis_data/agcrn_seq_length_{seq_length}_predictions_10_nodes.csv", index=False)

    return results

# Example Usage
run_experiments(save_csv=True, repetitions=10)
