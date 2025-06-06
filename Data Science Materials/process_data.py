"""
This script splits the data according to relevant columns and dimensionalizes them properly for the GNN.
"""
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

def process_data_mtgnn(df, unique_id, graph_type="gnn_learned", seq_length=5): 
    """
    Prepares the data for the MTGNN.
    Input:
    - df: Time-series dataframe.
    - unique_id: Subject ID for idiographic setup.
    - seq_length: Length of input sequences.
    - forecast_steps: Number of future time steps to predict.
    Output:
    - num_nodes: number of symptoms we are predicting
    - seq_length: repeating for the MTGNN parameter setup.
    - x_train, y_train, x_test, y_test: the split data for validation.
    - A_tilde: if a pre-calculated graph is used.
    """
    df = df[df["Nr"] == unique_id].copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')
    relevant_columns = ['date', 'Energie', 'Valenz', 'Ruhe', 'PA', 'Angst', 'Depr', 
                        'PB', 'TB', 'Hoff', 'suge']
    df = df[relevant_columns]
    df = df.dropna()

    # Train-Test split on time steps
    num_time_steps = df.shape[0]
    split_index = int(num_time_steps * 0.7)
    
    train_df = df.iloc[:split_index, 1:]  # Remove date
    test_df = df.iloc[split_index:, 1:]
    train_data = train_df.T.to_numpy()
    test_data = test_df.T.to_numpy()
    
    scaler = MinMaxScaler()
    train_df[:] = scaler.fit_transform(train_df)
    test_df[:] = scaler.transform(test_df)

    forecast_steps = 1
    if graph_type == "gnn_learned":
        A_tilde = None 
    else:
        raise ValueError("Other graph types are not supported at this time.")
    
    num_nodes = train_data.shape[0]
    x_train = torch.tensor(train_data[:, -(seq_length + forecast_steps):-forecast_steps], dtype=torch.float).unsqueeze(0).unsqueeze(0)
    x_test = torch.tensor(test_data[:, -(seq_length + forecast_steps):-forecast_steps], dtype=torch.float).unsqueeze(0).unsqueeze(0)

    y_train = torch.tensor(train_data[:, -forecast_steps:], dtype=torch.float).unsqueeze(0).unsqueeze(0)
    y_test = torch.tensor(test_data[:, -forecast_steps:], dtype=torch.float).unsqueeze(0).unsqueeze(0)
    
    return num_nodes, seq_length, x_train, y_train, x_test, y_test, A_tilde

def process_data_agcrn(df, unique_id, seq_length=5, forecast_steps=1):
    """
    Prepares the data for the AGCRN.
    Input:
    - df: Time-series dataframe.
    - unique_id: Subject ID for idiographic setup.
    - seq_length: Length of input sequences.
    - forecast_steps: Number of future time steps to predict.
    Output:
    - num_nodes: number of symptoms we are predicting
    - node_embeddings: maps from higher dimensional to lower dimensional space.
    - x_train, y_train, x_test, y_test: the split data for validation.
    """
    df = df[df["Nr"] == unique_id].copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')
    relevant_columns = ['date', 'Energie', 'Valenz', 'Ruhe', 'PA', 'Angst', 'Depr', 
                        'PB', 'TB', 'Hoff', 'suge']
    df = df[relevant_columns].dropna()

    # Train-Test split on time steps (same as before)
    num_time_steps = df.shape[0]
    split_index = int(num_time_steps * 0.7)
    train_df = df.iloc[:split_index, 1:]
    test_df = df.iloc[split_index:, 1:]
    
    scaler = MinMaxScaler()
    train_df[:] = scaler.fit_transform(train_df)
    test_df[:] = scaler.transform(test_df)
    
    train_data = train_df.T.to_numpy()
    test_data = test_df.T.to_numpy()
    
    # Requires a slightly different dimensionality
    x_train = torch.tensor(train_data[:, -(seq_length + forecast_steps):-forecast_steps], dtype=torch.float).unsqueeze(0)
    x_test = torch.tensor(test_data[:, -(seq_length + forecast_steps):-forecast_steps], dtype=torch.float).unsqueeze(0)
    
    y_train = torch.tensor(train_data[:, -forecast_steps:], dtype=torch.float).unsqueeze(0)
    y_test = torch.tensor(test_data[:, -forecast_steps:], dtype=torch.float).unsqueeze(0)
    
    # Create random node embeddings
    num_nodes = len(relevant_columns)-1
    node_embeddings = torch.randn(num_nodes, 1)  # (nodes, embedding_dim)

    return num_nodes, node_embeddings, x_train, y_train, x_test, y_test