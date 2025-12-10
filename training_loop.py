# training_loop.py - FINAL CLEAN VERSION: AI Model Training Simulation
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

def train_imitation_model(model, data_df, epochs=5, learning_rate=0.001):
    """
    Simulates the training loop for the ImitationModel using dummy data.
    
    Args:
        model (torch.nn.Module): The PyTorch model to train.
        data_df (pd.DataFrame): DataFrame containing features and target actions.
        epochs (int): Number of training epochs.
        learning_rate (float): Optimizer learning rate.
        
    Returns:
        list: List of loss values recorded during training.
    """
    if data_df.empty:
        return []

    # --- Prepare Dummy Data ---
    num_features = 50
    num_actions = 6
    
    # Create random target actions (as we don't have real ground truth)
    actions_columns = [f'target_action_{i}' for i in range(num_actions)]
    # Ensure data_df is large enough for indexing before setting target actions
    if len(data_df) > 0:
        data_df[actions_columns] = np.random.uniform(-1.0, 1.0, (len(data_df), num_actions))
    
    # Extract features (skipping the 'task_type' column at index 0)
    X = torch.from_numpy(data_df.iloc[:, 1:1+num_features].values).float()
    y = torch.from_numpy(data_df[actions_columns].values).float()

    # --- Training Setup ---
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []
    
    model.train() # Set model to training mode
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
    model.eval() # Set model back to evaluation mode
    return loss_history