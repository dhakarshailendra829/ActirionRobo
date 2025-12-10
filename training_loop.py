import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

def train_imitation_model(model, data_df, epochs=5, learning_rate=0.001):
    
    if data_df.empty:
        return []

    num_features = 50
    num_actions = 6
    
    actions_columns = [f'target_action_{i}' for i in range(num_actions)]
    if len(data_df) > 0:
        data_df[actions_columns] = np.random.uniform(-1.0, 1.0, (len(data_df), num_actions))
    
    X = torch.from_numpy(data_df.iloc[:, 1:1+num_features].values).float()
    y = torch.from_numpy(data_df[actions_columns].values).float()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []
    
    model.train() 
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
    model.eval() 
    return loss_history