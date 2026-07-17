import torch
import torch.nn as nn
import torch.optim as optim
from src.models import STGNN
from src.feature_engineering import FeatureEngineer
from src.graph_builder import build_edge_index
import numpy as np

class Trainer:
    def __init__(self, model, lr=0.001):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.fe = FeatureEngineer()

    def train(self, crime_df, weather_df, event_df, socio_df, coords, epochs=10):
        # 1. Feature Engineering
        processed_data = self.fe.run_pipeline(crime_df, weather_df, event_df, socio_df)
        
        # 2. Graph Construction
        edge_index = build_edge_index(coords, threshold=0.1)
        
        # Select numeric feature columns only, specifically excluding indexing columns
        features_to_drop = ['timestamp', 'nodes', 'region_id', 'event_id', 'crime']
        feature_df = processed_data.drop(columns=[c for c in features_to_drop if c in processed_data.columns])
        
        # Reshape to [batch=1, seq=1, nodes=10, features=...]
        # The input data is flat [nodes, features] for one time step
        num_features = feature_df.shape[1]
        data = torch.tensor(feature_df.values, dtype=torch.float32).reshape(1, 1, 10, num_features)
        
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            output = self.model(data, edge_index)
            # Dummy target
            target = torch.randn(output.shape)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")
