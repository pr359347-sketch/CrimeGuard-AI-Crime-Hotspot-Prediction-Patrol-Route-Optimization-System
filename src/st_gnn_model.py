"""Spatial-Temporal Graph Neural Network with LSTM for crime prediction"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config import config

class SpatialGraphConv(nn.Module):
    """Spatial Graph Convolution Layer"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_nodes, in_channels)
            adj: (num_nodes, num_nodes) adjacency matrix
        Returns:
            (batch_size, num_nodes, out_channels)
        """
        # Graph convolution: AXW + b
        out = torch.matmul(adj, x)  # (num_nodes, num_nodes) @ (batch, num_nodes, in_channels)
        out = torch.matmul(out, self.weight)  # @ (in_channels, out_channels)
        out = out + self.bias
        
        return out

class TemporalLSTM(nn.Module):
    """Temporal LSTM Layer"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            output: (batch_size, seq_len, hidden_size)
            (h_n, c_n): final hidden and cell states
        """
        return self.lstm(x)

class STGNNBlock(nn.Module):
    """Spatial-Temporal GNN Block combining GCN and LSTM"""
    
    def __init__(self, in_channels: int, out_channels: int, hidden_dim: int):
        super().__init__()
        
        # Spatial component
        self.spatial_conv = SpatialGraphConv(in_channels, out_channels)
        self.spatial_norm = nn.BatchNorm1d(out_channels)
        
        # Temporal component
        self.temporal_lstm = TemporalLSTM(
            input_size=out_channels,
            hidden_size=hidden_dim,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT
        )
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.DROPOUT)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, num_nodes, in_channels)
            adj: (num_nodes, num_nodes)
        Returns:
            (batch_size, seq_len, num_nodes, hidden_dim)
        """
        batch_size, seq_len, num_nodes, in_channels = x.shape
        
        # Apply spatial convolution at each time step
        spatial_out = []
        for t in range(seq_len):
            xt = x[:, t, :, :]  # (batch, num_nodes, in_channels)
            
            # Reshape for graph conv
            xt_flat = xt.view(-1, in_channels)  # (batch*num_nodes, in_channels)
            
            # Apply graph conv
            gcn_out = self.spatial_conv(xt.transpose(0, 1), adj)  # (num_nodes, batch, out_channels)
            gcn_out = gcn_out.transpose(0, 1)  # (batch, num_nodes, out_channels)
            
            spatial_out.append(gcn_out)
        
        # Stack temporal dimension
        spatial_out = torch.stack(spatial_out, dim=1)  # (batch, seq_len, num_nodes, out_channels)
        spatial_out = self.relu(spatial_out)
        spatial_out = self.dropout(spatial_out)
        
        # Apply LSTM on temporal dimension
        temporal_out = []
        for node_idx in range(num_nodes):
            node_sequence = spatial_out[:, :, node_idx, :]  # (batch, seq_len, out_channels)
            lstm_out, _ = self.temporal_lstm(node_sequence)
            temporal_out.append(lstm_out)
        
        # Stack back to spatial dimension
        temporal_out = torch.stack(temporal_out, dim=2)  # (batch, seq_len, num_nodes, hidden_dim)
        
        return temporal_out

class CrimeHotspotPredictor(nn.Module):
    """Complete ST-GNN model for crime hotspot prediction"""
    
    def __init__(self, num_nodes: int, input_dim: int, output_dim: int = 1):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # ST-GNN blocks
        self.st_block_1 = STGNNBlock(input_dim, 32, config.LSTM_HIDDEN_DIM)
        self.st_block_2 = STGNNBlock(config.LSTM_HIDDEN_DIM, 64, config.LSTM_HIDDEN_DIM)
        
        # Prediction head
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * config.PREDICTION_HORIZON, 128),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(64, output_dim)
        )
        
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, num_nodes, input_dim)
            adj: (num_nodes, num_nodes)
        Returns:
            predictions: (batch_size, num_nodes, output_dim)
        """
        # ST-GNN blocks
        out = self.st_block_1(x, adj)
        out = self.st_block_2(out, adj)
        
        # Prediction head
        batch_size, seq_len, num_nodes, hidden_dim = out.shape
        
        predictions = []
        for node_idx in range(num_nodes):
            node_out = out[:, :, node_idx, :]  # (batch, seq_len, hidden_dim)
            node_flat = node_out.reshape(batch_size, -1)  # (batch, seq_len*hidden_dim)
            pred = self.fc_layers(node_flat)  # (batch, output_dim)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=1)  # (batch, num_nodes, output_dim)
        predictions = torch.clamp(predictions, min=0)  # Ensure non-negative
        
        return predictions

class ModelTrainer:
    """Trainer for ST-GNN model"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE
        )
        self.criterion = nn.MSELoss()
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader, adj: torch.Tensor) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            adj = adj.to(self.device)
            
            self.optimizer.zero_grad()
            
            predictions = self.model(x, adj)
            loss = self.criterion(predictions, y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, val_loader, adj: torch.Tensor) -> float:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                adj = adj.to(self.device)
                
                predictions = self.model(x, adj)
                loss = self.criterion(predictions, y)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def fit(self, train_loader, val_loader, adj: torch.Tensor, epochs: int = config.EPOCHS):
        """Train model"""
        logger.info(f"Starting training for {epochs} epochs...")
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, adj)
            val_loss = self.validate(val_loader, adj)
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info("Training complete!")
    
    def predict(self, x: torch.Tensor, adj: torch.Tensor) -> np.ndarray:
        """Make predictions"""
        self.model.eval()
        
        with torch.no_grad():
            x = x.to(self.device)
            adj = adj.to(self.device)
            predictions = self.model(x, adj)
        
        return predictions.cpu().numpy()

if __name__ == "__main__":
    # Test the model
    num_nodes = config.GRID_SIZE * config.GRID_SIZE
    batch_size = config.BATCH_SIZE
    seq_len = config.HISTORY_WINDOW
    input_dim = 30  # Number of input features
    
    model = CrimeHotspotPredictor(num_nodes, input_dim)
    
    x = torch.randn(batch_size, seq_len, num_nodes, input_dim)
    adj = torch.randn(num_nodes, num_nodes)
    
    output = model(x, adj)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
