"""Spatial-temporal crime hotspot model and trainer."""
import torch
import torch.nn as nn
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.config import config

class CrimeHotspotPredictor(nn.Module):
    """Simple spatial-temporal model for hotspot prediction."""
    def __init__(self, num_nodes: int, num_features: int, history: int = config.HISTORY_WINDOW):
        super().__init__()
        self.num_nodes = num_nodes
        self.history = history
        self.num_features = num_features
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=history * num_features, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, adjacency: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [batch, time, grid, grid, features]
        batch_size, time_steps, grid_h, grid_w, features = x.shape
        x = x.reshape(batch_size, time_steps * features, grid_h, grid_w)
        out = self.conv(x)
        return out.squeeze(1)


class ModelTrainer:
    """Trainer for the crime hotspot model."""
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = torch.device(device if isinstance(device, str) else str(device))
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def fit(self, train_loader, val_loader, adjacency: torch.Tensor, epochs: int = 10):
        adjacency = adjacency.to(self.device)
        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(X_batch, adjacency)
                loss = self.criterion(preds, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * X_batch.size(0)

            avg_train_loss = total_loss / max(len(train_loader.dataset), 1)
            logger.info(f"Epoch {epoch}/{epochs} train loss: {avg_train_loss:.4f}")

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    preds = self.model(X_batch, adjacency)
                    loss = self.criterion(preds, y_batch)
                    val_loss += loss.item() * X_batch.size(0)

            avg_val_loss = val_loss / max(len(val_loader.dataset), 1)
            logger.info(f"Epoch {epoch}/{epochs} val loss: {avg_val_loss:.4f}")
