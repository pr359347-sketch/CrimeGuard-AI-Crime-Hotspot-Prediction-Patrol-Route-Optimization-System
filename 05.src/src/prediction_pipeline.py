"""Complete prediction pipeline for crime hotspot forecasting"""
import numpy as np
import pandas as pd
import torch
import logging
from typing import Tuple, Dict, List
import pickle
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.config import config
from src.data_loader import CrimeDataLoader
from src.feature_engineering import FeatureEngineer
try:
    from src.st_gnn_model import CrimeHotspotPredictor, ModelTrainer
except Exception:
    CrimeHotspotPredictor = None
    ModelTrainer = None
from src.route_optimizer import PatrolRouteOptimizer, ResourceAllocationOptimizer
from src.visualization import HotspotVisualizer, StatisticsVisualizer
from src.model_evaluation import PredictionEvaluator, HotspotDetectionValidator

class CrimeHotspotPipeline:
    """End-to-end pipeline for crime prediction and patrol optimization"""
    
    def __init__(self, city: str = 'Chicago'):
        """
        Args:
            city: 'Chicago' or 'LA'
        """
        self.city = city
        self.bounds = config.CHICAGO_BOUNDS if city == 'Chicago' else config.LA_BOUNDS
        
        self.data = None
        self.features_df = None
        self.model = None
        self.predictions = None
        self.hotspots = None
        self.routes = None
        
        logger.info(f"Initialized pipeline for {city}")
    
    def load_and_preprocess_data(self) -> pd.DataFrame:
        """Load and preprocess crime data"""
        logger.info(f"Loading and preprocessing {self.city} data...")
        
        loader = CrimeDataLoader()
        
        if self.city == 'Chicago':
            df = loader.download_chicago_crimes(num_records=10000)
        else:
            df = loader.download_la_crimes(num_records=10000)
        
        df = loader.preprocess_crimes(df)
        df = loader.create_spatial_grid(df, self.city)
        
        self.data = df
        logger.info(f"Data loaded: {df.shape[0]} records")
        
        return df
    
    def engineer_features(self) -> pd.DataFrame:
        """Engineer features for modeling"""
        logger.info("Engineering features...")
        
        if self.data is None:
            raise ValueError("Load data first with load_and_preprocess_data()")
        
        engineer = FeatureEngineer()
        
        # Aggregate to grid time series
        loader = CrimeDataLoader()
        df_grid = loader.aggregate_to_grid_time_series(self.data)
        
        # Engineer features
        features_df = engineer.engineer_features(df_grid)
        
        self.features_df = features_df
        logger.info(f"Features engineered: {features_df.shape[1]} features")
        
        return features_df
    
    def prepare_sequences(self, features_df: pd.DataFrame = None) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training sequences"""
        logger.info("Preparing sequences...")
        
        if features_df is None:
            features_df = self.features_df
        
        # Get feature columns (exclude metadata)
        feature_cols = [col for col in features_df.columns 
                       if col not in ['date', 'lat_grid', 'lon_grid', 'crime_count']]
        
        X_list = []
        y_list = []
        
        num_nodes = config.GRID_SIZE * config.GRID_SIZE
        
        # Create sequences for each grid cell
        for lat_idx in range(config.GRID_SIZE):
            for lon_idx in range(config.GRID_SIZE):
                cell_data = features_df[
                    (features_df['lat_grid'] == lat_idx) & 
                    (features_df['lon_grid'] == lon_idx)
                ].sort_values('date')
                
                if len(cell_data) > config.HISTORY_WINDOW + config.PREDICTION_HORIZON:
                    values = cell_data[feature_cols].values
                    targets = cell_data['crime_count'].values
                    
                    # Create sliding windows
                    for i in range(len(values) - config.HISTORY_WINDOW - config.PREDICTION_HORIZON + 1):
                        X_list.append(values[i:i+config.HISTORY_WINDOW])
                        y_list.append(targets[i+config.HISTORY_WINDOW:i+config.HISTORY_WINDOW+config.PREDICTION_HORIZON].mean())
        
        if not X_list:
            logger.warning("No sequences created. Using synthetic data.")
            return self._create_synthetic_sequences()
        
        X = np.array(X_list)
        y = np.array(y_list).reshape(-1, 1)
        
        logger.info(f"Sequences created: X={X.shape}, y={y.shape}")
        
        return X, y
    
    def _create_synthetic_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create synthetic sequences for testing"""
        logger.info("Creating synthetic training sequences...")
        
        num_sequences = 1000
        num_features = 30
        
        X = np.random.randn(num_sequences, config.HISTORY_WINDOW, num_features)
        y = np.random.exponential(scale=2, size=(num_sequences, 1))
        
        return X, y
    
    def create_4d_sequences(self, features_df: pd.DataFrame = None) -> Tuple[np.ndarray, np.ndarray]:
        """Create 4D sequences (batch, time, space, features) for ST-GNN"""
        logger.info("Creating 4D ST-GNN sequences...")
        
        if features_df is None:
            features_df = self.features_df
        
        feature_cols = [col for col in features_df.columns 
                       if col not in ['date', 'lat_grid', 'lon_grid', 'crime_count']]
        
        num_nodes = config.GRID_SIZE * config.GRID_SIZE
        num_features = len(feature_cols)
        
        # Create a 3D grid time series
        dates = sorted(features_df['date'].unique())
        
        X_list = []
        y_list = []
        
        for seq_idx in range(len(dates) - config.HISTORY_WINDOW - config.PREDICTION_HORIZON + 1):
            date_range = dates[seq_idx:seq_idx + config.HISTORY_WINDOW]
            
            # Create spatial-temporal tensor for this sequence
            seq_tensor = np.zeros((config.HISTORY_WINDOW, config.GRID_SIZE, config.GRID_SIZE, num_features))
            target_tensor = np.zeros((config.GRID_SIZE, config.GRID_SIZE))
            
            for t, date in enumerate(date_range):
                day_data = features_df[features_df['date'] == date]
                
                for _, row in day_data.iterrows():
                    lat_idx = int(row['lat_grid'])
                    lon_idx = int(row['lon_grid'])
                    
                    if 0 <= lat_idx < config.GRID_SIZE and 0 <= lon_idx < config.GRID_SIZE:
                        seq_tensor[t, lat_idx, lon_idx, :] = day_data[feature_cols].values[0]
            
            # Target is future week's crime count
            future_date_range = dates[seq_idx + config.HISTORY_WINDOW:seq_idx + config.HISTORY_WINDOW + config.PREDICTION_HORIZON]
            future_data = features_df[features_df['date'].isin(future_date_range)]
            
            for _, row in future_data.iterrows():
                lat_idx = int(row['lat_grid'])
                lon_idx = int(row['lon_grid'])
                if 0 <= lat_idx < config.GRID_SIZE and 0 <= lon_idx < config.GRID_SIZE:
                    target_tensor[lat_idx, lon_idx] += row['crime_count']
            
            X_list.append(seq_tensor)
            y_list.append(target_tensor)
        
        if not X_list:
            logger.warning("No 4D sequences created. Using synthetic data.")
            return self._create_synthetic_4d_sequences()
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        logger.info(f"4D sequences created: X={X.shape}, y={y.shape}")
        
        return X, y
    
    def _create_synthetic_4d_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create synthetic 4D sequences"""
        num_sequences = 100
        num_features = 30
        
        X = np.random.randn(num_sequences, config.HISTORY_WINDOW, config.GRID_SIZE, config.GRID_SIZE, num_features)
        y = np.random.exponential(scale=2, size=(num_sequences, config.GRID_SIZE, config.GRID_SIZE))
        
        return X, y
    
    def train_model(self, X: np.ndarray = None, y: np.ndarray = None, epochs: int = config.EPOCHS):
        """Train ST-GNN model"""
        logger.info("Training ST-GNN model...")
        
        if X is None or y is None:
            X, y = self.create_4d_sequences()
        
        # Create dummy adjacency matrix
        num_nodes = config.GRID_SIZE * config.GRID_SIZE
        adjacency = np.eye(num_nodes)  # Identity for now
        adjacency = torch.from_numpy(adjacency).float()
        
        # Prepare data
        num_features = X.shape[-1]
        
        # Simple train/val split
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create data loaders
        train_data = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).float()
        )
        val_data = torch.utils.data.TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_val).float()
        )
        
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=config.BATCH_SIZE)
        
        # Create and train model
        self.model = CrimeHotspotPredictor(num_nodes, num_features)
        trainer = ModelTrainer(self.model, device=config.DEVICE)
        
        trainer.fit(train_loader, val_loader, adjacency, epochs=epochs)
        
        logger.info("Model training complete")
        
        return self.model
    
    def predict_hotspots(self, X: np.ndarray = None, return_grid: bool = True) -> np.ndarray:
        """Make hotspot predictions"""
        logger.info("Making hotspot predictions...")
        
        if self.model is None:
            raise ValueError("Train model first with train_model()")
        
        if X is None:
            X, _ = self.create_4d_sequences()
            X = X[-1:]  # Use last sequence for prediction
        
        num_nodes = config.GRID_SIZE * config.GRID_SIZE
        adjacency = np.eye(num_nodes)
        adjacency = torch.from_numpy(adjacency).float()
        
        predictions = self.model(torch.from_numpy(X).float(), adjacency)
        predictions = predictions.detach().numpy()
        
        if return_grid:
            # Reshape to grid
            predictions = predictions[0].reshape(config.GRID_SIZE, config.GRID_SIZE)
            predictions = np.abs(predictions)  # Ensure positive
            predictions = predictions / (predictions.max() + 1e-8)  # Normalize to [0, 1]
        
        self.predictions = predictions
        logger.info(f"Predictions shape: {predictions.shape}")
        
        return predictions
    
    def detect_hotspots(self) -> List[Tuple[int, int]]:
        """Detect hotspot locations"""
        logger.info("Detecting hotspots...")
        
        if self.predictions is None:
            raise ValueError("Make predictions first with predict_hotspots()")
        
        hotspot_mask = self.predictions > config.HOTSPOT_THRESHOLD
        hotspot_indices = np.where(hotspot_mask)
        
        hotspots = list(zip(hotspot_indices[0], hotspot_indices[1]))
        
        self.hotspots = hotspots
        logger.info(f"Detected {len(hotspots)} hotspots")
        
        return hotspots
    
    def optimize_patrol_routes(self) -> Dict:
        """Optimize patrol routes for detected hotspots"""
        logger.info("Optimizing patrol routes...")
        
        if not self.hotspots:
            raise ValueError("Detect hotspots first with detect_hotspots()")
        
        optimizer = PatrolRouteOptimizer(self.hotspots)
        routes = optimizer.get_routes_coordinates(self.bounds)
        
        self.routes = routes
        logger.info(f"Optimized {len(routes)} patrol routes")
        
        return routes
    
    def generate_visualizations(self, output_dir: str = config.OUTPUT_DIR) -> Dict:
        """Generate all visualizations"""
        logger.info("Generating visualizations...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        visualizer = HotspotVisualizer(self.bounds, self.city)
        
        visualizations = {}
        
        # Heatmap
        if self.predictions is not None:
            heatmap_path = os.path.join(output_dir, f'{self.city.lower()}_heatmap.html')
            visualizer.create_heatmap(self.predictions, heatmap_path)
            visualizations['heatmap'] = heatmap_path
        
        # Patrol routes
        if self.routes:
            route_path = os.path.join(output_dir, f'{self.city.lower()}_routes.html')
            visualizer.create_patrol_route_map(self.routes, route_path)
            visualizations['routes'] = route_path
        
        logger.info(f"Visualizations saved to {output_dir}")
        
        return visualizations
    
    def run_complete_pipeline(self):
        """Execute complete pipeline"""
        logger.info(f"Running complete Crime Hotspot Pipeline for {self.city}...")
        
        # Data
        self.load_and_preprocess_data()
        self.engineer_features()
        
        # Model
        self.train_model(epochs=10)  # Quick training for demo
        
        # Predictions
        self.predict_hotspots()
        self.detect_hotspots()
        
        # Optimization
        self.optimize_patrol_routes()
        
        # Visualizations
        self.generate_visualizations()
        
        logger.info("Pipeline execution complete!")
        
        return {
            'predictions': self.predictions,
            'hotspots': self.hotspots,
            'routes': self.routes
        }
    
    def save_model(self, path: str = None):
        """Save trained model"""
        if path is None:
            path = os.path.join(config.MODELS_DIR, f'{self.city.lower()}_model.pt')
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model"""
        if not os.path.exists(path):
            logger.warning(f"Model file not found: {path}")
            return False
        
        num_nodes = config.GRID_SIZE * config.GRID_SIZE
        self.model = CrimeHotspotPredictor(num_nodes, 30)
        self.model.load_state_dict(torch.load(path))
        logger.info(f"Model loaded from {path}")
        
        return True

if __name__ == "__main__":
    pipeline = CrimeHotspotPipeline('Chicago')
    results = pipeline.run_complete_pipeline()
    print(f"Predictions shape: {results['predictions'].shape}")
    print(f"Hotspots detected: {len(results['hotspots'])}")
    print(f"Routes optimized: {len(results['routes'])}")
