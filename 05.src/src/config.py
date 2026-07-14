"""Configuration and constants for Crime Hotspot Prediction System"""
import os
from dataclasses import dataclass
from typing import Tuple

@dataclass
class SystemConfig:
    """System-wide configuration"""
    
    # Data paths
    DATA_DIR = "data"
    MODELS_DIR = "models"
    OUTPUT_DIR = "outputs"
    
    # Geographic boundaries (Chicago and LA)
    CHICAGO_BOUNDS = {
        'lat_min': 41.6,
        'lat_max': 42.0,
        'lon_min': -87.9,
        'lon_max': -87.5
    }
    
    LA_BOUNDS = {
        'lat_min': 33.7,
        'lat_max': 34.15,
        'lon_min': -118.7,
        'lon_max': -117.9
    }
    
    # Grid parameters
    GRID_SIZE = 50  # 50x50 spatial grid
    PREDICTION_HORIZON = 7  # 7 days ahead
    HISTORY_WINDOW = 60  # 60 days history
    
    # Model parameters
    ST_GNN_HIDDEN_DIM = 64
    LSTM_HIDDEN_DIM = 128
    NUM_LAYERS = 2
    DROPOUT = 0.2
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    EPOCHS = 50
    
    # Route optimization
    NUM_PATROL_UNITS = 10
    PATROL_SHIFT_HOURS = 8
    MAX_DISTANCE_KM = 50
    
    # Feature engineering
    WEATHER_FEATURES = ['temperature', 'precipitation', 'humidity', 'wind_speed']
    TIME_FEATURES = ['hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday']
    
    # Hotspot detection
    HOTSPOT_THRESHOLD = 0.7  # Probability threshold for hotspot
    MIN_HOTSPOT_SIZE = 3  # Minimum grid cells to form a hotspot
    
    # Device
    DEVICE = 'cpu'  # or 'cuda'
    
    # Random seed for reproducibility
    RANDOM_SEED = 42

config = SystemConfig()
