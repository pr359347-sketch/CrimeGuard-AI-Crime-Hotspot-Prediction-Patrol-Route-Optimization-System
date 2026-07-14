"""Feature engineering for spatial-temporal crime prediction"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.config import config

class FeatureEngineer:
    """Engineer temporal, spatial, and external features"""
    
    def __init__(self):
        self.weather_data = None
        self.event_data = None
        self.socioeconomic_data = None
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features"""
        logger.info("Adding temporal features...")
        
        df['date'] = pd.to_datetime(df['date'])
        
        # Time-based features
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Holiday features (US holidays)
        df['is_holiday'] = df['month'].isin([1, 7, 11, 12]).astype(int)
        
        # Cyclic encoding for temporal features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def add_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add synthetic weather features"""
        logger.info("Adding weather features...")
        
        np.random.seed(config.RANDOM_SEED)
        
        # Generate synthetic weather data
        n_records = len(df)
        df['temperature'] = np.random.normal(15, 8, n_records)  # Celsius
        df['precipitation'] = np.random.exponential(2, n_records)  # mm
        df['humidity'] = np.random.uniform(30, 90, n_records)  # %
        df['wind_speed'] = np.random.exponential(10, n_records)  # km/h
        
        # Add weather patterns that correlate with crime
        # Higher temperature slightly increases crime
        df['temperature'] = df['temperature'] + 5 * (df['crime_count'] if 'crime_count' in df.columns else 0)
        
        # Normalize weather features
        for col in ['temperature', 'precipitation', 'humidity', 'wind_speed']:
            df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
        
        return df
    
    def add_socioeconomic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add synthetic socioeconomic features by grid cell"""
        logger.info("Adding socioeconomic features...")
        
        np.random.seed(config.RANDOM_SEED)
        
        # Create synthetic socioeconomic data for each grid cell
        unique_grids = df[['lat_grid', 'lon_grid']].drop_duplicates()
        
        socio_data = []
        for _, row in unique_grids.iterrows():
            socio_data.append({
                'lat_grid': row['lat_grid'],
                'lon_grid': row['lon_grid'],
                'poverty_rate': np.random.uniform(0, 50),  # %
                'unemployment_rate': np.random.uniform(0, 20),  # %
                'population_density': np.random.uniform(100, 10000),  # per sq km
                'median_income': np.random.uniform(20000, 150000),  # $
                'education_level': np.random.uniform(0, 100)  # %
            })
        
        socio_df = pd.DataFrame(socio_data)
        
        # Merge with main dataframe
        df = df.merge(socio_df, on=['lat_grid', 'lon_grid'], how='left')
        
        # Normalize socioeconomic features
        for col in socio_df.columns:
            if col not in ['lat_grid', 'lon_grid']:
                df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
        
        return df
    
    def add_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add spatial features"""
        logger.info("Adding spatial features...")
        
        # Distance from grid center to city center (normalized)
        lat_center = (config.CHICAGO_BOUNDS['lat_min'] + config.CHICAGO_BOUNDS['lat_max']) / 2
        lon_center = (config.CHICAGO_BOUNDS['lon_min'] + config.CHICAGO_BOUNDS['lon_max']) / 2
        
        grid_size_lat = config.CHICAGO_BOUNDS['lat_max'] - config.CHICAGO_BOUNDS['lat_min']
        grid_size_lon = config.CHICAGO_BOUNDS['lon_max'] - config.CHICAGO_BOUNDS['lon_min']
        
        df['lat_normalized'] = (df['lat_grid'] / config.GRID_SIZE - 0.5) * 2
        df['lon_normalized'] = (df['lon_grid'] / config.GRID_SIZE - 0.5) * 2
        
        df['distance_from_center'] = np.sqrt(
            df['lat_normalized']**2 + df['lon_normalized']**2
        )
        
        return df
    
    def add_lagged_features(self, df: pd.DataFrame, lags: list = [1, 7, 14]) -> pd.DataFrame:
        """Add lagged crime count features"""
        logger.info(f"Adding lagged features with lags {lags}...")
        
        # Sort by grid and date
        df = df.sort_values(['lat_grid', 'lon_grid', 'date'])
        
        for lag in lags:
            df[f'crime_count_lag_{lag}'] = df.groupby(['lat_grid', 'lon_grid'])['crime_count'].shift(lag)
        
        # Forward fill missing values
        df = df.fillna(0)
        
        return df
    
    def add_rolling_statistics(self, df: pd.DataFrame, windows: list = [7, 14, 30]) -> pd.DataFrame:
        """Add rolling statistics"""
        logger.info(f"Adding rolling statistics with windows {windows}...")
        
        for window in windows:
            df[f'crime_mean_{window}d'] = df.groupby(['lat_grid', 'lon_grid'])['crime_count'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'crime_std_{window}d'] = df.groupby(['lat_grid', 'lon_grid'])['crime_count'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
        
        df = df.fillna(0)
        
        return df
    
    def create_adjacency_matrix(self) -> np.ndarray:
        """Create adjacency matrix for spatial graph"""
        logger.info("Creating spatial adjacency matrix...")
        
        n_nodes = config.GRID_SIZE * config.GRID_SIZE
        adjacency = np.zeros((n_nodes, n_nodes))
        
        for i in range(config.GRID_SIZE):
            for j in range(config.GRID_SIZE):
                node_idx = i * config.GRID_SIZE + j
                
                # Connect to 8 neighbors
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        
                        ni, nj = i + di, j + dj
                        if 0 <= ni < config.GRID_SIZE and 0 <= nj < config.GRID_SIZE:
                            neighbor_idx = ni * config.GRID_SIZE + nj
                            # Weight by distance (inverse)
                            distance = np.sqrt(di**2 + dj**2)
                            adjacency[node_idx, neighbor_idx] = 1.0 / distance
        
        # Normalize
        row_sum = adjacency.sum(axis=1, keepdims=True)
        adjacency = adjacency / (row_sum + 1e-8)
        
        return adjacency
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps"""
        logger.info("Starting feature engineering...")
        
        df = self.add_temporal_features(df)
        df = self.add_weather_features(df)
        df = self.add_socioeconomic_features(df)
        df = self.add_spatial_features(df)
        df = self.add_lagged_features(df, lags=[1, 7, 14])
        df = self.add_rolling_statistics(df, windows=[7, 14, 30])
        
        logger.info(f"Feature engineering complete. Features: {df.shape[1]}")
        
        return df

if __name__ == "__main__":
    engineer = FeatureEngineer()
