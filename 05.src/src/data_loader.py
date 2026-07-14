"""Data loading and preprocessing for crime datasets"""
import pandas as pd
import numpy as np
import requests
import os
from typing import Tuple, Dict
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.config import config

class CrimeDataLoader:
    """Load and preprocess crime data from Chicago and LA"""
    
    def __init__(self):
        self.chicago_data = None
        self.la_data = None
        
    def download_chicago_crimes(self, num_records: int = 50000) -> pd.DataFrame:
        """Download Chicago crime data from Socrata API"""
        logger.info(f"Downloading Chicago crime data ({num_records} records)...")
        
        try:
            # Chicago uses Socrata Open Data API
            url = "https://data.cityofchicago.org/api/views/ijzp-q8t2/rows.json"
            params = {
                "$limit": num_records,
                "$order": "date DESC"
            }
            
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                
                # Extract records
                records = []
                if 'data' in data:
                    for record in data['data']:
                        try:
                            records.append({
                                'date': record[16],
                                'latitude': float(record[18]) if record[18] else None,
                                'longitude': float(record[19]) if record[19] else None,
                                'crime_type': record[15],
                                'district': record[11],
                                'city': 'Chicago'
                            })
                        except (ValueError, IndexError, TypeError):
                            continue
                
                df = pd.DataFrame(records)
                df = df.dropna(subset=['latitude', 'longitude'])
                logger.info(f"Downloaded {len(df)} Chicago crime records")
                return df
            else:
                logger.warning(f"API request failed with status {response.status_code}")
                return self._generate_synthetic_chicago_data(num_records)
                
        except Exception as e:
            logger.warning(f"Error downloading Chicago data: {e}. Using synthetic data.")
            return self._generate_synthetic_chicago_data(num_records)
    
    def download_la_crimes(self, num_records: int = 50000) -> pd.DataFrame:
        """Download LA crime data from LA Open Data API"""
        logger.info(f"Downloading LA crime data ({num_records} records)...")
        
        try:
            # LA uses Socrata Open Data API
            url = "https://data.lacity.gov/api/views/2nrs-mtv8/rows.json"
            params = {
                "$limit": num_records,
                "$order": "date_occ DESC"
            }
            
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                
                records = []
                if 'data' in data:
                    for record in data['data']:
                        try:
                            records.append({
                                'date': record[10],
                                'latitude': float(record[27]) if record[27] else None,
                                'longitude': float(record[28]) if record[28] else None,
                                'crime_type': record[14],
                                'district': record[6],
                                'city': 'LA'
                            })
                        except (ValueError, IndexError, TypeError):
                            continue
                
                df = pd.DataFrame(records)
                df = df.dropna(subset=['latitude', 'longitude'])
                logger.info(f"Downloaded {len(df)} LA crime records")
                return df
            else:
                logger.warning(f"API request failed with status {response.status_code}")
                return self._generate_synthetic_la_data(num_records)
                
        except Exception as e:
            logger.warning(f"Error downloading LA data: {e}. Using synthetic data.")
            return self._generate_synthetic_la_data(num_records)
    
    def _generate_synthetic_chicago_data(self, num_records: int) -> pd.DataFrame:
        """Generate synthetic Chicago crime data for testing"""
        logger.info("Generating synthetic Chicago data...")
        
        np.random.seed(config.RANDOM_SEED)
        bounds = config.CHICAGO_BOUNDS
        
        dates = [datetime.now() - timedelta(days=int(x)) 
                for x in np.random.uniform(0, 365, num_records)]
        
        data = {
            'date': dates,
            'latitude': np.random.uniform(bounds['lat_min'], bounds['lat_max'], num_records),
            'longitude': np.random.uniform(bounds['lon_min'], bounds['lon_max'], num_records),
            'crime_type': np.random.choice(
                ['Theft', 'Robbery', 'Burglary', 'Assault', 'Motor Vehicle Theft'],
                num_records
            ),
            'district': np.random.choice(range(1, 23), num_records),
            'city': 'Chicago'
        }
        
        return pd.DataFrame(data)
    
    def _generate_synthetic_la_data(self, num_records: int) -> pd.DataFrame:
        """Generate synthetic LA crime data for testing"""
        logger.info("Generating synthetic LA data...")
        
        np.random.seed(config.RANDOM_SEED + 1)
        bounds = config.LA_BOUNDS
        
        dates = [datetime.now() - timedelta(days=int(x)) 
                for x in np.random.uniform(0, 365, num_records)]
        
        data = {
            'date': dates,
            'latitude': np.random.uniform(bounds['lat_min'], bounds['lat_max'], num_records),
            'longitude': np.random.uniform(bounds['lon_min'], bounds['lon_max'], num_records),
            'crime_type': np.random.choice(
                ['Theft', 'Robbery', 'Burglary', 'Assault', 'Motor Vehicle Theft'],
                num_records
            ),
            'district': np.random.choice(range(1, 22), num_records),
            'city': 'LA'
        }
        
        return pd.DataFrame(data)
    
    def preprocess_crimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess crime data"""
        logger.info("Preprocessing crime data...")
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        
        # Filter by bounding box based on city
        if df['city'].iloc[0] == 'Chicago':
            bounds = config.CHICAGO_BOUNDS
        else:
            bounds = config.LA_BOUNDS
        
        df = df[
            (df['latitude'] >= bounds['lat_min']) &
            (df['latitude'] <= bounds['lat_max']) &
            (df['longitude'] >= bounds['lon_min']) &
            (df['longitude'] <= bounds['lon_max'])
        ]
        
        # Add temporal features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        df['week'] = df['date'].dt.isocalendar().week
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        logger.info(f"Preprocessed data shape: {df.shape}")
        return df
    
    def create_spatial_grid(self, df: pd.DataFrame, city: str) -> pd.DataFrame:
        """Create spatial grid representation (50x50)"""
        logger.info(f"Creating spatial grid for {city}...")
        
        bounds = config.CHICAGO_BOUNDS if city == 'Chicago' else config.LA_BOUNDS
        
        # Create grid bins
        lat_bins = np.linspace(bounds['lat_min'], bounds['lat_max'], config.GRID_SIZE + 1)
        lon_bins = np.linspace(bounds['lon_min'], bounds['lon_max'], config.GRID_SIZE + 1)
        
        # Assign each crime to a grid cell
        df['lat_grid'] = pd.cut(df['latitude'], bins=lat_bins, labels=False)
        df['lon_grid'] = pd.cut(df['longitude'], bins=lon_bins, labels=False)
        
        return df
    
    def aggregate_to_grid_time_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate crimes to grid cells by day"""
        logger.info("Aggregating to grid time series...")
        
        # Group by date, lat_grid, lon_grid
        df['date_only'] = df['date'].dt.date
        
        grid_ts = df.groupby(['date_only', 'lat_grid', 'lon_grid']).size().reset_index(name='crime_count')
        grid_ts['date'] = pd.to_datetime(grid_ts['date_only'])
        grid_ts = grid_ts.drop('date_only', axis=1)
        
        # Fill missing grid cells with 0
        all_dates = pd.date_range(grid_ts['date'].min(), grid_ts['date'].max(), freq='D')
        full_grid = pd.MultiIndex.from_product(
            [all_dates, range(config.GRID_SIZE), range(config.GRID_SIZE)],
            names=['date', 'lat_grid', 'lon_grid']
        )
        
        grid_ts = grid_ts.set_index(['date', 'lat_grid', 'lon_grid']).reindex(full_grid, fill_value=0).reset_index()
        
        logger.info(f"Grid time series shape: {grid_ts.shape}")
        return grid_ts
    
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and preprocess all crime data"""
        
        # Download/generate data
        chicago_df = self.download_chicago_crimes()
        la_df = self.download_la_crimes()
        
        # Preprocess
        chicago_df = self.preprocess_crimes(chicago_df)
        la_df = self.preprocess_crimes(la_df)
        
        # Create spatial grids
        chicago_df = self.create_spatial_grid(chicago_df, 'Chicago')
        la_df = self.create_spatial_grid(la_df, 'LA')
        
        # Aggregate to grid time series
        chicago_grid = self.aggregate_to_grid_time_series(chicago_df)
        la_grid = self.aggregate_to_grid_time_series(la_df)
        
        return chicago_grid, la_grid

if __name__ == "__main__":
    loader = CrimeDataLoader()
    chicago_data, la_data = loader.load_all_data()
    
    print("Chicago data shape:", chicago_data.shape)
    print("LA data shape:", la_data.shape)
