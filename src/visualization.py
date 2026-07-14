"""Visualization utilities for crime hotspot predictions"""
import folium
from folium import plugins
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config import config

class HotspotVisualizer:
    """Create interactive visualizations for crime hotspots"""
    
    def __init__(self, bounds: Dict, city: str = 'Chicago'):
        """
        Args:
            bounds: Geographic bounds (lat_min, lat_max, lon_min, lon_max)
            city: City name
        """
        self.bounds = bounds
        self.city = city
        self.center_lat = (bounds['lat_min'] + bounds['lat_max']) / 2
        self.center_lon = (bounds['lon_min'] + bounds['lon_max']) / 2
    
    def create_heatmap(self, predictions: np.ndarray, output_path: str = None) -> folium.Map:
        """Create interactive heatmap from predictions
        
        Args:
            predictions: (grid_size, grid_size) array of hotspot probabilities
            output_path: Path to save HTML map
        
        Returns:
            folium.Map object
        """
        logger.info(f"Creating heatmap for {self.city}...")
        
        # Create base map
        m = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # Convert grid predictions to coordinates
        lat_step = (self.bounds['lat_max'] - self.bounds['lat_min']) / config.GRID_SIZE
        lon_step = (self.bounds['lon_max'] - self.bounds['lon_min']) / config.GRID_SIZE
        
        heat_data = []
        
        for lat_idx in range(config.GRID_SIZE):
            for lon_idx in range(config.GRID_SIZE):
                lat = self.bounds['lat_min'] + (lat_idx + 0.5) * lat_step
                lon = self.bounds['lon_min'] + (lon_idx + 0.5) * lon_step
                intensity = float(predictions[lat_idx, lon_idx])
                
                if intensity > config.HOTSPOT_THRESHOLD:
                    heat_data.append([lat, lon, intensity])
                    
                    # Add circle markers for high-probability hotspots
                    if intensity > 0.85:
                        folium.CircleMarker(
                            location=[lat, lon],
                            radius=5,
                            popup=f"Hotspot: {intensity:.2%}",
                            color='red',
                            fill=True,
                            fillColor='red',
                            fillOpacity=intensity,
                            weight=2
                        ).add_to(m)
                    elif intensity > 0.70:
                        folium.CircleMarker(
                            location=[lat, lon],
                            radius=3,
                            popup=f"Hotspot: {intensity:.2%}",
                            color='orange',
                            fill=True,
                            fillColor='orange',
                            fillOpacity=intensity * 0.7,
                            weight=1
                        ).add_to(m)
        
        # Add heatmap layer
        if heat_data:
            plugins.HeatMap(
                heat_data,
                name='Crime Hotspots',
                min_opacity=0.3,
                max_zoom=19,
                radius=20,
                blur=15,
                gradient={0.0: 'blue', 0.5: 'yellow', 0.7: 'orange', 1.0: 'red'}
            ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save if path provided
        if output_path:
            m.save(output_path)
            logger.info(f"Heatmap saved to {output_path}")
        
        return m
    
    def create_patrol_route_map(self, routes: List[Dict], output_path: str = None) -> folium.Map:
        """Create map with patrol routes
        
        Args:
            routes: List of route dictionaries with coordinates
            output_path: Path to save HTML map
        
        Returns:
            folium.Map object
        """
        logger.info(f"Creating patrol route map for {self.city}...")
        
        # Create base map
        m = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # Define colors for different routes
        colors = [
            'red', 'blue', 'green', 'purple', 'orange',
            'darkred', 'darkblue', 'darkgreen', 'darkpurple', 'cadetblue'
        ]
        
        for route_idx, route in enumerate(routes):
            color = colors[route_idx % len(colors)]
            coordinates = route['coordinates']
            
            if coordinates:
                # Add route line
                folium.PolyLine(
                    locations=coordinates,
                    color=color,
                    weight=3,
                    opacity=0.8,
                    popup=f"Vehicle {route['vehicle_id']}"
                ).add_to(m)
                
                # Add start point (depot)
                folium.CircleMarker(
                    location=coordinates[0],
                    radius=8,
                    popup=f"Start - Vehicle {route['vehicle_id']}",
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.8,
                    weight=2
                ).add_to(m)
                
                # Add waypoints
                for point_idx, coord in enumerate(coordinates[1:], 1):
                    folium.CircleMarker(
                        location=coord,
                        radius=5,
                        popup=f"Stop {point_idx} - Vehicle {route['vehicle_id']}",
                        color=color,
                        fill=True,
                        fillColor=color,
                        fillOpacity=0.6,
                        weight=1
                    ).add_to(m)
        
        folium.LayerControl().add_to(m)
        
        if output_path:
            m.save(output_path)
            logger.info(f"Route map saved to {output_path}")
        
        return m
    
    def create_comparison_map(self, predictions_past: np.ndarray, 
                             predictions_future: np.ndarray, 
                             output_path: str = None) -> folium.Map:
        """Create comparison map of past vs predicted hotspots"""
        logger.info(f"Creating comparison map for {self.city}...")
        
        m = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        lat_step = (self.bounds['lat_max'] - self.bounds['lat_min']) / config.GRID_SIZE
        lon_step = (self.bounds['lon_max'] - self.bounds['lon_min']) / config.GRID_SIZE
        
        # Add past hotspots in blue
        for lat_idx in range(config.GRID_SIZE):
            for lon_idx in range(config.GRID_SIZE):
                if predictions_past[lat_idx, lon_idx] > 0.5:
                    lat = self.bounds['lat_min'] + (lat_idx + 0.5) * lat_step
                    lon = self.bounds['lon_min'] + (lon_idx + 0.5) * lon_step
                    
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=3,
                        popup=f"Past: {predictions_past[lat_idx, lon_idx]:.2%}",
                        color='blue',
                        fill=True,
                        fillColor='blue',
                        fillOpacity=0.4,
                        weight=1
                    ).add_to(m)
        
        # Add future hotspots in red
        for lat_idx in range(config.GRID_SIZE):
            for lon_idx in range(config.GRID_SIZE):
                if predictions_future[lat_idx, lon_idx] > 0.5:
                    lat = self.bounds['lat_min'] + (lat_idx + 0.5) * lat_step
                    lon = self.bounds['lon_min'] + (lon_idx + 0.5) * lon_step
                    
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=4,
                        popup=f"Predicted: {predictions_future[lat_idx, lon_idx]:.2%}",
                        color='red',
                        fill=True,
                        fillColor='red',
                        fillOpacity=0.6,
                        weight=1
                    ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 200px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <b>Legend</b><br>
        <i class="fa fa-circle" style="color:blue"></i> Past Hotspots<br>
        <i class="fa fa-circle" style="color:red"></i> Predicted Hotspots<br>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        if output_path:
            m.save(output_path)
            logger.info(f"Comparison map saved to {output_path}")
        
        return m

class StatisticsVisualizer:
    """Visualize prediction statistics"""
    
    @staticmethod
    def get_hotspot_stats(predictions: np.ndarray) -> Dict:
        """Calculate statistics about hotspots"""
        high_prob = np.sum(predictions > config.HOTSPOT_THRESHOLD)
        max_prob = np.max(predictions)
        mean_prob = np.mean(predictions)
        
        return {
            'high_risk_cells': int(high_prob),
            'max_probability': float(max_prob),
            'mean_probability': float(mean_prob),
            'total_cells': predictions.size,
            'high_risk_percentage': float(100 * high_prob / predictions.size)
        }
    
    @staticmethod
    def get_top_hotspots(predictions: np.ndarray, top_n: int = 10) -> List[Dict]:
        """Get top N hotspots by probability"""
        grid_size = predictions.shape[0]
        flat_probs = predictions.flatten()
        top_indices = np.argsort(flat_probs)[::-1][:top_n]
        
        hotspots = []
        for rank, idx in enumerate(top_indices, 1):
            lat_grid = idx // grid_size
            lon_grid = idx % grid_size
            hotspots.append({
                'rank': rank,
                'lat_grid': int(lat_grid),
                'lon_grid': int(lon_grid),
                'probability': float(flat_probs[idx])
            })
        
        return hotspots

if __name__ == "__main__":
    # Test visualization
    visualizer = HotspotVisualizer(config.CHICAGO_BOUNDS, 'Chicago')
    test_predictions = np.random.uniform(0, 1, (config.GRID_SIZE, config.GRID_SIZE))
    
    # Create heatmap
    m = visualizer.create_heatmap(test_predictions, 'test_heatmap.html')
    print("Heatmap created successfully")
