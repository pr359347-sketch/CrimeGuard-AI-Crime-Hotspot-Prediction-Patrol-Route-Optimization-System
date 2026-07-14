"""Visualization utilities: folium heatmaps and route maps"""
import folium
from folium.plugins import HeatMap
import os
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.config import config

class HotspotVisualizer:
    def __init__(self, bounds: dict, city: str = 'Chicago'):
        self.bounds = bounds
        self.city = city
        self.center = ((bounds['lat_min'] + bounds['lat_max'])/2, (bounds['lon_min'] + bounds['lon_max'])/2)

    def create_heatmap(self, predictions, outpath: str):
        logger.info(f"Creating heatmap at {outpath}")
        m = folium.Map(location=self.center, zoom_start=11)
        
        # Flatten predictions into lat/lon pairs with weights
        heat_data = []
        grid_size = len(predictions)
        for i in range(grid_size):
            for j in range(grid_size):
                lat = self.bounds['lat_min'] + (i + 0.5) * (self.bounds['lat_max'] - self.bounds['lat_min']) / grid_size
                lon = self.bounds['lon_min'] + (j + 0.5) * (self.bounds['lon_max'] - self.bounds['lon_min']) / grid_size
                weight = float(predictions[i, j])
                heat_data.append([lat, lon, weight])
        
        HeatMap(heat_data, radius=10, blur=15, max_zoom=12).add_to(m)
        
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        m.save(outpath)
        logger.info(f"Heatmap saved: {outpath}")

    def create_patrol_route_map(self, routes: List[Dict], outpath: str):
        logger.info(f"Creating patrol route map at {outpath}")
        m = folium.Map(location=self.center, zoom_start=11)
        
        # Draw each route
        for r in routes:
            coords = r['coordinates']
            if not coords:
                continue
            folium.PolyLine(coords, color='blue', weight=3, opacity=0.8).add_to(m)
            for idx, c in enumerate(coords):
                folium.CircleMarker(location=c, radius=3, color='red', fill=True).add_to(m)
        
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        m.save(outpath)
        logger.info(f"Route map saved: {outpath}")

class StatisticsVisualizer:
    def plot_top_hotspots(self, predictions, top_n: int = 10):
        # placeholder for matplotlib/plotly summaries
        pass

if __name__ == "__main__":
    viz = HotspotVisualizer({'lat_min':41.6,'lat_max':42.05,'lon_min':-87.95,'lon_max':-87.5}, 'Chicago')
    import numpy as np
    preds = np.random.rand(50,50)
    viz.create_heatmap(preds, 'example_heatmap.html')
    viz.create_patrol_route_map([{'coordinates':[(41.8,-87.7),(41.85,-87.6)]}], 'example_routes.html')
