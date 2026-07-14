"""Route optimization fallback for patrol routing"""
import math
import itertools
import logging
from typing import List, Tuple, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.config import config

class PatrolRouteOptimizer:
    """Greedy round-robin patrol route optimizer when OR-Tools not available"""
    def __init__(self, hotspots: List[Tuple[int, int]], num_vehicles: int = None):
        self.hotspots = hotspots
        self.num_vehicles = num_vehicles or config.NUM_VEHICLES
        
    def _grid_to_coords(self, hotspot: Tuple[int, int], bounds: dict) -> Tuple[float, float]:
        lat_idx, lon_idx = hotspot
        lat = bounds['lat_min'] + (lat_idx + 0.5) * (bounds['lat_max'] - bounds['lat_min']) / config.GRID_SIZE
        lon = bounds['lon_min'] + (lon_idx + 0.5) * (bounds['lon_max'] - bounds['lon_min']) / config.GRID_SIZE
        return lat, lon
    
    def _euclidean_distance(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def get_routes_coordinates(self, bounds: dict) -> List[Dict]:
        """Assign hotspots to vehicles in round-robin and return GPS coordinates"""
        logger.info("Assigning hotspots to vehicles using greedy round-robin")
        
        # Convert hotspots to coordinates
        coords = [self._grid_to_coords(h, bounds) for h in self.hotspots]
        
        routes = {i: [] for i in range(self.num_vehicles)}
        distances = {i: 0.0 for i in range(self.num_vehicles)}
        
        # Simple round-robin assignment
        for idx, coord in enumerate(coords):
            vehicle_id = idx % self.num_vehicles
            if routes[vehicle_id]:
                distances[vehicle_id] += self._euclidean_distance(routes[vehicle_id][-1], coord)
            routes[vehicle_id].append(coord)
        
        # Convert to expected output format
        result = []
        for vid in routes:
            result.append({
                'vehicle_id': vid,
                'coordinates': routes[vid],
                'distance': float(distances[vid])
            })
        
        logger.info(f"Generated {len(result)} routes")
        return result

class ResourceAllocationOptimizer:
    """Simple resource allocation by hotspot severity"""
    
    def allocate_resources(self, hotspot_scores: List[float], num_resources: int) -> List[int]:
        """Return indices of hotspots to allocate resources to"""
        logger.info("Allocating resources to top hotspots")
        
        ranked = sorted(enumerate(hotspot_scores), key=lambda x: x[1], reverse=True)
        selected = [idx for idx, _ in ranked[:num_resources]]
        
        return selected

if __name__ == "__main__":
    optimizer = PatrolRouteOptimizer([(1,2),(3,4),(5,6),(10,11),(12,13)], num_vehicles=2)
    routes = optimizer.get_routes_coordinates({'lat_min':0,'lat_max':1,'lon_min':0,'lon_max':1})
    print(routes)
