"""Route optimization for police patrol units using OR-Tools"""
import numpy as np
from typing import List, Tuple, Dict
from ortools.linear_solver import pywraplp
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config import config

class PatrolRouteOptimizer:
    """Optimize patrol routes using vehicle routing problem (VRP)"""
    
    def __init__(self, hotspots: List[Tuple[int, int]], grid_size: int = config.GRID_SIZE):
        """
        Args:
            hotspots: List of (lat_grid, lon_grid) tuples for hotspot cells
            grid_size: Size of the spatial grid
        """
        self.hotspots = hotspots
        self.grid_size = grid_size
        self.num_vehicles = config.NUM_PATROL_UNITS
    
    def calculate_distance_matrix(self) -> np.ndarray:
        """Calculate Euclidean distance matrix between hotspots and depot"""
        logger.info("Calculating distance matrix...")
        
        # Add depot (city center)
        depot = (self.grid_size // 2, self.grid_size // 2)
        locations = [depot] + self.hotspots
        
        n = len(locations)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                lat_diff = locations[i][0] - locations[j][0]
                lon_diff = locations[i][1] - locations[j][1]
                distance_matrix[i, j] = np.sqrt(lat_diff**2 + lon_diff**2)
        
        return distance_matrix.astype(int)
    
    def optimize_routes(self) -> Dict:
        """Optimize patrol routes using OR-Tools"""
        logger.info(f"Optimizing routes for {len(self.hotspots)} hotspots...")
        
        if not self.hotspots:
            logger.warning("No hotspots to optimize")
            return {'routes': [], 'total_distance': 0}
        
        distance_matrix = self.calculate_distance_matrix()
        
        # Create routing index manager
        manager = pywraplp.RoutingIndexManager(len(distance_matrix), self.num_vehicles, 0)
        
        # Create routing model
        routing = pywraplp.RoutingModel(manager)
        
        # Define distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]
        
        transit_callback_index = routing.SetArcCostEvaluatorOfAllVehicles(distance_callback)
        
        # Add distance dimension
        dimension_name = 'Distance'
        routing.AddDimension(
            transit_callback_index,
            0,  # slack
            config.MAX_DISTANCE_KM * 1000,  # vehicle max distance
            True,
            dimension_name
        )
        distance_dimension = routing.GetDimensionOrDie(dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(100)
        
        # Set first solution strategy
        search_parameters = pywraplp.RoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            pywraplp.RoutingSearchParameters.AUTOMATIC
        )
        search_parameters.local_search_metaheuristic = (
            pywraplp.RoutingSearchParameters.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = 5
        
        # Solve
        solution = routing.SolveFromAssignmentWithParameters(
            routing.DefaultRoutingSearchParameters(), None
        )
        
        if not solution:
            logger.warning("No solution found")
            return {'routes': [], 'total_distance': 0}
        
        # Extract routes
        routes = []
        total_distance = 0
        
        for vehicle_id in range(self.num_vehicles):
            route = []
            index = routing.Start(vehicle_id)
            route_distance = 0
            
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                if node > 0:  # Skip depot
                    route.append(self.hotspots[node - 1])
                
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
            
            if route:
                routes.append({
                    'vehicle_id': vehicle_id,
                    'route': route,
                    'distance': route_distance
                })
                total_distance += route_distance
        
        logger.info(f"Optimization complete. Total distance: {total_distance}, Routes: {len(routes)}")
        
        return {
            'routes': routes,
            'total_distance': total_distance,
            'num_vehicles_used': len(routes),
            'avg_distance_per_vehicle': total_distance / len(routes) if routes else 0
        }
    
    def get_routes_coordinates(self, bounds: Dict) -> List[Dict]:
        """Convert grid coordinates to lat/lon"""
        logger.info("Converting routes to coordinates...")
        
        routes_data = self.optimize_routes()
        lat_min, lat_max = bounds['lat_min'], bounds['lat_max']
        lon_min, lon_max = bounds['lon_min'], bounds['lon_max']
        
        lat_step = (lat_max - lat_min) / self.grid_size
        lon_step = (lon_max - lon_min) / self.grid_size
        
        routes_coords = []
        for route_info in routes_data['routes']:
            coords = []
            for lat_grid, lon_grid in route_info['route']:
                lat = lat_min + (lat_grid + 0.5) * lat_step
                lon = lon_min + (lon_grid + 0.5) * lon_step
                coords.append([lat, lon])
            
            routes_coords.append({
                'vehicle_id': route_info['vehicle_id'],
                'coordinates': coords,
                'distance': route_info['distance']
            })
        
        return routes_coords

class ResourceAllocationOptimizer:
    """Optimize resource allocation based on hotspot density"""
    
    def __init__(self, hotspot_predictions: np.ndarray):
        """
        Args:
            hotspot_predictions: (grid_size, grid_size) array of hotspot probabilities
        """
        self.hotspot_predictions = hotspot_predictions
    
    def allocate_resources(self) -> Dict:
        """Allocate patrol units based on hotspot density"""
        logger.info("Allocating resources...")
        
        # Flatten and sort by probability
        grid_size = self.hotspot_predictions.shape[0]
        flat_probs = self.hotspot_predictions.flatten()
        sorted_indices = np.argsort(flat_probs)[::-1]
        
        # Select top cells as resource allocation targets
        num_targets = min(config.NUM_PATROL_UNITS * 5, len(sorted_indices))
        target_indices = sorted_indices[:num_targets]
        
        allocation = []
        for idx in target_indices:
            lat_grid = idx // grid_size
            lon_grid = idx % grid_size
            prob = flat_probs[idx]
            
            if prob > config.HOTSPOT_THRESHOLD:
                allocation.append({
                    'lat_grid': lat_grid,
                    'lon_grid': lon_grid,
                    'probability': float(prob),
                    'priority': 'high' if prob > 0.85 else 'medium'
                })
        
        logger.info(f"Resource allocation: {len(allocation)} high-priority targets")
        
        return {
            'allocations': allocation,
            'total_targets': len(allocation),
            'high_priority_count': sum(1 for a in allocation if a['priority'] == 'high')
        }

class DynamicPatrolScheduler:
    """Schedule dynamic patrol routes based on time and predictions"""
    
    def __init__(self, num_shifts: int = 3):
        """
        Args:
            num_shifts: Number of shifts per day (typically 3 for 24-hour coverage)
        """
        self.num_shifts = num_shifts
    
    def create_shift_schedule(self, hotspot_predictions_by_hour: np.ndarray) -> List[Dict]:
        """Create patrol schedule for different hours
        
        Args:
            hotspot_predictions_by_hour: (24, grid_size, grid_size) array
        
        Returns:
            List of shift schedules
        """
        logger.info("Creating dynamic patrol schedule...")
        
        schedule = []
        
        # Define shift times
        shifts = [
            {'name': 'Night', 'hours': range(0, 8)},
            {'name': 'Day', 'hours': range(8, 16)},
            {'name': 'Evening', 'hours': range(16, 24)}
        ]
        
        for shift_idx, shift in enumerate(shifts):
            # Average predictions for shift hours
            shift_predictions = hotspot_predictions_by_hour[list(shift['hours'])].mean(axis=0)
            
            schedule.append({
                'shift_name': shift['name'],
                'hours': list(shift['hours']),
                'peak_hour': int(np.argmax(shift_predictions.sum(axis=1)) + shift['hours'][0]),
                'avg_intensity': float(shift_predictions.mean()),
                'recommended_units': max(1, int(config.NUM_PATROL_UNITS * shift_predictions.mean() / 0.5))
            })
        
        logger.info(f"Schedule created for {len(schedule)} shifts")
        
        return schedule

if __name__ == "__main__":
    # Test route optimizer
    hotspots = [(10, 10), (15, 15), (20, 20), (25, 25), (30, 30)]
    optimizer = PatrolRouteOptimizer(hotspots)
    routes = optimizer.get_routes_coordinates(config.CHICAGO_BOUNDS)
    print(f"Generated {len(routes)} routes")
