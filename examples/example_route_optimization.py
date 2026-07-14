"""
Example: Route optimization for detected hotspots
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from route_optimizer import PatrolRouteOptimizer, ResourceAllocationOptimizer
from visualization import HotspotVisualizer
from config import config

def main():
    print("="*70)
    print("ROUTE OPTIMIZATION EXAMPLE")
    print("="*70)
    
    # Generate sample hotspots
    print("\n1. Generating sample hotspots...")
    np.random.seed(42)
    num_hotspots = 20
    hotspots = [
        (np.random.randint(10, 40), np.random.randint(10, 40))
        for _ in range(num_hotspots)
    ]
    print(f"   ✓ Generated {num_hotspots} hotspots")
    
    # Initialize optimizer
    print("\n2. Initializing route optimizer...")
    optimizer = PatrolRouteOptimizer(hotspots, config.GRID_SIZE)
    print("   ✓ Optimizer initialized")
    
    # Optimize routes
    print("\n3. Optimizing patrol routes...")
    routes_coords = optimizer.get_routes_coordinates(config.CHICAGO_BOUNDS)
    print(f"   ✓ Optimized {len(routes_coords)} routes")
    
    # Display route statistics
    print("\n   Route Statistics:")
    total_distance = 0
    for route in routes_coords:
        distance = route['distance']
        num_waypoints = len(route['coordinates'])
        total_distance += distance
        print(f"      Vehicle {route['vehicle_id']}: {num_waypoints} waypoints, {distance:.0f} units")
    
    print(f"\n   Total Distance: {total_distance:.0f} units")
    print(f"   Avg Distance per Vehicle: {total_distance/len(routes_coords):.0f} units")
    
    # Resource allocation
    print("\n4. Allocating resources based on hotspot density...")
    predictions = np.random.uniform(0, 1, (config.GRID_SIZE, config.GRID_SIZE))
    allocator = ResourceAllocationOptimizer(predictions)
    allocation = allocator.allocate_resources()
    
    print(f"   ✓ Allocated resources to {allocation['total_targets']} targets")
    print(f"   ✓ High-priority targets: {allocation['high_priority_count']}")
    
    print("\n   Top 5 Resource Allocation Targets:")
    for target in allocation['allocations'][:5]:
        print(f"      Cell ({target['lat_grid']}, {target['lon_grid']}) - Priority: {target['priority']} ({target['probability']:.2%})")
    
    # Visualization
    print("\n5. Creating visualizations...")
    visualizer = HotspotVisualizer(config.CHICAGO_BOUNDS, 'Chicago')
    
    # Create heatmap
    heatmap = visualizer.create_heatmap(predictions, 'example_heatmap.html')
    print("   ✓ Heatmap saved to example_heatmap.html")
    
    # Create route map
    route_map = visualizer.create_patrol_route_map(routes_coords, 'example_routes.html')
    print("   ✓ Route map saved to example_routes.html")
    
    print("\n" + "="*70)
    print("✓ ROUTE OPTIMIZATION EXAMPLE COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
