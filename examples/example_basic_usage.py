"""
Basic usage example: Load data and generate predictions
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from prediction_pipeline import CrimeHotspotPipeline
from config import config

def main():
    print("="*70)
    print("CRIME HOTSPOT PREDICTION - BASIC EXAMPLE")
    print("="*70)
    
    # Initialize pipeline for Chicago
    print("\n1. Initializing pipeline for Chicago...")
    pipeline = CrimeHotspotPipeline('Chicago')
    
    # Load and preprocess data
    print("\n2. Loading and preprocessing crime data...")
    pipeline.load_and_preprocess_data()
    print(f"   ✓ Loaded data shape: {pipeline.data.shape}")
    
    # Engineer features
    print("\n3. Engineering features...")
    features_df = pipeline.engineer_features()
    print(f"   ✓ Feature columns: {features_df.shape[1]}")
    
    # Train model
    print("\n4. Training spatial-temporal model...")
    pipeline.train_model(epochs=5)  # Quick training for demo
    print("   ✓ Model training complete")
    
    # Make predictions
    print("\n5. Generating hotspot predictions...")
    predictions = pipeline.predict_hotspots()
    print(f"   ✓ Predictions shape: {predictions.shape}")
    
    # Detect hotspots
    print("\n6. Detecting hotspots...")
    hotspots = pipeline.detect_hotspots()
    print(f"   ✓ Hotspots detected: {len(hotspots)}")
    
    # Display top hotspots
    if hotspots:
        print("\n   Top 10 hotspots (grid coordinates):")
        from visualization import StatisticsVisualizer
        top_hotspots = StatisticsVisualizer.get_top_hotspots(predictions, top_n=10)
        for hs in top_hotspots:
            print(f"      Rank {hs['rank']}: Cell ({hs['lat_grid']}, {hs['lon_grid']}) - {hs['probability']:.2%}")
    
    print("\n" + "="*70)
    print("✓ EXAMPLE COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
