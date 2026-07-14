"""Main entry point for Crime Hotspot Prediction & Patrol Optimizer"""
import sys
import os
import argparse
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '05.src', 'src'))

from src.config import config
from src.prediction_pipeline import CrimeHotspotPipeline
from src.visualization import HotspotVisualizer
from src.route_optimizer import PatrolRouteOptimizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Crime Hotspot Prediction & Patrol Optimizer'
    )
    
    parser.add_argument(
        '--city',
        choices=['Chicago', 'LA'],
        default='Chicago',
        help='City to analyze (default: Chicago)'
    )
    
    parser.add_argument(
        '--mode',
        choices=['predict', 'optimize', 'visualize', 'full'],
        default='full',
        help='Operation mode (default: full)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs (default: 10)'
    )
    
    parser.add_argument(
        '--output-dir',
        default=config.OUTPUT_DIR,
        help='Output directory for results (default: outputs/)'
    )
    
    args = parser.parse_args()
    
    logger.info(f"Starting Crime Hotspot Pipeline for {args.city}")
    logger.info(f"Mode: {args.mode}")
    
    # Initialize pipeline
    pipeline = CrimeHotspotPipeline(args.city)
    
    try:
        if args.mode in ['predict', 'full']:
            logger.info("="*60)
            logger.info("PHASE 1: DATA LOADING & PREPROCESSING")
            logger.info("="*60)
            pipeline.load_and_preprocess_data()
            
            logger.info("="*60)
            logger.info("PHASE 2: FEATURE ENGINEERING")
            logger.info("="*60)
            pipeline.engineer_features()
            
            logger.info("="*60)
            logger.info("PHASE 3: MODEL TRAINING")
            logger.info("="*60)
            pipeline.train_model(epochs=args.epochs)
            
            logger.info("="*60)
            logger.info("PHASE 4: HOTSPOT PREDICTION")
            logger.info("="*60)
            predictions = pipeline.predict_hotspots()
            hotspots = pipeline.detect_hotspots()
            
            logger.info(f"✓ Generated predictions for {args.city}")
            logger.info(f"✓ Detected {len(hotspots)} hotspots")
        
        if args.mode in ['optimize', 'full']:
            if pipeline.hotspots is None or len(pipeline.hotspots) == 0:
                logger.warning("No hotspots detected. Skipping route optimization.")
            else:
                logger.info("="*60)
                logger.info("PHASE 5: ROUTE OPTIMIZATION")
                logger.info("="*60)
                
                # Sample hotspots if too many
                hotspots_to_optimize = pipeline.hotspots
                if len(hotspots_to_optimize) > 100:
                    import numpy as np
                    np.random.seed(42)
                    indices = np.random.choice(len(hotspots_to_optimize), 100, replace=False)
                    hotspots_to_optimize = [hotspots_to_optimize[i] for i in indices]
                
                bounds = pipeline.bounds
                optimizer = PatrolRouteOptimizer(hotspots_to_optimize, config.GRID_SIZE)
                routes = optimizer.get_routes_coordinates(bounds)
                pipeline.routes = routes
                
                logger.info(f"✓ Optimized {len(routes)} patrol routes")
        
        if args.mode in ['visualize', 'full']:
            logger.info("="*60)
            logger.info("PHASE 6: VISUALIZATION")
            logger.info("="*60)
            
            viz_paths = pipeline.generate_visualizations(args.output_dir)
            logger.info(f"✓ Generated visualizations:")
            for viz_type, path in viz_paths.items():
                logger.info(f"  - {viz_type}: {path}")
        
        logger.info("="*60)
        logger.info("✓ PIPELINE EXECUTION COMPLETE")
        logger.info("="*60)
        
        # Save summary
        summary = {
            'city': args.city,
            'predictions_shape': pipeline.predictions.shape if pipeline.predictions is not None else None,
            'hotspots_detected': len(pipeline.hotspots) if pipeline.hotspots else 0,
            'routes_optimized': len(pipeline.routes) if pipeline.routes else 0,
            'output_directory': args.output_dir
        }
        
        logger.info("\nSUMMARY:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
