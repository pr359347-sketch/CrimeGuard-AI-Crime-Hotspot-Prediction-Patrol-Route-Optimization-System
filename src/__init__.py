"""Crime Hotspot Prediction & Patrol Optimizer Package"""

__version__ = "1.0.0"
__author__ = "AI Development Team"
__description__ = "Spatial-temporal ML for crime prediction and patrol route optimization"

from .config import config
from .data_loader import CrimeDataLoader
from .feature_engineering import FeatureEngineer
from .st_gnn_model import CrimeHotspotPredictor, ModelTrainer
from .route_optimizer import PatrolRouteOptimizer, ResourceAllocationOptimizer, DynamicPatrolScheduler
from .visualization import HotspotVisualizer, StatisticsVisualizer
from .model_evaluation import PredictionEvaluator, HotspotDetectionValidator
from .prediction_pipeline import CrimeHotspotPipeline

__all__ = [
    'config',
    'CrimeDataLoader',
    'FeatureEngineer',
    'CrimeHotspotPredictor',
    'ModelTrainer',
    'PatrolRouteOptimizer',
    'ResourceAllocationOptimizer',
    'DynamicPatrolScheduler',
    'HotspotVisualizer',
    'StatisticsVisualizer',
    'PredictionEvaluator',
    'HotspotDetectionValidator',
    'CrimeHotspotPipeline'
]
