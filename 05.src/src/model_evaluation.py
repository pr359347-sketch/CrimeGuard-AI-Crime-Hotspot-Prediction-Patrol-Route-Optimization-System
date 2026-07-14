"""Model evaluation metrics for crime prediction"""
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, precision_score, recall_score, f1_score
)
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.config import config

class PredictionEvaluator:
    """Evaluate model predictions"""
    
    @staticmethod
    def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate regression metrics"""
        logger.info("Computing regression metrics...")
        
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2)
        }
    
    @staticmethod
    def classification_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                               threshold: float = config.HOTSPOT_THRESHOLD) -> Dict:
        """Calculate classification metrics for hotspot detection"""
        logger.info("Computing classification metrics...")
        
        y_pred = (y_pred_proba > threshold).astype(int)
        y_true_binary = (y_true > threshold).astype(int)
        
        # Handle edge cases
        if len(np.unique(y_true_binary)) < 2:
            logger.warning("Only one class in true labels")
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'roc_auc': 0.0
            }
        
        try:
            precision = precision_score(y_true_binary, y_pred, zero_division=0)
            recall = recall_score(y_true_binary, y_pred, zero_division=0)
            f1 = f1_score(y_true_binary, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_true_binary, y_pred_proba)
        except Exception as e:
            logger.warning(f"Error computing classification metrics: {e}")
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'roc_auc': 0.0
            }
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'roc_auc': float(roc_auc)
        }
    
    @staticmethod
    def spatial_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate spatial-specific metrics"""
        logger.info("Computing spatial metrics...")
        
        # Hit rate: % of high-crime cells correctly predicted as high-risk
        true_hotspots = y_true > config.HOTSPOT_THRESHOLD
        pred_hotspots = y_pred > config.HOTSPOT_THRESHOLD
        
        if true_hotspots.sum() == 0:
            hit_rate = 0.0
        else:
            hit_rate = np.sum(true_hotspots & pred_hotspots) / true_hotspots.sum()
        
        # False alarm rate: % of low-crime cells incorrectly predicted as high-risk
        true_safe = y_true <= config.HOTSPOT_THRESHOLD
        if true_safe.sum() == 0:
            false_alarm_rate = 0.0
        else:
            false_alarm_rate = np.sum(pred_hotspots & true_safe) / true_safe.sum()
        
        # Coverage: % of total crime predicted to occur in predicted hotspots
        if y_true.sum() == 0:
            coverage = 0.0
        else:
            coverage = y_true[pred_hotspots].sum() / y_true.sum()
        
        return {
            'hit_rate': float(hit_rate),
            'false_alarm_rate': float(false_alarm_rate),
            'coverage': float(coverage)
        }
    
    @staticmethod
    def temporal_metrics(y_true_series: np.ndarray, y_pred_series: np.ndarray) -> Dict:
        """Calculate temporal prediction accuracy"""
        logger.info("Computing temporal metrics...")
        
        # Metric for different forecast horizons
        metrics = {}
        
        if len(y_true_series) >= 7:
            for horizon in [1, 3, 7]:
                if horizon <= len(y_true_series):
                    mae_h = mean_absolute_error(
                        y_true_series[:len(y_true_series)-horizon+1],
                        y_pred_series[horizon-1:len(y_pred_series)]
                    )
                    metrics[f'mae_{horizon}day'] = float(mae_h)
        
        return metrics
    
    @staticmethod
    def evaluate_comprehensive(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Comprehensive evaluation"""
        logger.info("Running comprehensive evaluation...")
        
        return {
            'regression': PredictionEvaluator.regression_metrics(y_true, y_pred),
            'classification': PredictionEvaluator.classification_metrics(y_true, y_pred),
            'spatial': PredictionEvaluator.spatial_metrics(y_true, y_pred)
        }

class HotspotDetectionValidator:
    """Validate hotspot detection performance"""
    
    @staticmethod
    def validate_hotspots(predictions: np.ndarray, min_size: int = config.MIN_HOTSPOT_SIZE) -> Dict:
        """Validate detected hotspots"""
        logger.info("Validating hotspot detection...")
        
        hotspot_cells = predictions > config.HOTSPOT_THRESHOLD
        num_hotspots = np.sum(hotspot_cells)
        
        # Analyze hotspot clustering
        from scipy import ndimage
        labeled_array, num_features = ndimage.label(hotspot_cells)
        
        cluster_sizes = []
        valid_clusters = 0
        
        for cluster_id in range(1, num_features + 1):
            cluster_size = np.sum(labeled_array == cluster_id)
            cluster_sizes.append(cluster_size)
            
            if cluster_size >= min_size:
                valid_clusters += 1
        
        return {
            'total_hotspot_cells': int(num_hotspots),
            'num_clusters': int(num_features),
            'valid_clusters': int(valid_clusters),
            'avg_cluster_size': float(np.mean(cluster_sizes)) if cluster_sizes else 0,
            'largest_cluster': int(np.max(cluster_sizes)) if cluster_sizes else 0,
            'cluster_coverage': float(num_hotspots / predictions.size)
        }

if __name__ == "__main__":
    # Test evaluation
    y_true = np.random.uniform(0, 1, (100, 100))
    y_pred = np.random.uniform(0, 1, (100, 100))
    
    evaluator = PredictionEvaluator()
    metrics = evaluator.evaluate_comprehensive(y_true, y_pred)
    
    print("Evaluation metrics:")
    print(f"Regression R2: {metrics['regression']['r2']:.4f}")
    print(f"Classification F1: {metrics['classification']['f1']:.4f}")
    print(f"Spatial hit rate: {metrics['spatial']['hit_rate']:.4f}")
