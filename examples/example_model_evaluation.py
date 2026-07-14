"""
Example: Model evaluation and metrics
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from model_evaluation import PredictionEvaluator, HotspotDetectionValidator
from config import config

def main():
    print("="*70)
    print("MODEL EVALUATION EXAMPLE")
    print("="*70)
    
    # Generate synthetic predictions
    print("\n1. Generating synthetic predictions...")
    np.random.seed(42)
    y_true = np.random.exponential(scale=2, size=(config.GRID_SIZE, config.GRID_SIZE))
    y_pred = y_true + np.random.normal(0, 0.5, (config.GRID_SIZE, config.GRID_SIZE))
    
    # Normalize to [0, 1]
    y_true = (y_true - y_true.min()) / (y_true.max() - y_true.min() + 1e-8)
    y_pred = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min() + 1e-8)
    
    print(f"   ✓ True predictions shape: {y_true.shape}")
    print(f"   ✓ Model predictions shape: {y_pred.shape}")
    
    # Regression metrics
    print("\n2. Computing regression metrics...")
    evaluator = PredictionEvaluator()
    reg_metrics = evaluator.regression_metrics(y_true, y_pred)
    
    print("   Regression Metrics:")
    print(f"      MSE:  {reg_metrics['mse']:.4f}")
    print(f"      MAE:  {reg_metrics['mae']:.4f}")
    print(f"      RMSE: {reg_metrics['rmse']:.4f}")
    print(f"      R²:   {reg_metrics['r2']:.4f}")
    
    # Classification metrics
    print("\n3. Computing classification metrics...")
    clf_metrics = evaluator.classification_metrics(y_true, y_pred, threshold=config.HOTSPOT_THRESHOLD)
    
    print("   Classification Metrics (Hotspot Detection):")
    print(f"      Precision: {clf_metrics['precision']:.4f}")
    print(f"      Recall:    {clf_metrics['recall']:.4f}")
    print(f"      F1 Score:  {clf_metrics['f1']:.4f}")
    print(f"      ROC-AUC:   {clf_metrics['roc_auc']:.4f}")
    
    # Spatial metrics
    print("\n4. Computing spatial metrics...")
    spatial_metrics = evaluator.spatial_metrics(y_true, y_pred)
    
    print("   Spatial Metrics:")
    print(f"      Hit Rate:         {spatial_metrics['hit_rate']:.4f}")
    print(f"      False Alarm Rate: {spatial_metrics['false_alarm_rate']:.4f}")
    print(f"      Coverage:         {spatial_metrics['coverage']:.4f}")
    
    # Hotspot validation
    print("\n5. Validating hotspot detection...")
    validator = HotspotDetectionValidator()
    validation = validator.validate_hotspots(y_pred)
    
    print("   Hotspot Validation:")
    print(f"      Total Hotspot Cells: {validation['total_hotspot_cells']}")
    print(f"      Number of Clusters:  {validation['num_clusters']}")
    print(f"      Valid Clusters:      {validation['valid_clusters']}")
    print(f"      Avg Cluster Size:    {validation['avg_cluster_size']:.1f}")
    print(f"      Largest Cluster:     {validation['largest_cluster']}")
    print(f"      Coverage:            {validation['cluster_coverage']:.2%}")
    
    # Comprehensive evaluation
    print("\n6. Running comprehensive evaluation...")
    comprehensive = evaluator.evaluate_comprehensive(y_true, y_pred)
    
    print("   Comprehensive Results:")
    print(f"      Regression R²: {comprehensive['regression']['r2']:.4f}")
    print(f"      Classification F1: {comprehensive['classification']['f1']:.4f}")
    print(f"      Spatial Hit Rate: {comprehensive['spatial']['hit_rate']:.4f}")
    
    # Summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Model Performance: {'✓ GOOD' if comprehensive['regression']['r2'] > 0.5 else '✗ NEEDS IMPROVEMENT'}")
    print(f"Hotspot Detection: {'✓ GOOD' if comprehensive['classification']['f1'] > 0.5 else '✗ NEEDS IMPROVEMENT'}")
    print(f"Spatial Accuracy: {'✓ GOOD' if comprehensive['spatial']['hit_rate'] > 0.5 else '✗ NEEDS IMPROVEMENT'}")
    print("="*70)

if __name__ == "__main__":
    main()
