import argparse
import pandas as pd
import numpy as np
import torch
from src.models import STGNN
from src.trainer import Trainer
from src.inference import Predictor
from src.graph_builder import build_edge_index

def run_cli_training(save_path=None):
    print("Initializing CrimeGuard-AI Training Pipeline...")
    
    # Mock data initialization
    crime_df = pd.DataFrame({'timestamp': [1]*10, 'nodes': range(10), 'crime': np.random.rand(10), 'region_id': [1]*10})
    weather_df = pd.DataFrame({'timestamp': [1], 'temp': [20]})
    event_df = pd.DataFrame({'timestamp': [1], 'event_type': ['test']})
    socio_df = pd.DataFrame({'region_id': [1], 'income': [50000]})
    coords = np.random.rand(10, 2)
    
    model = STGNN(num_nodes=10, num_features=5, hidden_dim=16, seq_len=3, out_dim=1)
    trainer = Trainer(model)
    
    print("Starting training...")
    trainer.train(crime_df, weather_df, event_df, socio_df, coords, epochs=5)
    
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Model weights saved to {save_path}")
        
    print("Training complete.")

def run_cli_predict():
    print("Running inference...")
    # Load model and run prediction
    predictor = Predictor('models/test_weights.pth', 10, 5, 16, 3)
    x = torch.randn(1, 3, 10, 5)
    edge_index = torch.randint(0, 10, (2, 20))
    predictions = predictor.predict_batch(x, edge_index)
    
    print("Predicted Crime Hotspots (Intensity):")
    print(predictions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CrimeGuard-AI CLI")
    parser.add_argument("--train", action="store_true", help="Trigger the full training pipeline")
    parser.add_argument("--predict", action="store_true", help="Run inference with saved weights")
    parser.add_argument("--save-model", type=str, help="Path to save trained weights")
    args = parser.parse_args()
    
    if args.train:
        run_cli_training(save_path=args.save_model)
    elif args.predict:
        run_cli_predict()
    else:
        print("Use --train to trigger training or --predict to run inference.")
