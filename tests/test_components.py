import unittest
import torch
import numpy as np
import pandas as pd
from src.models import STGNN
from src.route_optimizer import optimize_patrol_route

class TestCrimeGuardAI(unittest.TestCase):
    def test_stgnn_forward(self):
        # Parameters: 10 nodes, 3 features (calculated from feature_df shape), 16 hidden, 1 sequence, 1 output
        model = STGNN(num_nodes=10, num_features=3, hidden_dim=16, seq_len=1, out_dim=1)
        model.eval()
        
        # Batch: 1, Seq: 3, Nodes: 10, Features: 5
        x = torch.randn(1, 3, 10, 5)
        edge_index = torch.randint(0, 10, (2, 20))
        
        with torch.no_grad():
            output = model(x, edge_index)
        
        # STGNN output shape [batch, nodes, out_dim]
        self.assertEqual(output.shape, (1, 10, 1))

    def test_route_optimizer(self):
        # 3x3 distance matrix
        dist_matrix = np.array([
            [0, 1, 2],
            [1, 0, 3],
            [2, 3, 0]
        ])
        route = optimize_patrol_route(dist_matrix)
        
        self.assertIsNotNone(route)
        self.assertTrue(len(route) >= 3)
        self.assertEqual(route[0], 0) # Assumes depot is 0

    def test_feature_engineering(self):
        from src.feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        
        crime_data = pd.DataFrame({'timestamp': [1, 2], 'crime': [10, 20], 'region_id': [1, 1]})
        weather_data = pd.DataFrame({'timestamp': [1, 2], 'temp': [25, 26]})
        event_data = pd.DataFrame({'timestamp': [1], 'event_type': ['festival']})
        socio_data = pd.DataFrame({'region_id': [1], 'income': [50000]})
        
        processed_df = fe.run_pipeline(crime_data, weather_data, event_data, socio_data)
        
        self.assertIn('temp', processed_df.columns)
        self.assertIn('is_event', processed_df.columns)
        self.assertIn('income', processed_df.columns)
        self.assertEqual(processed_df.iloc[0]['is_event'], 1)
        self.assertEqual(processed_df.iloc[0]['income'], 50000)

    def test_inference_loading(self):
        from src.inference import Predictor
        import os
        
        model_path = 'models/test_weights.pth'
        if os.path.exists(model_path):
            predictor = Predictor(model_path, 10, 5, 16, 3)
            
            x = torch.randn(1, 3, 10, 5)
            edge_index = torch.randint(0, 10, (2, 20))
            
            output = predictor.predict_batch(x, edge_index)
            self.assertEqual(output.shape, (1, 10, 1))

    def test_graph_builder(self):
        from src.graph_builder import build_edge_index
        coords = np.array([
            [0, 0],
            [0.01, 0],
            [0.5, 0.5]
        ])
        edge_index = build_edge_index(coords, threshold=0.1)
        
        # Should have self-loops and connected pair (0,1)
        self.assertGreaterEqual(edge_index.shape[1], 3)
        self.assertEqual(edge_index.shape[0], 2)

    def test_full_trainer_pipeline(self):
        from src.trainer import Trainer
        from src.models import STGNN
        import pandas as pd
        import numpy as np

        model = STGNN(10, 5, 16, 3, 1)
        trainer = Trainer(model)
        
        # Fix TypeError by ensuring data contains only numeric types
        # crime_df nodes=10, timestamps=10
        crime_df = pd.DataFrame({'timestamp': [1]*10, 'nodes': range(10), 'crime': np.random.rand(10), 'region_id': [1]*10})
        weather_df = pd.DataFrame({'timestamp': [1]*10, 'temp': np.random.rand(10)})
        event_df = pd.DataFrame({'timestamp': [1]*10, 'event_id': np.zeros(10)})
        # Socio needs to be linked to nodes. 10 regions for 10 nodes.
        socio_df = pd.DataFrame({'region_id': range(10), 'income': [50000]*10})
        coords = np.random.rand(10, 2)

        # Should run without error
        trainer.train(crime_df, weather_df, event_df, socio_df, coords, epochs=2)

if __name__ == '__main__':
    unittest.main()
