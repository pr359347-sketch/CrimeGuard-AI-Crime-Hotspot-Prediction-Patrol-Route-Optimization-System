import torch
from src.models import STGNN

class Predictor:
    def __init__(self, model_path, num_nodes, num_features, hidden_dim, seq_len):
        self.model = STGNN(num_nodes, num_features, hidden_dim, seq_len, out_dim=1)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def predict_batch(self, x, edge_index):
        """
        Runs inference on a batch of data.
        x: [batch, seq, nodes, features]
        """
        with torch.no_grad():
            return self.model(x, edge_index)

def run_batch_prediction(model_path, x, edge_index):
    predictor = Predictor(model_path, num_nodes=x.shape[2], num_features=x.shape[3], hidden_dim=32, seq_len=x.shape[1])
    return predictor.predict_batch(x, edge_index)
