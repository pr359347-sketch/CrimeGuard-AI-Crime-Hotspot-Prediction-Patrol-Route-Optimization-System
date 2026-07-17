import torch
import numpy as np
from scipy.spatial import distance_matrix

def build_edge_index(coords, threshold=0.1):
    """
    Constructs edge_index for ST-GNN from coordinates.
    coords: numpy array of shape (num_nodes, 2)
    threshold: distance threshold for connectivity
    """
    dist = distance_matrix(coords, coords)
    adj = (dist < threshold).astype(int)
    
    # Create edge_index
    rows, cols = np.where(adj == 1)
    edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)
    return edge_index
