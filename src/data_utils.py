import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class CrimeDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        return self.data[idx:idx+self.seq_len], self.data[idx+self.seq_len]

def process_crime_data(df):
    # Basic normalization and transformation
    return df.values / df.values.max()
