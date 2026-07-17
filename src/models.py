import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class STGNN(nn.Module):
    def __init__(self, num_nodes, num_features, hidden_dim, seq_len, out_dim):
        super(STGNN, self).__init__()
        # GCNConv expects (in_channels, out_channels)
        # The number of input features MUST match the GCN layer's in_channels
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_dim)
        self.out_channels = hidden_dim

    def forward(self, x, edge_index):
        # x shape: [batch, seq, num_nodes, features]
        batch_size, seq_len, num_nodes, features = x.size()
        
        # Spatial: Apply GCN to each graph (batch * seq)
        # GCNConv expects [num_nodes, features]
        # We flatten batch*seq and process
        x_gcn = x.view(batch_size * seq_len, num_nodes, features)
        x_spatial = []
        for i in range(batch_size * seq_len):
            x_spatial.append(self.conv1(x_gcn[i], edge_index))
        
        # [batch * seq, nodes, hidden_dim]
        x_spatial = torch.stack(x_spatial).view(batch_size, seq_len, num_nodes, -1)
        
        # Temporal: Pool across nodes to get node-independent temporal representation
        # [batch, seq, hidden_dim]
        x_temporal = x_spatial.mean(dim=2) 
        
        # Temporal: LSTM [batch, seq, hidden_dim]
        lstm_out, _ = self.lstm(x_temporal)
        
        # Return [batch, num_nodes, out_dim]
        # Project LSTM hidden out to out_dim and expand to num_nodes
        return self.fc(lstm_out[:, -1, :]).unsqueeze(1).repeat(1, num_nodes, 1)
