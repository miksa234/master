#!/usr/bin/env python3

from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool
import torch.nn as nn
import torch.nn.functional as F

class GNNEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_heads,
        num_layers,
    ):
        super().__init__()

        self.conv_in = GATConv(
            in_channels, hidden_channels,
            heads=num_heads, concat=True, dropout=0.1
         )

        self.hidden_convs = nn.ModuleList(
            [GATConv(hidden_channels*num_heads, hidden_channels, heads=num_heads, concat=True) for i in range(num_layers-2)]
        )

        self.conv_out = GATConv(
            hidden_channels*num_heads, out_channels,
            concat=True, dropout=0.1
        )



    def forward(self, x, edge_index, edge_attr, batch=None):
        x = self.conv_in(x, edge_index, edge_attr)
        x = F.relu(x)

        for conv in self.hidden_convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)

        x = self.conv_out(x, edge_index, edge_attr)

        if batch is not None:
            x = global_mean_pool(x, batch)
        return x
