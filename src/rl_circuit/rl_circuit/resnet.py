#!/usr/bin/env python3

from .config import DEVICE
from .gnn import GNNEncoder

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import BatchNorm, GATv2Conv, Sequential, Linear

import logging
logger = logging.getLogger('rl_circuit')

class ResNet(nn.Module):
    def __init__(
        self,
        in_channels,
        emb_channels,
        num_heads=4,
        num_layers=3
    ):
        super().__init__()

        self.in_block = ResBlock(in_channels, emb_channels, num_heads)

        self.hidden_blocks = nn.ModuleList()
        for _ in range(num_layers-1):
            self.hidden_blocks.append(
                ResBlock(
                    emb_channels, emb_channels, num_heads
                )
            )

        self.policy_head = Sequential('x, edge_index', [
            (GATv2Conv(emb_channels, emb_channels, heads=1), 'x, edge_index -> x'),
            BatchNorm(emb_channels),
            nn.ReLU(),
            Linear(emb_channels, 1),
        ])

        self.value_head = Sequential('x, edge_index', [
            (GATv2Conv(emb_channels, emb_channels, heads=1), 'x, edge_index -> x'),
            BatchNorm(emb_channels),
            nn.ReLU(),
            Linear(emb_channels, 1),
            nn.Tanh(),
        ])


    def forward(self, node_attr, edge_index):
        x = self.in_block(node_attr, edge_index)
        for block in self.hidden_blocks:
            x = block(x, edge_index)

        policy = self.policy_head(x, edge_index).T
        value = torch.tensor([self.value_head(x, edge_index).mean()])

        return value, policy

class ResBlock(nn.Module):
    def __init__(self, in_channels, emb_channels, num_heads):
        super().__init__()
        self.in_layer = Sequential('x, edge_index', [
            (GATv2Conv(in_channels, emb_channels, heads=num_heads), 'x, edge_index -> x'),
            BatchNorm(emb_channels*num_heads),
            Linear(emb_channels*num_heads, emb_channels, bias=False)
        ])

        self.feed_forward = nn.Sequential(
            Linear(emb_channels, 512),
            nn.ReLU(),
            Linear(512, emb_channels)
        )
        self.batch_norm2 = BatchNorm(emb_channels)

    def forward(self, node_attr, edge_index):

        x = self.in_layer(node_attr, edge_index)

        res = x
        x = self.feed_forward(x)
        x = self.batch_norm2(res + x)

        return x
