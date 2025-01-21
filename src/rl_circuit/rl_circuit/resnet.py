#!/usr/bin/env python3

from .config import DEVICE
from .gnn import GNNEncoder

import torch
from torch import nn
from torch.nn import functional as F

import logging
logger = logging.getLogger('rl_circuit')

class ResNet(nn.Module):
    def __init__(
        self,
        game,
        num_res_blocks,
        num_hidden,
        gnn_in_channels,
        gnn_hidden_channels,
        gnn_out_channels,
        gnn_num_heads=4,
        gnn_num_layers=3
    ):
        super().__init__()

        self.game = game

        self.gnn = GNNEncoder(
            gnn_in_channels,
            gnn_hidden_channels,
            gnn_out_channels,
            gnn_num_heads,
            gnn_num_layers
        )

        self.start_block = nn.Sequential(
            nn.Conv2d(4, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )

        self.back_bone = nn.ModuleList(
            [ResBlock(num_hidden) for _ in range(num_res_blocks)]
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * len(game.nodes)**2, len(game.nodes)),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(num_hidden, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * len(game.nodes)**2, 1),
            nn.Tanh()
        )

    def forward(self, encoded_states, node_attr, edge_index, edge_attr):

        node_emb = self.gnn(node_attr, edge_index, edge_attr)
        embeddings = node_emb.repeat(1, 1 , len(self.game.nodes))

        if list(encoded_states.shape) == [3, *self.game.adj_matrix.shape]:
            x = torch.cat([encoded_states, embeddings], dim=0).unsqueeze(0).float()
        else:
            x= torch.stack([torch.cat([state, embeddings], dim=0).float() for state in encoded_states], dim=0)

        x = self.start_block(x)

        for res_block in self.back_bone:
            x = res_block(x)

        policy = self.policy_head(x)
        value = self.value_head(x)
        return value, policy


class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        res = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += res
        x = F.relu(x)
        return x
