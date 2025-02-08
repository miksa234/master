#!/usr/bin/env python3

from .config import DEVICE

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import BatchNorm, GATv2Conv, Sequential, Linear

import logging
logger = logging.getLogger('rl_circuit')

class ResNet(nn.Module):
    """
    A Residual Network (ResNet) model for graph-based data.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    emb_channels : int
        Number of embedding channels.
    num_heads : int, optional
        Number of attention heads. Default is 4.
    num_layers : int, optional
        Number of layers in the network. Default is 3.

    Attributes
    ----------
    in_block : ResBlock
        Initial residual block.
    hidden_blocks : nn.ModuleList
        List of hidden residual blocks.
    policy_head : Sequential
        Policy head for outputting policy logits.
    value_head : Sequential
        Value head for outputting value estimation.
    """
    def __init__(
        self,
        in_channels,
        emb_channels,
        num_heads=4,
        num_layers=3
    ):
        super().__init__()

        self.encoder = Linear(in_channels, emb_channels)

        self.in_block = ResBlock(emb_channels, emb_channels, num_heads)

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
            Linear(emb_channels, 2),
        ])

        self.value_head = Sequential('x, edge_index', [
            (GATv2Conv(emb_channels, emb_channels, heads=1), 'x, edge_index -> x'),
            BatchNorm(emb_channels),
            nn.ReLU(),
            Linear(emb_channels, 1),
            nn.Tanh(),
        ])


    def forward(self, node_attr, edge_index):
        """
        Forward pass of the ResNet model.

        Parameters
        ----------
        node_attr : Tensor
            Node attributes.
        edge_index : Tensor
            Edge indices.

        Returns
        -------
        tuple
            A tuple containing the value estimation and policy logits.
        """
        x = self.encoder(node_attr)
        x = self.in_block(x, edge_index)
        for block in self.hidden_blocks:
            x = block(x, edge_index)

        policy = self.policy_head(x, edge_index).flatten().unsqueeze(0)
        value = torch.tensor([self.value_head(x, edge_index).mean()])

        return value, policy

class ResBlock(nn.Module):
    """
    A Residual Block used in the ResNet model.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    emb_channels : int
        Number of embedding channels.
    num_heads : int
        Number of attention heads.

    Attributes
    ----------
    in_layer : Sequential
        Initial layer with GATv2Conv, BatchNorm, and Linear layers.
    feed_forward : nn.Sequential
        Feed-forward network with Linear and ReLU layers.
    batch_norm2 : BatchNorm
        Batch normalization layer.
    """
    def __init__(self, in_channels, emb_channels, num_heads):
        super().__init__()
        self.in_layer = Sequential('x, edge_index', [
            (GATv2Conv(in_channels, emb_channels, heads=num_heads), 'x, edge_index -> x'),
            Linear(emb_channels*num_heads, emb_channels, bias=False)
        ])

        self.batch_norm1 = BatchNorm(emb_channels)

        self.feed_forward = nn.Sequential(
            Linear(emb_channels, 512),
            nn.ReLU(),
            Linear(512, emb_channels)
        )
        self.batch_norm2 = BatchNorm(emb_channels)

    def forward(self, node_attr, edge_index):
        """
        Forward pass of the ResBlock.

        Parameters
        ----------
        node_attr : Tensor
            Node embeddings.
        edge_index : Tensor
            Edge indices.

        Returns
        -------
        Tensor
            Output tensor after applying the residual block.
        """
        res = node_attr
        x = self.in_layer(node_attr, edge_index)
        x = self.batch_norm1(res + x)

        res = x
        x = self.feed_forward(x)
        x = self.batch_norm2(res + x)

        return x
