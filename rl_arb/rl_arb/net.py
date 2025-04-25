#!/usr/bin/env python3

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import BatchNorm, GATConv, LayerNorm, Sequential, Linear

import logging

from torch_geometric.nn.glob import global_mean_pool
logger = logging.getLogger('rl_circuit')

from rl_arb.config import DEVICE

class PolicyNet(nn.Module):
    """
    A Residual Network model for graph-based data.

    Parameters
    ----------
    args : dict
        in_channels : int
            Number of input channels.
        emb_channels : int
            Number of embedding channels.
        num_heads : int, optional
            Number of attention heads. Default is 4.
        num_layers : int, optional
            Number of layers in the network. Default is 3.
        policy_mheads : int
            Number of attention heads for the policy head.
        value_mheads : int
            Number of attention heads for the value head.

    Attributes
    ----------
    encoder : Linear
        Linear layer for encoding input features.
    hidden_blocks : nn.ModuleList
        List of hidden residual blocks.
    policy_head : Sequential
        Policy head for outputting policy logits.
    """
    def __init__(
        self,
        args,
    ):
        super().__init__()

        self.encoder = Linear(args['in_channels'], args['emb_channels'])

        self.hidden_blocks = nn.ModuleList()
        for _ in range(args['num_layers']):
            self.hidden_blocks.append(
                ResCovBlock(args)
            )

        self.policy_head = Sequential('x, edge_index, batch', [
            (GATConv(
                3*args['emb_channels'],
                args['emb_channels'],
                heads=args['policy_mheads'],
            ), 'x, edge_index -> x'),
            (LayerNorm(args['emb_channels']*args['policy_mheads']), 'x, batch -> x'),
            nn.ReLU(),
            (Linear(args['emb_channels']*args['policy_mheads'], 2, bias=False), 'x -> x'),
        ])

    def forward(self, node_attr, edge_index, y, batch=None):
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
        x_1 = self.encoder(y)
        for block in self.hidden_blocks:
            x = block(x, edge_index, batch)

        mean = global_mean_pool(x, batch)
        if batch != None:
            x_n = []
            for i in batch.unique():
                count = (batch == i).sum()
                x_n.append(mean[i].repeat(count, 1))
            x_n = torch.cat(x_n, 0)
        else:
            x_n = mean.repeat(x.shape[0], 1)
        x_c = torch.cat((x, x_n, x_1), dim=1)

        policy = self.policy_head(x_c, edge_index, batch).flatten().unsqueeze(0)

        if batch != None:
            policy = policy.view(len(batch.unique()), -1)

        policy = torch.softmax(policy, dim=1)

        return policy

class ResCovBlock(nn.Module):
    """
    A Residual Block used in the ResNet model.

    Parameters
    ----------
    args : dict
        in_channels : int
            Number of input channels.
        emb_channels : int
            Number of embedding channels.
        num_heads : int
            Number of attention heads.
        ff_dim : int
            Dimension of the feed-forward network.

    Attributes
    ----------
    in_layer : Sequential
        Initial layer with GATConv, BatchNorm, and Linear layers.
    feed_forward : nn.Sequential
        Feed-forward network with Linear and ReLU layers.
    batch_norm1 : BatchNorm
        Batch normalization layer after the initial layer.
    batch_norm2 : BatchNorm
        Batch normalization layer after the feed-forward network.
    """
    def __init__(self, args):
        super().__init__()
        self.in_layer = Sequential('x, edge_index', [
            (GATConv(
                args['emb_channels'],
                args['emb_channels'],
                heads=args['num_heads']
            ), 'x, edge_index -> x'),
            Linear(
                args['emb_channels']*args['num_heads'],
                args['emb_channels'],
                bias=False
            )
        ])

        self.batch_norm1 = LayerNorm(args['emb_channels'])

        self.feed_forward = nn.Sequential(
            Linear(args['emb_channels'], args['ff_dim']),
            nn.ReLU(),
            Linear(args['ff_dim'], args['emb_channels'])
        )
        self.batch_norm2 = LayerNorm(args['emb_channels'])

    def forward(self, node_attr, edge_index, batch=None):
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
        x = self.batch_norm1(res + x, batch)

        res = x
        x = self.feed_forward(x)
        x = self.batch_norm2(res + x, batch)

        return x
