#!/usr/bin/env python3play

from networkx.classes import degree
from .game import NetGame
from .config import DEVICE
from .mcts import MCTS
from .resnet import ResNet
from .alphazero import AlphaZero

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(seed=0)

import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

import torch.multiprocessing as mp
try:
   mp.set_start_method('spawn', force=True)
except RuntimeError:
   pass

import logging
logger = logging.getLogger('rl_circuit')


def run():
#    pools = pd.read_csv(
#        'data/cached-pools.csv',
#        index_col=0,
#        names=[
#            "index", "addess", "version",
#             "token0", "token1", "fee", "block_number",
#             "timestamp", "tickspacing"
#       ]
#    ).sort_index()
#
#    G = nx.from_edgelist(list(tuple(zip(pools['token0'], pools['token1']))))
#    remove = [node for (node, d) in dict(G.degree()).items() if d <= 40]
#    G.remove_nodes_from(remove)
#
#    logger.info(f"{len(G.nodes())}")
#
#
#    mapping = {}
#    for i, node in enumerate(list(G.nodes())):
#        mapping[node] = i
#    G = nx.relabel_nodes(G, mapping)
#    for _, _, w in G.edges(data=True):
#        w['weight'] = round(np.random.uniform(0.7, 1.4), 4)


    weights = np.random.uniform(0.01, 100, size=(6, 6))
    weights = np.triu(1/weights, k=0) + np.tril(weights.T, k=0)
    weights = np.log(weights)
    np.fill_diagonal(weights, 0)
    G = nx.from_numpy_array(weights, create_using=nx.DiGraph)


# edge attributes:
#     * marginal exchange rate
#     * swap fee = 0.3%
#     * edge_used binary
#
# nodes should have
#     * degree
#     * current eth price (in wei)
#     * current node indicator binary

    # edge attributes
    edge_list = [list(e) for e in G.edges()]
    rates = [e[-1]['weight'] for e in G.edges(data=True)]
    fees = [0.003 for _ in range(len(edge_list))]
    used = [0 for _ in range(len(edge_list))]

    # node attributes
    degrees = [degree for _, degree in G.degree()]
    eth_rate = np.random.uniform(0.7, 1.4, size=len(G.nodes))
    indicator = np.zeros(len(G.nodes))

    node_attr  = torch.tensor(np.dstack([degrees, eth_rate, indicator])).float().squeeze(0)
    edge_index = torch.tensor(edge_list).t().contiguous()
    edge_attr = torch.tensor(np.dstack([rates, fees, used])).float().squeeze(0)

    data = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr).to(DEVICE)

    ## Just scramble code to test if shit works

    args = {
        'C': 10,
        'num_searches': 600,
        'num_iterations': 8,
        'num_self_play_iterations': 800,
        'num_parallel': 100,
        'num_epochs': 4,
        'batch_size': 128,
        'temperature': 1.25,
        'eps': 0.25,
        'dirichlet_alpha': 0.3,
        'num_processes': mp.cpu_count(),
    }

    game = NetGame(G, data)

    model = ResNet(
        in_channels=3,
        edge_dim=3,
        emb_channels=128,
        num_heads=8,
        num_layers=3
    ).to(DEVICE).share_memory()

#    state = [[1, 2], [2, 3]]
#    value, policy = [], []
#    for s in state:
#        v, p = model(
#            *game.encode_state(s)
#        )
#        value += v
#        policy += p
#    value = torch.stack(value).unsqueeze(1)
#    policy = torch.stack(policy)


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    mcts = MCTS(game, args, model)

#    alpha_zero = AlphaZero(model, optimizer, game, args)
#    alpha_zero.learn()

    model.load_state_dict(torch.load('./model/model_7.pt', weights_only=True))
    model.eval()
    for s in game.nodes:
        state = [s]
        while True:
#            x, edge_index, edge_attr = game.encode_state(state)
#
#            model_probs = torch.tensor(mcts.search(state))
#            model_probs = torch.softmax(model_probs, dim=0).numpy()
#            model_probs = game.mask_policy(model_probs, state)
            probs = mcts.search(state)

            action = [state[-1], int(np.random.choice(game.nodes, p=probs))]
            state = game.get_next_state(state, action)

            value, is_terminal = game.get_value_and_terminated(state)

            if is_terminal:
                profit = np.exp(value/1)
                if game.check_terminal(state):
                    if profit < 1:
                        logger.info(f'LOSS {state} with profit {profit}')
                    else:
                        logger.info(f'WON {state} with profit {profit}')
                else:
                    logger.info(f'LOSS {state} with profit {profit}')
                break
