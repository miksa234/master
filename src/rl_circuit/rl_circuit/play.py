#!/usr/bin/env python3play

from .game import NetGame
from .config import DEVICE
from .mcts import MCTS
from .resnet import ResNet
from .alphazero import AlphaZero

import networkx as nx
import pandas as pd
import numpy as np
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


    weights = np.random.uniform(0.7, 1.4, size=(6, 6))
    weights = np.triu(1/weights, k=0) + np.tril(weights.T, k=0)
    np.fill_diagonal(weights, 0)
    G = nx.from_numpy_array(weights, create_using=nx.DiGraph)

    ## Just scramble code to test if shit works

    args = {
        'C': 2,
        'num_searches': 100,
        'num_iterations': 5,
        'num_self_play_iterations': 40,
        'num_parallel': 10,
        'num_epochs': 4,
        'batch_size': 5,
        'temperature': 1.25,
        'eps': 0.25,
        'dirichlet_alpha': 0.3,
        'num_processes': mp.cpu_count(),
    }

    game = NetGame(G)

    model = ResNet(
        game,
        num_res_blocks=6,
        num_hidden=256,
        gnn_in_channels=1,
        gnn_hidden_channels=128,
        gnn_out_channels=1,
        gnn_num_heads=8,
        gnn_num_layers=4
    ).to(DEVICE).share_memory()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    mcts = MCTS(game, args, model)

    model.load_state_dict(torch.load('./model/model_9.pt', weights_only=True))

#    alpha_zero = AlphaZero(model, optimizer, game, args)
#    alpha_zero.learn()

    model.eval()
    for s in game.nodes:
        state = [s]
        while True:
            #logger.info(f"STATE: {state}")
            model_probs = torch.tensor(mcts.search(state))
            model_probs = torch.softmax(model_probs, dim=0).numpy()
            model_probs = game.mask_policy(model_probs, state)

            action = [state[-1], int(np.random.choice(game.nodes, p=model_probs))]
            state = game.get_next_state(state, action)


            value, is_terminal = game.get_value_and_terminated(state)

            #logger.info(f"Value: {value}")


            if is_terminal:
                profit = np.exp(value)
                if game.check_win(state):
                    if profit < 1:
                        logger.info(f'LOSS {state} with value {profit}')
                    else:
                        logger.info(f'WON {state} with value {profit}')
                else:
                    logger.info(f'LOSS {state} with valule {profit}')
                break
