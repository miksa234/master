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
#
    G = nx.fast_gnp_random_graph(6, 1, seed=1)
    for _, _, w in G.edges(data=True):
        w['weight'] = round(np.random.uniform(0.7, 1.4), 4)


    ## Just scramble code to test if shit works

    args = {
        'C': 2,
        'num_searches': 60,
        'num_iterations': 3,
        'num_self_play_iterations': 100,
        'num_epochs': 4,
        'batch_size': 32,
        'temperature': 1.25,
        'eps': 0.25,
        'dirichlet_alpha': 0.3,
        'num_processes': 4
    }

    game = NetGame(G)

    gnn = GATConv(
        in_channels=1,
        out_channels=1,
        heads=1,
        concat=True,
        negative_slope=0.2,
        dropout=0,
        add_self_loops=True,
        edge_dim=1,
        fill_value="mean",
        bias=False,
        residual=False
    ).to(DEVICE).share_memory()

    model = ResNet(game, gnn, 16, 256).to(DEVICE).share_memory()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    mcts = MCTS(game, args, model)

    data = game.get_net_data().to(DEVICE).share_memory_()

    model.load_state_dict(torch.load('./model/model_2.pt', weights_only=True))

#    alpha_zero = AlphaZero(model, optimizer, game, args)
#    alpha_zero.learn(data)

    for s in game.nodes:
        state = [s]
        while True:
            #logger.info(f"STATE: {state}")
            valid_actions = game.get_valid_actions(state)
            model_val, model_probs  = model(
                game.encode_state(state, state[-1]),
                data.x,
                data.edge_index,
                data.edge_attr
            )

            model_probs = torch.softmax(model_probs.squeeze(0), dim=0).detach().cpu().numpy()
            model_probs = game.mask_policy(model_probs, state)

            action = [state[-1], int(np.random.choice(game.nodes, p=model_probs))]
            state = game.get_next_state(state, action)


            value, is_terminal = game.get_value_and_terminated(state)
            #logger.info(f"Value: {value}")


            if is_terminal:
                if game.check_win(state):
                    logger.info(f'WON {state} with value {value}')
                else:
                    logger.info(f'LOSS {state} with valule {value}')
                break
