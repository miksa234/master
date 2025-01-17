#!/usr/bin/env python3

from .game import NetGame
from .config import DEVICE
from .mcts import MCTS
from .resnet import ResNet

import networkx as nx
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv


def run():
    pools = pd.read_csv(
        'data/cached-pools.csv',
        index_col=0,
        names=[
            "index", "addess", "version",
             "token0", "token1", "fee", "block_number",
             "timestamp", "tickspacing"
       ]
    ).sort_index()

    G = nx.from_edgelist(list(tuple(zip(pools['token0'], pools['token1']))))
    for _ in range(10):
        remove = [node for (node, d) in dict(G.degree()).items() if d <= 10]
        G.remove_nodes_from(remove)


    mapping = {}
    for i, node in enumerate(list(G.nodes())):
        mapping[node] = i
    G = nx.relabel_nodes(G, mapping)
    for _, _, w in G.edges(data=True):
        w['weight'] = round(np.random.uniform(0.7, 1.4), 4)

#    G = nx.fast_gnp_random_graph(100, 1, seed=1)
#    for _, _, w in G.edges(data=True):
#        w['weight'] = round(np.random.uniform(0.7, 1.4), 4)


    ## Just scramble code to test if shit works

    args = {
        'C': 2,
        'num_searches': 1000
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
    ).to(DEVICE)
    model = ResNet(game, gnn, 4, 64).to(DEVICE)
    model.eval()

    mcts = MCTS(game, args, model)

    data = game.get_net_data().to(DEVICE)

    state = [0]
    while True:
        print(state)
        valid_actions = game.get_valid_actions(state)
        mcts_probs = mcts.search(state)
        action = valid_actions[np.argmax(mcts_probs)]
        state = game.get_next_state(state, action)

        value, is_terminal = game.get_value_and_terminated(state)


        if is_terminal:
            print(state)
            if game.check_win(state):
                print('WON')
                print(value)
            else:
                print('LOSS')
                print(value)
            break
