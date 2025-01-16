#!/usr/bin/env python3

from .game import NetGame
from .config import DEVICE
from .mcts import MCTS
from .resnet import ResNet
#from .gnn import GNN

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

def run():
    G = nx.fast_gnp_random_graph(6, 1, seed=1)
    for _, _, w in G.edges(data=True):
        w['weight'] = round(np.random.uniform(0.7, 1.4), 4)


    ## Just scramble code to test if shit works

    game = NetGame(G)
    model = ResNet(game, 4, 64).to(DEVICE)
    gat_layer = GATConv(
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

    data = game.get_net_data().to(DEVICE)
    node_embeddings = gat_layer(data.x, data.edge_index, data.edge_attr).squeeze(1)

    state = [0]
    valid_actions = game.get_valid_actions(state)
    action_index = 0
    action = valid_actions[action_index]
    state = game.get_next_state(state, action)

    print(state)

    value, terminated = game.get_value_and_terminated(state)

    x = game.encode_state(state, node_embeddings).to(DEVICE)

    value, policy = model(x)
    policy = torch.softmax(game.mask_policy(policy.squeeze(0), state), dim=0)

    print(policy)
    print(value)



#    args = {'C': 1.41, 'num_searches': 1000}
#    mcts = MCTS(game, args)
#
#
#    while True:
#        valid_actions = game.get_valid_actions(state)
#        if len(valid_actions) == 0:
#            print("LOSS")
#            break
#
#        print("valid_actions", [(f"index {i}", j) for i, (_, j) in enumerate(valid_actions)])
#
#        action_probs = mcts.search(state)
#        action_index = np.argmax(action_probs)
#        action = valid_actions[action_index]
#
#
#        state = game.get_next_state(state, action)
#
#        value, terminated = game.get_value_and_terminated(state)
#
#        print(value, state)
#
#        if terminated == True:
#            if state[0] == state[-1]:
#                print("WON")
#            else:
#                print("LOSS")
#            break
#
#        print(state)










