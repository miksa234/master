#!/usr/bin/env python3play

from networkx.classes import degree
from .game import NetGame
from .config import DEVICE
from .mcts import MCTS
from .resnet import ResNet
from .rlearn import AgentRLearn
from .utils import *
from .brute_force import *

import networkx as nx
import pandas as pd
import numpy as np

import torch
from torch_geometric.data import Data

import torch.multiprocessing as mp
try:
   mp.set_start_method('spawn', force=True)
except RuntimeError:
   pass

import logging
logger = logging.getLogger('rl_circuit')


def run():


    pools, tokens = load_pools_and_tokens(
        '../data/pools/pools_deg_15_liq_100_block_18.csv',
        '../data/tokens/tokens.csv',
    )
    prices = pd.read_parquet(
        '../data/prices/prices_deg_15_liq_100_block_18.parquet'
    )
    pools, prices = filter_pools_with_no_gradient(pools, prices)
    print(len(pools))

    num_blocks = len(set(prices['block_number']))
    G = make_token_graph(pools, prices)

    # plots prices
#    import matplotlib.pyplot as plt
#    for e in G.edges(data=True):
#        plt.plot(e[2]['weight'])
#        pool_adr = pools[(pools['token0']== e[0]) & (pools['token1'] == e[1])].iloc[e[2]['k']]['address']
#        plt.savefig(f'./plots/{pool_adr}.png')
#        plt.close()
#
#    exit()

    G, token_mapping = linear_node_relabel(G)

    L = nx.line_graph(G, create_using=nx.Graph)
    for e in G.edges(data=True):
        nx.set_node_attributes(L, {(e[0], e[1], e[2]['k']): e[2]['weight']}, name='mexr')
#        nx.set_node_attributes(L, {(e[0], e[1], e[2]['k']): e[2]['fee']}, name='fee')
        nx.set_node_attributes(L, {(e[0], e[1], e[2]['k']): e[2]['address']}, name='address')


    L, line_mapping = linear_node_relabel(L)

    edges = list(G.edges(data=True))
    for (t0, t1, d) in edges:
        inverse_weights = [1/el for el in d['weight']]
        G.add_edge(t1, t0, k=d['k'], weight=inverse_weights, address=d['address'])


#    weights = np.random.uniform(0.01, 100, size=(6, 6))
#    weights = np.triu(1/weights, k=0) + np.tril(weights.T, k=0)
#    weights = np.log(weights)
#    np.fill_diagonal(weights, 0)
#    G = nx.from_numpy_array(weights, create_using=nx.DiGraph)
#    print(len(G.edges))
#
#    trails, dw = brute_force_search_trail(G, 0)
#    for trail, w in zip(trails, dw):
#        print(trail)
#
#    exit()

# pool attributes: (edge_attributes)
#     * marginal exchange rate of the pool from t0 -> t1 (t1/t0)
#     * swap fee = 0.3%
#     * used binary
#     * current t0 binary (if currently at t0)
#     * current t1 binary (if currently at t1)
#
# state should be a list[tuples] of the edges taken
# e.g. [(0, 0, 0), (1, 2, 0), (2 ,5, 0), (5, 0, 1)]
# where the last entry in the tuple represents which edge was exactly
# taken if there are multiple edges connecting two nodes
# and the first entry (0, 0, 0) just the indicator where the game starts
# if the game starts at node 1 then the state entry should be (1, 1, 0)
#
# policy is shape of len(edges) because there can be mulitple edges attached to
# the same pair of nodes this is the consistent way of choosing the right action

# The learning is going to take place not on the game graph but
# on its line graph. The current edges will be vertices and, ....
# see further definition of the line graph.


    # pool attributes
    edge_list = [list(e) for e in L.edges()]
    rates = np.array([np.log(e[-1]['mexr']) for e in L.nodes(data=True)]).T
    fees = [e[-1]['fee'] for e in L.nodes(data=True)]
    used = [0 for _ in range(len(L.nodes))]
    t0_using = [0 for _ in range(len(L.nodes))]
    t1_using = [0 for _ in range(len(L.nodes))]

    node_attr  = torch.tensor(np.dstack([*rates, fees, used, t0_using, t1_using])).float().squeeze(0)
    edge_index = torch.tensor(edge_list).t().contiguous()

    data = Data(x=node_attr, edge_index=edge_index).to(DEVICE)

    args_game = {
        'tau': 1,
        'M': 2,
    }

    game = NetGame(G, data, line_mapping, num_blocks, args_game)

    args_training = {
        'C': 2,
        'C_1/3' : 3.5,
        'C_2/3' : 2.0,
        'C_3/3' : 1.5,
        'num_iterations': 100,
        'num_searches': 50,
        'num_self_play_iterations': 25,
        'num_parallel': 5,
        'num_epochs': 10,
        'batch_size': 32,
        'temperature': 1.25,
        'eps': 0.25,
        'dirichlet_alpha': 0.3,
        'num_processes': 5,
        'multicore': True,
    }

    args_model = {
        'in_channels': 4,
        'emb_channels': 256,
        'num_heads': 16,
        'num_layers': 8,
        'ff_dim': 1024,
        'policy_mheads': 1,
        'value_mheads': 1
    }

    model = ResNet(
        args_model
    ).to(DEVICE).share_memory()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    optimizer.zero_grad()

#    model.load_state_dict(
#        torch.load(
#            './model/model_25.pt',
#            weights_only=True,
#            map_location=DEVICE
#        )
#    )
#    optimizer.load_state_dict(
#        torch.load(
#            './model/optimizer_24.pt',
#            weights_only=True,
#            map_location=DEVICE
#        )
#    )

    mcts = MCTS(game, args_training, model)

    rlearn = AgentRLearn(model, optimizer, game, args_training)
    rlearn.learn()
    exit()

#    model.eval()
#    for s in game.nodes:
#        state = [(s, s, 0)]
#        while True:
#            probs = mcts.search(state)
#
#            action_idx = np.random.choice(list(range(len(game.edge_list))), p=probs)
#            action = game.edge_list[action_idx]
#            state = game.get_next_state(state, action)
#
#            value, is_terminal = game.get_value_and_terminated(state)
#
#            if is_terminal:
#                profit = np.exp(value/1)
#                if game.check_terminal(state):
#                    if profit < 1:
#                        logger.info(f'LOSS {state} with profit {profit}')
#                    else:
#                        logger.info(f'WON {state} with profit {profit}')
#                else:
#                    logger.info(f'LOSS {state} with profit {profit}')
#                break


#   TEST
#    state = [(0, 0, 0), (0, 1, 0), (1, 2, 0)]
#    value, policy = [], []
#    e_x = game.encode_state(state)
#    v, p = model(
#        e_x,
#        data.edge_index
#    )
#    state = [(0, 0, 0)]
#    while True:
#        print("state: ", state)
#        valid_actions = game.get_valid_actions(state)
#        print(valid_actions)
#        i = int(input(f"Pick an index from 0 - {len(valid_actions)-1}: "))
#        action = valid_actions[i]
#        s = game.get_next_state(state, action)
#        value, terminated = game.get_value_and_terminated(state)
#        e_x = game.encode_state(state)
#        if terminated:
#            profit = np.exp(value/1)
#            if game.check_terminal(state):
#                if profit < 1:
#                    logger.info(f'LOSS {state} with profit {profit}')
#                else:
#                    logger.info(f'WON {state} with profit {profit}')
#            else:
#                logger.info(f'LOSS {state} with profit {profit}')
#            break
#    exit()
