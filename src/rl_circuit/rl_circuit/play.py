#!/usr/bin/env python3play

from .game import NetGame
from .config import *
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
    """
    Main function here is where everything happens.
    Initializing objects and Running training and testing.
    """


    ### LOAD GRAPH DATA AND MAKE LINE GRAPH ###
    pools, tokens = load_pools_and_tokens(
        '../data/pools/pools_deg_15_liq_100_block_18.csv',
        '../data/tokens/tokens.csv',
    )
    prices = pd.read_parquet(
        '../data/prices/prices_deg_15_liq_100_block_18.parquet'
    )
    pools, prices = filter_pools_with_no_gradient(pools, prices)

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
        nx.set_node_attributes(L, {(e[0], e[1], e[2]['k']): e[2]['address']}, name='address')


    L, line_mapping = linear_node_relabel(L)

    edges = list(G.edges(data=True))
    for (t0, t1, d) in edges:
        inverse_weights = [1/el for el in d['weight']]
        G.add_edge(t1, t0, k=d['k'], weight=inverse_weights, address=d['address'])
    ########################################


    ### INITIALIZE THE GRAPH DATA FOR LEARNING ###
    edge_list = [list(e) for e in L.edges()]
    rates = np.array([np.log(e[-1]['mexr']) for e in L.nodes(data=True)]).T
    used = [0 for _ in range(len(L.nodes))]
    t0_using = [0 for _ in range(len(L.nodes))]
    t1_using = [0 for _ in range(len(L.nodes))]

    node_attr  = torch.tensor(np.dstack([*rates, used, t0_using, t1_using])).float().squeeze(0)
    edge_index = torch.tensor(edge_list).t().contiguous()
    data = Data(x=node_attr, edge_index=edge_index).to(DEVICE)
    #############################################

    ### LEARN ##
    game = NetGame(G, data, line_mapping, num_blocks, ARGS_GAME)

    model = ResNet(
        ARGS_MODEL
    ).to(DEVICE).share_memory()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    optimizer.zero_grad()


    mcts = MCTS(game, ARGS_TRAINING, model)

    rlearn = AgentRLearn(model, optimizer, game, ARGS_TRAINING)
    rlearn.learn()
    ###########

#   TEST
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
