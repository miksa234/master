#!/usr/bin/env python3

import networkx as nx
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import logging
logger = logging.getLogger('rl_circuit')
import torch.multiprocessing as mp
try:
   mp.set_start_method('spawn', force=True)
except RuntimeError:
   pass

from rl_arb.mdp import MDP
from rl_arb.config import *
from rl_arb.mcts import MCTS
from rl_arb.net import Net
from rl_arb.rlearn import AgentRLearn
from rl_arb.utils import *
from rl_arb.brute_force import *


class Initializer():
    """
    Initializer object for the problem to load all the importaint variables
    through a class

    Attributes:
    ----------
    pools: pd.DataFrame
        Filtered pools data used for the problem.
    tokens: pd.DataFrame
        All tokens found from all pools.
    prices: pd.DataFrame
        Prices of the used pools in a time interval.
    graph: nx.MultiDiGraph
        Problem graph, where edges are pools, nodes tokens.
    token_mapping: dict
        Linear node relabeling of eth address to index.
    line_graph: nx.Graph
        The line graph of 'graph' object.
    line_mapping: dict
        Linear node relabeling.
    mdp: MDP
        Network Markov Decision Process to be explored for arbitrage.
    model: torch.nn.Module
        Torch geometric deep graph neural network.
    optimizer: torch.optim.Adam
        Torch optimizer used to update the weights.
    rlearn: AgentRLearn
        Mcts based reinforcement learning algorithm.
    mcts: MCTS
        Monte Carlo Tree search algorithm.

    """
    def __init__(self):
        pools, tokens = load_pools_and_tokens(
            '../data/pools/pools_deg_15_liq_100_block_18.csv',
            '../data/tokens/tokens.csv',
        )
        prices = pd.read_parquet(
            '../data/prices/prices_deg_15_liq_100_block_18.parquet'
        )
        pools, prices = filter_pools_with_no_gradient(pools, prices)

        G = make_token_graph(pools, prices)
        G, token_mapping = linear_node_relabel(G)

        L = nx.line_graph(G, create_using=nx.Graph)
        for e in G.edges(data=True):
            nx.set_node_attributes(
                L,
                {(e[0], e[1], e[2]['k']): e[2]['weight']},
                name='mexr'
            )
            nx.set_node_attributes(
                L,
                {(e[0], e[1], e[2]['k']): e[2]['address']},
                name='address'
            )
        L, line_mapping = linear_node_relabel(L)

        for (t0, t1, d) in G.edges(data=True):
            inverse_weights = [1/el for el in d['weight']]
            G.add_edge(t1, t0, k=d['k'], weight=inverse_weights, address=d['address'])

        edge_list = [list(e) for e in L.edges()]
        rates = np.array([np.log(e[-1]['mexr']) for e in L.nodes(data=True)]).T
        used = [0 for _ in range(len(L.nodes))]
        t0_using = [0 for _ in range(len(L.nodes))]
        t1_using = [0 for _ in range(len(L.nodes))]

        node_attr  = torch.tensor(np.dstack([*rates, used, t0_using, t1_using])).float().squeeze(0)
        edge_index = torch.tensor(edge_list).t().contiguous()
        data = Data(x=node_attr, edge_index=edge_index).to(DEVICE)

        mdp = MDP(G, data, line_mapping, ARGS_GAME)

        model = Net(
            ARGS_MODEL
        ).to(DEVICE).share_memory()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        optimizer.zero_grad()

        mcts = MCTS(mdp, ARGS_TRAINING, model)

        rlearn = AgentRLearn(model, optimizer, mdp, ARGS_TRAINING)

        self.pools = pools
        self.tokens = tokens
        self.prices = prices

        self.graph = G
        self.token_mapping = token_mapping
        self.line_graph = L
        self.line_mapping = line_mapping
        self.graph_data = data

        self.mdp = mdp
        self.model = model
        self.optimizer = optimizer
        self.rlearn = rlearn
        self.mcts = mcts # not parallel version
