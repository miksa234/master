#!/usr/bin/env python3

import os
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
from rl_arb.mcts import MCTS
from rl_arb.net import PolicyNet
from rl_arb.rlearn import AgentRLearn
from rl_arb.reinforce import Reinforce
from rl_arb.config import (
    DEVICE,
    ARGS_GAME,
    ARGS_MODEL,
    ARGS_TRAINING,
    WETH
)
from rl_arb.utils import (
    load_pools_and_tokens,
    make_token_graph,
    linear_node_relabel,
    filter_pools_with_no_gradient,
    send_telegram_message
)


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
            '../data/pools/pools_deg_5_liq_100_block_18_grad.csv',
            '../data/tokens/tokens.csv',
        )
        prices = pd.read_parquet(
            '../data/prices/prices_deg_5_liq_100_block_18_grad.parquet'
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

        edges = list(G.edges(data=True))
        for (t0, t1, d) in edges:
            inverse_weights = [1/el for el in d['weight']]
            G.add_edge(t1, t0, k=d['k'], weight=inverse_weights, address=d['address'])


        edge_list = [list(e) for e in L.edges()]
        rates = np.array([np.log(n[1]['mexr']) for n in L.nodes(data=True)]).T
        used = [0 for _ in range(len(L.nodes))]
        t0_using = [0 for _ in range(len(L.nodes))]
        t1_using = [0 for _ in range(len(L.nodes))]

        node_attr  = torch.tensor(np.dstack([*rates, used, t0_using, t1_using])).float().squeeze(0)
        edge_index = torch.tensor(edge_list).t().contiguous()
        data = Data(x=node_attr, edge_index=edge_index)

        mdp = MDP(G, data, line_mapping, ARGS_GAME, start_node=token_mapping[WETH])

        model = PolicyNet(
            ARGS_MODEL
        ).share_memory().to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        optimizer.zero_grad()

        mcts = MCTS(mdp, ARGS_TRAINING, model)

        rlearn = AgentRLearn(model, mdp, ARGS_TRAINING)

        reinforce = Reinforce(model, optimizer, mdp, mcts, ARGS_TRAINING)

        if not os.path.exists('./model'):
            os.mkdir('./model')

        if not os.path.exists('./baseline'):
            os.mkdir('./baseline')

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
        self.reinforce = reinforce
        self.mcts = mcts # not parallel version
