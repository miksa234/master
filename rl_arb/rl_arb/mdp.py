#!/usr/bin/env python3

import networkx as nx
import torch
import numpy as np
from torch_geometric.data import Data

from rl_arb.config import DEVICE, WETH
from rl_arb.logger import logging
logger = logging.getLogger('rl_circuit')

class MDP:
    """
    A class to represent the Markov Decision Process environment.

    Attributes:
    -----------
    G : nx.MultiDiGraph
        The graph representing the network.
    edges : dict
        A dictionary of edges with their weights.
    nodes : list
        A list of nodes in the graph.
    edge_list : list
        A list of edges in the graph.
    data : Data
        The data associated with the graph.
    line_mapping : dict
        A mapping of lines to their indices.
    num_blocks : int
        The number of blocks in the data.
    current_block : int
        The current block index.
    device: torch.device
        Device object.
    start_node:
        Starting node.
    token_pool_mapping:
        Dictonary with key=node value=line_mapping value.

    Methods:
    --------
    set_current_block(block):
        Sets the current block index.
    get_next_state(state, action):
        Returns the next state given the current state and action.
    get_valid_actions(state):
        Returns the valid actions from the current state.
    check_win(state):
        Checks if the current state is a terminal state.
    get_value_and_terminated(state, at_block):
        Returns the value and whether the state is terminal.
    encode_state(state, at_block):
        Encodes the current state into a tensor.
    mask_policy(policy, state):
        Masks the policy tensor based on valid actions.
    """

    def __init__(
            self,
            G: nx.MultiDiGraph,
            data,
            line_mapping,
            args,
            current_block=-1,
            start_node=0
    ):
        """
        Constructs all the necessary attributes for the MDP object.

        Parameters:
        -----------
        G : nx.MultiDiGraph
            The graph representing the network.
        data : torch_geometric.data.Data
            The data associated with the graph.
        line_mapping : dict
            A mapping of lines to their indices.
        num_blocks : int
            The number of blocks in the data.
        current_block : int, optional
            The current block index (default is -1).
        start_node: int, optional
            WETH index label in the problem graph (default is 0).
        """
        self.G = G
        self.args = args
        self.edges = {(i, j, w['k']): w['weight'] for i, j, w in G.edges(data=True)}
        self.nodes = list(G.nodes)
        self.edge_list = list(self.edges.keys())
        self.action_size = len(self.edge_list)

        self.data = data
        self.line_mapping = line_mapping
        self.line_mapping_keys = list(self.line_mapping.keys())
        self.num_blocks = len(list(self.edges.items())[0][1])-1
        self.current_block = self.num_blocks
        self.device = DEVICE
        self.start_node = start_node

        self.token_pool_mapping = {node: [] for node in self.nodes}
        for node in self.nodes:
            for k, v in line_mapping.items():
                if node == k[0]:
                    self.token_pool_mapping[node].append((v, 0))
                if node == k[1]:
                    self.token_pool_mapping[node].append((v, 1))

    def set_current_block(self, block_index):
        """
        Sets the current block index.
        Parameters:
        -----------
        block : int
            The current block index.
        """
        self.current_block = block_index

    def get_next_state(self, state, action):
        """
        Returns the next state given the current state and action.

        Parameters:
        -----------
        state : list
            The current state represented as a list of nodes.
        action : tuple
            The action represented as an edge.

        Returns:
        --------
        list
            The next state.
        """
        state.append(action)
        return state

    def get_valid_actions(self, state):
        """
        Returns the valid actions from the current state.

        Parameters:
        -----------
        state : list
            The current state represented as a list of nodes.

        Returns:
        --------
        list
            The valid actions represented as edges.
        """
        current_node = state[-1][1]
        filter_edges = [e for e in state[1:]] +\
            [(e[1], e[0], e[2]) for e in state[1:]]

        valid_actions = [e for e in self.G.edges(current_node, data='k') if e not in filter_edges]
        return valid_actions

    def check_win(self, state):
        """
        Checks if the state is a win.

        Parameters:
        -----------
        state : list
            The current state represented as a list of nodes.

        Returns:
        --------
        bool
            True if the state is terminal, False otherwise.
        """
        return state[0][1] == state[-1][1] and len(state) > 1

    def get_value_and_terminated(self, state, at_block):
        """
        Returns the value and whether the state is terminal.

        Parameters:
        -----------
        state : list
            The current state represented as a list of nodes.
        at_block: int
            Block of state.

        Returns:
        --------
        tuple
            The value and whether the state is terminal.
        """
        valid_actions = self.get_valid_actions(state)

        value = 0
        terminated = False

        if len(state) >= self.args['cutoff']:
            terminated = True

        if self.check_win(state):
            profit = self.calculate_profit(state, at_block)
            if profit > 3:
                profit = 1

            value += np.log(profit)*self.args['M']

            terminated = True
            return value, terminated

        if len(valid_actions) == 0:
            value = -1
            terminated = True

        return value, terminated

    def encode_state(self, state, at_block):
        """
        Encodes the current state into a tensor.

        Parameters:
        -----------
        state : list
            The current state represented as a list of nodes.
        at_block: int
            Block of state.

        Returns:
        --------
        torch.Tensor
            The encoded state tensor.
        """
        e_x = self.data.x.detach().clone()
        e_x= torch.dstack([
            e_x[:, at_block],       # exchange rate
            e_x[:, self.num_blocks+1],      # used binary
            e_x[:, self.num_blocks+2],      # t0 binary
            e_x[:, self.num_blocks+3],      # t1 binary
        ]).squeeze(0)

        # encode the current t0 or t1
        line_mapping_keys = list(self.line_mapping.keys())
        start_node = state[0][1]
        current_node= state[-1][1]

        for p, t01 in self.token_pool_mapping[start_node]:
            e_x[p, t01+2] = 1

        if len(state) > 1:
            profit = np.log(self.calculate_profit(state, at_block))
            if state[-1] in line_mapping_keys:
                line_state_current = state[-1]
            else:
                line_state_current = (state[-1][1], state[-1][0], state[-1][2])

            e_x[self.line_mapping[line_state_current],
                line_state_current[:2].index(current_node)+2] = profit

            # encode the used edges
            edge_idx = []
            for e in state[1:]:
                if e in line_mapping_keys:
                    edge_idx.append(self.line_mapping[e])
                if (e[1], e[0], e[2]) in line_mapping_keys:
                    edge_idx.append(self.line_mapping[(e[1], e[0], e[2])])
            e_x[edge_idx, 1] = 1

        return e_x.to(self.device)

    def mask_policy(self, policy, state):
        """
        Masks the policy tensor based on valid actions.

        Parameters:
        -----------
        policy : torch.Tensor
            The policy tensor.
        state : list
            The current state represented as a list of nodes.

        Returns:
        --------
        torch.Tensor
            The masked policy tensor.
        """
        valid_actions = self.get_valid_actions(state)
        valid_indices = torch.tensor(
            [self.edge_list.index(e) for e in valid_actions]
        )

        mask = torch.zeros_like(policy)
        mask[valid_indices] = 1

        masked_policy = policy * mask
        return masked_policy / masked_policy.sum()

    def calculate_profit(self, state, at_block):
       return np.prod(
            [self.edges[edge][at_block] for edge in state[1:]]
        )
