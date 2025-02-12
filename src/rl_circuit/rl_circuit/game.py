#!/usr/bin/env python3

from .config import DEVICE
from torch_geometric.data import Data
import networkx as nx
import torch
import numpy as np

class NetGame:
    """
    A class to represent a network game.

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

    Methods:
    --------
    set_current_block(block):
        Sets the current block index.
    get_next_state(state, action):
        Returns the next state given the current state and action.
    get_valid_actions(state):
        Returns the valid actions from the current state.
    check_terminal(state):
        Checks if the current state is a terminal state.
    get_value_and_terminated(state):
        Returns the value and whether the state is terminal.
    encode_state(state):
        Encodes the current state into a tensor.
    mask_policy(policy, state):
        Masks the policy tensor based on valid actions.
    """

    def __init__(
            self,
            G: nx.MultiDiGraph,
            data,
            line_mapping,
            num_blocks,
            args,
            current_block=-1
    ):
        """
        Constructs all the necessary attributes for the NetGame object.

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
        """
        self.G = G
        self.args = args
        self.edges = {(i, j, w['k']): w['weight'] for i, j, w in G.edges(data=True)}
        self.nodes = list(G.nodes)
        self.edge_list = list(self.edges.keys())

        self.data = data
        self.line_mapping = line_mapping
        self.num_blocks = num_blocks
        self.current_block = self.num_blocks-1

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
        last_node = state[-1][1]
        filter_edges = []
        for e in state[1:]:
            filter_edges.append((e[0], e[1], e[2]))
            filter_edges.append(e)

        valid_actions = [e for e in self.G.edges(last_node, data='k') if e not in filter_edges]
        return valid_actions

    def check_terminal(self, state):
        """
        Checks if the current state is a terminal state.

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

    def get_value_and_terminated(self, state):
        """
        Returns the value and whether the state is terminal.

        Parameters:
        -----------
        state : list
            The current state represented as a list of nodes.

        Returns:
        --------
        tuple
            The value and whether the state is terminal.
        """
        valid_actions = self.get_valid_actions(state)

        value = 0
        terminated = False

        if len(valid_actions) == 0:
            value = 0
            terminated = True
        if len(state) <= 1 and len(valid_actions) == 0:
            value = -1
            terminated = True
        if len(state) <= 1 and len(valid_actions) > 0:
            value = 0
            terminated = False

        if self.check_terminal(state):
            edge_list = state[1:]
            value = np.sum([np.log(self.edges[edge][self.current_block]*self.args['tau']) for edge in edge_list])*self.args['M']
            if value > 0:
                value /= len(edge_list)
            terminated = True
        return value, terminated

    def encode_state(self, state):
        """
        Encodes the current state into a tensor.

        Parameters:
        -----------
        state : list
            The current state represented as a list of nodes.

        Returns:
        --------
        torch.Tensor
            The encoded state tensor.
        """
        e_x = self.data.x.detach().clone()
        # get the right prices for the block
        e_x= torch.dstack([
            e_x[:, self.current_block],       # exchange rate
#            e_x[:, self.num_blocks-1+1],      # swap fee
            e_x[:, self.num_blocks-1+1],      # used binary
            e_x[:, self.num_blocks-1+2],      # t0 binary
            e_x[:, self.num_blocks-1+3],      # t1 binary
        ]).squeeze(0)

        # encode the current t0 or t1
        line_mapping_keys = list(self.line_mapping.keys())
        current_node = state[-1][1]

        if len(state) > 1:
            if state[-1] in line_mapping_keys:
                line_state = state[-1]
            else:
                line_state = (state[-1][1], state[-1][0], state[-1][2])

            e_x[self.line_mapping[line_state],
                line_state[:2].index(current_node)+2] = 1

            # encode the used edges
            edge_idx = []
            for e in state[1:]:
                if e in line_mapping_keys:
                    edge_idx.append(self.line_mapping[e])
                if (e[1], e[0], e[2]) in line_mapping_keys:
                    edge_idx.append(self.line_mapping[(e[1], e[0], e[2])])
            e_x[edge_idx, 1] = 1

        return e_x.to(DEVICE)

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

        for e in self.edge_list:
            if e not in valid_actions:
                policy[self.edge_list.index(e)] = 0

        return policy / policy.sum()

