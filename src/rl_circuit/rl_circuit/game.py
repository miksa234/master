#!/usr/bin/env python3

from .config import DEVICE
from torch_geometric.data import Data
import networkx as nx
import torch
import numpy as np

class NetGame:
    def __init__(self, G: nx.MultiDiGraph, data, line_mapping):
        self.G = G
        self.edges = {(i, j, w['k']): w['weight'] for i, j, w in G.edges(data=True)}
        self.nodes = list(G.nodes)
        self.adj_matrix = nx.adjacency_matrix(G).toarray()
        self.edge_list = list(self.edges.keys())

        self.data = data
        self.line_mapping = line_mapping

    def get_next_state(self, state, action):
        # state is a set of nodes, last one where we are
        # action is the edge
        # next state is the node on the other side of the edge
        state.append(action)
        return state

    def get_valid_actions(self, state):
        # check all available edges from last node in the state
        last_node = state[-1][1]
        filter_edges = []
        for e in state[1:]:
            filter_edges.append((e[0], e[1], e[2]))
            filter_edges.append(e)

        valid_actions = [e for e in self.G.edges(last_node, data='k') if e not in filter_edges]
        return valid_actions

    def check_terminal(self, state):
        return state[0][1] == state[-1][1] and len(state) > 1

    def get_value_and_terminated(self, state):
        valid_actions = self.get_valid_actions(state)

        value = 0
        terminated = False

        if len(valid_actions) == 0:
            value = 0
            terminated = True
        if len(state) <= 1 and len(valid_actions) == 0:
            value = -30
            terminated = True
        if len(state) <= 1 and len(valid_actions) > 0:
            value = 0
            terminated = False

        if self.check_terminal(state):
            trade_penalty = np.log(1 - 0.01) # 1 percent of the trade
            edge_list = state[1:]
            win_loss_amplifier = 1
            value = np.sum([np.log(self.edges[edge])+trade_penalty for edge in edge_list])*win_loss_amplifier
            terminated = True

        # otherwise check if we go back to eth if we are in profit add return this value
        # weth should be node index 0 (or main currency)

        return value, terminated

    def encode_state(self, state):
        e_x = self.data.x.detach().clone()
        line_mapping_keys = list(self.line_mapping.keys())
        current_node = state[-1][1]
        if state[-1] in line_mapping_keys:
            # TODO: assumees t0 is entry 0 and t1 is entry 1
            line_state = state[-1]
        else:
            line_state = (state[-1][1], state[-1][0], state[-1][2])

        e_x[line_state, line_state[:2].index(current_node)] = 1

        if len(state) > 1:
            edge_idx = []
            for e in state[1:]:
                if e in line_mapping_keys:
                    edge_idx.append(self.line_mapping[e])
                if (e[1], e[0], e[2]) in line_mapping_keys:
                    edge_idx.append(self.line_mapping[(e[1], e[0], e[2])])
            e_x[edge_idx, 2] = 1

        return e_x

    def mask_policy(self, policy, state):
        valid_actions = self.get_valid_actions(state)
        line_mapping_keys = list(self.line_mapping.keys())

        extended_policy = np.zeros(len(self.edge_list))
        mask = []
        for i, e in enumerate(self.edges):
            if e in valid_actions:
                if e in line_mapping_keys:
                    extended_policy[i] = policy[self.line_mapping[e]]
                else:
                    extended_policy[i] = policy[self.line_mapping[(e[1], e[0], e[2])]]

        extended_policy = torch.tensor(extended_policy).to(DEVICE)
        return extended_policy / extended_policy.sum()

