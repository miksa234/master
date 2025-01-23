#!/usr/bin/env python3

from .config import DEVICE
from torch_geometric.data import Data
import networkx as nx
import torch
import numpy as np

class NetGame:
    def __init__(self, G: nx.Graph, data):
        self.G = G
        self.edges = {(i, j): w['weight'] for i, j, w in G.edges(data=True)}
        self.nodes = list(G.nodes)
        self.adj_matrix = nx.adjacency_matrix(G).toarray()
        self.data = data
        self.edge_list = self.data.edge_index.T.tolist()

    def get_next_state(self, state, action):
        # state is a set of nodes, last one where we are
        # action is the edge
        # next state is the node on the other side of the edge
        state.append(action[1])
        return state

    def get_valid_actions(self, state):
        # check all available edges from last node in the state
        last_node = state[-1]
        state_path = [(state[i], state[i+1]) for i in range(len(state)-1)] \
                + [(state[i+1], state[i]) for i in range(len(state)-1)]

        valid_actions = [e for e in self.G.edges(last_node) if e not in state_path]
        return valid_actions

    def check_terminal(self, state):
        return state[0] == state[-1] and len(state) > 1

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
            trade_penalty = 1 - 0.01 # 1 percent of the trade
            edge_list = [(state[i], state[i+1]) for i in range(len(state)-1)]
            win_loss_amplifier = 10
            value = np.log(np.prod([self.G[edge[0]][edge[1]]['weight']*trade_penalty for edge in edge_list]))*win_loss_amplifier
            terminated = True

        return value, terminated

    def get_profit(self, state):
        edge_list = [(state[i], state[i+1]) for i in range(len(state)-1)]
        profit = np.prod([self.G[edge[0]][edge[1]]['weight'] for edge in edge_list])
        return profit


    def encode_state(self, state):
        e_x = self.data.x.detach().clone()
        e_edge_attr = self.data.edge_attr.detach().clone()

        e_x[state[-1], 2] = 1

        if len(state) > 1:
            edges = [self.edge_list.index([state[i], state[i+1]]) for i in range(len(state)-1)]
            e_edge_attr[edges, 2] = 1

        return e_x, self.data.edge_index, e_edge_attr

    def mask_policy(self, policy, state):
        valid_nodes = [i for _, i in self.get_valid_actions(state)]
        mask = [i for i in range(len(self.nodes)) if i not in valid_nodes]
        policy[mask] = 0
        return policy / policy.sum()

