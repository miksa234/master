#!/usr/bin/env python3

from .config import DEVICE
from torch_geometric.data import Data
import networkx as nx
import torch
import numpy as np

class NetGame:
    def __init__(self, G: nx.Graph):
        self.G = G
        self.edges = {(i, j): w['weight'] for i, j, w in G.edges(data=True)}
        self.nodes = list(G.nodes)
        self.adj_matrix = nx.adjacency_matrix(G).toarray()
        self.data = self.get_net_data().to(DEVICE)

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

    def check_win(self, state):
        return state[0] == state[-1] and len(state) > 1

    def get_value_and_terminated(self, state):
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

        if self.check_win(state):
            trade_penalty = 1 - 0.01 # 1 percent of the trade
            edge_list = [(state[i], state[i+1]) for i in range(len(state)-1)]
            value = np.log(np.prod([self.G[edge[0]][edge[1]]['weight']*trade_penalty for edge in edge_list]))
            terminated = True

        elif len(state) > 1:
            value = self.estimate_value(state)
            terminated = False

        return value, terminated

    def estimate_value(self, state):
        trade_penalty = 1 - 0.01 # 1 percent of the trade
        edge_list = [(state[i], state[i+1]) for i in range(len(state)-1)]
        r_cum = np.prod([self.G[edge[0]][edge[1]]['weight']*trade_penalty for edge in edge_list])

        r_rem = max((self.edges[edge] for edge in self.G.edges(state[-1]) if edge not in edge_list), default=1.0)
        return np.log(r_cum*r_rem)

    def get_profit(self, state):
        edge_list = [(state[i], state[i+1]) for i in range(len(state)-1)]
        profit = np.prod([self.G[edge[0]][edge[1]]['weight'] for edge in edge_list])
        return profit

    def get_net_data(self):
        edge_list = [list(e) for e in self.edges]
        weights = [self.edges[e] for e in self.edges]

        # TODO: NODE ATTRIBUTES ????
        # for now use the node degree
        node_attributes = [[degree] for _, degree in self.G.degree()]

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        node_attr  = torch.tensor(node_attributes, dtype=torch.float)
        edge_attr = torch.tensor(weights, dtype=torch.float)

        data = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)
        return data



    def encode_state(self, state):
        A = torch.tensor(self.adj_matrix).unsqueeze(0).to(DEVICE)

        V = torch.zeros(self.adj_matrix.shape)
        if len(state) > 1:
            path = tuple(zip(*[(state[i], state[i+1]) for i in range(len(state)-1)]))
            V[path] = 1

        V = V.unsqueeze(0).to(DEVICE)

        current_node = torch.tensor(state[-1]).unsqueeze(0).repeat(1, len(self.nodes), len(self.nodes)).to(DEVICE)

        encoded_state = torch.cat([A, V, current_node], dim=0).float()
        return encoded_state

    def mask_policy(self, policy, state):
        valid_nodes = [i for _, i in self.get_valid_actions(state)]
        mask = [i for i in range(len(self.nodes)) if i not in valid_nodes]
        policy[mask] = 0
        return policy / policy.sum()

