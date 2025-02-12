#!/usr/bin/env python3

import numpy as np
import networkx as nx



def brute_force_search_trail(graph, source):
    def dfs(path):
        current_node = path[-1]
        if len(path) > 1 and path[-1] == path[0]:
            trails.append(path)
            return
        if len(path) == 1:
            for neighbor in graph.neighbors(current_node):
                dfs(path + [neighbor])
        for neighbor in graph.neighbors(current_node):
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            if (path[-1], neighbor) not in path_edges:
                dfs(path + [neighbor])
        return

    trails = []
    dfs([source])
    weights = []
    trade_penalty = np.log(1 - 0.01)
    for trail in trails:
        w = 0
        print(trail)
        for i in range(len(trail)-1):
            w += graph[trail[i]][trail[i+1]]['weight']+trade_penalty
        weights.append(np.exp(w))

    return trails, weights



