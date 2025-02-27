#!/usr/bin/env python3

import numpy as np
import torch
import time
import sys
from torch_geometric.data import Data, Batch
from torch_geometric.nn import summary
sys.setrecursionlimit(1000000)

from rl_arb.initializer import Initializer
from rl_arb.config import DEVICE
from rl_arb.logger import logging
logger = logging.getLogger('rl_circuit')


def brute_force_search_trail(mdp, source, cap):
    def dfs(state):
        if len(state) > cap:
            return

        if mdp.check_win(state):
            trails.append(state)
            return

        valid_actions = mdp.get_valid_actions(state)
        for action in valid_actions:
            dfs(state + [action])
        return

    trails = []
    state = [(source, source, 0)]
    dfs(state)
    profits = [0]
    for trail in trails:
        profits.append(mdp.calculate_profit(trail, mdp.current_block))

    return trails, profits


@torch.no_grad()
def test_model():
    problem = Initializer()

    problem.mdp.data.to(DEVICE)
    problem.mdp.device = DEVICE
    problem.model.to(DEVICE)
    problem.model.load_state_dict(
        torch.load(
            "./model/model_49.pt",
            weights_only = True,
            map_location=DEVICE
        )
    )

    problem.model.eval()

#    state = [(1, 1, 0), (1, 0, 0), (0, 4, 1)]
#    data_list = [Data(
#        problem.mdp.encode_state(state, problem.mdp.current_block-1),
#        problem.mdp.data.edge_index,
#        y=problem.mdp.encode_state(state[:1], problem.mdp.current_block-1)
#    ) for _ in range(5)]
#    batch = Batch.from_data_list(data_list)
#    value, policy = problem.model.forward(
#        batch.x,
#        batch.edge_index,
#        batch.y,
#        batch.batch
#    )
#    policy = policy.view(batch.batch_size, -1).detach().cpu()
#    print(policy)
#    print(value)
#    exit()

#    np.random.seed(0)
    for b in range(problem.mdp.num_blocks):
        problem.mdp.current_block = b
        logger.info("\n")
        logger.info(f"Current BLOCK: {problem.mdp.current_block}")
        s = 0
        state = [(0, 0, 0)]
        t, p = brute_force_search_trail(problem.mdp, s, 7)
        max_p = np.max(p)
        logger.info(f"s: {s} trails {len(t)} max_profit: {max_p}")
        if len(t) > 0:
            logger.info(f"Max state {t[p.index(max_p)-1]}")

        while True:
            probs = problem.mcts.search(state)

            action_idx = np.random.choice(
                list(range(len(problem.mdp.edge_list))),
                p=probs
            )
            action = problem.mdp.edge_list[action_idx]
            state = problem.mdp.get_next_state(state, action)

            value, is_terminal = problem.mdp.get_value_and_terminated(state, problem.mdp.current_block)

            if is_terminal:
                if problem.mdp.check_win(state):
                    profit = problem.mdp.calculate_profit(state, problem.mdp.current_block)
                    if profit < 1:
                        logger.info(f'LOSS {state} with profit {profit}, {value}')
                    else:
                        logger.info(f'WON {state} with profit {profit}, {value}')
                else:
                    logger.info(f'LOSS {state} NO valid actions, {value}')
                break


def print_summary():

    problem = Initializer()

    problem.mdp.data.to(DEVICE)
    problem.mdp.device = DEVICE
    problem.model.to(DEVICE)

    state = [(1, 1, 0), (1, 2, 0)]
    e_x = problem.mdp.encode_state(state).to(DEVICE)
    edge_index = problem.mdp.data.edge_index

    print(summary(model, e_x, edge_index))

