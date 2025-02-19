#!/usr/bin/env python3

import numpy as np
import torch
import time
import sys
from torch_geometric.data import Data, Batch
sys.setrecursionlimit(1000000)

from rl_arb.initializer import Initializer
from rl_arb.config import DEVICE
from rl_arb.logger import logging
logger = logging.getLogger('rl_circuit')


def brute_force_search_trail(mdp, source):
    def dfs(state):
        if len(state) > 5:
            return

        if mdp.check_win(state):
            trails.append(state)
            return

        valid_actions = mdp.get_valid_actions(state)
        np.random.shuffle(valid_actions)
        for action in valid_actions:
            dfs(state + [action])
        return

    trails = []
    state = [(source, source, 0)]
    dfs(state)
    profits = [0]
    for trail in trails:
        profits.append(mdp.calculate_profit(trail))

    return trails, profits

def test_model():
    problem = Initializer()

    problem.mdp.data.to(DEVICE)
    problem.mdp.device = DEVICE
    problem.model.to(DEVICE)

    problem.model.load_state_dict(
        torch.load(
            "./model/model_99.pt",
            weights_only = True,
            map_location=DEVICE
        )
    )
    problem.model.eval()

    problem.mdp.current_block = np.random.choice(problem.mdp.num_blocks)
    logger.info(f"Current BLOCK: {problem.mdp.current_block}")
    for s in problem.mdp.nodes[1:]:
        logger.info("\n")
        state = [(s, s, 0)]
        t, p = brute_force_search_trail(problem.mdp, s)
        max_p = np.max(p)
        logger.info(f"s: {s} trails {len(t)} max_profit: {max_p}")
        if len(t) > 0:
            logger.info(f"Max state {t[p.index(max_p)]}")


        while True:
            probs = problem.mcts.search(state)

            action_idx = np.random.choice(
                list(range(len(problem.mdp.edge_list))),
                p=probs
            )
            action = problem.mdp.edge_list[action_idx]
            state = problem.mdp.get_next_state(state, action)

            value, is_terminal = problem.mdp.get_value_and_terminated(state)

            if is_terminal:
                if problem.mdp.check_win(state):
                    profit = problem.mdp.calculate_profit(state)
                    if profit < 1:
                        logger.info(f'LOSS {state} with profit {profit}')
                    else:
                        logger.info(f'WON {state} with profit {profit}')
                else:
                    logger.info(f'LOSS {state} NO valid actions')
                break

