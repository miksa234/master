#!/usr/bin/env python3

import numpy as np
import torch
import time
import sys
from torch_geometric.data import Data, Batch
from torch_geometric.nn import summary

from rl_arb.utils import send_telegram_message
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
            "../model_1000.pt",
            weights_only = True,
            map_location=DEVICE
        ),
        strict = False
    )
    problem.model.share_memory()
    problem.model.eval()


#    for name, param in problem.model.named_parameters():
#        print(name)
#        print(param.shape)
#        print(param)
#    exit()

#    state = [(1, 1, 0), (1, 0, 0), (0, 4, 1)]
#    actions = [(0, 4, 1) for _ in range(5)]
#    data_list = [Data(
#        problem.mdp.encode_state(state, problem.mdp.current_block-1),
#        problem.mdp.data.edge_index,
#        y=problem.mdp.encode_state(state[:1], problem.mdp.current_block-1)
#    ) for _ in range(5)]
#    batch = Batch.from_data_list(data_list).to(DEVICE)
#    policy = problem.model.forward(
#        batch.x,
#        batch.edge_index,
#        batch.y,
#        batch.batch
#    )
#    policy = policy.view(batch.batch_size, -1)
#    values = torch.tensor([0.4, 0.5, 0.5, 0.5, 0.5]).float().to(DEVICE)
#
#    one_hot = torch.zeros_like(policy)
#    for a, p in zip(actions, one_hot):
#        p[problem.mdp.edge_list.index(a)] = 1
#
#    cross = torch.nn.functional.cross_entropy(policy, one_hot, reduction='none') * values
#    print(cross)
#    exit()

#    print(values.shape)
#
#    loss = torch.tensor(0.0).to(DEVICE)
#    loss += -torch.log(policy[0, problem.mdp.edge_list.index(state[-1])]) * values[0]
#    print(loss)
#    exit()


#    np.random.seed(0)


    loss = []
    values = []
    for b in range(problem.mdp.num_blocks):
        problem.mdp.current_block = b
        logger.info("\n")
        logger.info(f"Current BLOCK: {problem.mdp.current_block}")
        s = 0
        state = [(0, 0, 0)]
        t, p = brute_force_search_trail(problem.mdp, s, 5)
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
            values.append(value)

            if is_terminal:

                vs = np.array(values)
                vs_mean = np.mean(values, where=(vs!=0))
                vs_mean = torch.tensor(vs_mean)
                logger.info(f'mean values {vs_mean}, max_values = {np.max(values)}')
                if problem.mdp.check_win(state):
                    profit = problem.mdp.calculate_profit(state, problem.mdp.current_block)
                    if max_p < 4 and max_p > 0 :
                        loss.append(1-profit/max_p)
                    if profit < 1:
                        logger.info(f'LOSS {state} with profit {profit}, {value}')
                    else:
                        logger.info(f'WON {state} with profit {profit}, {value}')
                else:
                    logger.info(f'LOSS {state} NO valid actions, {value}')
                break

        logger.info(f"AVERAGE loss {np.mean(loss)}")

