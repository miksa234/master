#!/usr/bin/env python3

import numpy as np
import torch
import time
import os
import sys
import pickle
from torch_geometric.data import Data, Batch
from torch_geometric.nn import summary

from rl_arb.mcts import MCTSParallel, PMemory
from rl_arb.utils import send_telegram_message
sys.setrecursionlimit(1000000)

from rl_arb.initializer import Initializer
from rl_arb.config import ARGS_TRAINING, DEVICE
from rl_arb.logger import logging
logger = logging.getLogger('rl_circuit')

def brute_force_search_trail(mdp, source, cap):
    """
    Brute force solution of the MDP problem, by doing recursive depth first
    search and capping the length of the state by 'cap'

    Parameters
    ---------
        mdp: MDP
            Class of of the MDP.
        source: int
            Source node.
        cap: int
            Boundary on solution length.

    Returns
    -------
    tuple
        - list of solutions
        - list of values of solutions performance (profit)
    """
    def dfs(state):
#        if len(state) > cap:
#            return

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

    return trails, np.max(profits)


@torch.no_grad()
def test_model():
    """
        Testing function
    """
    problem = Initializer()

    problem.mdp.data.to(DEVICE)
    problem.mdp.device = DEVICE
    problem.model.to(DEVICE)
    problem.model.share_memory()
    problem.model.eval()

    mcts_parallel = MCTSParallel(problem.mdp, ARGS_TRAINING)

    if not os.path.exists('./test'):
        os.mkdir('./test')

    loss = {}
    values = {}
    brute_values = {}
    np.random.seed(0)
    selected_blocks = np.random.choice(problem.mdp.num_blocks, 100, replace=False)

    start = problem.mdp.start_node
    times = []
    for b in selected_blocks:
        problem.mdp.current_block = b
        problem.mcts.mdp.current_block = b
        st = time.time()
        _, brute_profit = brute_force_search_trail(problem.mdp, start, 10)
        et = time.time()
        print(et-st)
        times.append(et-st)
        brute_values[b] = np.log(brute_profit)
    print(np.mean(times))
    exit()

    for i in range(100):
#        problem.model.load_state_dict(
#            torch.load(
#                f"./model/model_{i}.pt",
#                weights_only = True,
#                map_location=DEVICE
#            ),
#            strict = False
#        )

        logger.info(f"Iterations {i}/100")
#        send_telegram_message(f"Iterations {i}/100")
        loss[i] = {}
        values[i] = {}

        step = 25
        for block in range(step, len(selected_blocks)+step, step):

            p_memory = [
                PMemory(mcts_parallel.mdp, b) for b in selected_blocks[block-step: block]
            ]

            st = time.time()
            while len(p_memory) > 0:
                states = [mem.state for mem in p_memory]

                mcts_parallel.search(problem.model, states, p_memory)
                for m in range(len(p_memory))[::-1]:
                    mem = p_memory[m]
                    probs = np.zeros(len(mcts_parallel.mdp.edge_list))

                    for child in mem.root.children:
                        probs[
                            mcts_parallel.mdp.edge_list.index(child.action_taken)
                        ] = child.value_best

                    if np.sum(probs) == 0:
                        for child in mem.root.children:
                            probs[
                                mcts_parallel.mdp.edge_list.index(child.action_taken)
                            ] = child.visit_count

                    probs /= np.sum(probs)

                    action = mcts_parallel.mdp.edge_list[np.argmax(probs)]
                    mem.state = mcts_parallel.mdp.get_next_state(mem.state, action)

                    value, is_terminal = mcts_parallel.mdp.get_value_and_terminated(
                        mem.state,
                        mem.current_block
                    )

                    if is_terminal:
                        cb = mem.current_block
                        values[i][cb] = value
                        loss[i][cb] = 1-value/brute_values[cb]

                        del p_memory[m]
                et = time.time()
                print((et-st)/25)


        logger.info(f"AVERAGE loss {np.mean([loss[i][k] for k in loss[i].keys()])}")
#        send_telegram_message(f"AVERAGE loss {np.mean([loss[i][k] for k in loss[i].keys()])}")

        with open(f'./test/test.pickle', "wb") as f:
            pickle.dump([loss, values, brute_values], f)


