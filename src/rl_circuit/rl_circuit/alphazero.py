#!/usr/bin/env python3

from .mcts import MCTS, MCTSParallel, PMemory
from .game import NetGame
from .config import DEVICE


import numpy as np
import torch
import torch.multiprocessing as mp
from torch.nn import functional as F

from tqdm import tqdm

import time

import logging
logger = logging.getLogger('rl_circuit')


class AlphaZero:
    def __init__(self, model, optimizer, game: NetGame, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTSParallel(self.game, self.args, self.model)


    def self_play(self):
        return_mem = []
        p_memory = [PMemory(self.game) for _ in range(self.args['num_parallel'])]

        while len(p_memory) > 0:
            states = [mem.state for mem in p_memory]
            self.mcts.search(states, p_memory)

            for i in range(len(p_memory))[::-1]:
                mem = p_memory[i]

                probs = np.zeros(len(self.game.nodes))
                for child in mem.root.children:
                    probs[child.action_taken[1]] = child.visit_count
                probs /= np.sum(probs)

                mem.memory.append((
                    mem.root.state,
                    probs
                ))

                temp_probs = probs ** (1 / self.args['temperature'])
                temp_probs /= temp_probs.sum()

                action = [mem.state[-1], np.random.choice(self.game.nodes, p=temp_probs)]
                mem.state = self.game.get_next_state(mem.state, action)
                value, is_terminal = self.game.get_value_and_terminated(mem.state)

                if is_terminal:
                    for hist_state, hist_probs in mem.memory:
                        return_mem.append((
                            hist_state,
                            hist_probs,
                            value

                        ))
                    del p_memory[i]
        return return_mem


    def train(self, memory):
        np.random.shuffle(memory)
        for batch_idx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batch_idx:np.min([len(memory) - 1, batch_idx + self.args['batch_size']])]
            states , policy_targets, value_targets = zip(*sample)

            policy_targets, value_targets = np.array(policy_targets), np.array(value_targets)
            policy_targets, value_targets = torch.tensor(policy_targets).to(DEVICE).float(), torch.tensor(value_targets).to(DEVICE).float()

            value_outs, policy_outs = [], []
            for s in states:
                v, p = self.model(
                    *self.game.encode_state(s)
                )
                value_outs += v
                policy_outs += p

            value_outs = torch.stack(value_outs).to(DEVICE).unsqueeze(1)
            policy_outs = torch.stack(policy_outs).to(DEVICE)

            policy_loss = F.cross_entropy(policy_outs, policy_targets)
            value_loss = F.mse_loss(value_outs, value_targets.unsqueeze(1))
            loss = policy_loss + value_loss


            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self):
        for iteration in range(self.args['num_iterations']):
            logger.info(f"Iterations {iteration+1}/{self.args['num_iterations']}")
            memory = []

            self.model.eval()
#            start = time.time()
#            for play_iter in tqdm(range(self.args['num_self_play_iterations']//self.args['num_parallel']), desc="self_play"):
#                memory += self.self_play()
#            end = time.time()
#            logger.info(f"normal execution {end-start}")

            play_iter = self.args['num_self_play_iterations']
            num_parallel = self.args['num_parallel']
            num_processes = self.args['num_processes']

            per_processor = play_iter//num_parallel//num_processes

            start = time.time()
            with mp.Pool(processes=num_processes) as pool:
                results = pool.starmap(
                    self_play_num_times,
                    [(self, per_processor) for _ in range(num_processes)]
                )
                pool.terminate()
            end = time.time()

            for result in results:
                memory += result

            logger.info(f"{len(memory)}")

            self.model.train()
            for epoch_iter in tqdm(range(self.args['num_epochs']), desc="epochs"):
                self.train(memory)

            torch.save(self.model.state_dict(), f"./model/model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"./model/optimizer_{iteration}.pt")


def self_play_num_times(alpha_zero, times=100):
    memory = []
    for _ in range(times):
        memory += alpha_zero.self_play()
    return memory
