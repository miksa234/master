#!/usr/bin/env python3

from .mcts import MCTS
from .game import NetGame
from .config import DEVICE


import numpy as np
import torch
import torch.multiprocessing as mp
from torch.nn import functional as F

import time

import logging
logger = logging.getLogger('rl_circuit')


class AlphaZero:
    def __init__(self, model, optimizer, game: NetGame, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(self.game, self.args, self.model)


    def self_play(self):
        memory = []
        state = [np.random.choice(self.game.nodes)]

        while True:
            #valid_actions = self.game.get_valid_actions(state)
            probs = self.mcts.search(state)
            memory.append((state, probs))

            temp_probs = probs ** (1 / self.args['temperature'])
            temp_probs /= temp_probs.sum()

            action = [state[-1], np.random.choice(self.game.nodes, p=temp_probs)]
            state = self.game.get_next_state(state, action)
            value, is_terminal = self.game.get_value_and_terminated(state)

            if is_terminal:
                return_mem = []
                for hist_state, hist_probs in memory:
                    return_mem.append((
                        self.game.encode_state(hist_state, hist_state[-1]).share_memory_(),
                        hist_probs,
                        value

                    ))
                return return_mem


    def train(self, memory, data):
        np.random.shuffle(memory)
        for batch_idx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batch_idx:np.min([len(memory) - 1, batch_idx + self.args['batch_size']])]

            encoded_states, policy_targets, value_targets = zip(*sample)

            policy_targets, value_targets = np.array(policy_targets), np.array(value_targets)
            policy_targets, value_targets = torch.tensor(policy_targets).to(DEVICE).float(), torch.tensor(value_targets).to(DEVICE).float()


            policy_outs, value_outs = [], []
            for encoded_state in encoded_states:
                out_value, out_policy = self.model(encoded_state, data.x, data.edge_index, data.edge_attr)
                out_policy = torch.softmax(out_policy.squeeze(0), dim=0)
                policy_outs.append(out_policy.unsqueeze(0))
                value_outs.append(out_value)

            policy_outs = torch.cat(policy_outs, dim=0)
            value_outs = torch.cat(value_outs, dim=0).squeeze(1)

            policy_loss = F.cross_entropy(policy_outs, policy_targets)
            value_loss = F.mse_loss(value_outs, value_targets)
            loss = policy_loss + value_loss


            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()




    def learn(self, data):
        for iteration in range(self.args['num_iterations']):
            logger.info(f"Iterations {iteration+1}/{self.args['num_iterations']}")
            memory = []
            self.model.eval()

#            start = time.time()
#            for play_iter in range(self.args['num_self_play_iterations']):
#                logger.info(f"SelfPlay {play_iter+1}/{self.args['num_self_play_iterations']}")
#                memory += self.self_play()
#            end = time.time()
#            logger.info(f"normal execution {end-start}")

            play_iter = self.args['num_self_play_iterations']
            num_processes = self.args['num_processes']
            start = time.time()
            with mp.Pool(processes=num_processes) as pool:
                results = pool.starmap(
                    self_play_num_times,
                    [(self, play_iter//num_processes) for _ in range(num_processes)]
                )
                pool.terminate()
            end = time.time()
            logger.info(f"paralle execution {end-start}")

            for result in results:
                memory += result

            self.model.train()
            for epoch_iter in range(self.args['num_epochs']):
                logger.info(f"Epoch {epoch_iter}/{self.args['num_epochs']}")
                self.train(memory, data)

            torch.save(self.model.state_dict(), f"./model/model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"./model/optimizer_{iteration}.pt")


def self_play_num_times(alpha_zero, times=100):
    memory = []
    for _ in range(times):
        memory += alpha_zero.self_play()
    return memory
