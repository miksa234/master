#!/usr/bin/env python3

from .mcts import MCTS
from .game import NetGame

import numpy as np
import torch
from torch.nn import functional as F


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
            valid_actions = self.game.get_valid_actions(state)
            probs = self.mcts.search(state)
            memory.append((state, probs))
            action = np.random.choice(valid_actions, p=probs)
            state = self.game.get_next_state(state, action)
            value, is_terminal = self.game.get_value_and_terminated(state)

            if is_terminal:
                return_mem = []
                for hist_state, hist_probs in memory:
                    return_mem.append((
                        hist_state,
                        hist_probs,
                        value

                    ))
                return return_mem

    def train(self, memory, data):
        np.random.shuffle(memory)
        for batch_idx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batch_idx:np.min(len(memory) - 1, batch_idx + self.args['batch-size'])]
            state, policy_targets, value_targets = (torch.tensor(it, dtype=torch.float32) for it in zip(*sample))

            out_value, out_policy = self.model(state, data.x, data.edge_index, data.edge_attr)
            out_policy = torch.softmax(out_policy.squeeze(0), dim=0)
            out_policy = self.game.mask_policy(out_policy, state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()



    def learn(self, data):
        for iteration in range(self.args['num_iterations']):
            memory = []
            self.model.eval()
            for _ in range(self.args['num_self_play_iterations']):
                memory += self.self_play()

            self.model.train()
            for _ in range(self.args['num_epochs']):
                self.train(memory, data)

            torch.save(self.model.state_dict(), f"./model/model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"./model/optimizer_{iteration}.pt")
