#!/usr/bin/env python3

from .config import DEVICE
import numpy as np
import torch

class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0):
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken

        self.children = []
        self.expandable_actions = game.get_valid_actions(state)

        self.visit_count = 0
        self.value_sum = 0

        self.prior = prior

    def is_fully_expanded(self):
        # node is fully expanded if there are no valid moves
        # node is fully expanded if there are children
        return len(self.expandable_actions) == 0 and len(self.children) > 0

    def select(self, ucb_method='alphazero'):
        # calculate ucb for all children
        # select child with best ucb
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            if ucb_method == 'alphazero':
                ucb = self.get_ucb_alphazero(child)
            else:
                ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child):
        # exploration vs exploitation
        q_value = ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * np.sqrt(np.log(self.visit_count)/child.visit_count)

    def get_ucb_alphazero(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = ((child.value_sum / child.visit_count) + 1) / 2

        return q_value + self.args['C'] * np.sqrt(self.visit_count)/(child.visit_count + 1) * child.prior

    def expand(self, game):
        action_index = np.random.choice(range(len(self.expandable_actions)))
        action = self.expandable_actions[action_index]
        self.expandable_actions.pop(action_index)

        child_state = self.state.copy()
        child_state = game.get_next_state(child_state, action)

        child = Node(game, self.args, child_state, self, action)
        self.children.append(child)
        return child

    def expand_alphazero(self, policy, game):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = game.get_next_state(child_state, (self.state[-1], action))

                child = Node(game, self.args, child_state, self, (self.state[-1], action), prob)
                self.children.append(child)
        return child

    def simulate(self, game):
        value, is_terminal = game.get_value_and_terminated(self.state)
        if is_terminal:
            return value

        rollout_state = self.state.copy()
        while True:
            valid_actions = game.get_valid_actions(rollout_state)
            action_index = np.random.choice(range(len(valid_actions)))
            action = valid_actions[action_index]

            rollout_state = game.get_next_state(rollout_state, action)
            value, is_terminal = game.get_value_and_terminated(rollout_state)
            if is_terminal:
                return value

    def backpropagete(self, value):
        self.value_sum += value
        self.visit_count += 1
        if self.parent is not None:
            self.parent.backpropagete(value)



class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model


    @torch.no_grad()
    def search(self, state):
        root = Node(self.game, self.args, state)

        for search in range(self.args['num_searches']):
            node = root
            # selection
            while node.is_fully_expanded():
                node = node.select()

            value, is_terminal = self.game.get_value_and_terminated(node.state)
            if not is_terminal:
                value, policy = self.model(
                    torch.tensor(node.state).to(DEVICE),
                    self.game.node_emb,
                    self.game.data.edge_index,
                    self.game.data.edge_attr
                )
                policy = torch.softmax(policy.squeeze(0), dim=0)
                policy = self.game.mask_policy(policy, node.state)

                value = value.item()

                # expansion
                node = node.expand_alphazero(policy, self.game)

                # simulation not needed for AlphaZero
                # node = node.expand()
                # value = node.simulate()



            # backpropagation
            node.backpropagete(value)

        # return visit_count distribution
        valid_actions = self.game.get_valid_actions(state)
        action_probs = np.zeros(len(valid_actions))
        for child in root.children:
            action_probs[valid_actions.index(child.action_taken)] = child.visit_count

        action_probs /= np.sum(action_probs)
        return action_probs

