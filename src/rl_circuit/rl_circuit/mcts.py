#!/usr/bin/env python3

from .config import DEVICE
import numpy as np
import torch

import logging
logger = logging.getLogger('rl_circuit')

class Node:
    """
    A class representing a node in the Monte Carlo Tree Search (MCTS).

    Attributes
    ----------
    args : dict
        Arguments for the MCTS.
    state : object
        The state of the game at this node.
    parent : Node, optional
        The parent node of this node.
    action_taken : object, optional
        The action taken to reach this node.
    prior : float, optional
        The prior probability of selecting this node.
    visit_count : int, optional
        The number of times this node has been visited.
    children : list
        The list of child nodes.
    expandable_actions : list
        The list of actions that can be taken from this node.
    value_sum : float
        The sum of values backpropagated through this node.

    Methods
    -------
    is_fully_expanded():
        Checks if the node is fully expanded.
    select(ucb_method='alphazero'):
        Selects the child node with the highest UCB value.
    get_ucb(child):
        Calculates the UCB value for a child node.
    get_ucb_alphazero(child):
        Calculates the UCB value for a child node using the AlphaZero method.
    expand(game):
        Expands the node by adding a child node.
    expand_alphazero(policy, game):
        Expands the node using the AlphaZero policy.
    simulate(game):
        Simulates a rollout from the current state.
    backpropagete(value):
        Backpropagates the value through the tree.
    """
    def __init__(
            self,
            game,
            args,
            state,
            parent=None,
            action_taken=None,
            prior=0,
            visit_count=0
    ):
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken

        self.children = []
        self.expandable_actions = game.get_valid_actions(state)

        self.visit_count = visit_count
        self.value_sum = 0

        self.prior = prior

    def is_fully_expanded(self):
        """
        Checks if the node is fully expanded.
        Node is fully expanded if there are no valid moves,
        or node is fully expanded if there are children.

        Returns
        -------
        bool
            True if the node is fully expanded, False otherwise.
        """
        return len(self.children) > 0

    def select(self, ucb_method='alphazero'):
        """
        Calculate ucb for all children.
        Selects the child node with the highest UCB value.

        Parameters
        ----------
        ucb_method : str, optional
            The method to calculate UCB value. Default is 'alphazero'.

        Returns
        -------
        Node
            The child node with the highest UCB value.
        """
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
        """
        Calculates the UCB value for a child node.
        Exploration vs Exploitation constant is self.args['C']

        Parameters
        ----------
        child : Node
            The child node.

        Returns
        -------
        float
            The UCB value for the child node.
        """
        q_value = ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * np.sqrt(np.log(self.visit_count)/child.visit_count)

    def get_ucb_alphazero(self, child):
        """
        Calculates the UCB value for a child node using the AlphaZero method.
        Exploration vs Exploitation constant is self.args['C']

        Parameters
        ----------
        child : Node
            The child node.

        Returns
        -------
        float
            The UCB value for the child node using the AlphaZero method.
        """
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = ((child.value_sum / child.visit_count) + 1) / 2

        return q_value + self.args['C'] * np.sqrt(self.visit_count)/(child.visit_count + 1) * child.prior

    def expand(self, game):
        """
        Expands the node by adding a child node.

        Parameters
        ----------
        game : object
            The game object.

        Returns
        -------
        Node
            The newly added child node.
        """
        action_index = np.random.choice(range(len(self.expandable_actions)))
        action = self.expandable_actions[action_index]

        self.expandable_actions.pop(action_index)

        child_state = self.state.copy()
        child_state = game.get_next_state(child_state, action)

        child = Node(game, self.args, child_state, self, action)
        self.children.append(child)
        return child

    def expand_alphazero(self, policy, game):
        """
        Expands the node using the AlphaZero policy.

        Parameters
        ----------
        policy : array-like
            The policy probabilities for each action.
        game : object
            The game object.
        """
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = game.get_next_state(child_state, game.edge_list[action])

                child = Node(game, self.args, child_state, self, game.edge_list[action], prob)
                self.children.append(child)

    def simulate(self, game):
        """
        Simulates a rollout from the current state.

        Parameters
        ----------
        game : object
            The game object.

        Returns
        -------
        float
            The value of the terminal state reached by the rollout.
        """
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
        """
        Backpropagates the value through the tree.

        Parameters
        ----------
        value : float
            The value to backpropagate.
        """
        self.value_sum += value
        self.visit_count += 1
        if self.parent is not None:
            self.parent.backpropagete(value)



class MCTS:
    """
    A class representing the Monte Carlo Tree Search (MCTS) algorithm.

    Attributes
    ----------
    game : object
        The game object.
    args : dict
        Arguments for the MCTS.
    model : object
        The model used for policy and value predictions.

    Methods
    -------
    search(state):
        Performs the MCTS search from the given state.
    """
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model


    @torch.no_grad()
    def search(self, state):
        """
        Performs the MCTS search from the given state.

        Parameters
        ----------
        state : object
            The state of the game.

        Returns
        -------
        array-like
            The visit count distribution over actions.
        """
        root = Node(self.game, self.args, state, visit_count=1)

        _, policy = self.model(
            self.game.encode_state(root.state),
            self.game.data.edge_index,
        )
        policy = torch.softmax(policy.squeeze(0), dim=0).detach().cpu().numpy()

        # dirichlet random noise
        policy = (1-self.args['eps']) * policy \
            + self.args['eps'] * np.random.dirichlet([self.args['dirichlet_alpha']]) * np.ones(policy.shape)
        policy = self.game.mask_policy(policy, root.state)

        root.expand_alphazero(policy, self.game)

        for s in range(self.args['num_searches']):
            node = root
            # selection
            while node.is_fully_expanded():
                node = node.select()


            value, is_terminal = self.game.get_value_and_terminated(node.state)

            if not is_terminal:
                value, policy = self.model(
                    self.game.encode_state(root.state),
                    self.game.data.edge_index
                )
                policy = torch.softmax(policy.squeeze(0), dim=0)
                policy = self.game.mask_policy(policy, node.state)

                value = value.item()

                # expansion
                node.expand_alphazero(policy, self.game)

                # simulation not needed for AlphaZero
                # node = node.expand()
                # value = node.simulate()



            # backpropagation
            node.backpropagete(value)

        # return visit_count distribution
        valid_actions = self.game.get_valid_actions(state)
        action_probs = np.zeros(len(self.game.edge_list))
        for child in root.children:
            action_probs[
                self.game.edge_list.index(child.action_taken)
            ] = child.visit_count

        action_probs /= np.sum(action_probs)
        return action_probs


class MCTSParallel:
    """
    A class representing the parallel version of
    the Monte Carlo Tree Search (MCTS) algorithm.

    Attributes
    ----------
    game : object
        The game object.
    args : dict
        Arguments for the MCTS.
    model : object
        The model used for policy and value predictions.

    Methods
    -------
    search(states, p_memory):
        Performs the parallel MCTS search from the given states.
    """
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model


    @torch.no_grad()
    def search(self, states, p_memory):
        """
        Performs the parallel MCTS search from the given states.

        Parameters
        ----------
        states : list
            The list of states to search from.
        p_memory : list
            The list of PMemory objects for parallel search.
        """
        policy = []
        for i, s in enumerate(states):
            policy += self.model(
                self.game.encode_state(s),
                self.game.data.edge_index
            )[1]
        policy = torch.stack(policy)

        policy = torch.softmax(policy, dim=1).detach().cpu().numpy()

        # dirichlet random noise
        policy = (1-self.args['eps']) * policy \
            + self.args['eps'] * np.random.dirichlet([self.args['dirichlet_alpha']]) * np.ones(policy.shape)

        for i, mem in enumerate(p_memory):
            p_policy = torch.tensor(policy[i])
            p_policy = self.game.mask_policy(p_policy, states[i])

            mem.root = Node(self.game, self.args, states[i], visit_count=1)
            mem.root.expand_alphazero(p_policy, self.game)

        for _ in range(self.args['num_searches']):
            for mem in p_memory:
                mem.node = None
                node = mem.root
                while node.is_fully_expanded():
                    node = node.select()

                value, is_terminal = self.game.get_value_and_terminated(node.state)

                if is_terminal:
                    node.backpropagete(value)
                else:
                    mem.node = node

            expandable = [i for i in range(len(p_memory)) if p_memory[i].node != None]

            if len(expandable) > 0:
                value, policy = [], []
                for i in expandable:
                    v, p = self.model(
                        self.game.encode_state(p_memory[i].node.state),
                        self.game.data.edge_index
                    )
                    value += v
                    policy += p
                policy = torch.stack(policy)
                value = torch.stack(value)

                policy = torch.softmax(policy, dim=1)

            for i, idx in enumerate(expandable):
                node = p_memory[idx].node
                p_value, p_policy = value[i], policy[i]

                p_policy = self.game.mask_policy(p_policy, node.state)

                node.expand_alphazero(p_policy, self.game)
                node.backpropagete(p_value)


class PMemory:
    """
    A class representing the memory used in parallel
    MCTS during parallel search.

    Attributes
    ----------
    state : list
        The state of the game.
    current_block : int
        The current block in the game.
    memory : list
        The memory of the search.
    root : Node
        The root node of the search tree.
    node : Node
        The current node in the search tree.
    """
    def __init__(self, game, at_block):
        node = int(np.random.choice(game.nodes))
        self.state = [(node, node, 0)]
        self.current_block = at_block
        self.memory = []
        self.root = None
        self.node = None

