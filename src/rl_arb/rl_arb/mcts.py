#!/usr/bin/env python3

import numpy as np
import torch
from torch_geometric.data import Batch, Data

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
        The state of the mdp at this node.
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
    value_best: float
        The best of values backpropagated through this node.

    Methods
    -------
    is_fully_expanded():
        Checks if the node is fully expanded.
    select(ucb_method='alphazero'):
        Selects the child node with the highest UCB value.
    get_ucb(child):
        Calculates the UCB value for a child node.
    expand(mdp):
        Expands the node by adding a child node.
    expand_alphazero(policy, mdp):
        Expands the node using the AlphaZero policy.
    simulate(mdp):
        Simulates a rollout from the current state.
    backpropagete(value):
        Backpropagates the value through the tree.
    """
    def __init__(
            self,
            mdp,
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
        self.expandable_actions = mdp.get_valid_actions(state)

        self.visit_count = visit_count
        self.value_best = 0

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

    def select(self):
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
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child):
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
        q_value = child.value_best
        u_value = np.sqrt(self.visit_count)/(child.visit_count + 1) * child.prior

        return q_value + self.args['C'] * u_value

    def expand(self, policy, mdp):
        """
        Expands the node using the AlphaZero policy.

        Parameters
        ----------
        policy : array-like
            The policy probabilities for each action.
        mdp : object
            The mdp object.
        """
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = mdp.get_next_state(child_state, mdp.edge_list[action])

                child = Node(mdp, self.args, child_state, self, mdp.edge_list[action], prob)
                self.children.append(child)

    def simulate(self, mdp):
        """
        Simulates a rollout from the current state.

        Parameters
        ----------
        mdp : object
            The mdp object.

        Returns
        -------
        float
            The value of the terminal state reached by the rollout.
        """
        value, is_terminal = mdp.get_value_and_terminated(self.state)
        if is_terminal:
            return value

        rollout_state = self.state.copy()
        while True:
            valid_actions = mdp.get_valid_actions(rollout_state)
            action_index = np.random.choice(range(len(valid_actions)))
            action = valid_actions[action_index]

            rollout_state = mdp.get_next_state(rollout_state, action)
            value, is_terminal = mdp.get_value_and_terminated(rollout_state)
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
        self.visit_count += 1

        if self.value_best < value:
            self.value_best = value

        if self.parent is not None:
            self.parent.backpropagete(value)



class MCTS:
    """
    A class representing the Monte Carlo Tree Search (MCTS) algorithm.

    Attributes
    ----------
    mdp : object
        The mdp object.
    args : dict
        Arguments for the MCTS.
    model : object
        The model used for policy and value predictions.

    Methods
    -------
    search(state):
        Performs the MCTS search from the given state.
    """
    def __init__(self, mdp, args, model):
        self.mdp = mdp
        self.args = args
        self.model = model


    @torch.no_grad()
    def search(self, state):
        """
        Performs the MCTS search from the given state.

        Parameters
        ----------
        state : object
            The state of the mdp.

        Returns
        -------
        array-like
            The visit count distribution over actions.
        """
        root = Node(self.mdp, self.args, state, visit_count=1)

        _, policy = self.model(
            self.mdp.encode_state(root.state),
            self.mdp.data.edge_index,
        )
        policy = torch.softmax(policy.squeeze(0), dim=0).detach().cpu()

#         dirichlet random noise
#        policy = (1-self.args['eps']) * policy \
#            + self.args['eps'] * np.random.dirichlet([self.args['dirichlet_alpha']]*len(self.mdp.edges))

        policy = self.mdp.mask_policy(policy, root.state)

        root.expand(policy, self.mdp)

        for s in range(self.args['num_searches']-len(state)):
            node = root
            while node.is_fully_expanded():
                node = node.select()


            value, is_terminal = self.mdp.get_value_and_terminated(node.state)

            if not is_terminal:
                value, policy = self.model(
                    self.mdp.encode_state(root.state),
                    self.mdp.data.edge_index
                )
                policy = torch.softmax(policy.squeeze(0), dim=0)
                policy = self.mdp.mask_policy(policy, node.state)

                value = value.item()

                node.expand(policy, self.mdp)


            node.backpropagete(value)

        # return visit_count distribution
        valid_actions = self.mdp.get_valid_actions(state)
        action_probs = np.zeros(len(self.mdp.edge_list))
        for child in root.children:
            action_probs[
                self.mdp.edge_list.index(child.action_taken)
            ] = child.visit_count

        action_probs /= np.sum(action_probs)
        return action_probs


class MCTSParallel:
    """
    A class representing the parallel version of
    the Monte Carlo Tree Search (MCTS) algorithm.

    Attributes
    ----------
    mdp : MDP
        The mdp object.
    args : dict
        Arguments for the MCTS.
    model : nn.Module
        The model used for policy and value predictions.

    Methods
    -------
    set_C(C):
        Sets the exploration parameter for the MCTS.
    search(states, p_memory):
        Performs the parallel MCTS search from the given states.
    """
    def __init__(self, mdp, args, model):
        self.mdp = mdp
        self.args = args
        self.model = model

    def set_C(self, C):
        """
        Set the exploration parameter for the MCTS algorithm.

        Parameters
        ----------
        self : object
            The instance of the class.
        C : float
            The exploration parameter to be set.
        """
        self.args['C'] = C

    @torch.no_grad()
    def search(self, states, p_memory):
        """
        Performs the parallel MCTS search from the given states.

        Parameters
        ----------
        states : list[tuple]
            The list of states to search from.
        p_memory : list[Pmemory]
            The list of PMemory objects for parallel search.
        """

        # batch data
        data_list = [
            Data(x=self.mdp.encode_state(s), edge_index=self.mdp.data.edge_index)\
            for s in states
        ]
        batch = Batch.from_data_list(data_list).to(self.mdp.device)
        _, policy = self.model.forward(batch.x, batch.edge_index, batch.batch)
        policy = policy.view(batch.batch_size, -1)
        del batch
        del data_list
        policy = torch.softmax(policy, dim=1).detach().cpu().numpy()

        # dirichlet random noise
        policy = (1-self.args['eps']) * policy \
            + self.args['eps'] \
            * np.random.dirichlet(
                    [self.args['dirichlet_alpha']]*len(self.mdp.edges),
                    size=policy.shape[0]
            )

        for i, mem in enumerate(p_memory):
            p_policy = torch.tensor(policy[i])
            p_policy = self.mdp.mask_policy(p_policy, states[i])

            mem.root = Node(self.mdp, self.args, states[i], visit_count=1)
            mem.root.expand(p_policy, self.mdp)

        for _ in range(self.args['num_searches']-len(states[0])+1):
            for mem in p_memory:
                mem.node = None
                node = mem.root
                while node.is_fully_expanded():
                    node = node.select()

                value, is_terminal = self.mdp.get_value_and_terminated(node.state)

                if is_terminal:
                    node.backpropagete(value)
                else:
                    mem.node = node

            expandable = [i for i in range(len(p_memory)) if p_memory[i].node != None]

            if len(expandable) > 0:
                data_list = [
                    Data(
                        x=self.mdp.encode_state(p_memory[i].node.state),
                        edge_index=self.mdp.data.edge_index
                    )\
                    for i in expandable
                ]
                batch = Batch.from_data_list(data_list).to(self.mdp.device)
                value, policy = self.model.forward(
                    batch.x,
                    batch.edge_index,
                    batch.batch
                )
                policy = policy.view(batch.batch_size, -1).detach().cpu()
                policy = torch.softmax(policy, dim=1)
                del data_list
                del batch

            for i, idx in enumerate(expandable):
                node = p_memory[idx].node
                p_value, p_policy = value[i], policy[i]
                p_policy = self.mdp.mask_policy(p_policy, node.state)

                node.expand(p_policy, self.mdp)
                node.backpropagete(p_value)


class PMemory:
    """
    A class representing the memory used in parallel
    MCTS during parallel search.

    Attributes
    ----------
    state : list[tuple]
        The state of the mdp.
    current_block : int
        The current block in the mdp.
    memory : list
        The memory of the search.
    root : Node
        The root node of the search tree.
    node : Node
        The current node in the search tree.
    """
    def __init__(self, mdp, at_block):
        node = int(np.random.choice(mdp.nodes))
        self.state = [(node, node, 0)]
        self.current_block = at_block
        self.memory = []
        self.root = None
        self.node = None
