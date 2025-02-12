#!/usr/bin/env python3

import numpy as np
import torch
from torch.nn import functional as F
import torch.multiprocessing as mp
from tqdm.contrib.telegram import tqdm
import logging
logger = logging.getLogger('rl_circuit')

from rl_arb.mcts import *
from rl_arb.mdp import *
from rl_arb.config import *
from rl_arb.utils import *


class AgentRLearn:
    def __init__(self, model, optimizer, mdp: MDP, args):
        """
        RL algorithm class for training and self-play using Monte Carlo Tree
        Search (MCTS).

        Parameters
        ----------
        model : torch.nn.Module
            The neural network model used for policy and value prediction.
        optimizer : torch.optim.Optimizer
            The optimizer used for training the model.
        mdp : MDP
            The MDP environment that provides the logic and state transitions.
        args : dict
            A dictionary of arguments and hyperparameters for training and self-play.

        Methods
        -------
        self_play()
            Executes self-play to generate training data.
        train(memory)
            Trains the model using the provided memory of mdp states and outcomes.
        learn()
            Main loop for the learning process, including self-play and training.
        """
        self.model = model
        self.optimizer = optimizer
        self.mdp = mdp
        self.args = args
        self.mcts = MCTSParallel(self.mdp, self.args, self.model)


    def self_play(self):
        """
        Executes self-play to generate training data.

        Returns
        -------
        list
            A list of tuples containing mdp states, policy probabilities,
            value estimates, and block indices.
        """
        return_mem = []
        at_block = np.random.choice(self.mdp.num_blocks)
        self.mdp.current_block = at_block
        p_memory = [PMemory(self.mdp, at_block) for _ in range(self.args['num_parallel'])]

        total = len(p_memory)
        pbar = tqdm(
            total=total, desc='terminal states',
            token=TELEGRAM_TOKEN, chat_id=TELEGRAM_CHAT_ID,
            disable=self.args['telegram']
        )
        while len(p_memory) > 0:
            if len(p_memory) < total:
                pbar.update(1)
                total = len(p_memory)

            states = [mem.state for mem in p_memory]
            self.mcts.search(states, p_memory)

            for i in range(len(p_memory))[::-1]:
                mem = p_memory[i]

                probs = np.zeros(len(self.mdp.edge_list))
                for child in mem.root.children:
                    probs[
                        self.mdp.edge_list.index(child.action_taken)
                    ] = child.visit_count
                probs /= np.sum(probs)

                mem.memory.append((
                    mem.root.state,
                    probs
                ))

                temp_probs = probs ** (1 / self.args['temperature'])
                temp_probs /= temp_probs.sum()

                action_idx = np.random.choice(
                    list(range(len(self.mdp.edge_list))),
                    p=probs
                )
                action = self.mdp.edge_list[action_idx]

                mem.state = self.mdp.get_next_state(mem.state, action)
                value, is_terminal = self.mdp.get_value_and_terminated(mem.state)

                if is_terminal:
                    for hist_state, hist_probs in mem.memory:
                        return_mem.append((
                            hist_state,
                            hist_probs,
                            value,
                            at_block,
                        ))
                    del p_memory[i]
        return return_mem


    def train(self, memory, iteration, epoch_iter):
        """
        Trains the model using the provided memory of mdp states and outcomes.

        Parameters
        ----------
        memory : list
            A list of tuples containing mdp states, policy probabilities,
            value estimates, and block indices.
        """
        np.random.shuffle(memory)
        for batch_idx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batch_idx:np.min([len(memory) - 1, batch_idx + self.args['batch_size']])]
            try:
                states , policy_targets, value_targets, block_indices = zip(*sample)
            except ValueError:
                print(f"batch_idx {batch_idx}")
                print(f"From-TO : {[batch_idx, np.min([len(memory) - 1, batch_idx + self.args['batch_size']])]}")
                print(f"len memory {len(memory)}")
                print(f"len sample {len(sample)}")
                continue

            policy_targets, value_targets = np.array(policy_targets), np.array(value_targets)
            policy_targets, value_targets = torch.tensor(policy_targets).to(DEVICE).float(), torch.tensor(value_targets).to(DEVICE).float()


            value_outs, policy_outs = [], []
            for i, s in enumerate(states):
                self.mdp.current_block = block_indices[i]
                v, p = self.model(
                    self.mdp.encode_state(s),
                    self.mdp.data.edge_index
                )
                value_outs += v
                policy_outs += p

            value_outs = torch.stack(value_outs).to(DEVICE)
            policy_outs = torch.stack(policy_outs).to(DEVICE)

            policy_loss = F.cross_entropy(policy_outs, policy_targets)
            value_loss = F.mse_loss(value_outs, value_targets)
            loss = policy_loss + value_loss


            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if epoch_iter == self.args['num_epochs']-1:
            save_loss_and_states_and_update_me(
                policy_loss, value_loss, states,
                iteration, self.args['telegram']
            )

    def adapt_exploration_parameter(self, iteration):
        """
        Adapts the exploration parameter of MCTS based on the current iteration.

        Parameters
        ----------
        iteration : int
            The current iteration number.
        """
        if iteration <= self.args['num_iterations']:
            self.mcts.set_C(self.args['C_3/3'])

        if iteration <= self.args['num_iterations']*2//3:
            self.mcts.set_C(self.args['C_2/3'])

        if iteration <= self.args['num_iterations']//3:
            self.mcts.set_C(self.args['C_1/3'])


    def learn(self):
        """
        Main loop for the learning process, including self-play and training.
        """
        for iteration in range(self.args['num_iterations']):
            logger.info(f"Iterations: {iteration+1}/{self.args['num_iterations']}")
            if self.args['telegram']:
                send_telegram_message(f"Iterations: {iteration+1}/{self.args['num_iterations']}")

            self.adapt_exploration_parameter(iteration)

            memory = []
            self.model.eval()
            if not self.args['multicore']:
                # single core self-play
                for play_iter in tqdm(
                    range(self.args['num_self_play_iterations']//self.args['num_parallel']),
                    desc="self play",
                    token=TELEGRAM_TOKEN,
                    chat_id=TELEGRAM_CHAT_ID,
                    disable=self.args['telegram']
                ):
                    memory += self.self_play()

            else:
                # multicore self-play
                play_iter = self.args['num_self_play_iterations']
                num_parallel = self.args['num_parallel']
                num_processes = self.args['num_processes']
                per_processor = play_iter//num_parallel//num_processes
                # at each self play choose prices from a different block number
                with mp.Pool(processes=num_processes) as pool:
                    results = pool.starmap(
                        self_play_num_times,
                        [(self, per_processor) for _ in range(num_processes)]
                    )
                    pool.terminate()

                for result in results:
                    memory += result

            self.model.train()
            for epoch_iter in tqdm(
                range(self.args['num_epochs']),
                desc="epochs",
                token=TELEGRAM_TOKEN,
                chat_id=TELEGRAM_CHAT_ID,
                disable=self.args['telegram']
            ):
                self.train(memory, iteration, epoch_iter)

            torch.save(self.model.state_dict(), f"./model/model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"./model/optimizer_{iteration}.pt")



def self_play_num_times(rlearn, times=100):
    """
    Helper function to perform self-play multiple times.

    Parameters
    ----------
    rlearn: AgentRLearn
        The AgentRLearn instance to use for self-play.
    times : int, optional
        The number of self-play iterations to perform (default is 100).

    Returns
    -------
    list
        A list of tuples containing mdp states, policy probabilities, value estimates, and block indices.
    """
    memory = []
    for _ in range(times):
        memory += rlearn.self_play()
    return memory
