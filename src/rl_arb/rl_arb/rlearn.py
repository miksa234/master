#!/usr/bin/env python3

import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.data import Batch, Data
import torch.multiprocessing as mp
from tqdm.contrib.telegram import tqdm
import logging
logger = logging.getLogger('rl_circuit')

from rl_arb.mdp import MDP
from rl_arb.mcts import MCTSParallel, PMemory
from rl_arb.config import (
    DEVICE,
    TELEGRAM_TOKEN,
    TELEGRAM_CHAT_ID,
)
from rl_arb.utils import (
        update_me,
        send_telegram_message,
        save_loss
)


class AgentRLearn():
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
            Trains the model using the training data from self_play.
        learn()
            Main loop for the learning process, including self_play and training.
        """
        self.model = model
        self.optimizer = optimizer
        self.mdp = mdp
        self.args = args
        self.mcts = MCTSParallel(self.mdp, self.args, self.model)

        self.loss = []
        self.avg_state_len = []
        self.pbar_play = False


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


        if self.pbar_play:
            pbar = tqdm(
                total=self.args['num_parallel'], desc='self_play',
                token=TELEGRAM_TOKEN, chat_id=TELEGRAM_CHAT_ID,
                disable=not self.args['telegram']
            )

        while len(p_memory) > 0:

            states = [mem.state for mem in p_memory]
            if self.pbar_play:
                pbar.set_description(f"self_play state_len {len(states[0])}")

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
                    p=temp_probs
                )
                action = self.mdp.edge_list[action_idx]

                mem.state = self.mdp.get_next_state(mem.state, action)
                value, is_terminal = self.mdp.get_value_and_terminated(mem.state)

                if is_terminal:
                    if self.pbar_play:
                        pbar.update(1)
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
        epoch_policy_loss = []
        epoch_value_loss = []
        np.random.shuffle(memory)
        for batch_idx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batch_idx:np.min([len(memory) - 1, batch_idx + self.args['batch_size']])]
            try:
                states , policy_targets, value_targets, block_indices = zip(*sample)
            except ValueError: # batch is len 0.
                print(f"batch_idx {batch_idx}")
                print(f"From-TO : {[batch_idx, np.min([len(memory) - 1, batch_idx + self.args['batch_size']])]}")
                print(f"len memory {len(memory)}")
                print(f"len sample {len(sample)}")
                continue

            policy_targets, value_targets = np.array(policy_targets), np.array(value_targets)
            policy_targets = torch.tensor(policy_targets).to(DEVICE).float()
            value_targets = torch.tensor(value_targets).to(DEVICE).float().unsqueeze(1)


            data_list = [
                Data(
                    x=self.mdp.encode_state(s),
                    edge_index=self.mdp.data.edge_index
                 )\
                for s in states
            ]
            batch = Batch.from_data_list(data_list)
            value_outs, policy_outs = self.model.forward(
                batch.x,
                batch.edge_index,
                batch.batch
            )
            policy_outs = policy_outs.view(batch.batch_size, -1)
            del batch
            del data_list

            policy_loss = F.cross_entropy(policy_outs, policy_targets)
            value_loss = F.mse_loss(value_outs, value_targets)
            loss = policy_loss + value_loss

            self.loss.append([policy_loss.item(), value_loss.item()])
            epoch_policy_loss.append(policy_loss.item())
            epoch_value_loss.append(value_loss.item())
            self.avg_state_len.append(
                np.mean(np.array([len(s) for s in states]))
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.args['telegram']:
            update_me(
                np.mean(epoch_policy_loss), np.mean(epoch_value_loss),
                self.avg_state_len[-1],
                epoch_iter, iteration
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
                self.model.to(DEVICE)
                self.mdp.data.to(DEVICE)
                self.mdp.device = DEVICE
                for play_iter in tqdm(
                    range(self.args['num_self_play_iterations']//self.args['num_parallel']),
                    desc="self play",
                    token=TELEGRAM_TOKEN,
                    chat_id=TELEGRAM_CHAT_ID,
                    disable=not self.args['telegram']
                ):
                    memory += self.self_play()

            else:
                cpu = torch.device('cpu')
                self.model.to(cpu)
                self.mdp.data.to(cpu)
                self.mdp.device = cpu
                # multicore self-play
                play_iter = self.args['num_self_play_iterations']
                num_parallel = self.args['num_parallel']
                num_processes = self.args['num_processes']
                per_processor = play_iter//num_parallel//num_processes
                # at each self play choose prices from a different block number
                with mp.Pool(processes=num_processes) as pool:
                    results = pool.starmap(
                        self_play_num_times,
                        [(self, per_processor, 'cuda:0', True)] +\
                        [(self, per_processor, 'cuda:0', False) for _ in range(num_processes//2 - 1)] +\
                        [(self, per_processor, 'cuda:1', False) for _ in range(num_processes//2 - 1)] +\
                        [(self, per_processor, 'cuda:1', True)]
                    )
                    pool.terminate()

                self.model.to(DEVICE)
                self.mdp.data.to(DEVICE)
                self.mdp.device = DEVICE

                for result in results:
                    memory += result

            self.model.train()
            for epoch_iter in tqdm(
                range(self.args['num_epochs']),
                desc="epochs",
                token=TELEGRAM_TOKEN,
                chat_id=TELEGRAM_CHAT_ID,
                disable=not self.args['telegram']
            ):
                self.train(memory, iteration, epoch_iter)

            torch.save(self.model.state_dict(), f"./model/model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"./model/optimizer_{iteration}.pt")

        save_loss(self.loss, self.avg_state_len)
        if self.args['telegram']:
            send_telegram_message("DONE!")



def self_play_num_times(rlearn, times=100, device='cpu', pbar=False):
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

    rlearn.pbar_play = pbar
    rlearn.model.to(device)
    rlearn.mdp.data.to(device)
    rlearn.mdp.device = device
    for _ in range(times):
        memory += rlearn.self_play()
    return memory
