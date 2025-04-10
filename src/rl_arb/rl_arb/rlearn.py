#!/usr/bin/env python3

import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.data import Batch, Data
import torch.multiprocessing as mp
from tqdm.contrib.telegram import tqdm
import logging
logger = logging.getLogger('rl_circuit')
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import distributed as dist
import os
import pickle

from rl_arb.net import PolicyNet
from rl_arb.mcts import MCTSParallel, PMemory
from rl_arb.config import (
    DEVICE,
    TELEGRAM_TOKEN,
    TELEGRAM_CHAT_ID,
    ARGS_MODEL,
)
from rl_arb.utils import (
    update_me,
    send_telegram_message,
)


class AgentRLearn():
    def __init__(self, model, mdp, args):
        """
        RL algorithm class for training and self-play using Monte Carlo Tree
        Search (MCTS).

        Parameters
        ----------
        model : torch.nn.Module
            The neural network model used for policy and value prediction.
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
        track_baseline(values, gamma_factors, blocks)
            Updates the baseline_tracker and give baseline values for current iteration.
        """
        self.model = model
        self.args = args
        self.mcts = MCTSParallel(mdp, self.args)

        self.pbar_play = False
        self.values = []
        self.baseline_tracker = {}
        self.avg_itr_value = [0.0]
        self.test_values= []


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
        at_block = np.random.choice(self.mcts.mdp.num_blocks)
        self.mcts.mdp.current_block = at_block
        p_memory = [PMemory(self.mcts.mdp, np.random.choice(self.mcts.mdp.num_blocks)) for _ in range(self.args['num_parallel'])]

        if self.pbar_play:
            pbar = tqdm(
                total=self.args['num_parallel'],
                desc='self_play',
                token=str(TELEGRAM_TOKEN),
                chat_id=str(TELEGRAM_CHAT_ID),
                disable=not self.args['telegram']
            )

        while len(p_memory) > 0:

            states = [mem.state for mem in p_memory]
            if self.pbar_play:
                pbar.set_description(f"self_play state_len {len(states[0])}")

            self.mcts.search(self.model, states, p_memory)

            for i in range(len(p_memory))[::-1]:
                mem = p_memory[i]
                probs = np.zeros(len(self.mcts.mdp.edge_list))
                for child in mem.root.children:
                    probs[
                        self.mcts.mdp.edge_list.index(child.action_taken)
                    ] = child.q_value

                if np.sum(probs) == 0:
                    for child in mem.root.children:
                        probs[
                            self.mcts.mdp.edge_list.index(child.action_taken)
                        ] = child.visit_count

                probs /= np.sum(probs)

                mem.memory.append((
                    mem.state.copy(),
                    probs,
                    mem.current_block
                ))

                action_idx = np.random.choice(
                    list(range(len(self.mcts.mdp.edge_list))),
                    p=probs
                )
                action = self.mcts.mdp.edge_list[action_idx]
                mem.state = self.mcts.mdp.get_next_state(mem.state, action)

                value, is_terminal = self.mcts.mdp.get_value_and_terminated(mem.state, mem.current_block)

                if is_terminal:
                    if self.pbar_play:
                        pbar.update(1)
                    for hist_state, hist_probs, current_block in mem.memory:
                        return_mem.append((
                            hist_state,
                            hist_probs,
                            value,
                            self.args['gamma']**(len(mem.state)-(len(hist_state)+1)),
                            action,
                            current_block,
                        ))
                    del p_memory[i]

        return return_mem

    def track_baseline(self, values, gamma_factors, blocks):
        bs = np.array(blocks)
        vs = np.array(values)
        baseline = np.zeros_like(bs, dtype=float)

        for block in np.unique(bs):
            idxs = np.where(bs==block)[0]
            max_val = np.max(vs[idxs])
            if block not in self.baseline_tracker or max_val > self.baseline_tracker[block]:
                self.baseline_tracker[block] = max_val

            baseline[idxs] = self.baseline_tracker[block]

        return baseline*np.array(gamma_factors)


    def learn(self, optimizer):
        """
        Main loop for the learning process, including self-play and training.
        """
        torch.save(self.model.state_dict(), "./model/model_0.pt")
        torch.save(optimizer.state_dict(), "./model/optimizer_0.pt")
        for iteration in range(self.args['num_iterations']):
            logger.info(f"Iterations: {iteration+1}/{self.args['num_iterations']}")
            if self.args['telegram']:
                send_telegram_message(f"Iterations: {iteration+1}/{self.args['num_iterations']}")

            memory = []
            self.model.eval()
            if not self.args['multicore']:
                # single core self-play
                self.model.to(DEVICE)
                self.mcts.mdp.device = DEVICE
                for play_iter in tqdm(
                    range(self.args['num_self_play_iterations']//self.args['num_parallel']),
                    desc="self play",
                    token=str(TELEGRAM_TOKEN),
                    chat_id=str(TELEGRAM_CHAT_ID),
                    disable=not self.args['telegram'],
                ):
                    memory += self.self_play()

            else:
                cpu = torch.device('cpu')
                self.model.to(cpu)
                self.model.load_state_dict(
                    torch.load(
                        f"./model/model_{iteration}.pt",
                        weights_only = True,
                        map_location=cpu
                    )
                )
                self.model.share_memory()
                self.mcts.mdp.device = cpu
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

                for result in results:
                    memory += result

            states, _, values, gamma_factors, _, blocks = zip(*memory)
            baseline = self.track_baseline(values, gamma_factors, blocks)

            memory = [(*it, float(baseline[i])) for i, it in enumerate(memory)]

            terminal_idxs = np.where(np.array(gamma_factors)==1)[0]
            terminal_values = np.array(values)[terminal_idxs]
            terminal_states = np.array([len(s)+1 for s in states])[terminal_idxs]

            self.values.append((np.mean(terminal_values), np.mean(terminal_states)))
            if self.args['telegram']:
                send_telegram_message(f"""
                Average values {np.mean(terminal_values)}
                Average state_len {np.mean(terminal_states)}
                """)

            world_size = torch.cuda.device_count()
            mp.spawn(
                train,
                args=(world_size, memory, self.mcts, iteration),
                nprocs=world_size,
                join=True
            )

            with open("values.pickle", "wb") as f:
                pickle.dump(self.values, f)

            test_values = self.test_model(iteration)
            average_value = float(np.mean(test_values))
            with open("test_values.pickle", "wb") as f:
                pickle.dump(test_values, f)

            logger.info(values)
            send_telegram_message(f"Average profit {average_value}")

            #if self.avg_itr_value[-1] > average_value:
            #    self.override_update(iteration)
            #else:
            #    self.avg_itr_value.append(average_value)

        if self.args['telegram']:
            send_telegram_message("DONE!")


    def test_model(self, iteration):
        cpu = torch.device('cpu')
        self.model.to(cpu)
        self.model.load_state_dict(
            torch.load(
                f"./model/model_{iteration+1}.pt",
                weights_only = True,
                map_location=cpu
            )
        )
        self.model.share_memory()
        self.mcts.mdp.device = cpu
        # multicore self-play
        play_iter = self.mcts.mdp.num_blocks-1
        num_parallel = self.args['num_parallel']
        num_processes = play_iter//num_parallel
        per_processor = play_iter//num_parallel//num_processes

        block_ranges = [
            list(range(num_parallel)),
            list(range(num_parallel, (num_parallel*num_processes)//2 -1)),
            list(range((num_parallel*num_processes)//2, play_iter-num_parallel)),
            list(range(play_iter-num_parallel, play_iter)),
        ]
        logger.info(f"{block_ranges}")

        with mp.Pool(processes=num_processes) as pool:
            results = pool.starmap(
                test_blocks,
                [(self, block_ranges[0], per_processor, 'cuda:0', True)] +\
                [(self, block_ranges[1], per_processor, 'cuda:0', False) for _ in range(num_processes//2 - 1)] +\
                [(self, block_ranges[2], per_processor, 'cuda:1', False) for _ in range(num_processes//2 - 1)] +\
                [(self, block_ranges[3], per_processor, 'cuda:1', True)]
            )
            pool.terminate()

        values =[]
        for result in results:
            values += result

        return values

    def test_play(self, block_range):
        """
        Executes self-play to generate training data.

        Returns
        -------
        list
            A list of tuples containing mdp states, policy probabilities,
            value estimates, and block indices.
        """
        return_mem = []
        p_memory = [PMemory(self.mcts.mdp, block) for block in block_range]

        if self.pbar_play:
            pbar = tqdm(
                total=self.args['num_parallel'],
                desc='test_play',
                token=str(TELEGRAM_TOKEN),
                chat_id=str(TELEGRAM_CHAT_ID),
                disable=not self.args['telegram']
            )

        while len(p_memory) > 0:
            states = [mem.state for mem in p_memory]

            if self.pbar_play:
                pbar.set_description(f"test_play state_len {len(states[0])}")

            self.mcts.search(self.model, states, p_memory)
            for i in range(len(p_memory))[::-1]:
                mem = p_memory[i]
                probs = np.zeros(len(self.mcts.mdp.edge_list))
                for child in mem.root.children:
                    probs[
                        self.mcts.mdp.edge_list.index(child.action_taken)
                    ] = child.q_value

                if np.sum(probs) == 0:
                    for child in mem.root.children:
                        probs[
                            self.mcts.mdp.edge_list.index(child.action_taken)
                        ] = child.visit_count

                probs /= np.sum(probs)

                action_idx = np.random.choice(
                    list(range(len(self.mcts.mdp.edge_list))),
                    p=probs
                )

                action = self.mcts.mdp.edge_list[action_idx]
                mem.state = self.mcts.mdp.get_next_state(mem.state, action)

                value, is_terminal = self.mcts.mdp.get_value_and_terminated(mem.state, mem.current_block)

                if is_terminal:
                    if self.pbar_play:
                        pbar.update(1)
                    return_mem.append(value)
                    del p_memory[i]

        return return_mem

    def override_update(self, iteration):

        self.model.load_state_dict(
            torch.load(
                f"./model/model_{iteration}.pt",
                weights_only = True,
            )
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        optimizer.load_state_dict(
            torch.load(
                f"./model/optimizer_{iteration}.pt",
                weights_only = True,
            )
        )

        torch.save(self.model.state_dict(), f"./model/model_{iteration+1}.pt")
        torch.save(optimizer.state_dict(), f"./model/optimizer_{iteration+1}.pt")


def test_blocks(rlearn, block_range, times=100, device='cpu', pbar=False):
    """
    Helper function to perform self-play multiple times.

    Parameters
    ----------
    rlearn: AgentRLearn
        The AgentRLearn instance to use for self-play.
    times : int, optional
        The number of self-play iterations to perform (default is 100).
    device: string, optional
        Device to execute the code.
    pbar: bool, optional
        Print a tqdm bar or not.

    Returns
    -------
    list
        A list of tuples containing mdp states, policy probabilities, value estimates, and block indices.
    """
    profits = []

    rlearn.pbar_play = pbar
    rlearn.model.to(device)
    rlearn.mcts.mdp.device = device
    for _ in range(times):
        profits += rlearn.test_play(block_range)
    return profits


def self_play_num_times(rlearn, times=100, device='cpu', pbar=False):
    """
    Helper function to perform self-play multiple times.

    Parameters
    ----------
    rlearn: AgentRLearn
        The AgentRLearn instance to use for self-play.
    times : int, optional
        The number of self-play iterations to perform (default is 100).
    device: string, optional
        Device to execute the code.
    pbar: bool, optional
        Print a tqdm bar or not.

    Returns
    -------
    list
        A list of tuples containing mdp states, policy probabilities, value estimates, and block indices.
    """
    memory = []

    rlearn.pbar_play = pbar
    rlearn.model.to(device)
    rlearn.mcts.mdp.device = device
    for _ in range(times):
        memory += rlearn.self_play()
    return memory


def train(rank, world_size, memory, mcts, iteration):
    """
    Distributed multi gpu training in pytorch.

    Parameters
    ----------
    world_size: int
        Number of cpus
    memory : list
        A list of containing mdp states, policy probabilities,
        value estimates, and block indices.
    mcts: MCTSParallel
        Monte Carlo Tree search object.
    iteration: int
        Current search iteration
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    dist.barrier()
    model = PolicyNet(ARGS_MODEL)
    model.load_state_dict(
        torch.load(
            f"./model/model_{iteration}.pt",
            weights_only = True,
        )
    )
    model.to(rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.load_state_dict(
        torch.load(
            f"./model/optimizer_{iteration}.pt",
            weights_only = True,
        )
    )

    model = DDP(model, device_ids=[rank])

    lm = len(memory)
    memory = [memory[i: i+lm//world_size] for i in range(0, lm, lm//world_size)][rank]

    epoch_loss = []
    avg_state_len = []
    avg_rewards = []
    for epoch_iter in tqdm(
        range(mcts.args['num_epochs']),
        desc="epochs",
        token=str(TELEGRAM_TOKEN),
        chat_id=str(TELEGRAM_CHAT_ID),
        disable=(not mcts.args['telegram'] and rank != 0),
    ):
        np.random.shuffle(memory)
        for batch_idx in range(0, len(memory), mcts.args['batch_size']):
            sample = memory[batch_idx:np.min([len(memory) - 1, batch_idx + mcts.args['batch_size']])]
            try:
                states , policy_targets, values, gamma_factors, actions, block_indices, baseline = zip(*sample)
            except ValueError: # batch is len 0.
                print(f"batch_idx {batch_idx}")
                print(f"From-TO : {[batch_idx, np.min([len(memory) - 1, batch_idx + mcts.args['batch_size']])]}")
                print(f"len memory {len(memory)}")
                print(f"len sample {len(sample)}")
                continue

            policy_targets, value_targets = np.array(policy_targets), np.array(values)*np.array(gamma_factors)
            policy_targets = torch.tensor(policy_targets).to(rank).float()
            value_targets = torch.tensor(value_targets).to(rank).float()
            baseline = torch.tensor(baseline).to(rank)

            data_list = [
                Data(
                    x=mcts.mdp.encode_state(s, b),
                    edge_index=mcts.mdp.data.edge_index,
                    y=mcts.mdp.encode_state(s[:1], b),
                 )\
                for s, b  in zip(states, block_indices)
            ]
            batch = Batch.from_data_list(data_list).to(rank)

            policy_outs = model.forward(
                batch.x,
                batch.edge_index,
                batch.y,
                batch.batch,
            )

            del batch
            del data_list

            one_hot = torch.zeros_like(policy_outs).to(rank)
            for a, p in zip(actions, one_hot):
                p[mcts.mdp.edge_list.index(a)] = 1

            cross_entropy = torch.mean(policy_outs * one_hot)
            loss = torch.sum(cross_entropy * (value_targets - baseline))

            epoch_loss.append(loss.item())
            avg_rewards.append(value_targets.mean().item())
            avg_state_len.append(
                np.mean(np.array([len(s) for s in states]))
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if mcts.args['telegram'] and rank == 0:
            update_me(
                np.mean(epoch_loss),
                np.mean(avg_state_len),
                np.mean(avg_rewards),
                epoch_iter,
                iteration,
            )

    if rank == 0:
        torch.save(model.module.state_dict(), f"./model/model_{iteration+1}.pt")
        torch.save(optimizer.state_dict(), f"./model/optimizer_{iteration+1}.pt")

    dist.destroy_process_group()
