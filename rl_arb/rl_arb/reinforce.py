#!/usr/bin/env python3

import torch
import numpy as np
import pickle
from torch_geometric.data import Data, Batch

from rl_arb.config import DEVICE

import logging
logger = logging.getLogger('rl_circuit')

class Reinforce:
    def __init__(self, model, optimizer, mdp, mcts, args):
        self.model = model
        self.mdp = mdp
        self.mcts = mcts
        self.optimizer = optimizer
        self.args = args

    @torch.no_grad()
    def play(self, state, at_block, mode="learn"):
        mcts_probs = []
        actions, rewards, transitions, g_values =[], [], [], []
        suc_states = 0
        while True:
            probs = self.model(
                self.mdp.encode_state(state, at_block),
                self.mdp.data.edge_index,
                self.mdp.encode_state(state[:1], at_block)
            )

            probs = probs.squeeze(0).detach().cpu()

            probs = self.mdp.mask_policy(probs, state).numpy()

            if mode == "test":
                action_idx = np.argmax(probs)
            else:
                action_idx = np.random.choice(list(
                    range(len(self.mdp.edge_list))
                ), p=probs/probs.sum())

            action = self.mdp.edge_list[action_idx]

            actions.append(action)
            mcts_probs.append(probs)

            state = self.mdp.get_next_state(state, action)

            value, is_terminal = self.mdp.get_value_and_terminated(
                state,
                at_block
            )

            if is_terminal:
                rewards.append(value)

                if value != -1:
                    suc_states +=1

                for i in range(1, len(state)):
                    transitions.append((at_block, state[:i]))
                    g_values.append(
                        self.args['gamma']**(
                            len(state)-len(state[:i])
                        )*value
                    )
                break
        return mcts_probs, actions, rewards, transitions, g_values, suc_states

    def learn(self):
        self.model.to(DEVICE)

        mcts_probs = []
        rewards = []
        g_values = []
        actions = []
        transitions = []
        suc_states = 0
        max_reward = 0
        mean_rewards = 0

        suc_states_hist = []
        mean_rewards_hist = []
        test_rewards_hist = []

        for itr in range(self.args['num_reinforce']):
            at_block = 10
            self.mdp.current_block = at_block
            self.mcts.mdp.current_block = at_block

            state = [(self.mdp.start_node, self.mdp.start_node, 0)]
            m, a, r, t, g, suc = self.play(state, at_block)
            suc_states += suc
            transitions += t
            g_values += g
            rewards += r
            actions += a
            mcts_probs += m

            if itr%300==0 and itr > 0:
                state = [(self.mdp.start_node, self.mdp.start_node, 0)]
                _, _, tr, _, _, _ = self.play(state, at_block, mode="test")
                logger.info(f"mean rewards: {mean_rewards}")
                logger.info(f"max rewards: {max_reward}")
                logger.info(f"test rewards: {tr}")
                logger.info(f"suc_states : {suc_states}")

                mean_rewards = np.mean(rewards)

                if max_reward < np.max(rewards):
                    max_reward = np.max(rewards)

                suc_states_hist.append(suc_states)
                mean_rewards_hist.append(mean_rewards)
                test_rewards_hist.append(tr)

                rewards = []
                suc_states = 0


                data_list = [
                    Data(
                        x=self.mdp.encode_state(s, b),
                        edge_index=self.mdp.data.edge_index,
                        y=self.mdp.encode_state(s[:1], b),
                     )\
                    for b, s in transitions
                ]
                batch = Batch.from_data_list(data_list).to(DEVICE)

                policy_outs = self.model.forward(
                    batch.x,
                    batch.edge_index,
                    batch.y,
                    batch.batch,
                )

                del batch
                del data_list

                one_hot = torch.zeros_like(policy_outs).to(DEVICE)
                for a, p in zip(actions, one_hot):
                    p[self.mdp.edge_list.index(a)] = 1

                g_tensor = torch.tensor(g_values).to(DEVICE)

                ce = torch.mean(torch.log(policy_outs) * one_hot, dim=1)
                loss = -torch.mean(ce * (g_tensor-mean_rewards))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                message = f"""
                        ITR: {itr}
                        Mean CE Loss: {torch.mean(ce)}
                        REINFORCE Loss: {loss}
                """
                logger.info(message)

                mcts_probs = []
                g_values = []
                actions = []
                transitions = []

                with open(f"./reinforce.pickle", "wb") as f:
                    pickle.dump(
                        [
                            suc_states_hist,
                            mean_rewards_hist,
                            test_rewards_hist
                        ],
                        f
                    )
