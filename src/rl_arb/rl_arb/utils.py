#!/usr/bin/env python3

import pandas as pd
import networkx as nx
import numpy as np
import torch
import os
import pickle
import requests

from rl_arb.config import TELEGRAM_CHAT_ID, TELEGRAM_SEND_URL

def load_pools_and_tokens(path_pools, path_tokens):
    """
    Load pool & token data into pd.DataFrame/s from csv files.

    Parameters
    ----------
    path_pools : str
        File path to csv file for the pools
    path_tokens : str
        File path to csv file for the tokens

    Returns
    -------
    tuple
        A tuple containing:
        - pools (pd.DataFrame): pool data.
        - tokens (pd.DataFrame): token data.
    """
    pools = pd.read_csv(
        path_pools,
        names = ['index', 'address', 'version', 'token0', 'token1', 'fee', 'block_number', 'time_stamp', 'tick_spacing'],
        header = None,
    ).sort_index().drop_duplicates()
    tokens = pd.read_csv(
        path_tokens,
        header = None,
        names = ['index', 'address', 'name', 'symbol', 'decimals']
    ).sort_index().drop_duplicates()


    return pools, tokens


def make_price(price):
    """
    Calculate the price from price pd.DataFrame
    Uniswapv3 price is t1/t0 -> sqrt_price_x96 = sqrt(reserve1/reserve0) * 2**96
    Uniswapv2 price we will define also as t1/t0

    Parameters
    ----------
    price : pd.DataFrame

    Returns
    -------
    list: prices.
    """
    block_price = []
    for _, p in price.iterrows():
        if (spx96 := p['sqrt_price_x96']) != None:
            block_price.append((int(spx96) / 2**96)**2)
        else:
            t0 = int(p['reserve_t0'])
            t1 = int(p['reserve_t1'])
            block_price.append(t1/t0)
    return block_price


def pools_to_edge_list(pools, prices):
    """
    Makes an edge list from pools & prices for the problem graph.

    Parameters
    ----------
    pools : pd.DataFrame
        pool data
    prices : pd.DataFrame
        price data

    Returns
    -------
    list
        List of edges in the form of a tuple (token0, token1, attributes).
        Where the attributes is a type dict with keys:
        - 'k' (int): Repeated count of the pool with the same tokens.
        - 'weight' (list): The historical prices of the specific pool.
    """
    edge_list = []
    cache = []
    for (_, pool) in pools.iterrows():

        t0 = pool['token0']
        t1 = pool['token1']
        p = make_price(prices[prices['pool_address'] == pool['address']])


        k = 0
        for e in cache:
            if (t0, t1) == e or (t1, t0) == e:
                k += 1
        edge_list.append(
            (t0, t1,
             {'k': k, 'weight': p, 'address': pool['address'], 'fee': int(pool['fee'])/1e6})
        )
        cache.append((t0, t1))

    return edge_list


def make_token_graph(pools, prices):
    """
    Make a directed multi graph from pool and price data.

    Parameters
    ----------
    pools : pd.DataFrame
        Pool data.
    prices : pd.DataFrame
        Price data.

    Returns
    -------
    nx.MultiDiGraph
        A directed multi graph. Nodes represent tokens
        edges represent pools with attributes generated
        by the function pools_to_edge_list
    """
    edge_list = pools_to_edge_list(pools, prices)

    G = nx.MultiDiGraph()
    G.add_edges_from(edge_list)
    return G

def linear_node_relabel(G):
    """
    Relabel the nodes linearly. Input node labels
    are ETH addresses which are relabeled in a chronological order
    to integer values starting from 0.

    Parameters
    ----------
    G : nx.MultiDiGraph
        Input graph.

    Returns
    -------
    tuple
        A tuple containing:
        - G (nx.MultiDiGraph): Graph with relabeled nodes (automatically edges).
        - mapping (dict): Mapping dictionary.
    """
    mapping = {}
    for i, node in enumerate(list(G.nodes())):
        mapping[node] = i
    G = nx.relabel_nodes(G, mapping)
    return G, mapping


def filter_pools_with_no_gradient(pools, prices):
    """
    Filter out pools that have no change in price by computing the gradient of
    the historical prices.

    Parameters
    ----------
    pools : pd.DataFrame
        Pool data.
    prices : pd.DataFrame
        Price data

    Returns
    -------
    tuple
        A tuple containing:
        - filtered_pools (pd.DataFrame): Filtered pool data.
        - filtered_prices (pd.DataFrame): Filtered price data.
    """
    pools = pools[pools['address'].isin(set(prices['pool_address']))]
    ticks = len(prices['block_number'].unique())
    mask = []
    for _, pool in pools.iterrows():
        t0 = pool['token0']
        t1 = pool['token1']
        p = make_price(prices[prices['pool_address'] == pool['address']])
        mask.append(np.count_nonzero(np.gradient(p)) > ticks*2//3 )

    pools = pools[mask]
    prices = prices[prices['pool_address'].isin(list(pools['address']))]
    return pools, prices


def save_loss_and_states_and_update_me(
    policy_loss,
    value_loss,
    states,
    iteration,
    telegram
):
    """
    Saves policy, value losses and all the states run through the in the batch
    at a given epoch. Also sends a message through a telegram bot on current
    training state.

    Parameters
    ----------
    policy_loss: torch.float
        Policy loss.
    value_loss: torch.float
        Value loss.
    states: list[list[tuple]].
        List of states.
    iteration: int
        Current iteration number.
    telegram: bool
        Send message via telegram or not.
    """

    if not os.path.exists('./model'):
        os.mkdir('./model')
    if not os.path.exists('./model/states'):
        os.mkdir('./model/states')
    if not os.path.exists('./model/loss'):
        os.mkdir('./model/loss')

    torch.save(
        policy_loss,
        f'./model/loss/policy_loss_{iteration}.pt'
    )
    torch.save(
        policy_loss,
        f'./model/loss/value_loss_{iteration}.pt'
    )
    with open(f'./model/states/state_{iteration}.pickle', "wb") as f:
        pickle.dump(states, f)

    avg_state_len = np.mean(np.array([len(s) for s in states]))

    if telegram:
        message = f"""
            ITR: {iteration} | EPOCH: {epoch_iter} | BATCH_IDX: {batch_idx}
            Policy loss: {policy_loss}
            Value loss: {value_loss}
            Total loss: {policy_loss + value_loss}
            Average state length: {avg_state_len}
        """
        send_telegram_message(message)


def send_telegram_message(message):
    """
    Send a message through a telegram-bot.

    Parameters
    ----------
    message: str
        String containing the message to be sent by the bot.
    """
    requests.post(TELEGRAM_SEND_URL, json={'chat_id': TELEGRAM_CHAT_ID, 'text': message})
