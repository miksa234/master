#!/usr/bin/env python3

import pandas as pd
import networkx as nx
import numpy as np


def load_pools_and_tokens(path_pools, path_tokens):
    """
    Load pool and token data from CSV files into pandas DataFrames.

    Parameters
    ----------
    path_pools : str
        The path to the CSV file containing pool data.
    path_tokens : str
        The path to the CSV file containing token data.

    Returns
    -------
    tuple
        A tuple containing two pandas DataFrames:
        - pools (pd.DataFrame): DataFrame containing the pool data.
        - tokens (pd.DataFrame): DataFrame containing the token data.
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
    Calculate the price from the given price data.
    Uniswapv3 price is t1/t0 -> sqrt_price_x96 = sqrt(reserve1/reserve0) * 2**96
    Uniswapv2 price we will define also as t1/t0

    Parameters
    ----------
    price : pd.DataFrame
        DataFrame containing price data with columns
        'sqrt_price_x96', 'reserve_t0', and 'reserve_t1'.

    Returns
    -------
    list
        A list of calculated prices.
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
    Convert pool and price data into an edge list.

    Parameters
    ----------
    pools : pd.DataFrame
        DataFrame containing pool data with
        columns 'token0', 'token1', and 'address'.
    prices : pd.DataFrame
        DataFrame containing price data with
        columns 'pool_address' and other price-related columns.

    Returns
    -------
    list
        A list of edges where each edge is represented
        as a tuple (token0, token1, attributes).
        The attributes dictionary contains:
        - 'k' (int): The count of occurrences of the token pair in the cache.
        - 'weight' (float): The price associated with the pool.
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
             {'k': k, 'weight': p, 'address': pool['address'], 'fee': int(pool['fee'])/10e6})
        )
        cache.append((t0, t1))

    return edge_list


def make_token_graph(pools, latest_prices):
    """
    Create a token graph from pool and price data.

    Parameters
    ----------
    pools : pd.DataFrame
        DataFrame containing pool data.
    latest_prices : pd.DataFrame
        DataFrame containing the latest price data.

    Returns
    -------
    networkx.MultiDiGraph
        A directed multigraph where nodes represent
        tokens and edges represent pools with attributes.
    """
    edge_list = pools_to_edge_list(pools, latest_prices)

    G = nx.MultiDiGraph()
    G.add_edges_from(edge_list)
    return G

def linear_node_relabel(G):
    """
    Relabel the nodes of the graph with consecutive integers.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        The input graph with nodes to be relabeled.

    Returns
    -------
    tuple
        A tuple containing:
        - G (networkx.MultiDiGraph): The graph with relabeled nodes.
        - mapping (dict): A dictionary mapping the original
        node labels to the new labels.
    """
    mapping = {}
    for i, node in enumerate(list(G.nodes())):
        mapping[node] = i
    G = nx.relabel_nodes(G, mapping)
    return G, mapping


def filter_pools_with_no_gradient(pools, prices):
    pools = pools[pools['address'].isin(set(prices['pool_address']))]
    mask = []
    for _, pool in pools.iterrows():
        t0 = pool['token0']
        t1 = pool['token1']
        p = make_price(prices[prices['pool_address'] == pool['address']])
        mask.append(np.gradient(p).sum() != 0)

    pools = pools[mask]
    prices = prices[prices['pool_address'].isin(list(pools['address']))]
    return pools, prices


