#!/usr/bin/env python3
import pandas as pd


def load_pools_and_tokens(path_pools, path_tokens):
    pools = pd.read_csv(
        path_pools,
        names=["index", "addess", "version", "token0", "token1", "fee", "block_number", "timestamp", "tickspacing"]
    ).sort_index().drop_duplicates()
    tokens = pd.read_csv(
        path_tokens,
        index_col=0,
        names=["index", "address", "name", "symbol", "decimals"]
    ).sort_index().drop_duplicates()

    return pools, tokens


def pools_to_edge_list(pools):
    t0t1 = pools[['token0', 'token1']].to_numpy()
    edge_list = []
    cache = []
    for (t0, t1) in t0t1:
        k = 0
        for e in cache:
            if (t0, t1) == e or (t1, t0) == e:
                k += 1
        edge_list.append((t0, t1, {'k': k}))
        cache.append((t0, t1))
    return edge_list


