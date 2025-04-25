#!/usr/bin/env python3

import torch
from dotenv import dotenv_values

env = dotenv_values(".env")

TELEGRAM_TOKEN = env["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = env["TELEGRAM_CHAT_ID"]
TELEGRAM_SEND_URL = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"

ARGS_GAME = {
    'tau': 1,
    'M': 1,
    'cutoff': 25
}

ARGS_MODEL = {
    'in_channels': 4,
    'emb_channels': 128,
    'num_heads': 4,
    'num_layers': 4,
    'ff_dim': 512,
    'policy_mheads': 2,
}

ARGS_TRAINING = {
    'C': 1.4,
    'num_reinforce': 100000,
    'num_iterations': 100,
    'num_searches': 100,
    'num_rollouts': 100,
    'num_self_play_iterations': 200,
    'num_parallel': 10,
    'num_epochs': 1,
    'batch_size': 25,
    'gamma': 0.98,
    'temperature': 1.25,
    'eps': 0.25,
    'dirichlet_alpha': 0.03,
    'num_processes': 20,
    'multicore': True,
    'telegram': True,
}
