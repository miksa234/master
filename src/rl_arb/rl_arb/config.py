#!/usr/bin/env python3

import torch
from dotenv import dotenv_values

env = dotenv_values(".env")

TELEGRAM_TOKEN = env["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = env["TELEGRAM_CHAT_ID"]
TELEGRAM_SEND_URL = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ARGS_GAME = {
    'tau': 1,
    'M': 1,
    'cutoff': 25
}

ARGS_MODEL = {
    'in_channels': 4,
    'emb_channels': 256,
    'num_heads': 16,
    'num_layers': 20,
    'ff_dim': 1024,
    'policy_mheads': 3,
    'value_mheads': 3
}

ARGS_TRAINING = {
    'C': 1.4,
    'C_1/3' : 1.3,
    'C_2/3' : 1.3,
    'C_3/3' : 1.3,
    'num_iterations': 1000,
    'num_searches': 200,
    'num_rollouts': 0,
    'num_self_play_iterations': 300,
    'num_parallel': 10,
    'num_epochs': 1,
    'batch_size': 50,
    'gamma': 0.98,
    'temperature': 1.25,
    'eps': 0.25,
    'dirichlet_alpha': 0.03,
    'num_processes': 30,
    'multicore': True,
    'telegram': True,
}
