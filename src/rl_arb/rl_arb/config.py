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
    'cutoff': 20
}

ARGS_MODEL = {
    'in_channels': 4,
    'emb_channels': 256,
    'num_heads': 16,
    'num_layers': 8,
    'ff_dim': 1024,
    'policy_mheads': 1,
    'value_mheads': 1
}

ARGS_TRAINING = {
    'C': 2,
    'C_1/3' : 5,
    'C_2/3' : 2.5,
    'C_3/3' : 1,
    'num_iterations': 50,
    'num_searches': 50,
    'num_self_play_iterations': 4000,
    'num_parallel': 100,
    'num_epochs': 20,
    'batch_size': 64,
    'temperature': 1,
    'eps': 0.25,
    'dirichlet_alpha': 0.3,
    'num_processes': 20,
    'multicore': True,
    'telegram': True,
}
