#!/usr/bin/env python3

import sys
import os
import subprocess

from rl_arb.initializer import Initializer
from rl_arb.utils import print_summary, send_telegram_message
from rl_arb.brute_force import test_model
from rl_arb.logger import logging
logger = logging.getLogger('rl_circuit')

def run():
    """
    Small cli for interaction.
    """
    logger.info("Started...")
    help = """
        flags:
            learn: initializes the training process

            test: Does some testing with the model.

            vsc5: intitializes the training on the vsc5 node

            mycomp: intitializes the training on the private comp
    """

    if sys.argv[1] == "mycomp":
        hostname = os.uname()[1]
        if hostname == "frame":
            cwd = os.getcwd()
            logger.info("Copying code...")
            subprocess.getoutput(f"rsync -Pr {cwd} mycomp:~/")
            logger.info("Executing code remotely...")
            if sys.argv[2] == "test":
                cmd = "test"
            else:
                cmd = "learn"
            with subprocess.Popen(
                f"ssh -4 mycomp 'source py_env/bin/activate && cd ./rl_arb && ./main.py {cmd}'",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=True
            ) as process:
                for line in process.stdout:
                  logger.info(line.decode('utf8'))
        else:
            logger.info("Not on frame, closing...")

    elif sys.argv[1] == "vsc5":
        cwd = os.getcwd()
        data_path = "/gpfs/data/fs70700/miksa234"
        logger.info("Copying code...")
        subprocess.getoutput(f"rsync -Pr {cwd} vsc5:{data_path}")
        logger.info(subprocess.getoutput(f"ssh -4 vsc5 'cd {data_path} && sbatch rl.job'"))

    elif sys.argv[1] == "learn":
        problem = Initializer()
        problem.rlearn.learn(problem.optimizer)

    elif sys.argv[1] == "reinforce":
        problem = Initializer()
        problem.reinforce.learn()

    elif sys.argv[1] == "info":
        problem = Initializer()
        print_summary(problem)

    elif sys.argv[1] == "test":
        test_model()

    elif sys.argv[1] == "help":
        logger.info(help)

    else:
        logger.info("No valid input detected")
        logger.info(help)

    logger.info("END")
