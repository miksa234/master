#!/usr/bin/env python3

import sys
import os
import subprocess

from rl_arb.initializer import Initializer
from rl_arb.logger import logging
logger = logging.getLogger('rl_circuit')


def run():
    """
    Small cli interface.
    """
    logger.info("Started...")
    help = """
        flags:
            learn: initializes the training process

            test: Does some testing with the model.

            vsc5: intitializes the training on the vsc5 node

            mycomp: intitializes the training on the private comp
    """

    try:
        if sys.argv[1] == "mycomp":
            hostname = os.uname()[1]
            if hostname == "frame":
                cwd = os.getcwd()
                logger.info("Copying code...")
                subprocess.getoutput(f"rsync -Pr {cwd} mycomp:~/")
                logger.info("Executing code remotely...")
                with subprocess.Popen(
                    "ssh -4 mycomp 'source py_env/bin/activate && cd ./rl_arb && ./main.py learn'",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    shell=True
                ) as process:
                    for line in process.stdout:
                      logger.info(line.decode('utf8'))
            else:
                logger.info("Not on frame, closing...")

        elif sys.argv[1] == "vsc5":
            logger.info("Copying code...")
            subprocess.getoutput(f"rsync -Pr {cwd} vsc5:/gpfs/data/fs70700/miksa234")
            subprocess.getoutput("ssh -4 vsc5 'cd $DATA && sbatch rl.job")

        elif sys.argv[1] == "learn":
            problem = Initializer()
            problem.rlearn.learn()
        else:
            logger.info("No valid input detected")
            logger.info(help)
    except IndexError as e:
        logger.info("No valid input detected")
        logger.info(help)

    logger.info("END")
