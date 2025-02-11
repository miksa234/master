#!/usr/bin/env python3

from rl_circuit.play import run
from rl_circuit.logger import logging

logger = logging.getLogger('rl_circuit')

import subprocess
import os

def main():
    hostname = os.uname()[1]

#    if hostname == "frame":
#        cwd = os.getcwd()
#        print("\nCopying code...\n")
#        subprocess.getoutput(f"rsync -Pr {cwd} mycomp:~/")
#        print("\nExecuting code remotely...\n")
#        with subprocess.Popen("ssh -4 mycomp 'source py_env/bin/activate && cd ./rl_circuit && ./main.py'", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True) as process:
#            for line in process.stdout:
#              print(line.decode('utf8'))
#        exit()

    logger.info("Started Logging")
    run()

if __name__ == '__main__':
    main()
