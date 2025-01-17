#!/usr/bin/env python3

from src import *

import subprocess
import os

hostname = os.uname()[1]

if hostname == "frame":
    cwd = os.getcwd()
    print(subprocess.getoutput(f"rsync -Pr {cwd} mycomp:~/"))
    print("\nExecuting code remotely...\n")
    print(subprocess.getoutput("ssh mycomp 'source py_env/bin/activate && cd ./rl_circuit && ./main.py'"))
    exit()

run()
