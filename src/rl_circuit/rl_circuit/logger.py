#!/usr/bin/env python3

import logging
import sys

format = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=format)

logger = logging.getLogger('rl_circuit')
#ch = logging.StreamHandler(sys.stdout)
#logger.addHandler(ch)
logger.setLevel(logging.DEBUG)
