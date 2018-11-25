This code was adapted from github.com/aaksham/frozenlake.git.

In order to run code for the 16x16 grid, user must download a local version of 'gym' and replace the frozen_lake.py file with the frozen_lake.py file in this repository.  The filepath in gym is "gym/envs/toy_text/frozen_lake.py".

VI, PI, and Q-Learning can be run using the FrozenLake_Engine.py file.  This can be executed using Python3.

Depending on if a grid world of 4x4, 8x8 or 16x16 is to be generated, include or copy out the following lines from the top of FrozenLake_Engine.py.

from deeprl_hw1.rl1 import *
from deeprl_hw1.rl8 import *
from deeprl_hw1.rl16 import *

Note:
rl1 is for 4x4
rl8 is for 8x8
rl16 is for 16x16

Input the desired envname into FrozenLake_Engine.py.  A list of envname's are shown below:

Stochastic-4x4-FrozenLake-v0
Stochastic-8x8-FrozenLake-v0
Stochastic-16x16-FrozenLake-v0

Furthermore, the probability model of the environment can be changed from 0.8 to 0.333 by updating the lakeEnv variable in the frozen_lake.py file.

