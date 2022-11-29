# %% 
import argparse
import os
import random
import re
import time
from dataclasses import dataclass
from distutils.util import strtobool
from typing import Iterable, Optional, Tuple, Union

import gym
import gym.envs.registration
import numpy as np
import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from gym.spaces import Box, Discrete
from numpy.random import Generator
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

MAIN = __name__ == "__main__"
os.environ["SDL_VIDEODRIVER"] = "dummy"

t.set_default_dtype(t.float32)

class QNetwork(nn.Module):

    def __init__(self,
                 dim_observation: int,
                 num_actions: int,
                 hidden_sizes: list[int] = [120, 84]):
        super().__init__()
        self.dim_observation = dim_observation
        self.num_actions = num_actions
        self.hidden_sizes = hidden_sizes

        self.linear1 = nn.Linear(dim_observation, hidden_sizes[0])
        self.linear2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.linear3 = nn.Linear(hidden_sizes[1], self.num_actions)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

# %% 

# write a function which samples the output of q_network for random inputs as the hidden layer sizes are varied 

for hidden_sizes in [[256, 128], [128, 64], [64, 32], [32, 16], [16, 8], [8, 4], [4, 2], [2, 1]]:
    q_network = QNetwork(4, 2, hidden_sizes)
    total_activation = 0
    for i in range(10000):
        x = t.rand(4)*10-5
        y = q_network(x)
        total_activation += y.sum()
    average_activation = total_activation / 10000
    print(f"Average activation for hidden sizes {hidden_sizes} is {average_activation}")


# %%
