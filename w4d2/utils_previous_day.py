import numpy as np
from typing import List
import torch as t
import gym
import random
Arr = np.ndarray

def sum_rewards(rewards : List[int], gamma : float = 1):
    """
    Computes the total discounted sum of rewards for an episode.
    By default, assume no discount
    Input:
        rewards [r1, r2, r3, ...] The rewards obtained during an episode
        gamma: Discount factor
    Output:
        The sum of discounted rewards 
        r1 + gamma*r2 + gamma^2 r3 + ...
    """
    total_reward = 0
    for r in rewards[:0:-1]: #reverse, excluding first
        total_reward += r
        total_reward *= gamma
    total_reward += rewards[0]
    return total_reward

def cummean(arr: Arr):
    """
    Computes the cumulative mean
    """
    return np.cumsum(arr) / np.arange(1, len(arr) + 1)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)


def make_env(env_id: str, seed: int, idx: int, capture_video: bool, run_name: str):
    """Return a function that returns an environment after setting up boilerplate."""
    
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    
    return thunk
