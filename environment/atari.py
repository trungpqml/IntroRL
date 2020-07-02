import gym
import atari_py
import random
import numpy as np


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Clip rewards to be either -1, 0, or +1 based on the sign
        """
        return np.sign(reward)
