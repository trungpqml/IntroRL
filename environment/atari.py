import gym
import atari_py
import random
import numpy as np
import cv2
from gym.spaces.box import Box


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Clip rewards to be either -1, 0, or +1 based on the sign
        """
        return np.sign(reward)


def process_frame_84(frame, conf):
    frame = frame[conf['crop1']:conf['crop2']+160, :160]
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0/255.0)
    frame = cv2.resize(frame, (84, conf["dimension2"]))
    frame = cv2.resize(frame, (84, 84))
    frame = np.reshape(frame, [1, 84, 84])
    return frame


class AtariRescale(gym.ObservationWrapper):
    def __init(self, env, env_conf):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = Box(0.0, 1.0, [1, 84, 84])
        self.conf = env_conf

    def observation(self, observation):
        return process_frame_84(observation, self.conf)
