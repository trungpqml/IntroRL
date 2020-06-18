from config import cfg
from numpy import np


class Q_learner(object):
    def __init__(self, env):
        self.obs_shape = env.observation_space.shape
        self.obs_high = env.observation_space.high
        self.obs_low = env.observation_space.low
        # number of bin to discretize each observation dim
        self.obs_bins = cfg.num_discrete_bins
        self.bin_width = (self.obs_high-self.obs_low)/self.obs_bins
        self.action_shape = env.action_shape.n

        # Q-value of size 31 x 31 x 3
        self.Q = np.zeros(
            (self.obs_bins+1, self.obs_bins+1, self.action_shape))
        self.alpha = cfg.alpha
        self.gamma = cfg.gamma
        self.epsilon = cfg.epsilon_max

    def discretize(self, obs):
        return tuple(((obs - self.obs_low)/self.bin_width).astype(int))

    def get_action(self, obs):
        
