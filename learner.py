from config import cfg
import numpy as np


class Q_learner(object):
    def __init__(self, env):
        self.obs_shape = env.observation_space.shape
        self.obs_high = env.observation_space.high
        self.obs_low = env.observation_space.low
        # number of bin to discretize each observation dim
        self.obs_bins = cfg.num_discrete_bins
        self.bin_width = (self.obs_high-self.obs_low)/self.obs_bins
        self.action_shape = env.action_space.n

        # Q-value of size 31 x 31 x 3
        self.Q = np.zeros(
            (self.obs_bins+1, self.obs_bins+1, self.action_shape))
        self.alpha = cfg.alpha
        self.gamma = cfg.gamma
        self.epsilon_min = cfg.epsilon_min
        self.epsilon = cfg.epsilon_max
        self.epsilon_decay = cfg.epsilon_decay

    def discretize(self, obs):
        return tuple(((obs - self.obs_low)/self.bin_width).astype(int))

    def get_action(self, obs):
        discretized_obs = self.discretize(obs)
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        if np.random.random() > self.epsilon:
            return np.argmax(self.Q[discretized_obs])
        else:
            return np.random.choice([a for a in range(self.action_shape)])

    def learn(self, obs, action, reward, next_obs):
        discretized_obs = self.discretize(obs)
        discretized_next_obs = self.discretize(next_obs)
        td_target = reward + self.gamma*np.max(self.Q[discretized_next_obs])
        td_error = td_target - self.Q[discretized_obs][action]
        self.Q[discretized_obs][action] += self.alpha*td_error
