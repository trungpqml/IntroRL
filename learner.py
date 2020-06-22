import numpy as np
from config import cfg


class Q_Learner(object):
    def __init__(self, env):
        self.obs_shape = env.observation_space.shape
        self.obs_high = env.observation_space.high
        self.obs_low = env.observation_space.low
        self.obs_bin = cfg.num_discrete_bins
        self.bin_width = (self.obs_high - self.obs_low)/self.obs_bin
        self.action_shape = env.action_space.n

        self.Q = np.zeros((self.obs_bin+1, self.obs_bin+1,
                           self.obs_bin+1, self.obs_bin+1, self.action_shape))

        self.alpha = cfg.alpha
        self.gamma = cfg.gamma
        self.epsilon = cfg.epsilon_max

    def get_action(self, obs):
        discretized_obs = self.discretize_obs(obs)
        if self.epsilon > cfg.epsilon_min:
            self.epsilon -= cfg.epsilon_decay
        if np.random.random() > self.epsilon:
            return np.argmax(self.Q[discretized_obs])
        else:
            return np.random.choice([a for a in range(self.action_shape)])

    def discretize_obs(self, obs):
        return tuple(((obs - self.obs_low)/self.bin_width).astype(int))

    def learn(self, obs, action, reward, next_obs):
        discretized_obs = self.discretize_obs(obs)
        discretized_next_obs = self.discretize_obs(next_obs)
        td_target = reward + self.gamma * np.max(self.Q[discretized_next_obs])
        td_error = td_target - self.Q[discretized_obs][action]
        self.Q[discretized_obs][action] += self.alpha * td_error
