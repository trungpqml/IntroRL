import numpy as np
import torch
from config import cfg


class DeepQLearner:
    def __init__(self, state_shape, action_shape, device):
        self.Q = self.DQN(state_shape, action_shape, device).to(device)
        self.Q_optimizer = torch.optim.Adam(
            self.Q.parameters(), lr=cfg.learning_rate)
        if self.params['use_target_network']:
            self.Q_target = self.DQN(state_shape, action_shape, device)

    def get_action(self):
        pass

    def learn_from_batch_experience(self, experiences):
        pass
