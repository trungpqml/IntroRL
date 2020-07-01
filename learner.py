import numpy as np
import torch
from config import cfg
from utils.experience_memory import Experience, ExperienceMemory


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
        batch_xp = Experience(*zip(*experiences))
        obs_batch = np.array(batch_xp.obs)
        action_batch = np.array(batch_xp.action)
        reward_batch = np.array(batch_xp.reward)
        next_obs_batch = np.array(batch_xp.next_obs)
        done_batch = np.array(batch_xp.done)

        if self.params['use_target_network']:
            if self.step_num % self.params['target_network_update_freq'] == 0:
                self.Q_target.load_state_dict(self.Q.state_dict())
            td_target = reward_batch + ~done_batch * \
                np.tile(self.gamma, len(next_obs_batch)) * \
                self.Q_target(next_obs_batch).max(1)[0].data
        else:
            td_target = reward_batch + ~done_batch * \
                np.tile(self.gamma, len(next_obs_batch)) * \
                self.Q(next_obs_batch).detach().max(1)[0].data
        td_target = td_target.to(device)
        action_idx = torch.from_numpy(action_batch).to(device)
        td_error = torch.nn.functional.mse_loss(
            self.Q(obs_batch).gather(1, action_idx.view(-1, 1)), td_target.float().unsqueeze(1))
        self.Q_optimizer.zero_grad()
        td_error.mean().backward()
        writer.add_scalar("DQL/td_error", td_error.mean(), self.step_num)
        self.Q_optimizer.step()
