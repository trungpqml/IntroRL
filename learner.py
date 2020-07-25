from os.path import join, exists
from os import makedirs

import gym
import torch
import random
import numpy as np

from utils.decay_schedule import LinearDecayScheduler
from utils.experience_memory import Experience, ExperienceMemory
import utils.weights_initializer
from function_approximator.perceptron import SLP
from function_approximator.cnn import CNN


class DeepQLearner:
    def __init__(self, state_shape, action_shape, params, writer, device="cpu"):
        """
        self.Q is the Action-value function.This agent represents Q using a Neural Network
        If the input is a single dimensional vector, use a Single Layer Perceptron else if the input is 3 dimensional image, use a Convolutional Neural Network
        :param state_shape: Shape (tuple) of the observation/state
        :param action_shape: Shape (number) of the discrete action space
        :param params: A dictionary containing various Agent configuration parameters and hyper-parameters 
        """
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.params = params
        self.gamma = self.params['gamma']
        self.learning_rate = self.params['lr']
        self.best_mean_reward = -float('inf')
        self.best_reward = -float('inf')
        self.training_steps_completed = 0
        self.writer = writer
        self.device = device

        if len(self.state_shape) == 1:
            self.DQN = SLP
        elif len(self.state_shape) == 3:
            self.DQN = CNN

        self.Q = self.DQN(state_shape, action_shape,
                          self.device).to(self.device)
        self.Q.apply(utils.weights_initializer.xavier)
        self.Q_optimizer = torch.optim.Adam(
            self.Q.parameters(), lr=self.learning_rate)

        if self.params['use_target_network']:
            self.Q_target = self.DQN(
                state_shape, action_shape, self.device).to(self.device)
        self.policy = self.epsilon_greedy_Q
        self.epsilon_max = params['epsilon_max']
        self.epsilon_min = params['epsilon_min']
        self.epsilon_decay = LinearDecayScheduler(
            initial_value=self.epsilon_max, final_value=self.epsilon_min, max_steps=self.params['epsilon_decay_final_step'])
        self.step_num = 0

        self.memory = ExperienceMemory(capacity=int(
            self.params['experience_memory_capacity']))

    def get_action(self, obs):
        obs = np.array(obs)
        obs = obs/255.0
        if len(obs.shape) == 3:
            if obs.shape[2] < obs.shape[0]:
                obs = obs.reshape(obs.shape[2], obs.shape[1], obs.shape[0])
            obs = np.expand_dims(obs, 0)
        return self.policy(obs)

    def epsilon_greedy_Q(self, obs):
        self.writer.add_scalar(
            'DQL/epsilon', self.epsilon_decay(self.step_num), self.step_num)
        self.step_num += 1
        if random.random() < self.epsilon_decay(self.step_num) and not self.params['test']:
            action = random.choice([i for i in range(self.action_shape)])
        else:
            action = np.argmax(self.Q(obs).data.to(self.device).numpy())
        return action

    def learn(self, obs, action, reward, next_obs, done):
        if done:
            td_target = reward + 0.0
        else:
            td_target = reward + self.gamma * torch.max(self.Q(next_obs))
        td_error = td_target - self.Q(s)[a]
        self.Q_optimizer.zero_grad()
        td_error.backward()
        self.Q_optimizer.step()

    def learn_from_batch_experience(self, experiences):
        batch_xp = Experience(*zip(*experiences))
        obs_batch = np.array(batch_xp.obs)
        action_batch = np.array(batch_xp.action)

        reward_batch = np.array(batch_xp.reward)
        if self.params['clip_rewards']:  # Clip the rewards
            reward_batch = np.sign(reward_batch)

        next_obs_batch = np.array(batch_xp.next_obs)/255.0
        done_batch = np.array(batch_xp.done)

        if self.params['use_target_network']:
            if self.step_num % self.params['target_network_update_freq'] == 0:
                self.Q_target.load_state_dict(self.Q.state_dict())
            td_target = reward_batch + ~done_batch * \
                np.tile(self.gamma, len(next_obs_batch)) * \
                self.Q_target(next_obs_batch).max(
                    1)[0].data.to(self.device).numpy()
        else:
            td_target = reward_batch + ~done_batch * \
                np.tile(self.gamma, len(next_obs_batch)) * \
                self.Q(next_obs_batch).detach().max(
                    1)[0].data.to(self.device).numpy()

        td_target = torch.from_numpy(td_target).to(self.device)
        action_idx = torch.from_numpy(action_batch).to(self.device)
        td_error = torch.nn.functional.mse_loss(
            self.Q(obs_batch).gather(1, action_idx.view(-1, 1)), td_target.float().unsqueeze(1))

        self.Q_optimizer.zero_grad()
        td_error.mean().backward()
        self.writer.add_scalar('DQL/td_error', td_error.mean(), self.step_num)
        self.Q_optimizer.step()

    def replay_experience(self, batch_size=None):
        batch_size = batch_size if batch_size is not None else self.params['replay_batch_size']
        experience_batch = self.memory.sample(batch_size)
        self.learn_from_batch_experience(experience_batch)
        self.training_steps_completed += 1

    def save(self, env_name):
        if not exists(self.params['save_dir']):
            makedirs(self.params['save_dir'])
        file_name = join(self.params['save_dir'], 'DQL_' + env_name + '.ptm')
        agent_state = {'Q': self.Q.state_dict(),
                       'best_mean_reward': self.best_mean_reward,
                       'best_reward': self.best_reward
                       }
        torch.save(agent_state, file_name)
        print(f"Agent's state saved to {file_name}")

    def load(self, env_name):
        file_name = self.params['load_dir'] + 'DQL_' + env_name + '.ptm'
        agent_state = torch.load(
            file_name, map_location=lambda storage, loc: storage)
        self.Q.load_state_dict(agent_state['Q'])
        self.Q.to(self.device)
        self.best_mean_reward = agent_state['best_mean_reward']
        self.best_reward = agent_state['best_reward']
        print(
            f'Loaded Q model state from {file_name} which fetched a best mean reward of {self.best_mean_reward:.3f} and an all time best reward of {self.best_reward}')
