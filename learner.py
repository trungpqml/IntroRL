import torch
import gym
import random
import numpy as np

import environment.atari as Atari
from utils.params_manager import ParamsManager
from utils.decay_schedule import LinearDecayScheduler
from utils.experience_memory import Experience, ExperienceMemory
from function_approximator.perceptron import SLP
from function_approximator.cnn import CNN
from tensorboardX import SummaryWriter
from datetime import datetime
from argparse import ArgumentParser

args = ArgumentParser("learner")
args.add_argument(
    "--params-file", help="Path to the parameters JSON file. Default is parameters.json", default="parameters.json", type=str, metavar="PFILE")
args.add_argument(
    "--env-name", help="ID of the Atari environment available in OpenAI Gym. Default is Pong-v0", default="Pong-v0", type=str, metavar="ENV")
args = args.parse_args()

params_manager = ParamsManager(args.params_file)
seed = params_manager.get_agent_params()['seed']
summary_file_path_prefix = params_manager.get_agent_params()[
    'summary_file_path_prefix']
summary_file_name = summary_file_path_prefix + args.env_name + \
    "_" + datetime.now().strftime("%y-%m-%d-%H-%M")
writer = SummaryWriter(summary_file_name)
global_step_num = 0
use_cuda = params_manager.get_agent_params()['use_cuda']
device = torch.device("cuda" if torch.cuda.is_available()
                      and use_cuda else "cpu")
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available() and use_cuda:
    torch.cuda.manual_seed_all(seed)


class DeepQLearner:
    def __init__(self, state_shape, action_shape, params):
        """
        self.Q is the Action-value function.This agent represents Q using a Neural Network
        If the input is a single dimensional vector, use a Single Layer Perceptron else if the input is 3 dimensional image, use a Convolutional Neural Network
        :param state_shape: Shape (tuple) of the observation/state
        :param action_shape: Shaoe (number) of the discrete action space
        :param params: A dictionary containing various Agent configuration parameters and hyper-parameters 
        """
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.params = params
        self.gamma = self.params['gamma']
        self.learning_rate = self.params['lr']
        if len(self.state_shape) == 1:
            self.DQN = SLP
        elif len(self.state_shape) == 3:
            self.DQN = CNN

        self.Q = self.DQN(state_shape, action_shape, device).to(device)
        self.Q_optimizer = torch.optim.Adam(
            self.Q.parameters(), lr=self.learning_rate)

        if self.params['use_target_network']:
            self.Q_target = self.DQN(
                state_shape, action_shape, device).to(device)
        self.policy = self.epsilon_greedy_Q
        self.epsilon_max = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = LinearDecayScheduler(
            initial_value=self.epsilon_max, final_value=self.epsilon_min, max_steps=self.params['epsilon_decay_final_step'])
        self.step_num = 0

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
