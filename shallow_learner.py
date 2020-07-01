import numpy as np
from config import cfg
from function_approximator.perceptron import SLP
import torch
from utils.decay_schedule import LinearDecayScheduler
import gym


class Shallow_Q_Learner(object):
    def __init__(self, state_shape, action_shape):
        self.state_shape = state_shape
        self.action_shape = action_shape

        self.Q = SLP(self.state_shape, self.action_shape)
        self.Q_optimizer = torch.optim.Adam(
            self.Q.parameters(), lr=cfg.learning_rate)

        self.gamma = cfg.gamma
        self.epsilon_max = cfg.epsilon_max
        self.epsilon_min = cfg.epsilon_min
        self.epsilon_decay = LinearDecayScheduler(
            initial_value=self.epsilon_max, final_value=self.epsilon_min, max_steps=cfg.max_num_steps)
        self.step_num = 0
        self.policy = self.epsilon_greedy_Q

    def get_action(self, obs):
        return self.policy(obs)

    def epsilon_greedy_Q(self, obs):
        if np.random.random() >= self.epsilon_decay(self.step_num):
            return np.argmax(Q(obs).data.numpy())
        else:
            return np.random.choice([a for a in range(self.action_shape)])

    def learn(self, obs, action, reward, next_obs):
        td_target = reward + self.gamma * torch.max(self.Q(next_obs))
        td_error = torch.nn.functional.mse_loss(td_target, self.Q(obs)[action])
        self.Q_optimizer.zero_grad()
        td_error.backward()
        self.Q_optimizer.step()


if __name__ == "__main__":
    env = gym.make(cfg.env_name)
    observation_shape = env.observation_space.shape
    action_shape = env.action_space.n
    agent = Shallow_Q_Learner(observation_shape, action_shape)
    first_episode = True
    episode_reward_list = list()
    for episode in range(cfg.max_num_episodes):
        obs = env.reset()
        cumulative_reward = 0.0
        for step in range(cfg.max_num_steps):
            env.render()
            action = agent.get_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.learn(obs, action, reward, next_obs)
            obs = next_obs
            cumulative_reward += reward

            if done:
                if first_episode:
                    max_reward = cumulative_reward
                    first_episode = False
                episode_reward_list.append(cumulative_reward)
                if cumulative_reward > max_reward:
                    max_reward = cumulative_reward
                print(
                    f"\nEpisode# {episode}\tended in {step+1}\tsteps. Reward = {cumulative_reward}\tmean reward = {np.mean(episode_reward_list):.2f}\tbest reward = {max_reward}")
                break
    env.close()
