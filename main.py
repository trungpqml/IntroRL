import numpy as np
from config import cfg
from learner import Q_Learner
import gym
from os.path import join, exists
from os import mkdir


def train(agent, env):
    best_reward = -float('inf')
    for episode in range(cfg.max_num_episodes):
        done = False
        obs = env.reset()
        total_reward = 0.0
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.learn(obs, action, reward, next_obs)
            next_obs = obs
            total_reward += reward
        if total_reward > best_reward:
            print(
                f"\tEpisode #{episode}\treward: {total_reward}\tbest reward: {best_reward}\teps: {agent.epsilon}")
        return np.argmax(agent.Q, axis=2)


def test(agent, env, policy):
    done = False
    obs = env.reset()
    total_reward = 0.0
    while not done:
        action = policy[agent.discretize_obs(obs)]
        next_obs, reward, done, info = env.step(action)
        obs = next_obs
        total_reward += reward
    return total_reward


def set_up_output(env):
    path = join(".", cfg.output_dir)
    if not exists(path):
        mkdir(path=path)
    return gym.wrappers.Monitor(env, path, force=True)


if __name__ == "__main__":
    env = gym.make(cfg.env_name)
    agent = Q_Learner(env)
    learned_policy = train(agent, env)
    env = set_up_output(env)
    for _ in range(1000):
        test(agent, env, learned_policy)
    env.close()
