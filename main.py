from config import cfg
import gym
from learner import Q_learner
from os.path import join, exists
from os import makedirs


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
            obs = next_obs
            total_reward += reward
        if total_reward > best_reward:
            best_reward = total_reward
        print(
            f"Episode #{episode}\treward: {total_reward},\tbest reward: {best_reward},\tepsilon: {agent.epsilon:.2f}")
    return np.argmax(agent.Q, axis=2)


def test(agent, env, policy):
    done = False
    obs = env.reset()
    total_reward = 0.0
    while not done:
        action = policy[agent.discretize(obs)]
        next_obs, reward, done, info = env.step(action)
        obs = next_obs
        total_reward += reward
    return total_reward


if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    agent = Q_learner(env)
    learned_policy = train(agent, env)

    gym_monitor_path = join(".", "gym_monitor_output")
    if not exists(gym_monitor_path):
        makedirs(gym_monitor_path)

    env = gym.wrappers.Monitor(env, gym_monitor_path, force=True)
    for _ in range(cfg.num_test_episodes):
        test(agent, env, learned_policy)
    env.close()
