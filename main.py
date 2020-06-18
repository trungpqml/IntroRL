from config import cfg


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
