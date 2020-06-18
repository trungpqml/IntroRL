import gym
env = gym.make("MountainCar-v0")
MAX_NUM_EPISODES = 5000

for episode in range(MAX_NUM_EPISODES):
    done = False
    obs = env.reset()
    # To keep track of total reward in each episode
    total_reward = 0.0
    step = 0
    while not done:
        env.render()
        # Take a random action from sample action space
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        total_reward += reward
        step += 1
        obs = next_obs
    print(
        f"Episode #{episode} \tended in {step} steps. Total reward={total_reward}")
env.close()
