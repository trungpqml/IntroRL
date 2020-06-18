class Config:
    epsilon_min = 0.005
    epsilon_max = 1.0
    max_num_episodes = 5000
    steps_per_episode = 200
    max_num_steps = max_num_episodes * steps_per_episode
    epsilon_decay = 500 * epsilon_min/max_num_steps
    alpha = 0.05  # learning rate
    gamma = 0.98  # discount factor
    num_discrete_bins = 30


cfg = Config()
