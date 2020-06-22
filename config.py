class Config(object):
    max_num_episodes = 50000
    steps_per_episode = 200
    max_num_steps = max_num_episodes*steps_per_episode
    epsilon_min = 0.5
    epsilon_max = 1.0
    epsilon_decay = 500 * epsilon_min/max_num_steps
    alpha = 0.5
    gamma = 0.98
    num_discrete_bins = 30
    env_name = "CartPole-v0"
    output_dir = "gym_monitor_output"


cfg = Config()

if __name__ == "__main__":
    print(cfg)
