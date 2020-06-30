class LinearDecayScheduler:
    def __init__(self, initial_value, final_value, max_steps):
        assert initial_value > final_value, "initial_value should not be < final_value"
        self.initial_value = initial_value
        self.final_value = final_value
        self.decay_factor = (initial_value-final_value)/max_steps

    def __call__(self, step_num):
        current_value = self.initial_value - self.decay_factor*step_num
        if current_value < self.final_value:
            current_value = self.final_value
        return current_value


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    epsilon_initial = 1.0
    epsilon_final = 0.05
    max_num_episodes = 100000
    steps_per_episode = 300
    scheduler = LinearDecayScheduler(
        initial_value=epsilon_initial, final_value=epsilon_final, max_steps=max_num_episodes*steps_per_episode)
    epsilon_list = [scheduler(step) for step in range(
        max_num_episodes*steps_per_episode)]
    plt.plot(epsilon_list)
    plt.show()
