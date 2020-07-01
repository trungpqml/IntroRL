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
