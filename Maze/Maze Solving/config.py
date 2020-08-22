import numpy as np


class Config:
    visited_mark = 0.8  # color value of visited mark
    rat_mark = 0.5  # color value of rat position
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3

    action_dict = {
        LEFT: 'left',
        UP: 'up',
        RIGHT: 'right',
        DOWN: 'down'
    }

    num_actions = len(action_dict)
    epsilon = 0.1  # exporation factor
    n_epochs = 15000
    max_memory = 1000
    data_size = 50
    weights_file = ''

    maze = np.array([
        [1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.],
        [1.,  0.,  1.,  1.,  1.,  0.,  1.,  1.],
        [1.,  1.,  1.,  1.,  0.,  1.,  0.,  1.],
        [1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
        [1.,  1.,  0.,  1.,  1.,  1.,  1.,  1.],
        [1.,  1.,  1.,  0.,  1.,  0.,  0.,  0.],
        [1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
        [1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.]
    ])


cfg = Config()
