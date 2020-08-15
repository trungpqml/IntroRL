class Config:
    visited_mark = 0.8
    rat_mark = 0.5
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


cfg = Config()
