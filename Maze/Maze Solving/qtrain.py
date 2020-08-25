from helpers import format_time
from config import cfg
import datetime
from qmaze import Qmaze
from experience import Experience
import numpy as np
import random


def qtrain(model, maze, **opt):
    n_epoch = opt.get('n_epoch', cfg.n_epoch)
    max_memory = opt.get('max_memory', cfg.max_memory)
    data_size = opt.get('data_size', cfg.data_size)
    weights_file = opt.get('weights_file', cfg.weights_file)
    name = opt.get('name', 'model')
    start_time = datetime.datetime.now()

    if weights_file:
        print(f'loading weights from file: {weights_file}')
        model.load_weights(weights_file)

    qmaze = Qmaze(maze)

    experience = Experience(model, max_memory=max_memory)

    win_history = []
    n_free_cells = len(qmaze.free_cells)
    hsize = qmaze.maze.size//2
    win_rate = 0.0
    imctr = 1

    for epoch in range(n_epoch):
        loss = 0.0
        rat_cell = random.choice(qmaze.free_cells)
        qmaze.reset(rat_cell)
        game_over = False

        envstate = qmaze.observe()

        n_episodes = 0
        while not game_over:
            valid_actions = qmaze.valid_actions()
            if not valid_actions:
                break
            prev_envstate = envstate
            if np.random.rand() < cfg.epsilon:
                action = ra
            else:
                action = np.argmax(experience.predict(envstate))
