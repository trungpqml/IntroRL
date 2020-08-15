import matplotlib.pyplot as plt
import numpy as np


def show(qmaze):
    plt.grid('on')
    nrows, ncols = qmaze.maze.shape
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, nrows, 1))
    ax.set_yticks(np.arange(0.5, ncols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    canvas = np.copy(qmaze.maze)

    for row, col in qmaze.visited:
        canvas[row, col] = 0.6

    rat_row, rat_col, _ = qmaze.state
    canvas[rat_row, rat_col] = 0.3  # agent cell
    canvas[nrows-1, ncols-1] = 0.9  # cheese cell
    img = plt.imshow(canvas, interpolation='none', cmap='gray')
    return img


def play_game(model, qmaze, rat_cell):
    qmaze.reset(rat_cell)
    envstate = qmaze.observe()
    while True:
        prev_envstate = envstate
        q = model.predict(prev_envstate)
        action = np.argmax(q[0])

        envstate, reward, game_status = qmaze.act(action)

        if game_status == 'win':
            return True
        elif game_status == 'lose':
            return False


def completion_check(model, qmaze):
    for cell in qmaze.free_cells:
        if not qmaze.valid_actions(cell):
            return False
        if not play_game(model, qmaze, cell):
            return False
    return True
