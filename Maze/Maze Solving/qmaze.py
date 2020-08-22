# Maze is a 2D Numpy array of floats between 0.0 and 1.0
# 1.0 corresponds to a free cell, 0.0 is an occupied cell
# rat = (row, col) initial rat position (defaults to (0, 0))
# target = (row, col), always be (row-1, col-1)


import numpy as np
from config import cfg


class Qmaze(object):
    def __init__(self, maze, rat=(0, 0)):
        self._maze = np.array(maze)
        nrows, ncols = self._maze.shape
        self.target = (nrows-1, ncols-1)
        self.free_cells = [(r, c) for r in range(nrows)
                           for c in range(ncols) if self._maze[r, c] == 1.0]
        self.free_cells.remove(self.target)
        if self._maze[self.target] == 0.0:
            raise Exception('Invalid maze, target cell cannot be blocked!')
        if not rat in self.free_cells:
            raise Exception('Invalid rat location: must sit on a free cell.')
        self.reset(rat)

    def reset(self, rat):
        '''
            Init/reset the system
        '''
        self.rat = rat
        self.maze = np.copy(self._maze)
        nrows, ncols = self.maze.shape
        row, col = rat
        self.maze[row, col] = cfg.rat_mark
        self.state = (row, col, 'start')
        self.min_reward = -0.5 * self.maze.size
        self.total_reward = 0
        self.visited = set()

    def act(self, action):
        self.update_state(action)
        reward = self.get_reward()

        self.total_reward += reward

        status = self.game_status()
        envstate = self.observe()

        return envstate, reward, status

    def update_state(self, action):
        '''
            Update value of state with new action
            input   action  : action will be performed
            output  None
        '''

        # Getting the current state
        nrows, ncols = self.maze.shape
        nrow, ncol, nmode = rat_row, rat_col, mode = self.state

        # Adding current position to visited cells.
        if self.maze[rat_row, rat_col] > 0.0:
            self.visited.add((rat_row, rat_col))

        # Get list of valid action
        valid_actions = self.valid_actions()

        if not valid_actions:
            nmode = 'blocked'
        elif action in valid_actions:
            nmode = 'valid'
            if action == cfg.LEFT:
                ncol -= 1
            elif action == cfg.UP:
                nrow -= 1
            elif action == cfg.RIGHT:
                ncol += 1
            elif action == cfg.DOWN:
                nrow += 1
            else:
                mode = 'invalid'

        self.state = (nrow, ncol, nmode)

    def get_reward(self):
        '''
            Calculate reward
            input None
            return reward (float)   : reward of agent
        '''
        rat_row, rat_col, mode = self.state
        nrows, ncols = self.maze.shape
        if rat_row == nrows - 1 and rat_col == ncols-1:
            return 1
        if mode == 'blocked':
            return self.min_reward - 1
        if (rat_row, rat_col) in self.visited:
            return -0.25
        if mode == 'invalid':
            return -0.75
        if mode == 'valid':
            return -0.04

    def game_status(self):
        '''
            Return the status of game at current state
            return status (string)  : status of game can be lose, win or not_over
        '''
        if self.total_reward < self.min_reward:
            return 'lose'
        rat_row, rat_col, mode = self.state
        nrows, ncols = self.maze.shape
        if rat_row == nrows-1 and rat_col == ncols-1:
            return 'win'

        return 'not_over'

    def observe(self):
        '''
            Getting the current state of the maze
            return envstate (list)  : current state of the maze
        '''
        canvas = self.draw_env()
        envstate = canvas.reshape((1, -1))
        return envstate

    def valid_actions(self, cell=None):
        '''
            A function to get all valid actions with current state of agent
            :input  cell    : postion of agent, default is None, means the current position
            :return actions : a list of valid actions can be performed 
        '''
        if cell is None:
            row, col, mode = self.state
        else:
            row, col = cell

        actions = [0, 1, 2, 3]  # left, up, right, down
        nrows, ncols = self.maze.shape
        if row == 0:  # Agent at first row, can not go up
            actions.remove(1)
        elif row == nrows - 1:  # Agent at last row, can not go down
            actions.remove(3)

        if col == 0:  # Agent at first column, can not go left
            actions.remove(0)
        elif col == ncols-1:  # Agent at last column can not go right
            actions.remove(2)

        # Other case when the next cell is occupied
        if row > 0 and self.maze[row-1, col] == 0.0:
            actions.remove(1)
        if row < nrows-1 and self.maze[row+1, col] == 0:
            actions.remove(3)

        if col > 0 and self.maze[row, col-1] == 0.0:
            actions.remove(0)
        if col < ncols-1 and self.maze[row, col+1] == 0.0:
            actions.remove(2)

        return actions

    def draw_env(self):
        canvas = np.copy(self.maze)
        nrows, ncols = self.maze.shape
        for r in range(nrows):
            for c in range(ncols):
                if canvas[r, c] > 0.0:
                    canvas[r, c] = 1.0
        row, col, valid = self.state
        canvas[row, col] = cfg. rat_mark
        return canvas