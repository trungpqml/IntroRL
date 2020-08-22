from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import MSE
import tensorflow as tf
from config import cfg


def build_model(maze, lr=1e-3):
    model = Sequential([
        Dense(maze.size, input_shape=(maze.size,), activation='relu'),
        Dense(maze.size, activation='relu'),
        Dense(cfg.num_actions)
    ])

    model.compile(
        optimizer='adam',
        loss='mse',
    )

    return model


if __name__ == "__main__":
    model = build_model(cfg.maze)
    print(model.summary())
