import gym
import sys
from gym.spaces import *
from config import cfg

def run_gym_env(env):
    env.reset()
    for _ in range(50):
        env.render()
        env.step(env.action_space.sample())
    env.close()


def print_spaces(space):
    print("\t", space)
    if isinstance(space, Box):
        print(f"\t\tspace.low: {space.low} of type {type(space.low)}")
        print(f"\t\tspace.high: {space.high} of type {type(space.high)}")


def print_observation(env):
    print("Observation Space:")
    print_spaces(env.observation_space)
    print("Action Space:")
    print_spaces(env.action_space)
    try:
        print(
            f"\t\tAction description/meaning: {env.unwrapped.get_action_meaning()}")
    except AttributeError:
        pass


def overview(env):
    print_observation(env)
    run_gym_env(env)


if __name__ == "__main__":
    env = gym.make(cfg.env_name)
    overview(env)
