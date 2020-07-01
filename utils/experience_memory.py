from collections import namedtuple
import random

Experience = namedtuple(
    'Experience', ['obs', 'action', 'reward', 'next_obs', 'done'])


class ExperienceMemroy:
    def __init__(self):
        pass
