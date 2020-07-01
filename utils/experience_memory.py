from collections import namedtuple
import random

Experience = namedtuple(
    'Experience', ['obs', 'action', 'reward', 'next_obs', 'done'])


class ExperienceMemory:
    """A cyclic buffer based Experience Memory implementation"""

    def __init__(self, capacity=int(1e6)):
        """
            :param capacity: Total capacity (Max number of Experiences)
            :return :
            """
        self.capacity = capacity
        self.mem_idx = 0
        self.memory = []

    def store(self, experience):
        """Adding new experience to memory
        :param experience: The Experience object to be stored
        : return :
        """
        self.memory.insert(self.mem_idx % self.capacity, experience)
        self.mem_idx += 1

    def sample(self, batch_size):
        """ Get a batch of experiences from memory
        :param batch_size: Sample batch_size
        :return: A list of batch_size number of Experiences sampled at random from mem 
        """
        assert batch_size <= len(
            self.memory), "Sample batch_size is more than the number of available experience in memory"
        return random.sample(self.memory, batch_size)

    def get_size(self):
        """Helper method to find out how many experiences are already stored in the memory
        :return: Number of experiences in memory
        """
        return len(self.memory)
