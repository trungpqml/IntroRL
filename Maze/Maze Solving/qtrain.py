from helpers import format_time
from config import cfg
import datetime


def qtrain(model, maze, **opt):
    n_epoch = opt.get('n_epoch', cfg.n_epoch)
    max_memory = opt.get('max_memory', cfg.max_memory)
    data_size = opt.get('data_size', cfg.data_size)
    weights_file = opt.get('weights_file', cfg.weights_file)
    name = opt.get('name', 'model')
    start_time = datetime.datetime.now()
