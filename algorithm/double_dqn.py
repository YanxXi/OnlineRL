import torch
import numpy as np
from utils.nn import *
from .dqn import DQNPolicy


def check(input):
    input = torch.from_numpy(input) if type(input) == np.ndarray else input
    return input


class DoubleDQNPolicy(DQNPolicy):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super().__init__(args, obs_space, act_space, device)

    @property
    def name(self):
        return 'double_dqn'

    
