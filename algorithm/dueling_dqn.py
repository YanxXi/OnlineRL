from copy import deepcopy
import torch
from torch import nn
import numpy as np
from utils.nn import *
from utils.flatten import build_flattener
from .dqn import DQNPolicy


def check(input):
    input = torch.from_numpy(input) if type(input) == np.ndarray else input
    return input


class DuelingDQNPolicy(DQNPolicy):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        self.args = args
        self.obs_space = obs_space
        self.act_space = act_space
        self.device = device
        self.lr = args.lr
        self._tau = args.tau
        self.epsilon = args.epsilon
        self.q_net = DuelingQNet(args, act_space, obs_space, device)
        self.q_net_target = deepcopy(self.q_net)
        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(), lr=self.lr)

    @property
    def name(self):
        return 'dueling_dqn'

class DuelingQNet(nn.Module):
    def __init__(self, args, act_space, obs_space, device=torch.device("cpu")):
        super(DuelingQNet, self).__init__()
        self.hidden_size = args.hidden_size
        self.use_batch_normalization = args.use_batch_normalization
        self.tpdv = dict(dtype=torch.float32, device=device)
        self._input_dim = build_flattener(obs_space).size
        self._act_dim = act_space.n
        if self.use_batch_normalization:
            self.state_batch_norm = nn.BatchNorm1d(self._input_dim)
        self.mlp_net = MLPBase(self._input_dim,
                               self.hidden_size,
                               self.use_batch_normalization)
        self.value_out  = nn.Linear(self.mlp_net.output_dim, 1)
        self.advantage_out = nn.Linear(self.mlp_net.output_dim, self._act_dim)

    def forward(self, x):
        x = check(x).to(**self.tpdv)
        if self.use_batch_normalization:
            x = self.state_batch_norm(x)
        x = self.mlp_net(x)
        s_values = self.value_out(x)
        advantages = self.advantage_out(x)

        return s_values, advantages

    def get_values(self, obs, actions):
        '''
        get Q values of corresponding actions
        '''
        s_values, advantages = self(obs)
        q_values = s_values + advantages - advantages.mean(dim=-1, keepdim=True)

        q_values_pred = q_values.gather(1, actions.long())

        return q_values_pred
    
    def get_max_q_values(self, obs):

        s_values, advantages = self(obs)
        q_values = s_values + advantages - advantages.mean(dim=-1, keepdim=True)
        max_q_values = q_values.max(dim=-1, keepdim=True)[0]

        return max_q_values

    def get_max_actions(self, obs):
        '''
        get action with maximum Q values(equals to maximum advantages)
        '''

        _, advantages = self(obs)
        actions = torch.max(advantages, dim=-1, keepdim=True)[1]

        return actions

