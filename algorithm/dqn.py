from re import M
from typing import OrderedDict
import torch
from torch import nn
import numpy as np
from utils.nn import *
from utils.flatten import build_flattener
import time
from copy import deepcopy


def check(input):
    input = torch.from_numpy(input) if type(input) == np.ndarray else input
    return input


class DQNPolicy():
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        self.args = args
        self.obs_space = obs_space
        self.act_space = act_space
        self.device = device
        self.lr = args.lr
        self._tau = args.tau
        self.epsilon = args.epsilon
        self.q_net = QNet(args, act_space, obs_space, device)
        self.q_net_target = deepcopy(self.q_net)
        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(), lr=self.lr)

    @property
    def obs_size(self):
        return build_flattener(self.obs_space).size

    @property
    def act_size(self):
        return build_flattener(self.act_space).size

    @property
    def name(self):
        return 'dqn'

    def get_actions(self, obs):
        '''
        obtain actions with epsilon-greedy (used for collecting samples)

        '''
        max_actions = self.q_net.get_max_actions(obs)
        if np.random.rand() >= self.epsilon:
            actions = np.random.choice(range(self.act_space.n), 1)
        else:
            actions = max_actions

        return actions.flatten()

    def act(self, obs):
        '''
        obtain actions with deterministic policy (used for evaluating policy)
        '''

        actions = self.q_net.get_max_actions(obs)

        return actions.flatten()

    def get_values(self, obs, actions):
        '''
        obtain Q values of (state, action)
        '''

        q_values_pred = self.q_net.get_values(obs, actions)

        return q_values_pred

    def soft_update(self):
        '''
        soft update, modify parameters slowly
        '''
        for target, eval in zip(self.q_net_target.parameters(), self.q_net.parameters()):
            target.data.copy_((1-self._tau) * target.data +
                              self._tau * eval.data)

    def hard_update(self):
        '''
        hard update, copy parameters directly
        '''
        self.q_net_target.load_state_dict(self.q_net.state_dict())

    def save_param(self, save_path):
        '''
        save parameters in Q value network
        '''

        torch.save(self.q_net.state_dict(), save_path + time.strftime('%m_%d_%H_%M',
                   time.localtime(time.time())) + '.pkl')

    def load_param(self, env_name, save_time):
        '''
        load parameters in Q value network
        '''

        load_path = './model_save/' + str(self.name) + '/' + \
            str(env_name) + '/' + save_time + '.pkl'
        state_dict = torch.load(load_path)

        self.q_net.load_state_dict(state_dict)


class QNet(nn.Module):
    def __init__(self, args, act_space, obs_space, device=torch.device("cpu")):
        super(QNet, self).__init__()
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

        self.q_out = nn.Linear(self.mlp_net.output_dim, self._act_dim)

    def forward(self, x):
        x = check(x).to(**self.tpdv)
        if self.use_batch_normalization:
            x = self.state_batch_norm(x)
        x = self.mlp_net(x)
        q_values = self.q_out(x)

        return q_values

    def get_values(self, obs, actions):
        '''
        get Q values of corresponding actions
        '''
        q_values = self(obs)

        q_values_pred = q_values.gather(1, actions.long())

        return q_values_pred

    def get_max_actions(self, obs):
        '''
        get action with maximun Q values
        '''

        q_values = self(obs)
        actions = torch.max(q_values, dim=-1, keepdim=True)[1]

        return actions

