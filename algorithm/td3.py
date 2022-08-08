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


class TD3Actor(nn.Module):
    def __init__(self, args, act_space, obs_space, device=torch.device("cpu")):
        super(TD3Actor, self).__init__()
        self.act_hidden_size = args.act_hidden_size
        self.use_batch_normalization = args.use_batch_normalization
        self.tpdv = dict(dtype=torch.float32, device=device)
        self._input_dim = build_flattener(obs_space).size
        self._act_dim = act_space.shape[0]

        if self.use_batch_normalization:
            self.state_batch_norm = nn.BatchNorm1d(self._input_dim)
        
        # mlp modoule
        self.mlp_net = MLPBase(input_size=self._input_dim,
                               hidden_size=self.act_hidden_size,
                               use_batch_norm=self.use_batch_normalization)  
        # action module
        self.act_out = nn.Linear(self.mlp_net.output_dim, self._act_dim)

        self.to(device)

    def forward(self, x):
        x = check(x).to(**self.tpdv)
        
        if self.use_batch_normalization:
            x = self.state_batch_norm(x)

        x = self.mlp_net(x)
        actions = self.act_out(x).tanh()

        return actions


class TD3Critic(nn.Module):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super(TD3Critic, self).__init__()
        self.value_hidden_size = args.value_hidden_size
        self.use_batch_normalization = args.use_batch_normalization
        self.tpdv = dict(dtype=torch.float32, device=device)
        self._input_dim = build_flattener(obs_space).size + act_space.shape[0] 
        self._output_dim = 1

        if self.use_batch_normalization:
            self.state_batch_norm = nn.BatchNorm1d(self._input_dim)

        self.mlp1_net = MLPBase(self._input_dim, 
                               self.value_hidden_size,
                               self.use_batch_normalization)
        self.mlp2_net = MLPBase(self._input_dim, 
                               self.value_hidden_size,
                               self.use_batch_normalization)
        self.q1_net = nn.Linear(self.mlp1_net.output_dim, self._output_dim)
        self.q2_net = nn.Linear(self.mlp2_net.output_dim, self._output_dim)
        self.Q1 = nn.Sequential(self.mlp1_net, self.q1_net)
        self.Q2 = nn.Sequential(self.mlp2_net, self.q2_net)

        self.to(device)

    def forward(self, state, action):
        '''
        only calculate Q1 value
        '''
        state = check(state).to(**self.tpdv)
        action = check(action).to(**self.tpdv)

        x = torch.cat([state, action], dim=-1)
        values = self.Q1(x)

        return values

    def get_q1_q2_values(self, state, action):

        state = check(state).to(**self.tpdv)
        action = check(action).to(**self.tpdv)

        if self.use_batch_normalization:
            state = self.state_batch_norm(state)

        x = torch.cat([state, action], dim=-1)
        q1_values = self.Q1(x)
        q2_values = self.Q2(x)

        return q1_values, q2_values

    def get_min_q_values(self, state, action):

        return torch.min(*self.get_q1_q2_values(state, action))


class TD3Policy():
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        assert isinstance(act_space, gym.spaces.Box), "Only Solve Continuous Control Problem"
        self.args = args
        self.obs_space = obs_space
        self.act_space = act_space
        self.device = device
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.explore_noise = args.explore_noise
        self._tau = args.tau
        self.actor = TD3Actor(args, act_space, obs_space, device)
        self.critic = TD3Critic(args, obs_space, act_space, device)
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr)

    @property
    def act_size(self):
        return build_flattener(self.act_space).size

    @property
    def obs_size(self):
        return build_flattener(self.obs_space).size

    @property
    def name(self):
        return 'td3'

    def get_actions(self, obs):
        '''
        obtain actions with epsilon-greedy (used for collecting samples)
        '''

        actions = self.actor(obs)

        # noises = torch.normal(0, self.explore_noise, size=tuple(actions.shape))
        noises = torch.randn_like(actions) * self.explore_noise

        actions = (actions + noises).clamp(-1, 1).flatten()

        return actions

    def act(self, obs):
        '''
        obtain actions with deterministic policy (used for evaluating policy)
        '''
        actions = self.actor(obs).flatten()

        return actions

    def actor_soft_update(self):
        for target, eval in zip(self.actor_target.parameters(), self.actor.parameters()):
            target.data.copy_((1-self._tau) * target.data +
                              self._tau * eval.data)

    def critic_soft_update(self):
        for target, eval in zip(self.critic_target.parameters(), self.critic.parameters()):
            target.data.copy_((1-self._tau) * target.data +
                              self._tau * eval.data)
                              

    def save_param(self, save_path):
        '''
        save parameters in actor network and critic network
        '''

        state_dict = OrderedDict()
        state_dict['actor'] = self.actor.state_dict()
        state_dict['critic'] = self.critic.state_dict()

        torch.save(state_dict, save_path + time.strftime('%m_%d_%H_%M',
                   time.localtime(time.time())) + '.pkl')

    def load_param(self, env_name, save_time):
        '''
        load parameters in actor and critic network
        '''

        load_path = './model_save/td3/' + \
            str(env_name) + '/' + save_time + '.pkl'
        state_dict = torch.load(load_path)
        actor_state_dict = state_dict['actor']
        critic_state_dict = state_dict['critic']
        self.actor.load_state_dict(actor_state_dict)
        self.critic.load_state_dict(critic_state_dict)
