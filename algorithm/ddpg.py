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


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class DDPGActor(nn.Module):
    def __init__(self, args, act_space, obs_space, device=torch.device("cpu")):
        super(DDPGActor, self).__init__()
        self.act_hidden_size = args.act_hidden_size
        self.use_batch_normalization = args.use_batch_normalization
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.obs_flattener = build_flattener(obs_space)
        self._input_dim = self.obs_flattener.size
        self._act_dim = act_space.shape[0]

        if self.use_batch_normalization:
            self.state_batch_norm = nn.BatchNorm1d(self._input_dim)

        self.mlp_net = MLPBase(self._input_dim,
                               self.act_hidden_size,
                               self.use_batch_normalization)
        # action module
        self.act_out = nn.Linear(self.mlp_net.output_dim, self._act_dim)

        self.init_parameters()

        self.to(device)

    def forward(self, x):
        x = check(x).to(**self.tpdv)

        if self.use_batch_normalization:
            x = self.state_batch_norm(x)

        x = self.mlp_net(x)
        actions = self.act_out(x).tanh()

        return actions

    def init_parameters(self):
        if self.use_batch_normalization:
            self.mlp_net._modules[0].weight.data.uniform_(
                *hidden_init(self.func1._modules[0]))
            self.func2._modules[3].weight.data.uniform_(
                *hidden_init(self.func2._modules[3]))
            self.act_out.weight.data.uniform_(-3e-3, 3e-3)
        else:
            self.mlp_net._modules['fc']._modules['0'].weight.data.uniform_(
                *hidden_init(self.mlp_net._modules['fc']._modules['0']))
            self.mlp_net._modules['fc']._modules['2'].weight.data.uniform_(
                *hidden_init(self.mlp_net._modules['fc']._modules['2']))
            self.act_out.weight.data.uniform_(-3e-3, 3e-3)


class DDPGCritic(nn.Module):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super(DDPGCritic, self).__init__()
        self.value_hidden_size = args.value_hidden_size
        self.use_batch_normalization = args.use_batch_normalization
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.obs_flattener = build_flattener(obs_space)
        self.act_flattener = build_flattener(act_space)
        self._input_dim = self.obs_flattener.size
        self._act_dim = self.act_flattener.size
        self._hidden_size = list(map(int, self.value_hidden_size.split(' ')))
        self.acti_func = nn.ReLU()

        if self.use_batch_normalization:
            self.state_batch_norm = nn.BatchNorm1d(self._input_dim)
            self.mlp1_net = nn.Sequential(
                nn.Linear(self._input_dim, self._hidden_size[0]),
                self.acti_func,
                nn.BatchNorm1d(self._hidden_size[0])
            )
        else:
            self.mlp1_net = nn.Sequential(
                nn.Linear(self._input_dim, self._hidden_size[0]),
                self.acti_func
            )

        self.mlp2_net = nn.Sequential(
            nn.Linear(self._act_dim + self._hidden_size[0], self._hidden_size[1]),
            self.acti_func
        )

        # value module
        self.value_out = nn.Linear(self._hidden_size[-1], 1)

        self.init_parameters()

        self.to(device)

    def forward(self, state, action):
        state = check(state).to(**self.tpdv)
        action = check(action).to(**self.tpdv)

        if self.use_batch_normalization:
            state = self.state_batch_norm(state)

        state = self.mlp1_net(state)
        x = torch.cat([state, action], dim=-1)
        x = self.mlp2_net(x)
        values = self.value_out(x)

        return values

    def init_parameters(self):
        self.mlp1_net._modules['0'].weight.data.uniform_(
            *hidden_init(self.mlp1_net._modules['0']))
        self.mlp2_net._modules['0'].weight.data.uniform_(
            *hidden_init(self.mlp2_net._modules['0']))
        self.value_out.weight.data.uniform_(-3e-3, 3e-3)


class DDPGPolicy():
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        self.args = args
        self.obs_space = obs_space
        self.act_space = act_space
        self.device = device
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self._tau = args.tau
        # explore noise of action (OrnsteinUhlenbeckNoise)
        self.epsilon = args.epsilon
        self.explore_noise = args.explore_noise
        self.actor = DDPGActor(args, act_space, obs_space, device)
        self.critic = DDPGCritic(args, obs_space, act_space, device)
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr)
        self.ou_noise = OrnsteinUhlenbeckNoise(
            self.act_space, sigma=self.explore_noise)

    @property
    def act_size(self):
        return build_flattener(self.act_space).size

    @property
    def obs_size(self):
        return build_flattener(self.obs_space).size


    @property
    def name(self):
        return 'ddpg'

    def get_actions(self, obs):
        '''
        obtain actions with epsilon-greedy (used for collecting samples)
        '''

        actions = self.actor(obs)
        if np.random.rand() < self.epsilon:
            ou_noises = torch.as_tensor(
                self.ou_noise(), dtype=torch.float32, device=self.device).unsqueeze(0)
            actions = (actions + ou_noises).clamp(-1, 1)

        return actions.flatten()

    def act(self, obs):
        '''
        obtain actions with deterministic policy (used for evaluating policy)
        '''
        actions = self.actor(obs).flatten()

        return actions

    def get_values(self, obs, action):
        '''
        obtain predicetd value (used for collecting sample)
        '''

        values = self.critic(obs, action)

        return values

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

        load_path = './model_save/ddpg/' + \
            str(env_name) + '/' + save_time + '.pkl'
        state_dict = torch.load(load_path)
        actor_state_dict = state_dict['actor']
        critic_state_dict = state_dict['critic']
        self.actor.load_state_dict(actor_state_dict)
        self.critic.load_state_dict(critic_state_dict)


class OrnsteinUhlenbeckNoise:  # NOT suggest to use it
    def __init__(self, act_space, theta=0.15, sigma=0.2, ou_noise=0.0, dt=1e-2):
        """
        The noise of Ornstein-Uhlenbeck Process
        It makes Zero-mean Gaussian Noise more stable.
        It helps agent explore better in a inertial system.
        Don't abuse OU Process. OU process has too much hyper-parameters and over fine-tuning make no sense.
        :int size: the size of noise, noise.shape==(-1, action_dim)
        :float theta: related to the not independent of OU-noise
        :float sigma: related to action noise std
        :float ou_noise: initialize OU-noise
        :float dt: derivative
        """
        self.theta = theta
        self.sigma = sigma
        self.ou_noise = ou_noise
        self.dt = dt
        act_flattener = build_flattener(act_space)
        self.size = act_flattener.size

    def __call__(self) -> float:
        """
        output a OU-noise
        :return array ou_noise: a noise generated by Ornstein-Uhlenbeck Process
        """
        noise = self.sigma * np.sqrt(self.dt) * \
            np.random.normal(size=self.size)
        self.ou_noise -= self.theta * self.ou_noise * self.dt + noise
        return self.ou_noise
