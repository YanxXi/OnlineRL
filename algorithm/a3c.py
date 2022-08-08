from dis import dis
from typing import OrderedDict
import torch
from torch import nn
import numpy as np
from utils.nn import *
from utils.flatten import build_flattener
import time
from ..utils.optimizer import SharedAdam

def check(input):
    input = torch.from_numpy(input) if type(input) == np.ndarray else input
    return input

class A3CActor(nn.Module):
    def __init__(self, args, act_space, obs_space, device=torch.device("cpu")):
        super(A3CActor, self).__init__()
        self.act_hidden_size = args.act_hidden_size
        self.use_batch_normalization = args.use_batch_normalization
        self.tpdv = dict(dtype=torch.float32, device=device)
        self._input_dim = build_flattener(obs_space).size

        # mlp modoule
        self.mlp_net = MLPBase(self._input_dim,
                               self.act_hidden_size,
                               self.use_batch_normalization)
        # action module
        if isinstance(act_space, gym.spaces.Box):
            self._continuous = True
            self._act_dim = act_space.shape[0]
            self.act_mean_out = nn.Linear(self.mlp_net.output_dim, self._act_dim)
            self.act_std_out = nn.Sequential(
                nn.Linear(self._hidden_size[-1], self._act_dim),
                nn.Softplus(self._act_dim))
        elif isinstance(act_space, gym.spaces.Discrete):
            self._act_dim = act_space.n
            self.act_logits = nn.Linear(self.mlp_net.output_dim, self._act_dim)
        else:
            raise NotImplementedError

        self.to(device)
     

    def forward(self, obs):
        '''
        output the distribution not specific value

        Args:
            obs: observation or state
        
        Returns:
            Normal or Categotial
        '''
        obs = check(obs).to(**self.tpdv)
        obs = self.mlp_net(obs)
        if self._continuous:
            action_mean = self.act_mean(obs)
            action_std = self.act_std(obs)
            dist = torch.distributions.Normal(action_mean, action_std)
        else:
            action_logits = self.act_logits(obs)
            dist = torch.distributions.Categorical(action_logits)

        return dist

    def get_actions(self, obs, deterministic=False):
        dist = self(obs)

        if deterministic:
            if self._continuous:
                actions = dist.mean
            else:
                actions = dist.probs.argmax(dim=-1, keepdim=True)
        else:
            if self._continuous:
                actions = dist.sample().tanh()
            else:
                actions = dist.sample()

        return actions
        
        
    def get_log_probs(self, obs, actions):
        dist = self(obs)

        if self._continuous:
            log_probs = dist.log_probs(actions)
        else:
            log_probs = dist.logits.gather(1, actions.long())

        return log_probs


class A3CCritic(nn.Module):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super(A3CCritic, self).__init__()
        self.value_hidden_size = args.value_hidden_size
        self.use_batch_normalization = args.use_batch_normalization
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.obs_flattener = build_flattener(obs_space)
        self._input_dim = self.obs_flattener.size

        # mlp modoule
        self.mlp_net = MLPBase(self._input_dim,
                               self.value_hidden_size,
                               self.use_batch_normalization)
        # value module
        self.value_net = nn.Linear(self.mlp_net.output_dim, 1)
    
        self.to(device)

    def forward(self, state):
        '''
        only calculate Q1 value
        '''
        state = check(state).to(**self.tpdv)

        state = self.mlp_net(state)

        values = self.value_net(state)

        return values


class A3CPolicy():
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        self.args = args
        self.obs_space = obs_space
        self.act_space = act_space
        self.act_size = build_flattener(self.act_space).size
        self.obs_size = build_flattener(self.obs_space).size
        self.device = device
        self.actor_lr = args.lr
        self.critic_lr = args.lr                                  
        self.actor = A3CActor(args, act_space, obs_space, device)
        self.critic = A3CCritic(args, obs_space, act_space, device)
        self.actor_optimizer = SharedAdam(
            self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = SharedAdam(
            self.critic.parameters(), lr=self.critic_lr)

    @property
    def name(self):
        return 'a3c'

    def get_actions(self, obs):
        '''
        obtain actions with epsilon-greedy (used for collecting samples)
        '''

        actions = self.actor.get_actions(obs)

        return actions.flatten()

    def act(self, obs):
        '''
        obtain actions with deterministic policy (used for evaluating policy)
        '''
        actions = self.actor.get_actions(obs, deterministic=True)

        return actions.flatten()                              

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

        load_path = './model_save/a3c/' + \
            str(env_name) + '/' + save_time + '.pkl'
        state_dict = torch.load(load_path)
        actor_state_dict = state_dict['actor']
        critic_state_dict = state_dict['critic']
        self.actor.load_state_dict(actor_state_dict)
        self.critic.load_state_dict(critic_state_dict)

