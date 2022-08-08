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

class SACActor(nn.Module):
    '''
    Actor of both continuous and discrete policy
    '''
    def __init__(self, args, act_space, obs_space, device=torch.device("cpu")):
        super(SACActor, self).__init__()
        self._continuous = False
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
            self.act_log_std_out = nn.Linear(self.mlp_net.output_dim, self._act_dim)
        elif isinstance(act_space, gym.spaces.Discrete):
            self._act_dim = act_space.n
            self.act_logits = nn.Linear(self.mlp_net.output_dim, self._act_dim)
        else:
            raise NotImplementedError

        self.to(device)

    def forward(self, obs):
        '''
        Return the distribution of the policy, either Normal or Categorial
        '''
        obs = check(obs).to(**self.tpdv)
        obs = self.mlp_net(obs)
        if self._continuous:
            mean = self.act_mean_out(obs)
            log_std = self.act_log_std_out(obs)
            std = log_std.exp()
            dist = torch.distributions.normal.Normal(mean, std)
        else:
            action_logits = self.act_logits(obs)
            dist = torch.distributions.Categorical(logits=action_logits)

        return dist

    def get_action(self, obs, deterministic=False):
        '''
        obtain action given state accroding to policy

        Args:
            obs: state or observation
            deterministic: choose exploratory action or deterministic action 

        '''
        dist = self(obs)

        if deterministic:
            if self._continuous:
                actions = dist.mean.tanh()
            else:
                actions = dist.probs.argmax(dim=-1, keepdim=True)
        else:
            if self._continuous:
                actions = dist.sample().tanh()
            else:
                actions = dist.sample()

        return actions
        
    def get_action_log_prob(self, obs):
        '''
        get actions and log probs for continuous policy 
        get prob and log probs for discrete policy
        '''
        dist = self(obs)
        if self._continuous:
            # Reparameterization
            reparam_actions = dist.rsample()
            actions = reparam_actions.tanh()
            log_probs = dist.log_prob(reparam_actions)
            log_probs = log_probs - (1 + 1e-6 - actions.pow(2)).log().sum(dim=1, keepdim=True)
            return actions, log_probs
        else:
            probs = dist.probs
            log_probs = dist.logits
            return probs, log_probs

class SACCritic(nn.Module):
    '''
    Actor of both continuous and discrete policy
    '''
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super(SACCritic, self).__init__()
        self._continuous = False
        self.value_hidden_size = args.value_hidden_size
        self.use_batch_normalization = args.use_batch_normalization
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.obs_flattener = build_flattener(obs_space)
        self.act_flattener = build_flattener(act_space)

        if isinstance(act_space, gym.spaces.Box):
            self._continuous = True
            self._input_dim = self.obs_flattener.size  + self.act_flattener.size
            self._output_dim = 1
        elif isinstance(act_space, gym.spaces.Discrete):
            self._input_dim = self.obs_flattener.size
            self._output_dim = act_space.n
        else:
            raise NotImplementedError
        
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

        if self._continuous:
            x = torch.cat([state, action], dim=-1)
            values = self.Q1(x)
        else:
            values = self.Q1(state)

        return values

    def get_q1_q2_values(self, state, action=None):
        '''
        obtain values of two Q networks
        Args:
            action: if action is None, return the q values of all actions of state,
                    else return the q values of (state, action)
        '''

        state = check(state).to(**self.tpdv)
        action = check(action).to(**self.tpdv) if action is not None else None

        if self._continuous:
            x = torch.cat([state, action], dim=-1)
            q1_values = self.Q1(x)
            q2_values = self.Q2(x)
        else:
            # obtain q values of all actions
            q1_values = self.Q1(state)
            q2_values = self.Q2(state)
            # obtain q values of sepcific actions
            if action != None:
                q1_values = q1_values.gather(1, action.long())
                q2_values = q2_values.gather(1, action.long()) 

        return q1_values, q2_values

    def get_min_q_values(self, state, action=None):
        '''
        obtain the minimum q values
        '''

        return torch.min(*self.get_q1_q2_values(state, action))


class SACPolicy():
    '''
    SAC policy for both discrete and continuous
    '''
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        self.args = args
        self.obs_space = obs_space
        self.act_space = act_space
        self.act_size = build_flattener(self.act_space).size
        self.obs_size = build_flattener(self.obs_space).size
        self.device = device
        self.actor_lr = args.lr
        self.critic_lr = args.lr
        self.alpha_lr = args.lr
        self._tau = args.tau
        self.init_alpha = args.init_alpha
        # log temperature parameters
        self.alpha_log = torch.tensor(np.log(self.init_alpha),
                                      dtype=torch.float32,
                                      requires_grad=True,
                                      device=device)
        self.entropy_tgt = torch.tensor(-self.act_size, 
                                        dtype=torch.int32,
                                        requires_grad=False,
                                        device=device)                                    
        self.actor = SACActor(args, act_space, obs_space, device)
        self.critic = SACCritic(args, obs_space, act_space, device)
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr)
        self.alpha_optimizer = torch.optim.Adam(
            (self.alpha_log, ), lr=self.alpha_lr)

    @property
    def name(self):
        return 'sac'
    
    @property
    def continuous_action(self):
        return self.actor._continuous

    def get_actions(self, obs):
        '''
        obtain actions with explore policy
        '''

        actions = self.actor.get_action(obs)

        return actions.flatten()

    def act(self, obs):
        '''
        obtain actions with deterministic policy (used for evaluating policy)
        '''
        actions = self.actor.get_action(obs, deterministic=True)

        return actions.flatten()

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

        load_path = './model_save/sac/' + \
            str(env_name) + '/' + save_time + '.pkl'
        state_dict = torch.load(load_path)
        actor_state_dict = state_dict['actor']
        critic_state_dict = state_dict['critic']
        self.actor.load_state_dict(actor_state_dict)
        self.critic.load_state_dict(critic_state_dict)

