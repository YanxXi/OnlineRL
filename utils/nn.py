import torch
from torch import nn
from torch.functional import Tensor
import torch.nn.functional as F

# import sys
# print(sys.path)

from utils.flatten import build_flattener
from utils.distribution import Categorial, Normal
import gym.spaces


class MLPObs(nn.Module):
    '''
    MLP layer for extracting the feature of obervations
    '''
    def __init__(self, obs_space, hidden_size: str):
        super(MLPObs, self).__init__()
        self.obs_flattener = build_flattener(obs_space)
        self._input_dim = self.obs_flattener.size
        self._hidden_size = [self._input_dim] + list(map(int, hidden_size.split(' ')))
        self.act_func = nn.ReLU()
        self.feature_norm = nn.LayerNorm(self._input_dim)
        fc_h = []
        for i in range(len(self._hidden_size) - 1):
            fc_h += [nn.Linear(self._hidden_size[i], self._hidden_size[i+1]), self.act_func, nn.LayerNorm(self._hidden_size[i+1])]
        self.fc = nn.Sequential(*fc_h)


    def forward(self, x):
        # x = self.feature_norm(x)
        return self.fc(x)


    @property
    def output_dim(self):
        return self._hidden_size[-1]



class MLPBase(nn.Module):
    '''
    base mlp layer
    '''
    def __init__(self, input_size: int, hidden_size: str, use_batch_norm: bool):
        super(MLPBase, self).__init__()
      
        self._hidden_size = [input_size] + list(map(int, hidden_size.split(' ')))
        self.act_func = nn.ReLU()
        fc_h = []
        if use_batch_norm:
            for i in range(len(self._hidden_size) - 1):
                fc_h += [nn.Linear(self._hidden_size[i], self._hidden_size[i+1]),
                        self.act_func,
                        nn.BatchNorm1d(self._hidden_size[i+1])]
        else:
            for i in range(len(self._hidden_size) - 1):
                fc_h += [nn.Linear(self._hidden_size[i], self._hidden_size[i+1]),
                        self.act_func]

        self.fc = nn.Sequential(*fc_h)

    def forward(self, x):

        return self.fc(x)


    @property
    def output_dim(self):
        return self._hidden_size[-1]


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRU, self).__init__()
        self._hidden_size = hidden_size
        self._hidden_layers = num_layers
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)

        # N: batch size
        # L: sequence length
        # num_layer: number of layers
        # input_size: size of input vector
        # input:  x: [L, N, input_size], h_0: [num_layers, N, hidden_size]
        # output: x: (L, N, hidden_size), h_n: [num_layers, N, hidden_size]


    def forward(self, x: F.Tensor, h_0: F.Tensor):
        # L = 0 x: [N, input_size], h_0: [N, num_layers, hidden_size] for decision
        if x.size(0) == h_0.size(0):
            x = x.unsqueeze(0)
            x, h_n = self.gru(x, h_0.transpose(0, 1).contiguous())
            x = x.squeeze(0)
            h_n = h_n.transpose(0, 1).contiguous()
        # x: [L*N, input_size], h_0: [N, num_layers, hidden_size] for training
        else:
            N = h_0.size(0)
            L = x.size(0) // N
            x = x.view(L, N, -1)
            x, h_n = self.gru(x, h_0.transpose(0, 1).contiguous())
            x = x.view(L*N, -1)
            h_n = h_n.transpose(0, 1).contiguous()

        x = self.norm(x)
        
        return x, h_n


    @property
    def output_dim(self):
        return self._hidden_size


class ActLayer(nn.Module):
    def __init__(self, act_space, input_size, hidden_size):
        super(ActLayer, self).__init__()
        self._continuous_action = False
        self._multidiscrete_action = False
        self.use_mlp_act = False

        if len(hidden_size) > 0:
            self.use_mlp_act = True
            self.mlp = MLPBase(input_size, hidden_size)
            input_size = self.mlp.output_dim

        if isinstance(act_space, gym.spaces.Discrete):
            action_dim = act_space.n
            self.action_out = Categorial(input_size, action_dim)
        elif isinstance(act_space, gym.spaces.Box):
            self._continuous_action = True
            action_dim = act_space.shape[0]
            self.action_out = Normal(input_size, action_dim)
        elif isinstance(act_space, gym.spaces.MultiDiscrete):
            self._multidiscrete_action = True
            actin_outs = []
            action_dims = act_space.nvec
            for action_dim in action_dims:
                actin_outs.append(Categorial(input_size, action_dim))
            self.action_outs = nn.ModuleList(actin_outs)
        else:
            raise NotImplementedError


    def forward(self, x, deterministic=False):
        if self.use_mlp_act:
            x = self.mlp(x)
        if self._multidiscrete_action:
            actions = []
            action_log_probs = []
            for action_out in self.action_outs:
                action_dist = action_out(x)
                action = action_dist.deterministic if deterministic else  action_dist.sample()
                action_log_porb = action_dist.log_prob(action)
                actions.append(action)
                action_log_probs.append(action_log_porb)

            actions = torch.cat(actions, dim=-1)
            action_log_probs = torch.cat(action_log_probs, dim=-1)
        elif self._continuous_action:
            action_dist = self.action_out(x)
            actions = action_dist.deterministic if deterministic else  action_dist.sample()
            action_log_probs = action_dist.log_prob(actions)
        else:
            action_dist = self.action_out(x)
            actions = action_dist.deterministic if deterministic else  action_dist.sample()
            action_log_probs = action_dist.log_prob(actions)

        return actions, action_log_probs


    def eval_actions(self, x, actions):
        if self.use_mlp_act:
            x = self.mlp(x)
        if self._multidiscrete_action:
            actions = torch.transpose(actions, 2, 0, 1)
            action_log_probs = []
            dist_entropy = []
            for action_out, action in zip(self.action_outs, actions):
                action_dist = action_out(x)
                action_log_porb = action_dist.log_prob(action)
                actions.append(action)
                action_log_probs.append(action_log_porb)
                dist_entropy.append(action_dist.entropy().mean(-1, keepdim=True))
            action_log_probs = torch.cat(action_log_probs, dim=-1)
            dist_entropy = torch.cat(dist_entropy, dim=-1)
        elif self._continuous_action:
            action_dist = self.action_out(x)
            action_log_probs = action_dist.log_prob(actions)
            dist_entropy = action_dist.entropy().mean(-1, keepdim=True)
        else:
            action_dist = self.action_out(x)
            action_log_probs = action_dist.log_prob(actions.flatten()).unsqueeze(-1)
            dist_entropy = action_dist.entropy()

        return action_log_probs, dist_entropy


    def get_probs(self, x):
        if self._mlp_actlayer:
            x = self.mlp(x)
        if self._multidiscrete_action:
            action_probs = []
            for action_out in self.action_outs:
                action_dist = action_out(x)
                action_prob = action_dist.probs
                action_probs.append(action_prob)
            action_probs = torch.cat(action_probs, dim=-1)
        elif self._continuous_action:
            assert ValueError("Normal distribution has no `probs` attribute!")
        else:
            action_dists = self.action_out(x)
            action_probs = action_dists.probs
        return action_probs
