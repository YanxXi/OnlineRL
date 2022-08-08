from typing import OrderedDict
import torch
from torch import nn
from torch.functional import Tensor
import numpy as np
from utils.nn import *
from utils.flatten import build_flattener
import time


def check(input):
    input = torch.from_numpy(input) if type(input) == np.ndarray else input
    return input


class PPOActor(nn.Module):
    def __init__(self, args, act_space, obs_space, device=torch.device("cpu")):
        super(PPOActor, self).__init__()
        self.mlp_hidden_size = args.mlp_hidden_size
        self.act_hidden_size = args.act_hidden_size
        self.use_recurrent_layer = args.use_recurrent_layer
        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.recurrent_hidden_layers = args.recurrent_hidden_layers
        self.use_mlp_layer = False
        self.tpdv = dict(dtype=torch.float32, device=device)

        # pre-process module
        if len(self.mlp_hidden_size) > 0:
            self.use_mlp_layer = True
            self.mlp_layer = MLPObs(obs_space, self.mlp_hidden_size)
            input_size = self.mlp_layer.output_dim
        else:
            input_size = build_flattener(obs_space).size
        # rnn module
        if self.use_recurrent_layer:
            self.gru_layer = GRU(
                input_size, self.recurrent_hidden_size, self.recurrent_hidden_layers)
            input_size = self.gru_layer.output_dim
        # action module
        self.act_layer = ActLayer(act_space, input_size, self.act_hidden_size)

        self.to(device)

    def forward(self, x, rnn_states, deterministic=False):
        x = check(x).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        if self.use_mlp_layer:
            x = self.mlp_layer(x)
        if self.use_recurrent_layer:
            x, rnn_states = self.gru_layer(x, rnn_states)
        actions, action_log_probs = self.act_layer(x, deterministic)

        return actions, action_log_probs, rnn_states

    def eval_actions(self, x, rnn_states, actions):
        x = check(x).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        if self.use_mlp_layer:
            x = self.mlp_layer(x)

        if self.use_recurrent_layer:
            x, rnn_states = self.gru_layer(x, rnn_states)
        action_log_probs, dist_entropy = self.act_layer.eval_actions(
            x, actions)

        return action_log_probs, dist_entropy


class PPOCritic(nn.Module):
    def __init__(self, args, obs_space, device=torch.device("cpu")):
        super(PPOCritic, self).__init__()
        self.mlp_hidden_size = args.mlp_hidden_size
        self.value_hidden_size = args.value_hidden_size
        self.use_recurrent_layer = args.use_recurrent_layer
        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.recurrent_hidden_layers = args.recurrent_hidden_layers
        self.use_mlp_layer = False
        self.use_mlp_value = False
        self.tpdv = dict(dtype=torch.float32, device=device)

        # pre-process module
        if len(self.mlp_hidden_size) > 0:
            self.use_mlp_layer = True
            self.mlp_layer = MLPObs(obs_space, self.mlp_hidden_size)
            input_size = self.mlp_layer.output_dim
        else:
            input_size = build_flattener(obs_space).size
        # rnn module
        if self.use_recurrent_layer:
            self.gru_layer = GRU(
                input_size, self.recurrent_hidden_size, self.recurrent_hidden_layers)
            input_size = self.gru_layer.output_dim
        # value module
        if len(self.value_hidden_size) > 0:
            self.use_mlp_value = True
            self.mlp = MLPBase(input_size, self.value_hidden_size)
            input_size = self.mlp.output_dim
        self.value_out = nn.Linear(input_size, 1)

        self.to(device)

    def forward(self, x, rnn_states, deterministic=False):
        x = check(x).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        if self.use_mlp_layer:
            x = self.mlp_layer(x)
        if self.use_recurrent_layer:
            x, rnn_states = self.gru_layer(x, rnn_states)
        if self.use_mlp_value:
            x = self.mlp(x)

        values = self.value_out(x)

        return values, rnn_states


class PPOPolicy():
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        self.args = args
        self.obs_space = obs_space
        self.act_space = act_space
        self.device = device
        self.lr = args.lr
        self.actor = PPOActor(args, act_space, obs_space, device)
        self.critic = PPOCritic(args, obs_space, device)
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters()}
        ], lr=self.lr)

    @property
    def act_size(self):
        return build_flattener(self.act_space).size

    @property
    def obs_size(self):
        return build_flattener(self.obs_space).size

    @property
    def name(self):
        return 'ppo'

    def get_actions(self, obs, rnn_states):
        '''
        obtain actions by random sampling (used for collecting sample)
        '''

        actions, action_log_probs, rnn_states = self.actor(obs, rnn_states)

        return actions, action_log_probs, rnn_states

    def get_values(self, obs, rnn_states):
        '''
        obtain predicetd value (used for collecting sample)
        '''

        values, rnn_states = self.critic(obs, rnn_states)

        return values, rnn_states

    def eval_actions(self, obs, rnn_states_actor, rnn_states_critic, actions):
        '''
        evaluate the state values, log probality of given antions and policy entropy (used for calculating loss)
        '''

        action_log_probs, dist_entropy = self.actor.eval_actions(
            obs, rnn_states_actor, actions)
        values, _ = self.critic(obs, rnn_states_critic)

        return values, action_log_probs, dist_entropy

    def act(self, x, rnn_states_actor):
        '''
        obtain actions according to maximum probability (used for evaluating policy)
        '''

        actions, _, rnn_states_actor = self.actor(x, rnn_states_actor, True)
        return actions, rnn_states_actor

    def save_param(self, save_path):
        '''
        save parameter in actor network and critic network
        '''
        state_dict = OrderedDict()
        state_dict['actor'] = self.actor.state_dict()
        state_dict['critic'] = self.critic.state_dict()

        torch.save(state_dict, save_path + time.strftime('%m_%d_%H_%M',
                   time.localtime(time.time())) + '.pkl')

        # torch.save(self.actor.state_dict(), './model_save/ppo/actor/' + time.strftime('%m_%d_%H_%M', time.localtime(time.time())) + '.pkl')
        # torch.save(self.critic.state_dict(), './model_save/ppo/critic/' + time.strftime('%m_%d_%H_%M', time.localtime(time.time())) + '.pkl')

    def load_param(self, env_name, save_time):
        '''
        load parameters in actor and critic network
        '''

        load_path = './model_save/ppo/' + \
            str(env_name) + '/' + save_time + '.pkl'
        state_dict = torch.load(load_path)
        actor_state_dict = state_dict['actor']
        critic_state_dict = state_dict['critic']
        self.actor.load_state_dict(actor_state_dict)
        self.critic.load_state_dict(critic_state_dict)

        # actor_net_dir = './model_save/ppo/actor/' + save_time + '.pkl'
        # critic_net_dir = './model_save/ppo/critic/' + save_time + '.pkl'

        # actor_state_dict = torch.load(actor_net_dir)
        # self.actor.load_state_dict(actor_state_dict)
        # critic_state_dic = torch.load(critic_net_dir)
        # self.critic.load_state_dict(critic_state_dic)

