from copy import deepcopy
import torch
import numpy as np
from collections import namedtuple    
import torch
from ..algorithm.a3c import A3CPolicy
from .base_trainer import BaseTrainer
import os
import torch.multiprocessing as mp
from ..utils.optimizer import SharedAdam

def check(input):
    '''
    check whether the type of input is Tensor
    '''

    input = torch.from_numpy(input) if type(input) == np.ndarray else input
    return input

def _ensure_shared_grads(model, shared_model):
    '''
    eusure the model and shared_model refer to same grad, 
    we use '=' to make two grad point to same adresss
    '''
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def _t2n(x):
    '''
    change the tensor to numpy
    '''

    x = x.detach().cpu().numpy() if torch.is_tensor(x) else x
    return x

class Worker:

    '''
    Work interact with environemt seprately based on the paremeters from shared policy,
    and update the shared policy
    '''

    def __init__(self, env, args, device=torch.device("cpu")) -> None:
        self.env = env
        self.args = args
        self.device = device
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.value_loss_coef = args.value_loss_coef
        self.policy = A3CPolicy(self.env.observation_space, self.env.action_space)
        self.env_act_size = self.policy.act_size
        self.env_obs_size = self.policy.obs_size
        self.Transition = namedtuple(
            'transition', ['state', 'action', 'reward', 'done', 'next_state'])
        self.buffer = []

    def __call__(self, rank:int, shared_policy: A3CPolicy,
                       counter, lock, optimizer: SharedAdam):
        '''
        explore the environment and update shared_policy
        Args:
            rank: the id of worker
            shared_policy: the policy we aim to update
            counter(mp.Value): global counter
            lock(mp.Lock): global lock
            optimizer: we use ShareAdam to update network
        '''

        # set random seed
        torch.manual_seed(self.args.seed + rank)
        self.env.seed(self.args.seed + rank)

        state = self.env.reset()
        t_max = 200

        while True:
            # Sync with the shared model
            self.policy.load_state_dict(shared_policy.state_dict())

            for t in range(self.args.total_steps):
                t_start = t
                action = self.collect(state)

                next_state, reward, done, _ = self.env.step(action.numpy())
                transition = self.Transition(state, action, reward, done, next_state)
                self.buffer.append(transition)
                state = next_state

                with lock:
                    counter.value += 1

                if done or (t - t_start) >= t_max:
                    state = self.env.reset()
                    break

            # obtain samples to update 
            states = np.array([t.state for t in self.buffer], dtype=float)
            next_states = np.array([t.next_state for t in self.buffer], dtype=float)
            actions = np.array([t.action for t in self.buffer], dtype=float)
            rewards = np.array([t.reward for t in self.buffer], dtype=float)
            dones = np.array([t.done for t in self.buffer], dtype=bool)

            samples = states, next_states, actions, rewards, dones

            self.train(samples, optimizer, shared_policy)


    def train(self, samples, optimizer, shared_policy):
        '''
        train the parameters in shared_policy by samples collected by worker.policy
        '''
        
        self.prep_train()
        
        states, next_states, actions, rewards, dones = samples

        states = check(states).to(**self.tpdv).reshape(-1, self.env_obs_size)
        actions = check(actions).to(**self.tpdv).reshape(-1, self.env_act_size)
        rewards = check(rewards).to(**self.tpdv).reshape(-1, 1)
        dones = check(dones).to(**self.tpdv).reshape(-1, 1)
        next_states = check(next_states).to(
            **self.tpdv).reshape(-1, self.env_obs_size)

        values = self.policy.critic(states) # 1,2,……,T
        next_values = self.policy.critic(next_states) # 2,3,……,T+1
        
        returns = torch.zero((len(rewards) + 1, 1))
        gae = torch.zeros_like(returns)

        returns[-1] = next_values[-1].detach() # 1,2,……，T + 1
        gae[-1] = torch.zeros(1)

        # calculate returns and gae
        for i in reversed(range(len(rewards))):
            # Returns
            returns[i] = self.gamma * returns[i+1] * (1-dones[i]) + rewards[i]

            # Generalized Advantage Estimation
            delta_t = rewards[i] + self.gamma * \
                next_values[i] - values[i]
            gae[i] = gae[i+1] * self.gamma * self.gae_lambda + delta_t
    
        returns = returns[:-1].detach()
        gae = gae[:-1].detach()
        
        # calculate critic loss
        advantages = returns - values
        value_loss = 0.5 * advantages.pow(2).mean()
        
        # calculate actor loss
        log_probs = self.policy.actor.get_log_probs(states, actions)
        policy_loss = log_probs * gae.mean()

        # update both critic and actor network
        optimizer.zero_grad()

        (policy_loss + self.value_loss_coef * value_loss).backward()

        _ensure_shared_grads(self.policy, shared_policy)

        optimizer.step()

    @torch.no_grad()
    def collect(self, state):
        '''
        collect date and insert the data into the buffer for training
        '''

        self.prep_rollout()
        state = torch.FloatTensor(state).unsqueeze(0)
        # obtain all data in one transition
        action = self.policy.get_actions(state)

        action = np.array(_t2n(action))

        return action

    def prep_rollout(self):
        '''
        turn to eval mode
        '''

        self.policy.actor.eval()
        self.policy.critic.eval()

    def prep_train(self):
        '''
        turn to train mode
        '''

        self.policy.actor.train()
        self.policy.critic.train()


class A3CTrainer(BaseTrainer):
    def __init__(self, env, policy, args, device=torch.device("cpu")):
        super(A3CTrainer, self).__init__(args, device)
        self.policy = policy
        self.env = env
        self.env_name = args.env
        self.shared_policy = policy
        self.shared_policy.share_memory()
        optimizer = SharedAdam(self.shared_policy.parameters(), lr=args.lr)
        optimizer.share_memory()

        
    def run(self):

        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
       
        processes = []

        counter = mp.Value('i', 0)
        lock = mp.Lock()

        # i don't think this evluation is a good one
        # p = mp.Process(target=test, args=(args.num_processes, args, shared_model, counter))
        # p.start()
        # processes.append(p)

        for rank in range(0, self.args.num_processes):
            worker = Worker(deepcopy(self.env), self.args, self.device)
            p = mp.Process(target=worker(), args=(rank, self.shared_policy, counter, lock, self.optimizer))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()