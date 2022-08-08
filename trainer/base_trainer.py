from tkinter import N
import torch
import os
import csv
from collections import namedtuple
from itertools import count
import ray
import numpy as np
from copy import deepcopy

def _t2n(x):
    '''
    change the tensor to numpy
    '''

    x = x.detach().cpu().numpy() if torch.is_tensor(x) else x
    return x

class BaseTrainer():
    def __init__(self, env, policy, args, device=torch.device("cpu")):
        self.policy = policy
        self.env = env
        self.args = args
        self.eval_env = deepcopy(env)
        self.env_name = args.env
        self.env_act_size = self.policy.act_size
        self.env_obs_size = self.policy.obs_size
        self.num_episode = int(args.num_episode)
        self.epochs = int(args.epochs)
        self.steps_per_epoch = int(args.steps_per_epoch)
        self.total_steps = self.epochs * self.steps_per_epoch
        self.max_episode_steps =\
             args.max_episode_steps if args.max_episode_steps is not None else env._max_episode_steps
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.device = device
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.eval_interval = args.eval_interval
        self.eval_episodes = args.eval_episodes
        self.buffer = None
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.max_grad_norm = args.max_grad_norm
        self.eval_in_paral = args.eval_in_paral
        self.log_info = args.log_info
        self.save_model = args.save_model
        self.solved_reward = args.solved_reward
        self.start_eval = args.start_eval
        self.reward_scale = 1
        self._iteration = 0
        self.Transition = namedtuple(
            'transition', ['state', 'action', 'reward', 'done', 'next_state'])


    def update(self):
        '''
        update policy based on samples
        : _tgt means the value is calculated by target network
        : _pred meean the value is calculated by normal network 
        '''

        pass

    def train(self):
        '''
        train the policy based on all data in the buffer, in training mode, the policy 
        will choose an action randomly.
        '''

        pass


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


    def run(self):
        '''
        interact with the environment to train and evaluate the policy in traditional 
        way, we update the policy per step and each time sample a minibatch from buffer
        '''

        print('\n-- Start Running {} Training Process --\n'.format(
            str(self.policy.name).upper()))

        episode_idx = 0
        episode_reward = 0
        cur_steps = 0

        state = self.env.reset()

        if self.eval_in_paral:
            ray.init()

        for t in range(self.total_steps):
            cur_steps += 1
            # interact with the environment to collent data
            action = self.collect(state)
            if action.size == 1:
                action = action.item()
            next_state, reward, done, _ = self.env.step(action)
            episode_reward += reward
            # rescale reward
            reward = np.clip(reward, -1, 1)
            # store one step transition into buffer
            transition = self.Transition(
                state, action, reward, done, next_state)
            self.buffer.insert(transition)

            # at each time step, update the network by sampling a minibatch in the buffer
            if self.buffer.current_size >= self.batch_size:
                train_info = self.train()

            # for next step
            state = next_state

            if done or cur_steps == self.max_episode_steps:
                episode_idx += 1
                print('-- episode: {}, timestep: {}, episode reward: {} --'.format(
                    episode_idx + 1, cur_steps, episode_reward))
                state = self.env.reset()
                episode_reward = 0
                cur_steps = 0

            # evaluate the policy and check whether the problem is sloved
            if (t+1) % self.steps_per_epoch == 0:
                epoch = (t+1) // self.steps_per_epoch
                eval_average_episode_reward = self.eval()
                print("\n-- Epoch: {}, Average Evaluation Episode Reward: {} --\n".format(
                    epoch, eval_average_episode_reward))
                if eval_average_episode_reward >= self.solved_reward:
                    print("-- Problem Solved --\n")
                    break

            # # log infomation
            # if self.log_info:
            #     self.log(i+1, t+1, episode_reward)

        if self.save_model:
            print("-- Save Model --\n")
            self.save()
        else:
            print("-- Not Save Model --\n")

        print('-- End Running --\n')

    @torch.no_grad()
    def eval(self):
        '''
        evaluate current policy, in evaluation mode, the policy will choose an action 
        according to maximum probability.
        '''

        self.prep_rollout()

        if self.eval_in_paral:
            obj_ref_list = [self.async_eval_one_episode.remote(
                self.eval_env, self.policy) for _ in range(self.eval_episodes)]
            eval_episode_rewards = ray.get(obj_ref_list)
        else:
            eval_episode_rewards = []
            for _ in range(self.eval_episodes):
                eval_episode_reward = self.eval_one_episode(
                    self.eval_env, self.policy)
                eval_episode_rewards.append(eval_episode_reward)

        # calculate average reward
        eval_average_episode_reward = np.array(eval_episode_rewards).mean()

        return eval_average_episode_reward

    @staticmethod
    def eval_one_episode(env, policy):
        '''
        eval one episode and return episode reward
        '''
        with torch.no_grad():
            eval_state = env.reset()
            eval_episode_reward = 0
            for t in count():
                eval_state = torch.FloatTensor(eval_state).unsqueeze(0)
                eval_action = policy.act(eval_state)
                eval_action = _t2n(eval_action)

                if eval_action.size == 1:
                    eval_action = eval_action.item()

                eval_next_state, eval_reward, eval_done, _ = env.step(
                    eval_action)
                eval_episode_reward += eval_reward
                eval_state = eval_next_state
                if eval_done:
                    env.close()
                    break

        return eval_episode_reward

    @staticmethod
    @ray.remote(num_returns=1)
    def async_eval_one_episode(env, policy):
        '''
        eval one episode asynchronously and return episode reward
        '''
        with torch.no_grad():
            eval_state = env.reset()
            eval_episode_reward = 0
            for t in count():
                eval_state = torch.FloatTensor(eval_state).unsqueeze(0)
                eval_action = policy.act(eval_state)
                eval_action = _t2n(eval_action)

                eval_next_state, eval_reward, eval_done, _ = env.step(
                    eval_action)
                eval_episode_reward += eval_reward
                eval_state = eval_next_state
                if eval_done:
                    env.close()
                    break

        return eval_episode_reward

    def save(self):
        '''
        save parameter in policy
        '''
        save_path = './model_save/' + \
            str(self.policy.name) + '/' + str(self.env_name) + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.policy.save_param(save_path)

    def log(self, episode, time_step, episode_reward):
        '''
        log time step and episode reward
        '''
        log_dir = 'rl_results/' + str(self.policy.name) + '/'

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_path = log_dir + self.env_name + '.cvs'
        if episode == 1:
            with open(log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['time_step', 'episode_reward'])

        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([time_step, episode_reward])


    def prep_rollout(self):
        '''
        turn to eval mode
        '''
        pass

    def prep_train(self):
        '''
        turn to train mode
        '''
        pass