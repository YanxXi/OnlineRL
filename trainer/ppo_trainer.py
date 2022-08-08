from itertools import count
from collections import namedtuple
import torch
import torch.nn as nn
import numpy as np
from .buffer import ExperienceReplayBuffer
from .base_trainer import BaseTrainer


def check(input):
    '''
    check whether the type of input is Tensor
    '''

    input = torch.from_numpy(input) if type(input) == np.ndarray else input
    return input


def _t2n(x):
    '''
    change the tensor to numpy
    '''

    x = x.detach().cpu().numpy() if torch.is_tensor(x) else x
    return x


class PPOTrainer(BaseTrainer):
    def __init__(self, env, policy, args, device=torch.device("cpu")):
        super(PPOTrainer, self).__init__(env, policy, args, device)
        self.use_reccurent = args.use_recurrent_layer
        self.clip_param = args.clip_param
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.ppo_epoch = args.ppo_epoch
        self.chunk_length = args.chunk_length
        self.buffer = ExperienceReplayBuffer(args)
        if self.use_reccurent:
            self.sample_generator = self.buffer.recurrent_generator
        else:
            self.sample_generator = self.buffer.feedforward_generator

        self.Transition = namedtuple('transition', ['state', 'action', 'action_log_prob', 'reward', 'done',
                                                    'next_state', 'rnn_state_actor', 'rnn_state_critic', 'value'])

    @torch.no_grad()
    def compute_returns(self):
        '''
        compute returns of each state in the buffer
        '''

        self.prep_rollout()

        rewards = np.array(
            _t2n([t.reward for t in self.buffer.transitions])).reshape(-1, 1)
        dones = np.array(
            _t2n([t.done for t in self.buffer.transitions])).reshape(-1, 1)
        returns = np.zeros((len(rewards)+1, 1))
        next_state = torch.FloatTensor(
            self.buffer.transitions[-1].next_state).unsqueeze(0)
        rnn_state_critic = self.buffer.transitions[-1].rnn_state_critic

        # we use the predicted value to refer to the return of final state in the buffer
        next_value, _ = self.policy.get_values(next_state,
                                               rnn_state_critic)
        next_value = np.array(_t2n(next_value))

        returns[-1] = next_value.item()

        # if one state is the terminal state, the its return is reward.
        for i in reversed(range(len(rewards))):
            returns[i] = rewards[i] + \
                (1 - dones[i]) * self.gamma * returns[i+1]

        # drop the return of last state
        return returns[:-1]

    @torch.no_grad()
    def collect(self, state, rnn_state_actor, rnn_state_critic):
        '''
        override collect function, since we need run states
        collect date and insert the data into the buffer for training
        '''

        self.prep_rollout()
        state = torch.FloatTensor(state).unsqueeze(0)
        # obtain all data in one transition
        action, action_log_prob, rnn_state_actor = self.policy.get_actions(
            state, rnn_state_actor)
        value, rnn_state_critic = self.policy.get_values(
            state, rnn_state_critic)

        action = np.array(_t2n(action))
        action_log_prob = np.array(_t2n(action_log_prob))
        rnn_state_actor = np.array(_t2n(rnn_state_actor))
        rnn_state_critic = np.array(_t2n(rnn_state_critic))
        value = np.array(_t2n(value))

        return action, action_log_prob, rnn_state_actor, rnn_state_critic, value

    def update(self, samples):
        '''
        implement update function
        update policy based on samples
        '''

        states, rnn_states_actor, rnn_states_critic, old_action_log_probs,\
            advantages, returns, actions = samples

        states = check(states).to(**self.tpdv).view(-1, self.env_obs_size)
        actions = check(actions).to(**self.tpdv).view(-1, self.env_act_size)
        old_action_log_probs = check(old_action_log_probs).to(
            **self.tpdv).view(-1, self.env_act_size)
        rnn_states_actor = check(rnn_states_actor).to(
            **self.tpdv).view(-1, *rnn_states_actor.shape[-2:])
        rnn_states_critic = check(rnn_states_critic).to(
            **self.tpdv).view(-1, *rnn_states_critic.shape[-2:])
        advantages = check(advantages).to(**self.tpdv).view(-1, 1)
        returns = check(returns).to(**self.tpdv).view(-1, 1)

        advantages = (advantages - advantages.mean()) / \
            (advantages.std() + 1e-5)
        mask_tensor = torch.sign(
            torch.max(torch.abs(states), dim=-1, keepdim=True)[0])

        # calculate new data based on newly policy
        pred_values, action_log_probs, dist_entropy = self.policy.eval_actions(states,
                                                                               rnn_states_actor,
                                                                               rnn_states_critic,
                                                                               actions)
        # calculate actor loss
        ratio = torch.exp(action_log_probs -
                          old_action_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_param,
                            1 + self.clip_param) * advantages
        CLIP_loss = torch.sum(torch.min(surr1, surr2), -1, keepdim=True)
        CLIP_loss = -(CLIP_loss * mask_tensor).mean()

        # calculate value loss
        VF_loss = 0.5 * (returns - pred_values).pow(2)
        VF_loss = (VF_loss * mask_tensor).mean()

        # calculate entropy loss
        entropy_loss = -(dist_entropy * mask_tensor).mean()

        # calculate total loss in one update
        policy_loss = CLIP_loss + VF_loss * self.value_loss_coef + \
            entropy_loss * self.entropy_coef

        # update weights in the actor and critic network
        self.policy.optimizer.zero_grad()
        policy_loss.backward()

        nn.utils.clip_grad_norm_(
            self.policy.actor.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(
            self.policy.critic.parameters(), self.max_grad_norm)

        self.policy.optimizer.step()

        return CLIP_loss, VF_loss, entropy_loss, policy_loss

    def train(self, returns):
        '''
        implement train function
        train the policy based on all data in the buffer, in training mode, the policy 
        will choose an action randomly.
        '''

        self.prep_train()

        # average loss in training process
        train_info = {}
        train_info['actor_loss'] = 0
        train_info['critic_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['entropy_loss'] = 0

        # number of update
        update_step = 0

        # generate samples for update
        for _ in range(self.ppo_epoch):
            for batch_data in self.sample_generator(batch_size=self.batch_size,
                                                    chunk_length=self.chunk_length,
                                                    returns=returns):
                # if use recurrent layer, we sample sequential data in the buffer
                if self.use_reccurent:
                    # obtain states, values, actions, old log probs in transitions and reshape for rnn input
                    batch_transitions, batch_returns, batch_rnn_states_actor, batch_rnn_states_critic = batch_data
                    L = self.chunk_length
                    N = batch_transitions.shape[0] // self.chunk_length
                    batch_states = np.zeros((L*N, self.env_obs_size))
                    batch_values = np.zeros((L*N, 1))
                    batch_actions = np.zeros((L * N, self.env_act_size))
                    batch_old_action_log_probs = np.zeros(
                        (L * N, self.env_act_size))
                    # mask is index of valid transitions since some chunk might smaller than chunk length
                    mask = np.where(batch_transitions != 0)
                    batch_states[mask] = np.array(
                        [t.state for t in batch_transitions[mask]])
                    batch_values[mask] = np.array(
                        [t.value for t in batch_transitions[mask]]).reshape(-1, 1)
                    batch_actions[mask] = np.array(
                        [t.action for t in batch_transitions[mask]])
                    batch_old_action_log_probs[mask] = np.array(
                        [t.action_log_prob for t in batch_transitions[mask]])
                    # obtain advantages
                    batch_advantages = batch_returns - batch_values

                # we sample normal data in the buffer
                else:
                    # obtain states, values, actions, old log probs in transitions
                    batch_transitions, batch_indices = batch_data
                    batch_states = np.array(
                        [t.state for t in batch_transitions], dtype=float)
                    batch_rnn_states_critic = np.array(
                        [t.rnn_state_critic for t in batch_transitions], dtype=float)
                    batch_rnn_states_actor = np.array(
                        [t.rnn_state_actor for t in batch_transitions], dtype=float)
                    batch_actions = np.array(
                        [t.action for t in batch_transitions], dtype=float)
                    batch_values = np.array(
                        [t.value for t in batch_transitions], dtype=float).reshape(-1, 1)
                    batch_old_action_log_probs = np.array(
                        [t.action_log_prob for t in batch_transitions], dtype=float)
                    # obtain advantages
                    batch_returns = returns[batch_indices]
                    batch_advantages = (batch_returns - batch_values)

                # gather samples to update
                samples = batch_states, batch_rnn_states_actor, batch_rnn_states_critic, batch_old_action_log_probs,\
                    batch_advantages, batch_returns, batch_actions

                # update the policy
                actor_loss, critic_loss, entropy_loss, policy_loss = self.update(
                    samples)
                update_step += 1

                # store train information
                train_info['policy_loss'] += policy_loss.item()
                train_info['actor_loss'] += actor_loss.item()
                train_info['critic_loss'] += critic_loss.item()
                train_info['entropy_loss'] += entropy_loss.item()

        # average all loss information
        for k in train_info.keys():
            train_info[k] /= update_step

        return train_info

    def run(self):
        '''
        override run function
        '''

        print('\n-- Start Running {} Training Process --\n'.format(
            str(self.policy.name).upper()))

        for i in range(self.num_episode):
            episode_reward = 0
            state = self.env.reset()
            rnn_state_actor = np.zeros(
                (1, self.args.recurrent_hidden_layers, self.policy.args.recurrent_hidden_size))
            rnn_state_critic = np.zeros(
                (1, self.args.recurrent_hidden_layers, self.policy.args.recurrent_hidden_size))
            # run one episode
            for t in count():
                # interact with the environment to collent data
                action, action_log_prob, next_rnn_state_actor, next_rnn_state_critic, value = self.collect(
                    state, rnn_state_actor, rnn_state_critic)

                next_state, reward, done, _ = self.env.step(action.item())
                episode_reward += reward
                # rescale reward
                reward = np.clip(reward, -1, 1)

                # store one step transition into buffer
                transiton = self.Transition(state, action, action_log_prob, reward, done, next_state,
                                            rnn_state_actor, rnn_state_critic, value)
                self.buffer.insert(transiton)
                # for next step
                state = next_state
                rnn_state_actor = next_rnn_state_actor
                rnn_state_critic = next_rnn_state_critic

                # check whether the buffer is prepared for training
                if self.buffer.is_full:
                    returns = self.compute_returns()
                    train_info = self.train(returns)
                    self.buffer.after_update()

                if done:
                    print(
                        '-- episode: {}, timestep: {}, episode reward: {} --'.format(i+1, t+1, episode_reward))
                    break

            # log infomation
            if self.log_info:
                self.log(i+1, t+1, episode_reward)

            # check whether need to evaluate the policy
            if (i+1) % self.eval_interval == 0 and (i+1) >= self.start_eval:
                eval_average_episode_reward = self.eval()
                if eval_average_episode_reward >= self.solved_reward:
                    print("-- Problem Solved --\n")
                    break

        # save model or not
        if self.save_model:
            print("-- Save Model --\n")
            self.save()
        else:
            print("-- Not Save Model --\n")

        print('-- End Running --\n')

    @torch.no_grad()
    def eval(self):
        '''
        override eval function since we need rnn states
        evaluate current policy, in evaluation mode, the policy will choose an action 
        according to maximum probability.
        '''

        self.prep_rollout()
        eval_episode_rewards = []

        for i in range(self.eval_episodes):
            eval_episode_reward = 0
            eval_state = self.env.reset()
            eval_rnn_states = np.zeros((1,
                                        self.args.recurrent_hidden_layers,
                                        self.args.recurrent_hidden_size),
                                       dtype=np.float32)
            for t in count():
                eval_state = torch.FloatTensor(eval_state).unsqueeze(0)
                eval_actions, eval_rnn_states = self.policy.act(eval_state,
                                                                eval_rnn_states)
                eval_actions = np.array(_t2n(eval_actions))

                eval_next_state, eval_reward, eval_done, _ = self.env.step(
                    eval_actions.item())
                eval_episode_reward += eval_reward
                eval_state = eval_next_state
                if eval_done:
                    eval_episode_rewards.append(eval_episode_reward)
                    break

        self.env.close()

        # calculate average reward
        eval_average_episode_reward = np.array(eval_episode_rewards).mean()
        print("\n-- Average Evaluation Episode Reward: {} --\n".format(
            eval_average_episode_reward))

        return eval_average_episode_reward

    def prep_rollout(self):
        '''
        implement prep_rollout function
        turn to eval mode
        '''

        self.policy.actor.eval()
        self.policy.critic.eval()

    def prep_train(self):
        '''
        implement prep_rollout function
        turn to train mode
        '''

        self.policy.actor.train()
        self.policy.critic.train()
