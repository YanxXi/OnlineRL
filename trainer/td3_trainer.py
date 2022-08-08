import ray
import torch
import numpy as np
from .base_trainer import BaseTrainer
from .buffer import ExperienceReplayBuffer


def check(input):
    '''
    check whether the type of input is Tensor
    '''

    input = torch.from_numpy(input) if type(input) == np.ndarray else input
    return input


class TD3Trainer(BaseTrainer):
    def __init__(self, env, policy, args, device=torch.device("cpu")):
        super(TD3Trainer, self).__init__(env, policy, args, device)
        self.buffer = ExperienceReplayBuffer(args)
        self.sample_generator = self.buffer.uniform_generator
        self.delay_frequency = args.delay_frequency
        self.policy_noise = args.policy_noise
        self.policy_clip = args.policy_clip


    def update(self, samples):
        '''
        implement update function
        update policy based on samples        
        '''

        states, actions, rewards, dones, next_states = samples

        states = check(states).to(**self.tpdv).reshape(-1, self.env_obs_size)
        actions = check(actions).to(**self.tpdv).reshape(-1, self.env_act_size)
        rewards = check(rewards).to(**self.tpdv).reshape(-1, 1)
        dones = check(dones).to(**self.tpdv).reshape(-1, 1)
        next_states = check(next_states).to(
            **self.tpdv).reshape(-1, self.env_obs_size)

        # compute the target value
        with torch.no_grad():
            # noises = torch.normal(
            #     0, self.policy_noise, tuple(actions.shape)).clamp(-self.policy_clip , self.policy_clip)
            next_actions_tgt = self.policy.actor_target(next_states)
            noises = (torch.randn_like(next_actions_tgt) *
                      self.policy_noise).clamp(-self.policy_clip, self.policy_clip)
            smoothed_actions_tgt = (next_actions_tgt + noises).clamp(-1, 1)
            min_next_q = self.policy.critic_target.get_min_q_values(
                next_states, smoothed_actions_tgt)
            q_values_tgt = rewards + (1-dones) * self.gamma * min_next_q

        # compute the predicted value
        q_values_pred1, q_values_pred2 = self.policy.critic.get_q1_q2_values(
            states, actions)

        # update critic network
        td_errors1 = q_values_tgt - q_values_pred1
        td_errors2 = q_values_tgt - q_values_pred2

        critic_loss = 0.5 * (td_errors1.pow(2).mean() +
                             td_errors2.pow(2).mean())

        self.policy.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.policy.critic_optimizer.step()

        # update actor network and update the parameters in target network
        if self._iteration > 0 and self._iteration % self.delay_frequency == 0:
            actions_pred = self.policy.actor(states)
            actor_loss = -self.policy.critic(states, actions_pred).mean()

            self.policy.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.policy.actor_optimizer.step()

            self.policy.actor_soft_update()
            self.policy.critic_soft_update()

        return critic_loss

    def train(self):
        '''
        implement train function
        train the policy based on all data in the buffer, in training mode, the policy 
        will choose an action randomly.
        '''

        self.prep_train()

        # average loss in training process
        train_info = {}
        # train_info['actor_loss'] = 0
        train_info['critic_loss'] = 0

        # satrt to update the policy
        batch_transitions = self.sample_generator(self.batch_size)

        # obtain states, values, actions, rewards in transitions and reshape for nn input
        batch_states = np.array(
            [t.state for t in batch_transitions], dtype=float)
        batch_actions = np.array(
            [t.action for t in batch_transitions], dtype=float)
        batch_rewards = np.array(
            [t.reward for t in batch_transitions], dtype=float)
        batch_dones = np.array(
            [t.done for t in batch_transitions], dtype=bool)
        batch_next_states = np.array(
            [t.next_state for t in batch_transitions], dtype=float)

        samples = batch_states, batch_actions, batch_rewards, batch_dones, batch_next_states

        # update the policy
        critic_loss = self.update(samples)
        self._iteration += 1

        # store train information
        # train_info['actor_loss'] += actor_loss.item()
        train_info['critic_loss'] += critic_loss.item()

        return train_info

    def prep_rollout(self):
        '''
        implement prep_rollout fuction
        turn to eval mode
        '''

        self.policy.actor.eval()
        self.policy.critic.eval()

    def prep_train(self):
        '''
        implement prep_train function
        '''

        self.policy.actor.train()
        self.policy.critic.train()
