import torch
import numpy as np
from .base_trainer import BaseTrainer
from .buffer import PrioritizedExperienceReplayBuffer, ExperienceReplayBuffer


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


class DDPGTrainer(BaseTrainer):
    def __init__(self, env, policy, args, device=torch.device("cpu")):
        super(DDPGTrainer, self).__init__(env, policy, args, device)
        self.use_per = args.use_per
        if self.use_per:
            self.buffer = PrioritizedExperienceReplayBuffer(args)
            self.sample_generator = self.buffer.prioritized_generator
        else:
            self.buffer = ExperienceReplayBuffer(args)
            self.sample_generator = self.buffer.feedforward_generator


    def update(self, samples):
        '''
        implement update function
        update policy based on samples        
        '''
        if self.use_per:
            states, actions, rewards, dones, next_states, is_weights = samples
            is_weights = check(is_weights).to(**self.tpdv).reshape(-1, 1)
        else:
            states, actions, rewards, dones, next_states = samples

        states = check(states).to(**self.tpdv).reshape(-1, self.env_obs_size)
        actions = check(actions).to(**self.tpdv).reshape(-1, self.env_act_size)
        rewards = check(rewards).to(**self.tpdv).reshape(-1, 1)
        dones = check(dones).to(**self.tpdv).reshape(-1, 1)
        next_states = check(next_states).to(
            **self.tpdv).reshape(-1, self.env_obs_size)

        # compute the predicted value
        q_values_pred = self.policy.get_values(states, actions)

        # compute the target value
        with torch.no_grad():
            next_actions_tgt = self.policy.actor_target(next_states)
            next_values_tgt = self.policy.critic_target(
                next_states, next_actions_tgt)
            q_value_tgt = rewards + (1-dones) * self.gamma * next_values_tgt

        # update critic network
        td_errors = q_value_tgt - q_values_pred
        if self.use_per:
            critic_loss = (td_errors.pow(2) * is_weights).mean()
        else:
            critic_loss = td_errors.pow(2).mean()

        self.policy.critic_optimizer.zero_grad()
        critic_loss.backward()
        # nn.utils.clip_grad_norm_(
        #     self.policy.critic.parameters(), self.max_grad_norm)

        self.policy.critic_optimizer.step()

        # update actor network
        actions_pred = self.policy.actor(states)
        actor_loss = -self.policy.critic(states, actions_pred).mean()

        self.policy.actor_optimizer.zero_grad()
        actor_loss.backward()
        # nn.utils.clip_grad_norm_(
        #     self.policy.actor.parameters(), self.max_grad_norm)

        self.policy.actor_optimizer.step()

        return actor_loss, critic_loss, td_errors

    def train(self):
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

        # satrt to update the policy
        num_batch = 1
        for batch_data in self.sample_generator(num_batch=num_batch,
                                                batch_size=self.batch_size):

            # different generator generate diffenent data
            if self.use_per:
                batch_transitions, batch_tree_idx,  batch_IS_weight = batch_data
            else:
                batch_transitions, _ = batch_data

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

            # gather samples to update
            if self.use_per:
                samples = batch_states, batch_actions, batch_rewards, batch_dones, batch_next_states, batch_IS_weight
            else:
                samples = batch_states, batch_actions, batch_rewards, batch_dones, batch_next_states

            # update the policy
            actor_loss, critic_loss, td_error = self.update(samples)

            self.policy.actor_soft_update()
            self.policy.critic_soft_update()

            if self.use_per:
                self.buffer.update_sum_tree(
                    batch_tree_idx, _t2n(td_error.squeeze()))

            # store train information
            train_info['actor_loss'] += actor_loss.item()
            train_info['critic_loss'] += critic_loss.item()

        # average all loss information
        for k in train_info.keys():
            train_info[k] /= num_batch

        return train_info


    def prep_rollout(self):
        '''
        implement prep_rollout function
        turn to eval mode
        '''

        self.policy.actor.eval()
        self.policy.critic.eval()

    def prep_train(self):
        '''
        implement prep_train function
        turn to train mode
        '''

        self.policy.actor.train()
        self.policy.critic.train()
