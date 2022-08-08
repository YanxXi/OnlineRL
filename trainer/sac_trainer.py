import torch
import numpy as np
from .base_trainer import BaseTrainer
from .buffer import ExperienceReplayBuffer
import ray


def check(input):
    '''
    check whether the type of input is Tensor
    '''

    input = torch.from_numpy(input) if type(input) == np.ndarray else input
    return input



class SACTrainer(BaseTrainer):
    '''
    we interact with environment and collect samples to train both dicrete and continuous
    SAC policy in this trainer
    '''
    def __init__(self, env, policy, args, device=torch.device("cpu")):
        super(SACTrainer, self).__init__(env, policy, args, device)
        self.buffer = ExperienceReplayBuffer(args)
        self.sample_generator = self.buffer.uniform_generator
        self.reward_scale = int(1 / self.policy.init_alpha)

        # allocate update function for different policy 
        if self.policy.continuous_action:
            self.update = self.update_continuous
        else:
            self.update = self.update_discrete


    def update_continuous(self, samples):
        '''
        update continuous sac policy based on samples   
        : _tgt means the value is calculated by target network
        : _pred meean the value is calculated by normal network     
        '''

        states, actions, rewards, dones, next_states = samples

        states = check(states).to(**self.tpdv).reshape(-1, self.env_obs_size)
        actions = check(actions).to(**self.tpdv).reshape(-1, self.env_act_size)
        rewards = check(rewards).to(**self.tpdv).reshape(-1, 1)
        dones = check(dones).to(**self.tpdv).reshape(-1, 1)
        next_states = check(next_states).to(
            **self.tpdv).reshape(-1, self.env_obs_size)

        # temperature parameters
        alpha = self.policy.alpha_log.exp().detach()

        # compute the predicted value
        q1_values_pred, q2_values_pred = self.policy.critic.get_q1_q2_values(
            states, actions)

        # compute the target value
        with torch.no_grad():
            next_actions_tgt, next_log_probs_tgt = self.policy.actor_target.get_action_log_prob(
                next_states)
            next_q_values_tgt = self.policy.critic_target.get_min_q_values(
                next_states, next_actions_tgt)
            q_values_tgt = rewards + \
                (1-dones) * self.gamma * \
                (next_q_values_tgt - alpha * next_log_probs_tgt)

        # update critic network
        critic_loss = 0.5 * ((q1_values_pred - q_values_tgt).pow(2).mean() +
                             (q2_values_pred - q_values_tgt).pow(2).mean())

        self.policy.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.policy.critic_optimizer.step()

        # calculate actions and log probs used for policy gradient
        actions_pred, log_probs_pred = self.policy.actor.get_action_log_prob(
            states)
        # calculate q values used for policy gradient
        q_values_pred = self.policy.critic.get_min_q_values(
            states, actions_pred)

        # update actor network
        actor_loss = (alpha * log_probs_pred - q_values_pred).mean()

        self.policy.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.policy.actor_optimizer.step()

        # update temperature parameters alpha
        alpha_loss = -self.policy.alpha_log * \
            (log_probs_pred + self.policy.entropy_tgt).detach().mean()
        self.policy.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.policy.alpha_optimizer.step()

        return actor_loss, critic_loss

    def update_discrete(self, samples):
        '''
        update discrete sac policy based on samples   
        : _tgt means the value is calculated by target network
        : _pred meean the value is calculated by normal network     
        '''

        states, actions, rewards, dones, next_states = samples

        states = check(states).to(**self.tpdv).reshape(-1, self.env_obs_size)
        actions = check(actions).to(**self.tpdv).reshape(-1, self.env_act_size)
        rewards = check(rewards).to(**self.tpdv).reshape(-1, 1)
        dones = check(dones).to(**self.tpdv).reshape(-1, 1)
        next_states = check(next_states).to(
            **self.tpdv).reshape(-1, self.env_obs_size)

        # temperature parameters
        alpha = self.policy.alpha_log.exp().detach()

        # compute the predicted value
        q1_values_pred, q2_values_pred = self.policy.critic.get_q1_q2_values(
            states, actions)

        # compute the target value
        with torch.no_grad():
            next_probs_tgt, next_log_probs_tgt =\
                 self.policy.actor_target.get_action_log_prob(next_states)
            next_q_values_tgt = self.policy.critic_target.get_min_q_values(next_states)

            q_values_tgt = rewards + (1-dones) * self.gamma * \
                torch.mul(
                    next_probs_tgt,
                    (next_q_values_tgt - alpha * next_log_probs_tgt)
                    ).sum(dim=-1, keepdim=True)

        # update critic network
        critic_loss = 0.5 * ((q1_values_pred - q_values_tgt).pow(2).mean() +
                             (q2_values_pred - q_values_tgt).pow(2).mean())

        self.policy.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.policy.critic_optimizer.step()

        # calculate actions and log probs used for policy gradient
        probs_pred, log_probs_pred = self.policy.actor.get_action_log_prob(states)
        # calculate q values used for policy gradient
        q_values_pred = self.policy.critic.get_min_q_values(states)
        # update actor network
        actor_loss = torch.einsum(
            'ij,ij->i',
            probs_pred,
            (alpha * log_probs_pred - q_values_pred)
        ).mean()

        self.policy.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.policy.actor_optimizer.step()

        # update temperature parameters alpha
        alpha_loss = torch.einsum(
            'ij,ij->i',
            log_probs_pred.detach(),
            -self.policy.alpha_log * (log_probs_pred + self.policy.entropy_tgt).detach()
            ).mean()

        self.policy.alpha_optimizer.zero_grad() 
        alpha_loss.backward()
        self.policy.alpha_optimizer.step()

        return actor_loss, critic_loss

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
        batch_transitions = self.sample_generator(batch_size=self.batch_size)

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
        samples = batch_states, batch_actions, batch_rewards, batch_dones, batch_next_states

        # update the policy
        actor_loss, critic_loss = self.update(samples)

        self.policy.actor_soft_update()
        self.policy.critic_soft_update()

        # store train information
        train_info['actor_loss'] += actor_loss.item()
        train_info['critic_loss'] += critic_loss.item()

        return train_info

    def run(self):
        '''
        override train function, since sac train the poicy not per step
        interact with the environment to train and evaluate the policy
        '''

        print('\n-- Start Running {} Training Process --\n'.format(
            str(self.policy.name).upper()))

        cur_steps = 0
        episode_idx = 0
        episode_reward = 0

        state = self.env.reset()

        if self.eval_in_paral:
            ray.init()
        
        # In SAC, we update step after sampling one trajactory
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
            # store transitions
            transition = self.Transition(
                state, action, reward * self.reward_scale, done, next_state)
            self.buffer.insert(transition)
            # for next step
            state = next_state

            # trainning
            if cur_steps == self.max_episode_steps or done:
                episode_idx += 1
                print('-- episode: {}, timestep: {}, episode reward: {} --'.
                      format(episode_idx, cur_steps, episode_reward))

                for _ in range(cur_steps):
                    train_info = self.train()
                state = self.env.reset()
                cur_steps = 0
                episode_reward = 0

            # evaluate the policy and check whether the problem is sloved
            if (t+1) % self.steps_per_epoch == 0:
                epoch = (t+1) // self.steps_per_epoch
                eval_average_episode_reward = self.eval()
                print("\n-- Epoch: {}, Average Evaluation Episode Reward: {} --\n".format(
                    epoch, eval_average_episode_reward))
                if eval_average_episode_reward >= self.solved_reward:
                    print("-- Problem Solved --\n")
                    break

        if self.save_model:
            print("-- Save Model --\n")
            self.save()
        else:
            print("-- Not Save Model --\n")

        print('-- End Running --\n')


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
