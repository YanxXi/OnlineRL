import torch
from .dqn_trainer import DQNTrainer
import numpy as np

def check(input):
    '''
    check whether the type of input is Tensor
    '''

    input = torch.from_numpy(input) if type(input) == np.ndarray else input
    return input

class DuelingDQNTrainer(DQNTrainer):
    def __init__(self, env, policy, args, device=torch.device("cpu")):
        super().__init__(env, policy, args, device)


    def update(self, samples):
        '''
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
            # max Q value of next state
            next_max_q = self.policy.q_net_target.get_max_q_values(next_states)
            q_values_tgt = rewards + (1 - dones) * self.gamma * next_max_q

        # update Q network
        td_errors = q_values_tgt - q_values_pred
        if self.use_per:
            loss = 0.5 * (td_errors.pow(2) * is_weights).mean()
        else:
            loss = 0.5 * td_errors.pow(2).mean()

        self.policy.optimizer.zero_grad()
        loss.backward()
        self.policy.optimizer.step()

        return loss, td_errors

 

 