from re import T
import torch
from itertools import count
import numpy as np
import pandas as pd


def _t2n(x):
    '''
    change the tensor to numpy
    '''

    x = x.detach().cpu().numpy() if torch.is_tensor(x) else x
    return x


@torch.no_grad()
def eval_ppo(args, env, policy):
    '''
    evaluate ppo policy in given environment, the policy will choose an action without exploration
    '''

    print('\n-- Start Evaluating PPO -- \n')
    policy.actor.eval()
    policy.critic.eval()
    eval_episode_rewards = []

    for i in range(args.num_episode):
        print("\r-- Episode: {} --".format(i+1), end="", flush=True)
        eval_state = env.reset()
        eval_rnn_states = np.zeros((1,
                                    args.recurrent_hidden_layers,
                                    args.recurrent_hidden_size),
                                   dtype=np.float32)
        eval_state = env.reset()
        eval_episode_reward = 0
        for t in count():
            eval_state = torch.FloatTensor(eval_state).unsqueeze(0)
            eval_action, eval_rnn_states = policy.act(
                eval_state, eval_rnn_states)
            eval_action = _t2n(eval_action)

            eval_next_state, eval_reward, eval_done, _ = env.step(
                eval_action.item())
            eval_episode_reward += eval_reward
            eval_state = eval_next_state
            if eval_done:
                eval_episode_rewards.append(eval_episode_reward)
                break

    env.close()

    # calculate average reward
    eval_average_episode_reward = np.array(eval_episode_rewards).mean()
    print('')
    print("\n-- Average Test Episode Reward: {} --\n".format(eval_average_episode_reward))
    print('-- End Testing --')


@torch.no_grad()
def eval_rl_ac(args, env, policy):
    '''
    evaluate RL policy(without CNN&RNN) with actor-critic in given environment,
    the policy will choose an action without exploration
    '''

    print('\n-- Start Evaluating {} -- \n'.format(policy.name.upper()))
    policy.actor.eval()
    policy.critic.eval()
    eval_episode_rewards = []

    for i in range(args.num_episode):
        print("\r-- Episode: {} --".format(i+1), end="", flush=True)
        eval_state = env.reset()
        eval_episode_reward = 0
        for t in count():
            eval_state = torch.FloatTensor(eval_state).unsqueeze(0)
            eval_action = policy.act(eval_state)
            eval_action = _t2n(eval_action)

            if eval_action.size == 1:
                eval_action = eval_action.item()

            eval_next_state, eval_reward, eval_done, _ = env.step(eval_action)
            eval_episode_reward += eval_reward
            eval_state = eval_next_state
            if eval_done:
                eval_episode_rewards.append(eval_episode_reward)
                break
    env.close()

    # calculate average reward
    eval_average_episode_reward = np.array(eval_episode_rewards).mean()
    print('')
    print("\n-- Average Test Episode Reward: {} --\n".format(eval_average_episode_reward))
    print('-- End Testing --')


@torch.no_grad()
def eval_dqn(args, env, policy):
    '''
    evaluate dqn policy and its variant in given environment,
    the policy will choose an action without exploration
    '''

    print('\n-- Start Evaluating {} -- \n'.format(policy.name.upper()))
    policy.q_net.eval()
    eval_episode_rewards = []

    for i in range(args.num_episode):
        print("\r-- Episode: {} --".format(i+1), end="", flush=True)
        eval_state = env.reset()
        eval_episode_reward = 0
        for t in count():
            eval_state = torch.FloatTensor(eval_state).unsqueeze(0)
            eval_action = policy.act(eval_state)
            eval_action = _t2n(eval_action)

            if eval_action.size == 1:
                eval_action = eval_action.item()

            eval_next_state, eval_reward, eval_done, _ = env.step(eval_action)
            eval_episode_reward += eval_reward
            eval_state = eval_next_state
            if eval_done:
                eval_episode_rewards.append(eval_episode_reward)
                break
    env.close()

    # calculate average reward
    eval_average_episode_reward = np.array(eval_episode_rewards).mean()
    print('')
    print("\n-- Average Test Episode Reward: {} --\n".format(eval_average_episode_reward))
    print('-- End Testing --')


