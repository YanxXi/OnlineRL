import argparse
from config import *
import gym

def test_ppo(args, obs_space, act_space):
    from algorithm.ppo import PPOPolicy
    from eval_model import eval_ppo


    ppo_policy = PPOPolicy(args, obs_space, act_space)
    ppo_policy.load_param(args.env, args.save_time)
    eval_ppo(args, env, ppo_policy)


def test_ddpg(args, obs_space, act_space):
    from algorithm.ddpg import DDPGPolicy
    from eval_model import eval_rl_ac

    ddpg_policy = DDPGPolicy(args, obs_space, act_space)
    ddpg_policy.load_param(args.env, args.save_time)
    eval_rl_ac(args, env, ddpg_policy)

def test_td3(args, obs_space, act_space):
    from algorithm.td3 import TD3Policy
    from eval_model import eval_rl_ac

    td3_policy = TD3Policy(args, obs_space, act_space)
    td3_policy.load_param(args.env, args.save_time)
    eval_rl_ac(args, env, td3_policy)

def test_sac(args, obs_space, act_space):
    from algorithm.sac import SACPolicy
    from eval_model import eval_rl_ac

    sac_policy = SACPolicy(args, obs_space, act_space)
    sac_policy.load_param(args.env, args.save_time)
    eval_rl_ac(args, env, sac_policy)

def test_dqn(args, obs_space, act_space):
    from algorithm.dqn import DQNPolicy
    from eval_model import eval_dqn

    ddpg_policy = DQNPolicy(args, obs_space, act_space)
    ddpg_policy.load_param(args.env, args.save_time)
    eval_dqn(args, env, ddpg_policy)

def test_double_dqn(args, obs_space, act_space):
    from algorithm.double_dqn import DoubleDQNPolicy
    from eval_model import eval_dqn

    double_dqn_policy = DoubleDQNPolicy(args, obs_space, act_space)
    double_dqn_policy.load_param(args.env, args.save_time)
    eval_dqn(args, env, double_dqn_policy)

def test_dueling_dqn(args, obs_space, act_space):
    from algorithm.dueling_dqn import DuelingDQNPolicy
    from eval_model import eval_dqn

    dueling_dqn_policy = DuelingDQNPolicy(args, obs_space, act_space)
    dueling_dqn_policy.load_param(args.env, args.save_time)
    eval_dqn(args, env, dueling_dqn_policy)


# get train config
ppo_parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)
ppo_parser = get_ppo_config(ppo_parser)

ddpg_parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)
ddpg_parser = get_ddpg_config(ddpg_parser)

td3_parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)
td3_parser = get_td3_config(td3_parser)

sac_parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)
sac_parser = get_sac_config(sac_parser)

dqn_parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)
dqn_parser = get_dqn_config(dqn_parser)

double_dqn_parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)
double_dqn_parser = get_dqn_config(double_dqn_parser)

dueling_dqn_parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)
dueling_dqn_parser = get_dqn_config(dueling_dqn_parser)

test_parser = argparse.ArgumentParser(
    description='ALL Congfig in Testing Environment')
test_parser.add_argument('--env', type=str, required=True,
                         help='environment used for testing')
test_parser.add_argument('--save-time', type=str,
                         required=True, help='time to save the model')
test_parser.add_argument('--num-episode', type=int, default='100',
                         help='number of episodes in testing environment')

subparsers = test_parser.add_subparsers(
    help='choose one algorithm to start corresponding testing function', dest='algorithm')

subparsers.add_parser('ppo', help='ppo algorithm', parents=[ppo_parser])
ppo_parser.set_defaults(func=test_ppo)

subparsers.add_parser('ddpg', help='ddpg algorithm', parents=[ddpg_parser])
ddpg_parser.set_defaults(func=test_ddpg)

subparsers.add_parser('td3', help='td3 algorithm', parents=[td3_parser])
ddpg_parser.set_defaults(func=test_td3)

subparsers.add_parser('sac', help='sac algorithm', parents=[sac_parser])
ddpg_parser.set_defaults(func=test_sac)

subparsers.add_parser('dqn', help='dqn algorithm', parents=[dqn_parser])
dqn_parser.set_defaults(func=test_dqn)

subparsers.add_parser('doubledqn', help='double dqn algorithm', parents=[double_dqn_parser])
double_dqn_parser.set_defaults(func=test_double_dqn)

subparsers.add_parser('duelingdqn', help='dueling dqn algorithm', parents=[dueling_dqn_parser])
dueling_dqn_parser.set_defaults(func=test_dueling_dqn)


# test_args = test_parser.parse_args(['--env', 'LunarLanderContinuous-v2',
#                                     '--save-time', '01_22_11_16', 'ddpg', '--use-batch-normalization'])

# test_args = test_parser.parse_args(['--env', 'LunarLander-v2',
#                                     '--save-time', '01_21_21_12', 'ppo', '--use-recurrent-layer'])


test_args = test_parser.parse_args()


# test model
env = gym.make(test_args.env)
obs_space = env.observation_space
act_sapce = env.action_space
if test_args.algorithm == 'ppo':
    test_ppo(test_args, obs_space, act_sapce)
elif test_args.algorithm == 'ddpg':
    test_ddpg(test_args, obs_space, act_sapce)
elif test_args.algorithm == 'td3':
    test_td3(test_args, obs_space, act_sapce)
elif test_args.algorithm == 'sac':
    test_sac(test_args, obs_space, act_sapce)
elif test_args.algorithm == 'dqn':
    test_dqn(test_args, obs_space, act_sapce)
elif test_args.algorithm == 'doubledqn':
    test_double_dqn(test_args, obs_space, act_sapce)
elif test_args.algorithm == 'duelingdqn':
    test_dueling_dqn(test_args, obs_space, act_sapce)
else:
    raise NotImplementedError
