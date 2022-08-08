import gym
from config import *
import argparse
# pip install Box2D


def train_ppo(args, obs_space, act_space):
    from algorithm.ppo import PPOPolicy
    from trainer.ppo_trainer import PPOTrainer
    # define policy, trainer
    ppo_policy = PPOPolicy(args, obs_space, act_space)
    ppo_trainer = PPOTrainer(env, ppo_policy, args)
    ppo_trainer.run()

def train_ddpg(args, obs_space, act_space):
    from algorithm.ddpg import DDPGPolicy
    from trainer.ddpg_trainer import DDPGTrainer
    # define policy, trainer
    ddpg_policy = DDPGPolicy(args, obs_space, act_space)
    ddpg_trainer = DDPGTrainer(env, ddpg_policy, args)
    ddpg_trainer.run()

def train_td3(args, obs_space, act_space):
    from algorithm.td3 import TD3Policy
    from trainer.td3_trainer import TD3Trainer
    # define policy, trainer
    td3_policy = TD3Policy(args, obs_space, act_space)
    td3_trainer = TD3Trainer(env, td3_policy, args)
    td3_trainer.run()

def train_sac(args, obs_space, act_space):
    from algorithm.sac import SACPolicy
    from trainer.sac_trainer import SACTrainer
    # define policy, trainer
    sac_policy = SACPolicy(args, obs_space, act_space)
    sac_trainer = SACTrainer(env, sac_policy, args)
    sac_trainer.run()

def train_dqn(args, obs_space, act_space):
    from algorithm.dqn import DQNPolicy
    from trainer.dqn_trainer import DQNTrainer
    # define policy, trainer
    dqn_policy = DQNPolicy(args, obs_space, act_space)
    dqn_trainer = DQNTrainer(env, dqn_policy, args)
    dqn_trainer.run()

def train_double_dqn(args, obs_space, act_space):
    from algorithm.double_dqn import DoubleDQNPolicy
    from trainer.double_dqn_trainer import DoubleDQNTrainer
    # define policy, trainer
    double_dqn_policy = DoubleDQNPolicy(args, obs_space, act_space)
    double_dqn_trainer = DoubleDQNTrainer(env, double_dqn_policy, args)
    double_dqn_trainer.run()

def train_dueling_dqn(args, obs_space, act_space):
    from algorithm.dueling_dqn import DuelingDQNPolicy
    from trainer.dueling_dqn_trainer import DuelingDQNTrainer 
    # define policy, trainer
    dueling_dqn_policy = DuelingDQNPolicy(args, obs_space, act_space)
    dueling_dqn_trainer = DuelingDQNTrainer(env, dueling_dqn_policy, args)
    dueling_dqn_trainer.run()


# get train config
ppo_parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)
ppo_parser = get_train_ppo_config(ppo_parser)

ddpg_parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)
ddpg_parser = get_train_ddpg_config(ddpg_parser)

td3_parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)
td3_parser = get_train_td3_config(td3_parser)

sac_parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)
sac_parser = get_train_sac_config(sac_parser)

dqn_parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)
dqn_parser = get_train_dqn_config(dqn_parser)

double_dqn_parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)
double_dqn_parser = get_train_double_dqn_config(double_dqn_parser)

dueling_dqn_parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)
dueling_dqn_parser = get_train_dueling_dqn_config(dueling_dqn_parser)

train_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                       description='ALL Congfig in Trainning Environment')

train_parser.add_argument('--env', type=str, default=' ',
                          help='Environment')
train_parser.add_argument('--solved-reward', type=float, default=None,
                          help='threshold used for judging whether the problem is sloved')
train_parser.add_argument('--ppo-help', action='store_true', default=False,
                          help='print all help information of ppo')
train_parser.add_argument('--ddpg-help', action='store_true', default=False,
                          help='print all help information of ddpg')
train_parser.add_argument('--td3-help', action='store_true', default=False,
                          help='print all help information of td3')
train_parser.add_argument('--sac-help', action='store_true', default=False,
                          help='print all help information of sac')
train_parser.add_argument('--dqn-help', action='store_true', default=False,
                          help='print all help information of dqn')
train_parser.add_argument('--double-dqn-help', action='store_true', default=False,
                          help='print all help information of double dqn')
train_parser.add_argument('--dueling-dqn-help', action='store_true', default=False,
                          help='print all help information of double dqn')

subparsers = train_parser.add_subparsers(
    help='choose one algorithm to start corresponding trainning function', dest='algorithm')

subparsers.add_parser('ppo', help='ppo algorithm', parents=[ppo_parser])
ppo_parser.set_defaults(func=train_ppo)

subparsers.add_parser('ddpg', help='ddpg algorithm', parents=[ddpg_parser])
ddpg_parser.set_defaults(func=train_ddpg)

subparsers.add_parser('td3', help='td3 algorithm', parents=[td3_parser])
td3_parser.set_defaults(func=train_td3)

subparsers.add_parser('sac', help='sac algorithm', parents=[sac_parser])
td3_parser.set_defaults(func=train_sac)

subparsers.add_parser('dqn', help='dqn algorithm', parents=[dqn_parser])
dqn_parser.set_defaults(func=train_dqn)

subparsers.add_parser(
    'doubledqn', help='double dqn algorithm', parents=[double_dqn_parser])
double_dqn_parser.set_defaults(func=train_double_dqn)

subparsers.add_parser(
    'duelingdqn', help='dueling dqn algorithm', parents=[dueling_dqn_parser])
dueling_dqn_parser.set_defaults(func=train_dueling_dqn)


# train_args = train_parser.parse_args(['--env', 'LunarLanderContinuous-v2', 'ddpg',
#                                      '--use-per', '--use-batch-normalization', '--log-info', '--eval-in-paral', '--save-model',
#                                       '--start-eval', '100'])

# train_args = train_parser.parse_args('--env BipedalWalker-v3 sac --max-episode-steps 1000 --save-model'.split(' '))

# train_args = train_parser.parse_args('--env CartPole-v1 duelingdqn --steps-per-epoch 500'.split(' '))


train_args = train_parser.parse_args()

print('\n', train_args)

if train_args.ppo_help:
    ppo_parser.print_help()
elif train_args.ddpg_help:
    ddpg_parser.print_help()
elif train_args.td3_help:
    td3_parser.print_help()
elif train_args.sac_help:
    sac_parser.print_help()
elif train_args.dqn_help:
    dqn_parser.print_help()
elif train_args.double_dqn_help:
    double_dqn_parser.print_help()
else:
    pass

# set sloved reward for existing environment
if 'CartPole' in train_args.env:
    train_args.solved_reward = 475
elif 'LunarLander' in train_args.env:
    train_args.solved_reward = 200
elif 'BipedalWalker' in train_args.env:
    train_args.solved_reward = 300
else:
    print("Please Input Solved Reward of this Environemnt")

# define environment
env = gym.make(train_args.env)
obs_space = env.observation_space
act_space = env.action_space
# train model
if train_args.algorithm == 'ppo':
    train_ppo(train_args, obs_space, act_space)
elif train_args.algorithm == 'ddpg':
    train_ddpg(train_args, obs_space, act_space)
elif train_args.algorithm == 'td3':
    train_td3(train_args, obs_space, act_space)
elif train_args.algorithm == 'sac':
    train_sac(train_args, obs_space, act_space)
elif train_args.algorithm == 'dqn':
    train_dqn(train_args, obs_space, act_space)
elif train_args.algorithm == 'doubledqn':
    train_double_dqn(train_args, obs_space, act_space)
elif train_args.algorithm == 'duelingdqn':
    train_dueling_dqn(train_args, obs_space, act_space)
else:
    raise NotImplementedError
