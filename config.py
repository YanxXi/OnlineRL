import argparse

def get_train_ppo_config(parser: argparse.ArgumentParser):
    # get algorithm config
    parser = get_ppo_config(parser)
    parser = get_base_trainer_config(parser)
    parser = get_ppo_trainer_config(parser)

    return parser


def get_train_ddpg_config(parser: argparse.ArgumentParser):
    # get algorithm config
    parser = get_ddpg_config(parser)
    parser = get_base_trainer_config(parser)
    parser = get_ddpg_trainer_config(parser)

    return parser


def get_train_td3_config(parser: argparse.ArgumentParser):
    # get algorithm config
    parser = get_td3_config(parser)
    parser = get_base_trainer_config(parser)
    parser = get_td3_trainer_config(parser)

    return parser


def get_train_sac_config(parser: argparse.ArgumentParser):
    # get algorithm config
    parser = get_sac_config(parser)
    parser = get_base_trainer_config(parser)
    parser = get_sac_trainer_config(parser)

    return parser


def get_train_dqn_config(parser: argparse.ArgumentParser):
    # get algorithm config
    parser = get_dqn_config(parser)
    parser = get_base_trainer_config(parser)
    parser = get_dqn_trainer_config(parser)

    return parser

def get_train_double_dqn_config(parser: argparse.ArgumentParser):
    # get algorithm config
    parser = get_dqn_config(parser)
    parser = get_base_trainer_config(parser)
    parser = get_dqn_trainer_config(parser)

    return parser

def get_train_dueling_dqn_config(parser: argparse.ArgumentParser):
    # get algorithm config
    parser = get_dqn_config(parser)
    parser = get_base_trainer_config(parser)
    parser = get_dqn_trainer_config(parser)

    return parser


def get_ppo_config(parser: argparse.ArgumentParser):

    group = parser.add_argument_group(description='PPO Alogorithm Config')
    group.add_argument('--lr', type=float, default=5e-4,
                       help='learning rate')
    group.add_argument('--mlp-hidden-size', type=str, default='128',
                       help='hidden size of mlp layer to extract observation feature')
    group.add_argument('--act-hidden-size', type=str,
                       default='128', help='hidden size of mlp layer in action module')
    group.add_argument('--value-hidden-size', type=str,
                       default='128', help='hidden size of mlp layer in value module')
    group.add_argument('--use-recurrent-layer', action='store_true',
                       default=False, help='whether to use recurrent layer')
    group.add_argument('--recurrent-hidden-size',
                       type=int, default=128, help='hidden size of GRU')
    group.add_argument('--recurrent-hidden-layers', type=int,
                       default=1, help='number of hidden layers of GRU')

    return parser


def get_ddpg_config(parser: argparse.ArgumentParser):

    group = parser.add_argument_group(description='DDPG Alogorithm Config')
    group.add_argument('--actor-lr', type=float, default=1e-4,
                       help='learning rate of actor network (default 1e-4)')
    group.add_argument('--critic-lr', type=float, default=1e-3,
                       help='learning rate of critic network (default 1e-3)')
    group.add_argument('--tau', type=float, default=5e-3,
                       help='parameter used for updating target network (default 5e-3)')
    group.add_argument('--epsilon', type=float, default=1.0,
                       help='explore noise of action (default 1.0)')
    group.add_argument('--explore_noise', type=float, default=0.2,
                       help='explore noise of action (default 0.2)')
    group.add_argument('--act-hidden-size', type=str, default='400 200',
                       help='hidden size in actor network (default 400 200)')
    group.add_argument('--value-hidden-size', type=str, default='400 200',
                       help='hidden size in critic network (default 400 200)')
    group.add_argument("--use-batch-normalization", action='store_true', default=False,
                       help="Whether to apply Batch Normalization to the feature extraction inputs")
    return parser


def get_td3_config(parser: argparse.ArgumentParser):

    group = parser.add_argument_group(description='TD3 Alogorithm Config')
    group.add_argument('--actor-lr', type=float, default=1e-4,
                       help='learning rate of actor network ')
    group.add_argument('--critic-lr', type=float, default=1e-4,
                       help='learning rate of critic network')
    group.add_argument('--tau', type=float, default=5e-3,
                       help='parameter used for updating target network (default 5e-3)')
    group.add_argument('--explore-noise', type=float, default=0.1,
                       help='explore noise of action (default 0.1)')
    group.add_argument('--act-hidden-size', type=str, default='200 200',
                       help='hidden size in actor network')
    group.add_argument('--value-hidden-size', type=str, default='200 200',
                       help='hidden size in critic network')
    group.add_argument("--use-batch-normalization", action='store_true', default=False,
                       help="Whether to apply Batch Normalization to the feature extraction inputs")
    return parser


def get_sac_config(parser: argparse.ArgumentParser):

    group = parser.add_argument_group(description='SAC Alogorithm Config')
    group.add_argument('--lr', type=float, default=1e-4,
                       help='learning rate of critic network')
    group.add_argument('--tau', type=float, default=5e-3,
                       help='parameter used for updating target network (default 5e-3)')
    group.add_argument('--act-hidden-size', type=str, default='256 256',
                       help='hidden size in actor network')
    group.add_argument('--value-hidden-size', type=str, default='256 256',
                       help='hidden size in critic network')
    group.add_argument("--use-batch-normalization", action='store_true', default=False,
                       help="Whether to apply Batch Normalization to the feature extraction inputs")
    group.add_argument("--init-alpha", type=float, default=0.2,
                       help="temperature parameters")
                    
    return parser

def get_dqn_config(parser: argparse.ArgumentParser):

    group = parser.add_argument_group(description='DQN Alogorithm Config')
    group.add_argument('--lr', type=float, default=5e-3,
                       help='learning rate of Q network (default 1e-3)')
    group.add_argument('--tau', type=float, default=5e-3,
                       help='parameter used for updating target network (default 5e-3)')
    group.add_argument('--epsilon', type=float, default=0.9,
                       help='explore noise of action (default 0.9)')
    group.add_argument('--hidden-size', type=str, default='100 100',
                       help='hidden size in Q network')
    group.add_argument("--use-batch-normalization", action='store_true', default=False,
                       help="Whether to apply Batch Normalization to the feature extraction inputs")
    return parser


def get_base_trainer_config(parser: argparse.ArgumentParser):

    group = parser.add_argument_group(description='Base Trainer Config')
    group.add_argument('--gamma', type=float, default=0.99,
                       help='discount factor')
    group.add_argument('--gae-lambda', type=float, default=0.95,
                       help='factor used for GAE')
    group.add_argument('--num-episode', type=int, default=1e3,
                       help='number of episodes')
    group.add_argument('--epochs', type=int, default=1e2,
                       help='number of epochs, we test the policy per epoch')
    group.add_argument('--steps-per-epoch', type=int, default=5e3,
                       help='the interval between per epoch')
    group.add_argument('--max-episode-steps', type=int, default=None,
                       help='the maximum steps for one episode')
    group.add_argument("--max-grad-norm", type=float, default=4,
                       help='restrict the grad norm to max_grad_norm to prevent gradient explosion')
    group.add_argument("--eval-interval", type=float, default=10,
                       help='evaluate policy each eval_interval episode')
    group.add_argument("--eval-episodes", type=float, default=100,
                       help='episodes for evaluation')
    group.add_argument("--start-eval", type=int, default=0,
                       help='only enter evaluation mode if current episode excce this number (default 0)')
    group.add_argument("--eval-in-paral", action='store_true', default=False,
                       help='evaluate the policy in parallel')
    group.add_argument("--log-info", action='store_true', default=False,
                       help='log trainning infomation')
    group.add_argument("--save-model", action='store_true', default=False,
                       help='save model in the trainning')

    return parser


def get_ppo_trainer_config(parser: argparse.ArgumentParser):

    group = parser.add_argument_group(description='PPO Trainer Config')

    group.add_argument("--clip-param", type=float, default=0.2,
                       help='ppo clip parameter (default: 0.2)')
    group.add_argument("--value-loss-coef", type=float, default=1,
                       help='ppo value loss coefficient (default: 1)')
    group.add_argument("--entropy-coef", type=float, default=0.01,
                       help='entropy term coefficient (default: 0.01)')
    group.add_argument("--ppo-epoch", type=int, default=10,
                       help='the number of process of training all samples')
    group.add_argument("--buffer-size", type=int, default=1024,
                       help='minimum buffer size to complete one training process')
    group.add_argument("--batch-size", type=int, default=64,
                       help='the number of samples to complete one update')
    group.add_argument("--chunk-length", type=int, default=10,
                       help='the length of chunk data')
    return parser


def get_ddpg_trainer_config(parser: argparse.ArgumentParser):

    group = parser.add_argument_group(description='DDPG Trainer Config')

    group.add_argument("--use-per", action='store_true', default=False,
                       help='whether to use prioritized experience replay buffer')
    group.add_argument("--buffer-size", type=int, default=2 ** 20,
                       help='minimum buffer size to complete one training process')
    group.add_argument("--batch-size", type=int, default=2 ** 6,
                       help='the number of samples to complete one update')
    return parser


def get_td3_trainer_config(parser: argparse.ArgumentParser):

    group = parser.add_argument_group(description='TD3 Trainer Config')

    group.add_argument("--buffer-size", type=int, default=2 ** 20,
                       help='minimum buffer size to complete one training process')
    group.add_argument("--batch-size", type=int, default=2 ** 7,
                       help='the number of samples to complete one update')
    group.add_argument("--delay-frequency", type=int, default=2,
                       help="update actor and target critic network every d iterations (default=2)")
    group.add_argument("--policy-noise", type=float, default=0.2,
                       help="the noise added to the actions for smoothing target policy (default=0.2)")
    group.add_argument("--policy-clip", type=float, default=0.5,
                       help="the target policy is clipped to this range (default=0.5)")

    return parser


def get_sac_trainer_config(parser: argparse.ArgumentParser):

    group = parser.add_argument_group(description='SAC Trainer Config')

    group.add_argument("--buffer-size", type=int, default=2 ** 20,
                       help='minimum buffer size to complete one training process')
    group.add_argument("--batch-size", type=int, default=2 ** 7,
                       help='the number of samples to complete one update')
    return parser


def get_dqn_trainer_config(parser: argparse.ArgumentParser):

    group = parser.add_argument_group(description='DQN Trainer Config')

    group.add_argument("--use-per", action='store_true', default=False,
                       help='whether to use prioritized experience replay buffer')
    group.add_argument("--buffer-size", type=int, default=2 ** 20,
                       help='minimum buffer size to complete one training process')
    group.add_argument("--batch-size", type=int, default=2 ** 7,
                       help='the number of samples to complete one update')
    return parser


