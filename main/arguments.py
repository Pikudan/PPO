import argparse

def str2bool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')
def get_args():
    """Argument parse"""

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',dest='mode',type=str, default='train')
    parser.add_argument('--upswing', dest='upswing', help='This is a boolean flag', type=str2bool, default=False)
    parser.add_argument('--target', dest='target', help='This is a boolean flag', type=str2bool, default=False)
    parser.add_argument('--extended_observation', dest='extended_observation', help='This is a boolean flag', type=str2bool, default=False)
    parser.add_argument('--mass_use', dest='mass_use', help='This is a boolean flag', type=str2bool, default=False)
    parser.add_argument('--mass', dest='mass', type=float, default=None)
    parser.add_argument('--policy_model', dest='policy_model', type=str, default=None)
    parser.add_argument('--value_model', dest='value_model', type=str, default=None)
    parser.add_argument('--num_observations', dest='num_observations', type=int, default=1)
    parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=16)
    parser.add_argument('--num_runner_steps', dest='num_runner_steps', type=int, default=2048)
    parser.add_argument('--gamma', dest='gamma', type=float, default=0.99)
    parser.add_argument('--lambda', dest='lambda_', type=float, default=0.95)
    parser.add_argument('--num_minibatches', dest='num_minibatches', type=int, default=64)
    args = parser.parse_args()
    return args
