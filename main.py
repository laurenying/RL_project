import sys
import argparse
import gym
import numpy as np
from KukaEnv_10703 import KukaVariedObjectEnv as k_env
from canonical_plot import plot
from DDPG import DDPG


def parse_arguments():
    parser = argparse.ArgumentParser(description='DDPG Argument Parser')
    parser.add_argument('--state', dest='state_size', type=int, default=18)
    parser.add_argument('--action', dest='action_size', type=int, default=3)
    parser.add_argument('--range', dest='action_range', type=int, default=1)
    parser.add_argument('--aun', dest='actor_un', type=int, default=64)
    parser.add_argument('--alr', dest='actor_lr', type=float, default=1e-4)
    parser.add_argument('--cun', dest='critic_un', type=int, default=64)
    parser.add_argument('--clr', dest='critic_lr', type=float, default=1e-3)
    parser.add_argument('--mu', dest='mu', type=float, default=0)
    parser.add_argument('--theta', dest='theta', type=float, default=0.1)
    parser.add_argument('--sigma', dest='sigma', type=float, default=0.1)
    parser.add_argument('--buffer', dest='buffer_size', type=int, default=30000)
    parser.add_argument('--batch', dest='batch_size', type=int, default=32)
    parser.add_argument('--gamma', dest='gamma', type=float, default=1)
    parser.add_argument('--tau', dest='tau', type=float, default=1e-3)
    parser.add_argument('--num-episode', dest='training_episode', type=int, default=5e4)
    parser.add_argument('--log', dest='log', type=str,
                        default="/Users/zhengluo/Google Drive/10703 DRL/project/anding/ddpg_v2/ddpg/log/log.csv")
    parser.add_argument('--act-save', dest='actor_save', type=str,
                        default="/Users/zhengluo/Google Drive/10703 DRL/project/anding/ddpg_v2/ddpg/actor/actor.h5")
    parser.add_argument('--cri-save', dest='critic_save', type=str,
                        default="/Users/zhengluo/Google Drive/10703 DRL/project/anding/ddpg_v2/ddpg/critic/critic.h5")

    return parser.parse_args()


def main(args):
    # disable warning
    gym.logger.set_level(40)

    args = parse_arguments()

    # PATH="/Your/Own/Path/to/items"
    env = k_env("/Users/zhengluo/Google Drive/10703 DRL/project/10703-Manipulation-Env/items")
    # Windows path: "D:\Python Project\DeepRL\manipulation\items"
    # Ubuntu path: "/home/ubuntu/ddpg/items"
    # MacOS path: "/Users/dingan/Downloads/10703-Manipulation-Env-master/items"

    task = {
        "state_size": args.state_size,
        "action_size": args.action_size,
        "action_range": args.action_range,
        # "action_low": np.array([-2, -2, -2, -4]), # {dX: [-2, 2], dY: [-2, 2], dZ: [-2, 0], dA: [-4, 4]}
        # "action_high": np.array([2, 2, 0, 4]), 
        "action_low": np.array([-1, -1, -1]),
        "action_high": np.array([1, 1, 1]),
    }

    actor_setting = {
        "num_hunits": args.actor_un,
        "lr": args.actor_lr,
    }
    critic_setting = {
        "num_hunits": args.critic_un,
        "lr": args.critic_lr,
    }
    noise = {
        "mu": args.mu,
        "theta": args.theta,
        "sigma": args.sigma,
    }
    memory = {
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
    }

    gamma = args.gamma
    tau = args.tau
    num_training = args.training_episode

    log_path = args.log
    actor_save_path = args.actor_save
    critic_save_path = args.critic_save

    agent = DDPG(
        env,  # learning environment
        task,  # environment parameters
        actor_setting,  # actor network parameters
        critic_setting,  # actor network parameters
        noise,  # noise parameters
        memory,  # memory parameters
        gamma,  # discount factor
        tau,  # soft update parameter
        num_training,  # number of training episodes
        log_path,
        actor_save_path,  # actor model save path
        critic_save_path  # critic model save path
    )

    rewards_list = agent.train()
    plot("DDPG_HER", rewards_list)


if __name__ == "__main__":
    main(sys.argv)
