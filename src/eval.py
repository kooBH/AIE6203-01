import argparse
import os

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from models.dqn import DQN
from env import make_env
import eps

from utils import play_and_record, compute_td_loss, evaluate
from utils import ReplayBuffer

from ptUtils.hparams import HParam
from ptUtils.writer import MyWriter

if __name__ == "__main__" : 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,
                        help="yaml for configuration")
    parser.add_argument('--default', type=str, default=None,
                        help="default configuration")
    parser.add_argument('--version_name', '-v', type=str, required=True,
                        help="version of current training")
    parser.add_argument('--chkpt',type=str,required=True)
    parser.add_argument('--device','-d',type=str,required=False,default="cuda:0")

    args = parser.parse_args()

    hp = HParam(args.config,args.default)
    print("Config :: {} Based on {}".format(args.config,args.default))
    device =  args.device
    version = args.version_name
    torch.cuda.set_device(device)

    ## ENV
    ENV_NAME = "ALE/KungFuMaster-v5"
    env = make_env(ENV_NAME,clip_rewards=True)
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n

    ## Model
    if hp.train.type == "DQN" : 
        agent = DQN(state_shape=state_shape, n_actions=n_actions,dueling=hp.model.dueling).to(device)
    elif hp.train.type == "policy_gradient" : 
        agent = DQN(state_shape=state_shape, n_actions=n_actions,policy=True).to(device)
    else :
        exit(-1)

    if args.chkpt:
        agent.load_state_dict(torch.load(args.chkpt))
        print("LOAD::{}".format(args.chkpt))
    agent.eval()

    best_rw = 0

    n_eval = 10

    with torch.no_grad() : 
        sum_rw = 0
        for i in range(n_eval) : 
            rw = evaluate(make_env(ENV_NAME,clip_rewards=False), agent, greedy=True)
            sum_rw += rw
            if rw > best_rw : 
                best_rw = rw

        mean_rw = sum_rw / n_eval
        print("Eval {} for {} times | best {} | avg {}".format(version,n_eval,best_rw,mean_rw))