import argparse
import os
import re

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
    parser.add_argument('--chkpt',type=str,required=False,default=None)
    parser.add_argument('--step','-s',type=int,required=False,default=0)
    parser.add_argument('--device','-d',type=str,required=False,default="cuda:0")

    args = parser.parse_args()

    hp = HParam(args.config,args.default)
    device =  args.device
    version = args.version_name
    torch.cuda.set_device(device)

    batch_size = hp.train.batch_size
    total_steps = hp.train.total_steps
    timesteps_per_epoch = 4

    print("total_steps : {}".format(total_steps))

    log_dir = os.path.join(hp.log.root,"log",version)

    checkpoint_path = os.path.join(hp.log.root,"chkpt",version)
    os.makedirs(checkpoint_path,exist_ok = True)

    summary_interval = hp.train.summary_interval
    refresh_target_network_freq = hp.train.update_interval
    validation_interval = hp.train.validation_interval

    max_grad_norm = hp.train.max_grad
    replay_size = hp.train.replay_size
    gamma = hp.train.gamma

    writer = MyWriter(hp,log_dir)

    ## ENV
    ENV_NAME = "ALE/KungFuMaster-v5"
    env = make_env(ENV_NAME,clip_rewards=True)
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n

    step = 0
    state = env.reset()

    ## Model
    agent = DQN(state_shape=state_shape, n_actions=n_actions).to(device)
    target_network = DQN(state_shape=state_shape, n_actions=n_actions).to(device)

    if args.chkpt:
        agent.load_state_dict(torch.load(args.chkpt))
        target_network.load_state_dict(torch.load(args.chkpt))
 
    step = args.step

    if hp.eps.type == "linear" : 
        agent.epsilon = eps.linear_decay(hp,step)
    elif hp.eps.type == "fixed" :
        agent.epsilon = eps.fixed(hp)


    ### Replay Buffer

    play_steps = int(10e2)
    exp_replay = ReplayBuffer(replay_size, False)
    for i in tqdm(range(100)):
        play_and_record(state, agent, env, exp_replay, n_steps=play_steps)

    optim = torch.optim.Adam(agent.parameters(), lr=hp.train.adam)

    print("Experience Reply buffer : {}".format(len(exp_replay)))
    double_dqn = hp.train.double_dqn

    score = evaluate(make_env(ENV_NAME,clip_rewards=False), agent, greedy=True)
    print("Score without training: {}".format(score))

    env.reset()

    best_rw = 0

    ## Train

    for step in trange(step, total_steps + 1):
        if hp.eps.type == "linear" : 
            agent.epsilon = eps.linear_decay(hp, step)

        # play
        _, state = play_and_record(state, agent, env, exp_replay, timesteps_per_epoch)

        states_bn, actions_bn, rewards_bn, next_states_bn, is_done_bn, is_weight = exp_replay.sample(batch_size)
        optim.zero_grad()

        loss, error = compute_td_loss(states_bn, actions_bn, rewards_bn, next_states_bn, is_done_bn,
                                      agent, target_network, is_weight,
                                      gamma=gamma,
                                      check_shapes=False,
                                      device=device,
                                      double_dqn=double_dqn)

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
        optim.step()
        exp_replay.update_priority(error)

        if step % summary_interval == 0:
            td_loss = loss.data.cpu().item()
            grad_norm = grad_norm

            assert not np.isnan(td_loss)
            writer.add_scalar("Training/TD loss history", td_loss, step)
            writer.add_scalar("Training/Grad norm history", grad_norm, step)

        if step % refresh_target_network_freq == 0:
            target_network.load_state_dict(agent.state_dict())
            torch.save(agent.state_dict(), os.path.join(checkpoint_path, "agent_{}.pth".format(step)))

        if step % validation_interval == 0:
            mean_rw = evaluate(make_env(ENV_NAME,clip_rewards=False), agent, greedy=True)

            initial_state_q_values = agent.get_qvalues(
                make_env(ENV_NAME,seed=step).reset()
            )
            initial_state_v = np.max(initial_state_q_values)

            print("buffer size = %i, epsilon = %.5f, mean_rw=%.2f, initial_state_v= %.2f" % (
            len(exp_replay), agent.epsilon, mean_rw, initial_state_v))

            writer.add_scalar("Eval/Mean reward per life", mean_rw, step)
            writer.add_scalar("Eval/Initial state V", initial_state_v, step)
            writer.close()

            if mean_rw > best_rw :
                best_rw = mean_rw
                torch.save(agent.state_dict(), os.path.join(checkpoint_path, "bestmodel.pt"))

    torch.save(agent.state_dict(), os.path.join(checkpoint_path, "lastmodel.pt"))