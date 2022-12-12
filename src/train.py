import argparse
import os
import re

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from tqdm import trange
from tqdm.auto import tqdm

from models.dqn import DQN
from env import make_env
import eps

from utils import play_and_record, compute_td_loss, evaluate
from utils import ReplayBuffer

from ptUtils.hparams import HParam
from ptUtils.writer import MyWriter

if __name__ == "__main__" : 
############################ Configuration ##############################
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
    print("Config :: {} Based on {}".format(args.config,args.default))
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

    if hp.loss == "smooth_l1" : 
        criterion = F.smooth_l1_loss
    elif hp.loss == "huber" : 
        criterion = F.huber_loss
    else : 
        criterion = F.smooth_l1_loss


    writer = MyWriter(hp,log_dir)

    ## ENV
    ENV_NAME = "ALE/KungFuMaster-v5"
    env = make_env(ENV_NAME,clip_rewards=True)
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n

    step = 0
    state = env.reset()

############################ INIT::DQN ##############################
    if hp.train.type == "DQN" :
        print("DQN")
        agent = DQN(state_shape=state_shape, n_actions=n_actions,dueling=hp.model.dueling).to(device)
        target_network = DQN(state_shape=state_shape, n_actions=n_actions,dueling=hp.model.dueling).to(device)

        step = args.step

        if hp.eps.type == "linear" : 
            agent.epsilon = eps.linear_decay(hp,step)
        elif hp.eps.type == "fixed" :
            agent.epsilon = eps.fixed(hp)
        elif hp.eps.type == "linear_annealing" :
            agent.epsilon = eps.linear_annealing(hp,step)
        else : 
            raise Exception("ERROR::Unknown EPS : {}".format(hp.eps.type))

        ### Replay Buffer

        play_steps = int(10e2)
        exp_replay = ReplayBuffer(replay_size, priority_replay =  hp.train.priority_replay, alpha = hp.train.priority.alpha, beta = hp.train.priority.beta)
        for i in tqdm(range(100)):
            play_and_record(state, agent, env, exp_replay, n_steps=play_steps)

        print("Experience Reply buffer : {}".format(len(exp_replay)))
        double_dqn = hp.train.double_dqn
        print("Double DQN::{}".format(double_dqn))

########################### INIT::POLICY GRADEINT ############################
    elif hp.train.type == "policy_gradient" :
        print("POLICY_GRADIENT")

        num_steps = hp.train.policy_gradient.n_step

        agent = DQN(state_shape=state_shape, n_actions=n_actions,policy=True).to(device)
        value_net = DQN(state_shape=state_shape, n_actions=n_actions,value=True).to(device)

        optim_value = torch.optim.Adam(value_net.parameters(), lr=hp.train.adam)

########################### INIT:: A2C ############################
    elif hp.train.type == "A2C" :
        print("A2C")

        agent = DQN(state_shape=state_shape, n_actions=n_actions,policy=True).to(device)
        value_net = DQN(state_shape=state_shape, n_actions=n_actions,value=True).to(device)
        target_value_net = DQN(state_shape=state_shape, n_actions=n_actions,value=True).to(device)
        
        optim_value = torch.optim.Adam(value_net.parameters(), lr=hp.train.adam)

        play_steps = int(10e2)
        exp_replay = ReplayBuffer(replay_size, priority_replay =  hp.train.priority_replay, alpha = hp.train.priority.alpha, beta = hp.train.priority.beta)
        for i in tqdm(range(100)):
            play_and_record(state, agent, env, exp_replay, n_steps=play_steps)


        print("Experience Reply buffer : {}".format(len(exp_replay)))
    else :
        raise Exception("ERROR::Unknown training method : {}".format(hp.train.type))

############################ INIT::COMMON SECTION ###################
    optim = torch.optim.Adam(agent.parameters(), lr=hp.train.adam)
    env.reset()

    best_rw = 0

    cnt_refersh = 0
    cnt_validation = 0

    stat_action = np.zeros(n_actions,np.int32)

    ## Train

    with tqdm(total=total_steps) as pbar:
        while step < total_steps :
########################## TRAIN::DQN ##############################
            if hp.train.type == "DQN" :
                if hp.eps.type == "linear" : 
                    agent.epsilon = eps.linear_decay(hp, step)
                elif hp.eps.type == "linear_annealing" :
                    agent.epsilon = eps.linear_annealing(hp,step)
                # play
                _, state = play_and_record(state, agent, env, exp_replay, timesteps_per_epoch)

                states_bn, actions_bn, rewards_bn, next_states_bn, is_done_bn, is_weight = exp_replay.sample(batch_size)

                loss, error = compute_td_loss(states_bn, actions_bn, rewards_bn, next_states_bn, is_done_bn,
                                            agent, target_network, is_weight,
                                            gamma=gamma,
                                            check_shapes=False,
                                            device=device,
                                            double_dqn=double_dqn,
                                            criterion=criterion
                                            )

                exp_replay.update_priority(error)
                step +=1
                pbar.update(1)
                cnt_refersh+=1
                cnt_validation+=1

                optim.zero_grad()
                loss.backward(retain_graph=True)
                grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optim.step()

                if cnt_refersh > refresh_target_network_freq == 0:
                    target_network.load_state_dict(agent.state_dict())

######################### TRAIN::POLICY GRADEINT ############################
            elif hp.train.type == "policy_gradient" :
                log_policies = []
                values = []
                rewards = []

                # collect short trajectory
                for _ in range(num_steps):
                    logits = agent([state])
                    value = value_net([state])

                    policy = F.softmax(logits, dim=1)
                    log_policy = F.log_softmax(logits, dim=1)
                    action = agent.sample_actions(logits)

                    state, reward, done, _, _ = env.step(action.squeeze())


                    stat_action[action.squeeze()]+=1

                    if done:
                        state = env.reset()

                    values.append(value)
                    log_policies.append(log_policy.gather(1, action))
                    #rewards.append(np.sign(reward))
                    rewards.append(reward)

                    step +=1
                    pbar.update(1)
                    cnt_validation+=1

                    if done:
                        break

                R = torch.zeros((1, 1), dtype=torch.float).to(device)
                #if not done:
                #    R = value_net([state])

                actor_loss = 0
                value_loss = 0
                next_value = R

                for value, log_policy, reward, in list(zip(values, log_policies, rewards))[::-1]:

                    R = R * gamma + reward
                    actor_loss = actor_loss + log_policy * ( R - value )
                    value_loss = value_loss + (value - R)**2

                # omitted 1 / n_traj term due to short trajectory
                loss = - actor_loss
                value_loss = value_loss / n_actions

                optim.zero_grad()
                loss.backward(retain_graph=True)
                grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optim.step()

                optim_value.zero_grad()
                value_loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(value_net.parameters(), max_grad_norm)
                optim_value.step()

                #print("step {} | R {:.2f} | policy {:.2f} | value {:.2f} | {}".format(step,R.item(),loss.item(),value_loss.item(),stat_action))
######################### TRAIN::Actor Critic ###########################
            elif hp.train.type == "A2C" : 
                _, state = play_and_record(state, agent, env, exp_replay, timesteps_per_epoch)

                states_bn, actions_bn, rewards_bn, next_states_bn, is_done_bn, is_weight = exp_replay.sample(batch_size)

                with torch.no_grad() : 
                    value_target = torch.tensor(rewards_bn).to(device) + gamma* (~torch.tensor(is_done_bn)).to(device)*value_net(next_states_bn)[:,0]
                    advantage = value_target - value_net(states_bn)[:,0]

                value_target = value_target.float()
                advantage = advantage.float()

                # policy network
                logits = agent(states_bn)
                log_policy = F.log_softmax(logits, dim=1)
                loss = -log_policy * torch.unsqueeze(advantage,dim=1)
                loss = loss.mean()
                optim.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optim.step()

                # value network
                value_loss = F.mse_loss(value_target,value_net(states_bn)) 
                optim_value.zero_grad()
                value_loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(value_net.parameters(), max_grad_norm)
                optim_value.step()

                step +=1
                pbar.update(1)
                cnt_refersh+=1
                cnt_validation+=1


                if cnt_refersh > refresh_target_network_freq == 0:
                    target_value_net.load_state_dict(value_net.state_dict())

                if step % 100 == 0 :
                    print("{} | A {:.3f} | V {:.3f} | R {:.3f} | v {:.3f} | a {:.3f} | {:.3f}".format(step,loss.item(), value_loss.item(), rewards_bn.mean().item(),value_target.mean().item(),advantage.mean().item(),log_policy.mean().item()))
######################### TRAIN ::COMMON SECTION ###################

            ### log & update
            if cnt_validation > validation_interval :
                mean_rw = evaluate(make_env(ENV_NAME,clip_rewards=False), agent, greedy=True)
                writer.add_scalar("Eval/Mean reward per life", mean_rw, step)
                print("{} | step {} | mean_rw {}".format(version,step,mean_rw))
                cnt_validation = 0

                if mean_rw > best_rw :
                    best_rw = mean_rw
                    torch.save(agent.state_dict(), os.path.join(checkpoint_path, "bestmodel.pt"))

    torch.save(agent.state_dict(), os.path.join(checkpoint_path, "lastmodel.pt"))
    writer.close()