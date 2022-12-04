from hparams import HParam
from common import *
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from itertools import count
from tqdm.auto import tqdm

from DQN import Qnet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,
                        help="yaml for configuration")
    parser.add_argument('--default', '-f', type=str, required=True,
                        help="default yaml")
    parser.add_argument('--version_name', '-v', type=str, required=True,
                        help="version of current training")
    parser.add_argument('--chkpt',type=str,required=False,default=None)
    parser.add_argument('--step','-s',type=int,required=False,default=0)
    parser.add_argument('--device','-d',type=str,required=False,default="cuda:0")
    args = parser.parse_args()

    hp = HParam(args.config,args.default)
    print("NOTE::Loading configuration : "+args.config)

    device = args.device
    version = args.version_name
    torch.cuda.set_device(device)

    env = KungFuMaster()
    d_in  = np.prod(env.shape)
    n_action = env.n_action

    print("d_in : {}".format(d_in))
    print("n_action : {}".format(n_action))

    #activation="GELU"
    activation="ReLU"

    # param to change
    BATCH_SIZE = 256
    GAMMA = 0.99
    TARGET_UPDATE = 100

    interval_log = 200

    memory = ReplayMemory(100000)

    # Huber Loss
    criterion = nn.SmoothL1Loss()

    # Model
    policy_net = Qnet(d_in,n_action,activation=activation).to(device)
    policy_net.train()
    target_net = Qnet(d_in,n_action,activation=activation).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    #optimizer = optim.RMSprop(policy_net.parameters())
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)

    # https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    # https://en.wikipedia.org/wiki/Huber_loss
    # To minimise this error, we will use the Huber loss. The Huber loss acts like the mean squared error when the error is small, but like the mean absolute error when the error is large - this makes it more robust to outliers when the estimates of QQ are very noisy.


    ### Epsilon functions
    eps_func = epsilon_greedy(hp.eps)

    step = 0

    num_episodes = 2000
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        env.reset()
        state_last= env.state.to(device)
        state = env.state.to(device)
        reward = torch.tensor([0],device=device)

        for t in count():
            ## Render States

            # Select and perform an action
            action = policy_net.sample(state,step,eps_func)
            step +=1
            _, reward, done, _, _ = env.step(action)
            action = action.to(device)
            reward = torch.tensor([reward], device=device)

            # Observe new state
            state_last = state
            state = env.state.to(device)
            if not done:
                next_state = state
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state_last, action, state, reward)

            # Perform one step of the optimization (on the policy network)
            if len(memory) >= BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
                # detailed explanation). This converts batch-array of Transitions
                # to Transition of batch-arrays.
                batch = Transition(*zip(*transitions))

                # Compute a mask of non-final states and concatenate the batch elements
                # (a final state would've been the one after which simulation ended)
                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)), device=device, dtype=torch.bool)
                non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
                state_batch = torch.cat(batch.state)
                action_batch = torch.cat(batch.action)
                reward_batch = torch.cat(batch.reward)

                #non_final_next_states = torch.flatten(non_final_next_states,start_dim=1)
                #state_batch = torch.flatten(state_batch,start_dim=1)

                # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
                # columns of actions taken. These are the actions which would've been taken
                # for each batch state according to policy_net



                # Compute V(s_{t+1}) for all next states.
                # Expected values of actions for non_final_next_states are computed based
                # on the "older" target_net; selecting their best reward with max(1)[0].
                # This is merged based on the mask, such that we'll have either the expected
                # state value or 0 in case the state was final.

                q_a = policy_net(state_batch).gather(1,action_batch.type(torch.int64))[:,0]
                max_q_prime = torch.zeros(BATCH_SIZE, device=device)
                # DQN
                #max_q_prime[non_final_mask] = target_net(non_final_next_states).max(1)[0]

                # Double DQN
                argmax_Q = policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
                max_q_prime[non_final_mask] = target_net(non_final_next_states).gather(1,argmax_Q)[:,0]

                # Compute the expected Q values
                target = (max_q_prime * GAMMA) + torch.max(reward_batch)


                # Compute Huber loss
                loss = criterion(q_a, target)

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                for param in policy_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()

                if t%interval_log == 0 :
                    #print("{} {}".format(q_a,target))
                    mean_reward =  torch.mean(reward_batch).item()
                    max_reward = torch.max(reward_batch).item()
                    print("{:3d} | {:5d} | {:4d} | {:2.3f} | {:2.0f} | {:.3E}".format(i_episode,step,t,mean_reward,max_reward,loss.item()))

            if done:
                break

            # Update the target network, copying all weights and biases in DQN
            if t % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())