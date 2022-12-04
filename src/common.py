
import gym
import numpy as np

import collections
import torch
import random
from collections import namedtuple,deque


"""
https://www.gymlibrary.dev/environments/atari/kung_fu_master/
"""
class KungFuMaster():
    def __init__(self):
        self.env = gym.make("ALE/KungFuMaster-v5",obs_type="grayscale")

        self.h_s=100
        self.h_e = 175

        self.state = torch.from_numpy(self.env.reset()[0][self.h_s:self.h_e,:]).type(torch.float32).unsqueeze(0)/255

        print(self.state.shape)

        self.shape = self.state.shape
        self.n_action = self.env.action_space.n

    def step(self,action):
        # state, reward, done, truncatedm, info
        s,r,d,t,i =  self.env.step(int(action))

        self.state = torch.from_numpy(s[self.h_s:self.h_e,:]).type(torch.float32).unsqueeze(0)/255

        return s,r/100,d,t,i

    def reset(self):
        self.state = torch.from_numpy(self.env.reset()[0][self.h_s:self.h_e,:]).type(torch.float32).unsqueeze(0)

# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class epsilon_greedy():
    def __init__(self,args):
        if args.type == "fixed" : 
            self.f = self.eps_fixed
            self.eps = args.fixed.eps
        elif args.type == "linear" : 
            self.f = self.eps_linear
            self.start = args.linear.start
            self.end = args.linear.end
            self.decay = args.linear.decay
        else :
            raise Exception("ERROR::unkonwn epsilon_method : {}".format(eps_type))


    def __call__(self,step) : 
        return self.f(step)
    
    def eps_fixed(self,step):
        return self.eps

    def eps_linear(self,step):
        return self.end + (self.start- self.end)*np.exp(-step/self.decay)






