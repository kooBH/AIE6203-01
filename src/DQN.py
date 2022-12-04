from common import *

import random
import torch
import torch.nn as nn
import numpy as np



class Qnet(nn.Module):
    def __init__(self,d_in,d_out,activation="ReLU",dueling=True):
        super(Qnet, self).__init__()

        self.d_in  = d_in
        self.d_out = d_out
        self.dueling = dueling

        self.c1 = nn.Sequential(
            nn.Conv2d(1, 32, 4, 3),
            nn.ReLU(),
            nn.BatchNorm2d(32))
        self.c2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2),
            nn.ReLU(),
            nn.BatchNorm2d(64))
        self.c3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2),
            nn.ReLU(),
            nn.BatchNorm2d(128))

        self.fc1 = nn.Linear(7680, 256,bias=True)
        self.fc2 = nn.Linear(256, 128,bias=True)
        self.fc3 = nn.Linear(128, d_out,bias=True)

        if self.dueling : 
            self.v_fc1 = nn.Linear(7680,256,bias=True)
            self.v_fc2 = nn.Linear(256,1,bias=True)

        if activation == "ReLU": 
            self.activation = nn.ReLU()
        elif activation == "GELU" :
            self.activation = nn.GELU()
        else :
            raise Exception("ERROR::Unknown activation : {}".format(activation))

    # [B,H,W]
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = torch.flatten(x,start_dim=1)
        a = self.activation(self.fc1(x))
        a = self.activation(self.fc2(a))
        a = self.fc3(a)

        if self.dueling : 
            a_avg = torch.mean(a)
            v = self.v_fc1(x)
            v = self.activation(v)
            v = self.v_fc2(v)
            q = v + a - a_avg
        else :
            q = a
        return q
      
    def sample(self, state, step, eps_func):
        #decaying EPS

        eps_threshold = eps_func(step)

        coin = random.random()
        if coin < eps_threshold:
            return torch.tensor(random.randint(0,self.d_out-1),dtype=torch.float32).view(1,1)
        else : 
            #state = torch.flatten(state,start_dim=1)
            return self.forward(state).max(1)[1].view(1,1)
